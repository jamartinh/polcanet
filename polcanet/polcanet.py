import gc
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations, batched
from os.path import split
from random import random
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.core.display_functions import clear_output
from IPython.display import display
from absl.logging.converter import standard_to_cpp
from ipywidgets.widgets import Output
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from polcanet.aencoders import FastTensorDataLoader
from polcanet.utils import custom_float_format


class LossConflictAnalyzer:
    def __init__(self, loss_names: List[str] = None, rate=0.1, conflict_threshold: float = 0.1):
        """
        Initializes the LossConflictAnalyzer class.

        Args:
            loss_names (List[str], optional): List of loss names. Defaults to None.
            rate (float, optional): Rate of updates as a probability. Defaults to 0.1.
            conflict_threshold (float, optional): Threshold for conflict detection. Defaults to 0.1.
        """
        self.pairwise_similarities = {}
        self.pairwise_conflicts = {}
        self.pairwise_interactions = {}
        self.total_conflicts = 0
        self.total_interactions = 0
        self.loss_names = loss_names if loss_names else []
        # rate of updates as a probability <1 if a random number is less than this rate the analysis will be done
        # this is to limit the heavy computation of the analysis during training
        self.rate = rate
        self.conflict_threshold = conflict_threshold
        self.reset_statistics()
        self.enabled = True

    def reset_statistics(self):
        self.total_interactions = 0
        self.total_conflicts = 0
        self.pairwise_interactions = {}
        self.pairwise_conflicts = {}
        self.pairwise_similarities = {}

    def step(self, model, losses: List[torch.Tensor]) -> None:
        # if len(losses) != len(self.loss_names):
        #     raise ValueError("Number of losses must match number of loss names")

        # rate test
        if not self.enabled or np.random.rand() > self.rate:
            return

        grads = self.compute_grad(model, losses)
        self.analyze_conflicts(grads)

    @staticmethod
    def compute_grad(model, losses: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        grads = []
        for loss in losses:
            if loss == 0:
                grads.append(None)
                continue
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad_dict = {name: param.grad.clone() for name, param in model.named_parameters() if
                         param.grad is not None}
            grads.append(grad_dict)
        return grads

    def analyze_conflicts(self, grads: List[Dict[str, torch.Tensor]]) -> None:
        """
        Analyze conflicts between gradients of different losses.

        Args:
            grads (List[Dict[str, torch.Tensor]]): List of gradient dictionaries for each loss.
        """
        num_losses = len(grads)

        for (i, j) in combinations(range(num_losses), 2):
            conflict_key = self.get_conflict_key(i, j)
            g_i, g_j = grads[i], grads[j]

            if g_i is not None and g_j is not None:
                similarity = self.compute_similarity(g_i, g_j)

                self.pairwise_interactions[conflict_key] = self.pairwise_interactions.get(conflict_key, 0) + 1
                self.pairwise_similarities[conflict_key] = self.pairwise_similarities.get(conflict_key, []) + [
                    similarity]

                if similarity < -self.conflict_threshold:
                    self.total_conflicts += 1
                    self.pairwise_conflicts[conflict_key] = self.pairwise_conflicts.get(conflict_key, 0) + 1

        self.total_interactions += 1

    @staticmethod
    def compute_similarity(grad1: Dict[str, torch.Tensor], grad2: Dict[str, torch.Tensor]) -> float:
        """
        Compute the cosine similarity between two gradients.
        """
        dot_product = 0
        norm1 = 0
        norm2 = 0
        for name in grad1.keys():
            if name in grad2:
                g1, g2 = grad1[name], grad2[name]
                dot_product += torch.sum(g1 * g2).item()
                norm1 += torch.sum(g1 * g1).item()
                norm2 += torch.sum(g2 * g2).item()

        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))

    def get_conflict_key(self, i: int, j: int) -> Tuple[str, str]:
        return self.loss_names[i], self.loss_names[j]

    def report(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        overall_conflict_rate = self.total_conflicts / max(1, self.total_interactions)

        report_dict = {'total_interactions': self.total_interactions, 'total_conflicts': self.total_conflicts,
                       'overall_conflict_rate': overall_conflict_rate, 'pairwise_data': {}}

        df_data = []

        for (loss1, loss2), interactions in self.pairwise_interactions.items():
            conflicts = self.pairwise_conflicts.get((loss1, loss2), 0)
            similarities = self.pairwise_similarities[(loss1, loss2)]
            avg_similarity = np.mean(similarities)
            conflict_rate = conflicts / interactions if interactions > 0 else 0

            if avg_similarity > self.conflict_threshold:
                relationship = "Strongly Cooperative"
            elif avg_similarity > 0:
                relationship = "Weakly Cooperative"
            elif avg_similarity > -self.conflict_threshold:
                relationship = "Weakly Conflicting"
            else:
                relationship = "Strongly Conflicting"

            report_dict['pairwise_data'][(loss1, loss2)] = {'interactions': interactions, 'conflicts': conflicts,
                                                            'conflict_rate': conflict_rate,
                                                            'avg_similarity': avg_similarity,
                                                            'relationship': relationship}

            df_data.append({'loss1': loss1, 'loss2': loss2, 'interactions': interactions, 'conflicts': conflicts,
                            'conflict_rate': conflict_rate, 'avg_similarity': avg_similarity,
                            'relationship': relationship})

        df = pd.DataFrame(df_data)
        if self.total_conflicts > 0:
            df = df.sort_values('avg_similarity').reset_index(drop=True)

        return report_dict, df


class PolcaNetLoss(nn.Module):
    """
    PolcaNetLoss is a class that extends PyTorch's Module class. It represents the loss function used for
    Principal Latent Components Analysis (Polca).

    Attributes:
        r (float): The weight for the reconstruction loss.
        c (float): The weight for the classification loss.
        alpha (float): The weight for the orthogonality loss.
        beta (float): The weight for the center of mass loss.
        gamma (float): The weight for the low variance loss.
        class_labels (list): The list of class labels.
        loss_names (dict): The dictionary of loss names.
        i (torch.Tensor): The normalized axis.
        loss_analyzer (LossConflictAnalyzer): The loss conflict analyzer.
    """

    def __init__(self, latent_dim, r=1., c=1., alpha=1e-2, beta=1e-2, gamma=1e-4, class_labels=None,
                 analyzer_rate=0.05):
        """
        Initialize PolcaNetLoss with the provided parameters.
        """
        super().__init__()
        self.latent_dim = latent_dim
        c = c if class_labels is not None else (0, 0)  # classification loss weight

        # build two dictionaries for the loss weights and the optional loss probabilities
        self.r = r if isinstance(r, tuple) else (r, 1)
        self.c = c if isinstance(c, tuple) else (c, 1)
        self.alpha = alpha if isinstance(alpha, tuple) else (alpha, 1)
        self.beta = beta if isinstance(beta, tuple) else (beta, 1)
        self.gamma = gamma if isinstance(gamma, tuple) else (gamma, 1)
        self.class_labels = class_labels

        self.loss_names = {"loss": "Total Loss"}
        self.loss_names.update({
            "rec": "Reconstruction Loss" if self.r[0] != 0 else None,
            "ort": "Orthogonality Loss" if self.alpha[0] != 0 else None,
            "com": "Center of Mass Loss" if self.beta[0] != 0 else None,
            "var": "Variance Reduction Loss" if self.gamma[0] != 0 else None,
            "class": "Classification Loss" if self.class_labels is not None else None,

        })
        self.loss_names = {k: v for k, v in self.loss_names.items() if v is not None}

        # precompute the normalized axis
        i = (torch.arange(latent_dim, dtype=torch.float32) ** 1.25) / latent_dim
        # register i as a buffer
        self.register_buffer("i", i)
        self.loss_analyzer = LossConflictAnalyzer(loss_names=list(self.loss_names)[1:], rate=analyzer_rate)

    def forward(self, z, r, x, yp=None, target=None, classifier_model=None):
        """
        Variables:
        z: latent representation
        r: reconstruction
        x: input data
        v: variance of latent representation

        Losses:
        l_rec: Reconstruction loss
        l_ort: Orthogonality loss
        l_com: Center of mass loss
        l_var: Variance regularization loss
        l_class: Classification loss

        """
        # squeeze x and r
        x = x.squeeze()
        r = r.squeeze()

        v = torch.var(z, dim=0)
        w = F.normalize(v, p=1.0, dim=0)
        zw = z * w

        # random tests for the loss components
        t_rec = self.r[0] != 0 and self.r[1] > random()
        t_ort = self.alpha[0] != 0 and self.alpha[1] > random()
        t_com = self.beta[0] != 0 and self.beta[1] > random()
        t_var = self.gamma[0] != 0 and self.gamma[1] > random()
        t_class = self.c[0] != 0 and self.c[1] > random()

        # Robust reconstruction loss
        # The mask works by randomly zeroing out x% of the pixels in the input and the reconstruction.
        mask = torch.rand_like(r) > 0.1
        # Apply the mask to both the reconstruction and the original
        masked_r = r * mask
        masked_x = x * mask
        l_rec = F.huber_loss(masked_r, masked_x) if t_rec else torch.tensor(0.0, device=z.device, requires_grad=True)
        l_ort = self.orthogonality_loss(zw) if t_ort else 0
        l_com = self.center_of_mass_loss(w) if t_com else 0
        l_var = v.mean() if t_var else 0
        l_class = self.clustering_loss(zw, target) if t_class else 0
        # l_class = self.cross_entropy_loss(yp, target) if t_class else 0

        # Combine losses
        loss = (
                self.r[0] * l_rec +
                self.c[0] * l_class +
                self.alpha[0] * l_ort +
                self.beta[0] * l_com +
                self.gamma[0] * l_var
        )

        # dict of losses:
        loss_dict = {
            "rec": (l_rec, self.r),
            "ort": (l_ort, self.alpha),
            "com": (l_com, self.beta),
            "var": (l_var, self.gamma),
            "class": (l_class, self.c),
        }
        aux_losses = [l for l, (weight, prob) in loss_dict.values() if weight != 0]

        return loss, aux_losses

    def center_of_mass_loss(self, w):
        """
        Center of mass loss
        Purpose: Concentrate information in earlier latent dimensions
        Method: Minimize the weighted average of normalized variances, e.g., the center of mass.
        """
        # Compute and normalize variance
        l_com = torch.mean(self.i * w, dim=0)
        return l_com

    @staticmethod
    def orthogonality_loss(z, eps=1e-8):
        """ Orthogonality loss
        Purpose: Encourage latent dimensions to be uncorrelated
        Method: Penalize off-diagonal elements of the cosine similarity matrix
        """
        # Multiply by the weights and add a little additive noise to z for numerical stability.
        z = z + eps * torch.randn_like(z)
        # Normalize z along the batch dimension
        z_norm = F.normalize(z, p=2, dim=0)

        # Compute cosine similarity matrix
        s = torch.mm(z_norm.t(), z_norm)  # z_norm.t() @ z_norm = I
        s = s.clamp(-1 + eps, 1 - eps)  # clamp to avoid NaNs and numerical instability

        idx0, idx1 = torch.triu_indices(s.shape[0], s.shape[1], offset=1)  # indices of triu w/o diagonal
        cos_sim = s[idx0, idx1]
        loss = torch.mean(cos_sim.square())

        return loss

    @staticmethod
    def cross_entropy_loss(yp, target):
        """
        """
        if len(target.shape) == 1 or (len(target.shape) == 2 and target.shape[1] == 1):
            # Binary and multi-class classification
            l_cross_ent = F.cross_entropy(yp, target.view(-1).long())
        elif len(target.shape) == 2 and target.shape[1] > 1:
            # Multi-label binary classification
            l_cross_ent = F.binary_cross_entropy_with_logits(yp, target.float())
        else:
            raise ValueError("Unsupported target shape")
        return l_cross_ent


    def clustering_loss(self, z, labels, eps=1e-8):
        # Use precomputed weight vector w
        z_norm = F.normalize(z, p=2, dim=1)

        # Use precomputed num_classes
        num_classes = len(self.class_labels)

        # Labels are already indices from 0 to N
        if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1):
            # Single-label case
            labels = labels.view(-1)
            label_mask = F.one_hot(labels, num_classes=num_classes).float()
        else:
            # Multi-label case
            label_mask = labels.float()  # Labels are multi-hot vectors

        # Compute class sizes
        class_sizes = label_mask.sum(dim=0)

        # Identify non-empty classes and get their indices
        non_empty_classes = class_sizes > 0
        non_empty_class_indices = non_empty_classes.nonzero(as_tuple=False).squeeze(1)

        # Check for invalid indices
        if non_empty_class_indices.numel() == 0:
            return 0

        # Filter label_mask and class_sizes to include only non-empty classes
        label_mask = label_mask[:, non_empty_class_indices]
        class_sizes = class_sizes[non_empty_class_indices]

        # Compute centroids
        centroids = torch.matmul(label_mask.t(), z_norm)
        centroids = centroids / class_sizes.unsqueeze(1)
        centroids = F.normalize(centroids, p=2, dim=1)

        # Compute similarities
        all_similarities = torch.matmul(z_norm, centroids.t())
        all_similarities = all_similarities.clamp(-1 + eps, 1 - eps)

        # Compute intra-class similarities (normalized per sample)
        num_labels_per_sample = label_mask.sum(dim=1, keepdim=True) + eps
        intra_class_similarities = ((all_similarities * label_mask).sum(dim=1, keepdim=True)) / num_labels_per_sample
        intra_class_loss = (1 - intra_class_similarities.square()).sum()

        # Compute inter-class similarities
        inter_class_similarities = all_similarities * (1 - label_mask)
        inter_class_loss = inter_class_similarities.square().sum()

        # Total loss
        total_loss = intra_class_loss + inter_class_loss

        # Normalize the loss by the number of classes and batch size
        batch_size = z_norm.size(0)
        normalized_loss = total_loss / (num_classes * batch_size)

        return normalized_loss


class PolcaNet(nn.Module):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network encoder
    that is used for Principal Latent Components Analysis (Polca).

    Parameters:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        latent_dim (int): The number of latent dimensions.
        r (float): The weight for the reconstruction loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        c (float): The weight for the classification loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        alpha (float): The weight for the orthogonality loss.
        beta (float): The weight for the center of mass loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        gamma (float): The weight for the low variance loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        class_labels (list): The list of class labels.
        analyzer_rate (float): The rate of updates for the loss conflict analyzer.
        std_noise (float): The standard deviation of noise to add to the input data.
        blend_prob (float): The probability of blending the latent space with the mean.

    Attributes:
        device (str): The device to run the encoder on ("cpu" or "cuda").
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        latent_dim (int): The number of latent dimensions.
        class_labels (list): The list of class labels.
        polca_loss (PolcaNetLoss): The loss function.
        mu (torch.Tensor): The mean values of the latent space.
        std (torch.Tensor): The standard deviation values of the latent space.
        std_noise (float): The standard deviation of noise to add to the input data.
        blend_prob (float): The probability of blending the latent space with the mean.


    Methods:
        forward(x): Encodes the input and decodes the result.
        predict(in_x, w): Encodes and decodes the input.
        encode(in_x): Encodes the input and adds the mean to the latent space.
        decode(z, w): Decodes the latent space.
        r_error(in_x, w): Computes the reconstruction error.
        to_device(device): Moves the encoder to the given device.
        to(device): Moves the encoder to the given device.
        train_model(data, batch_size, num_epochs, report_freq, lr): Trains the model.
        fitter_data_loader(data_loader, num_epochs, lr): Fits the model using a DataLoader.
        fitter_in_memory(data, y, batch_size, num_epochs, lr): Fits the model using in-memory data.
        learn_step(x, optimizer, y): Performs a training step.
        val_step(x, y): Performs a validation step.
        add_noise(x): Adds noise to the input data when training.
        stochastic_latent_blend(z, mu, std, batch_prob, temperature): Transforms the latent space by
        probabilistically stochastically blending with the mean when training.


    Usage:
        data = np.random.rand(100, 784)
        latent_dim = 32
        model = PolcaNet(encoder, decoder, latent_dim=latent_dim, alpha=1.0, beta=1.0, gamma=1.0, class_labels=None)
        model.to_device("cuda")
        model.train_model(data, batch_size=256, num_epochs=10000, report_freq=20, lr=1e-3)
        model.r_error(data)
        latents, reconstructed = model.predict(data)

    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 latent_dim: int,
                 r: float = 1.,
                 c: float = 1.,
                 alpha: float | Tuple[float, float] = 1e-2,
                 beta: float | Tuple[float, float] = 1e-2,
                 gamma: float | Tuple[float, float] = 1e-8,
                 class_labels=None,
                 analyzer_rate: float = 0.05,
                 std_noise: float = 0.0,
                 blend_prob: float = 0.5,
                 ):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = class_labels
        self.std_noise = std_noise
        self.blend_prob = blend_prob

        self.encoder = EncoderWrapper(encoder, latent_dim=latent_dim, class_labels=class_labels)
        self.decoder = decoder

        if hasattr(self.encoder, "decoder"):
            # remove the attribute decoder from the encoder
            self.encoder.decoder = None
            del self.encoder.decoder

        if hasattr(self.decoder, "encoder"):
            # remove the attribute encoder from the decoder
            self.decoder.encoder = None
            del self.decoder.encoder

        self.polca_loss = PolcaNetLoss(
            latent_dim=latent_dim,
            r=r,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            class_labels=class_labels,
            analyzer_rate=analyzer_rate,
        )
        self.mu = None
        self.std = None

    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z)  if self.class_labels is None else self.decoder(z[0])
        return z, r

    def predict(self, in_x, mask=None, n_components=None):
        self.eval()
        z = self.encode(in_x)
        z = z[:, :n_components] if n_components is not None else z
        r = self.decode(z, mask=mask)
        return z, r

    def predict_batched(self, x, batch_size=1024, mask=None, n_components=None):
        results = [self.predict(np.array(batch), mask=mask, n_components=n_components) for batch in
                   batched(x, batch_size)]
        latents, reconstructions = map(np.concatenate, zip(*results))
        return latents, reconstructions

    def encode(self, in_x):
        self.eval()
        with torch.no_grad():
            if isinstance(in_x, torch.Tensor):
                x = in_x.detach().clone().to(self.device)
            else:
                x = torch.tensor(in_x, dtype=torch.float32, device=self.device, requires_grad=False)
            z = self.encoder(x)
            if self.class_labels is not None:
                return z[0].cpu().numpy()

        return z.cpu().numpy()

    def decode(self, z, mask=None):
        self.eval()
        with torch.no_grad():
            z = torch.as_tensor(z, dtype=torch.float32, device=self.device)

            # pad with mean values when the latent space is smaller than the expected size
            if z.size(1) < self.latent_dim:
                padding = self.mu[z.size(1):].unsqueeze(0).expand(z.size(0), -1)
                z = torch.cat([z, padding], dim=1)

            # pad with mean when mask is provided for the latent space
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
                if mask.size(0) < self.latent_dim:
                    mask = torch.nn.functional.pad(mask, (0, self.latent_dim - mask.size(0)))
                mask = mask.unsqueeze(0).expand(z.size(0), -1)
                z = mask * z + (1 - mask) * self.mu.unsqueeze(0)

            return self.decoder(z).cpu().numpy()

    def rec_error(self, in_x, mask=None, n_components=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, mask=mask, n_components=n_components)
        error = (in_x - r).reshape(in_x.shape[0], -1)
        return np.mean(error ** 2, axis=-1)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def train_model(self, data, y=None, val_data=None, val_y=None, batch_size=None,
                    num_epochs=100, report_freq=10, lr=1e-3, weight_decay=1e-2, verbose=1, optimizer=None):
        """
        Train the model using the given data.
        """
        train_metrics = list()
        val_metrics = list()
        # Create the optimizer if not provided
        free_optimizer = False
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                          weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-6)
            free_optimizer = True


        fitter = self.fitter(optimizer=optimizer,
                             data=data,
                             y=y,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             val_data=val_data,
                             val_y=val_y,
                             )

        epoch_progress = tqdm(fitter,
                              desc="epoch", leave=True, total=num_epochs, mininterval=1.0, disable=bool(not verbose))
        metrics = {}
        v_metrics = None
        epoch = 0
        output_widget = Output()
        display(output_widget)
        df_train_val = None
        try:
            for epoch, metrics, v_metrics in epoch_progress:
                train_metrics.append({**metrics, **{"epoch": epoch + 1, "split": "train"}})
                if v_metrics is not None:
                    val_metrics.append({**v_metrics, **{"epoch": epoch + 1, "split": "val"}})

                if epoch % report_freq == 0 and bool(verbose):
                    # epoch_progress.set_postfix(metrics)

                    metrics.update(**{"epoch": epoch + 1, "split": "train"})
                    if v_metrics is not None:
                        v_metrics.update(**{"epoch": epoch + 1, "split": "val"})

                    metrics_list = [metrics, v_metrics]
                    new_df_train_val = pd.DataFrame.from_records(metrics_list).set_index("split")
                    new_df_train_val.drop(columns=["epoch"], inplace=True, errors="ignore")
                    if df_train_val is None:
                        df_train_val = new_df_train_val
                    else:
                        # we will put only in df_train_val the values from new_df_train_val that are nonzero.
                        df_train_val = new_df_train_val.where(new_df_train_val != 0, df_train_val)

                    with output_widget:
                        clear_output(wait=True)
                        print(df_train_val.map(custom_float_format).to_string())

        except KeyboardInterrupt:
            print("Training interrupted by user.")

        finally:

            # Print final metrics if available
            if bool(verbose) and metrics:
                print("\nFinal Train metrics at epoch:", epoch + 1)
                print("=" * 40)
                for k, v in metrics.items():
                    if k not in ["epoch", "split"]:
                        print(f"{self.polca_loss.loss_names[k]}: {v:.4g}")

                if v_metrics:
                    print("\nFinal Validation metrics at epoch:", epoch + 1)
                    print("=" * 40)
                    for k, v in v_metrics.items():
                        if k not in ["epoch", "split"]:
                            print(f"{self.polca_loss.loss_names[k]}: {v:.4g}")

        if free_optimizer:
            del optimizer

        del metrics
        del v_metrics
        del fitter
        del df_train_val
        gc.collect()
        torch.cuda.empty_cache()

        return None

    def old_prepare_training_data(self,
                                  data: Union[torch.Tensor, np.ndarray, DataLoader],
                                  y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                  val_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                  val_y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                  batch_size: int = 256
                                  ) -> Tuple[DataLoader, Optional[DataLoader]]:

        # Prepare training data
        data_loader = data
        if not isinstance(data, DataLoader):
            data = torch.as_tensor(data, dtype=torch.float32, device=self.device)

            if y is not None:
                y = torch.as_tensor(y, dtype=torch.int64, device=self.device)
                dataset = TensorDataset(data, y)
            else:
                dataset = TensorDataset(data)

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )

        # Prepare validation data
        val_loader = val_data
        if val_data is not None and not isinstance(val_data, DataLoader):
            val_data = torch.as_tensor(val_data, dtype=torch.float32, device=self.device)

            if val_y is not None:
                val_y = torch.as_tensor(val_y, dtype=torch.int64, device=self.device)
                val_dataset = TensorDataset(val_data, val_y)
            else:
                val_dataset = TensorDataset(val_data)

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )

        return data_loader, val_loader

    def prepare_training_data(self,
                              data: Union[torch.Tensor, np.ndarray, DataLoader],
                              y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              val_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              val_y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              batch_size: int = 256,
                              use_fast: bool = True
                              ) -> Tuple[DataLoader, Optional[DataLoader]]:
        def create_loader(data, labels=None, shuffle=True):
            data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
            if labels is not None:
                labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)

            if use_fast:
                return FastTensorDataLoader(data,
                                            labels,
                                            batch_size=batch_size,
                                            shuffle=shuffle) if labels is not None else FastTensorDataLoader(data,
                                                                                                             batch_size=batch_size,
                                                                                                             shuffle=shuffle)
            else:
                dataset = TensorDataset(data, labels) if labels is not None else TensorDataset(data)
                return DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=0,
                                  pin_memory=False,
                                  drop_last=True,
                                  )

        # Prepare training data
        if isinstance(data, DataLoader):
            data_loader = data
        else:
            data_loader = create_loader(data, y, shuffle=True)

        # Prepare validation data
        val_loader = None
        if val_data is not None:
            if isinstance(val_data, DataLoader):
                val_loader = val_data
            else:
                val_loader = create_loader(val_data, val_y, shuffle=False)

        return data_loader, val_loader

    def fitter(self, optimizer, data, y=None, batch_size=512, num_epochs=100, val_data=None, val_y=None):
        self.train()
        is_data_loader = isinstance(data, torch.utils.data.DataLoader)
        if not is_data_loader:
            data = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.float32)
            data = data.to(self.device)

            if val_data is not None:
                val_data = val_data if isinstance(val_data, torch.Tensor) else torch.tensor(val_data,
                                                                                            dtype=torch.float32)
                val_data = val_data.to(self.device)

            if y is not None:
                y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.int64)
                y = y.to(self.device)

            if val_y is not None:
                val_y = val_y if isinstance(val_y, torch.Tensor) else torch.tensor(val_y, dtype=torch.int64)
                val_y = val_y.to(self.device)

            num_samples = len(data)
            adjusted_num_samples = (num_samples // batch_size) * batch_size

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            val_losses = defaultdict(list)

            if is_data_loader:
                batches = data
            else:
                indices = torch.randperm(num_samples)[:adjusted_num_samples]
                batches = range(0, adjusted_num_samples, batch_size)

            for batch in batches:
                if is_data_loader:
                    x = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if self.class_labels is not None else None
                else:
                    batch_indices = indices[batch:batch + batch_size]
                    x = data[batch_indices]
                    targets = y[batch_indices] if y is not None else None

                loss, aux_losses = self.learn_step(x=x, optimizer=optimizer, y=targets)
                losses["loss"].append(loss.item())
                for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                    l_value = 0 if not aux_losses[idx] else aux_losses[idx].item()
                    losses[name].append(l_value)

            metrics = {name: np.mean(losses[name]) for name in losses}

            val_metrics = None
            if val_data is not None:
                if is_data_loader:
                    val_batches = val_data
                    for val_batch in val_batches:
                        x = val_batch[0].to(self.device)
                        targets = val_batch[1].to(self.device) if self.class_labels is not None else None
                        val_loss, val_aux_losses = self.val_step(x=x, y=targets)
                        val_losses["loss"].append(val_loss.item())
                        for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                            l_value = 0 if not val_aux_losses[idx] else val_aux_losses[idx].item()
                            val_losses[name].append(l_value)
                else:
                    val_loss, val_aux_losses = self.val_step(x=val_data, y=val_y)
                    val_losses["loss"].append(val_loss.item())
                    for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                        l_value = 0 if not val_aux_losses[idx] else val_aux_losses[idx].item()
                        val_losses[name].append(l_value)

                val_metrics = {name: np.mean(val_losses[name]) for name in val_losses}

            yield epoch, metrics, val_metrics

        del losses, val_losses
        data = data.cpu()
        if val_data is not None:
            val_data = val_data.cpu()


    def oldfitter(self, optimizer, data, y=None, batch_size=512, num_epochs=100, val_data=None, val_y=None):
        data_loader, val_loader = self.prepare_training_data(data, y, val_data, val_y, batch_size)
        if isinstance(data_loader, FastTensorDataLoader):
            print("Using FastTensorDataLoader")
        for epoch in range(num_epochs):
            self.train()
            metrics = self.run_epoch(data_loader, optimizer)
            if epoch % 10 == 0:
                val_metrics = self.run_epoch(val_loader, None) if val_loader is not None else None
            else:
                val_metrics = {k:0 for k in metrics}

            yield epoch, metrics, val_metrics

    def run_epoch(self, data_loader, optimizer):

        losses = defaultdict(list)
        for batch in data_loader:
            x, targets = batch if isinstance(batch, Sequence) and len(batch) == 2 else (batch, None)
            x = x[0] if isinstance(x, Sequence) else x
            x = x.to(self.device)
            if targets is not None:
                targets = targets.to(self.device) if self.class_labels is not None else None

            # with torch.amp.autocast('cuda'):
            if optimizer is not None:
                loss, aux_losses = self.learn_step(x=x, optimizer=optimizer, y=targets)
            else:
                loss, aux_losses = self.val_step(x=x, y=targets)

            losses["loss"].append(loss.item())
            for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                l_value = 0 if not aux_losses[idx] else aux_losses[idx].item()
                losses[name].append(l_value)

        metrics = {name: np.mean(losses[name]) for name in losses}
        return metrics

    def learn_step(self, x, optimizer, y=None):
        self.train()
        # add small noise to x to make regularization and a little data augmentation x has values in 0,1
        x = self.add_noise(x) if self.std_noise > 0 else x
        z, r = self.forward(x)
        yp = z[1] if self.class_labels is not None else None
        z = z[0] if self.class_labels is not None else z

        with torch.no_grad():
            self.mu = torch.mean(z, dim=0) if self.mu is None else self.mu + 0.01 * (torch.mean(z, dim=0) - self.mu)
            self.std = torch.std(z, dim=0) if self.std is None else self.std + 0.01 * (torch.std(z, dim=0) - self.std)

        if random() < self.blend_prob:
            z = self.stochastic_latent_blend(z, self.mu, self.std, batch_prob=self.blend_prob, temperature=1.0)

        classifier_module = self.encoder.classifier if self.class_labels is not None else None
        loss, aux_losses = self.polca_loss(z, r, x, yp=yp, target=y, classifier_model=classifier_module)

        # Analyze loss conflicts
        self.polca_loss.loss_analyzer.step(self, aux_losses)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return loss, aux_losses

    def val_step(self, x, y=None):
        self.eval()
        with torch.no_grad():
            z, r = self.forward(x)
            yp = z[1] if self.class_labels is not None else None
            z = z[0] if self.class_labels is not None else z
            classifier_module = self.encoder.classifier if self.class_labels is not None else None
            loss, aux_losses = self.polca_loss(z, r, x, yp=yp, target=y, classifier_model=classifier_module)

            return loss, aux_losses

    def add_noise(self, x):
        x = x + torch.randn_like(x) * self.std_noise  # add noise to x
        return x

    @staticmethod
    def stochastic_latent_blend(z, mu, std, batch_prob=0.5, temperature=1.0):
        """
        Transform z by probabilistically blending with mu values

        Args:
        z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim)
        mu (torch.Tensor): Mean values of shape (latent_dim,)
        std (torch.Tensor): Standard deviation values of shape (latent_dim,)
        batch_prob (float): Probability of applying the transformation to each element in the batch
        temperature (float): Temperature for Gumbel-Softmax, controls the discreteness of the output

        Returns:
        torch.Tensor: Transformed z
        """
        device = z.device
        batch_size, latent_dim = z.shape

        # 1. Batch-level probability
        batch_probs = torch.full((batch_size, 1), batch_prob, device=device)
        batch_mask = torch.bernoulli(batch_probs)  # Still using bernoulli, but it's not in the computational graph

        # 2. Feature-level probability inversely proportional to std
        feature_probs = 1 / (std + 1e-6)
        feature_probs = feature_probs / feature_probs.sum()

        # Generate differentiable feature mask using Gumbel-Softmax
        feature_logits = torch.log(feature_probs.unsqueeze(0).expand(batch_size, -1))
        feature_mask = F.gumbel_softmax(feature_logits, tau=temperature, hard=False)

        # Combine batch and feature masks
        combined_mask = batch_mask * feature_mask

        # Apply the transformation as a soft blending
        z_transformed = combined_mask * mu + (1 - combined_mask) * z

        return z_transformed


class EncoderWrapper(nn.Module):
    """
    Wrapper class for an encoder module.
    A Softsign activation function is applied to the output of the encoder.
    Optionally, a classifier can be added to the output of the encoder when class_labels are provided.
    """

    def __init__(self, encoder: nn.Module, latent_dim, class_labels=None):
        """
        Initializes the encoder wrapper with the encoder module and the factor scale parameter.
        :param encoder: The encoder module.
        """
        super().__init__()
        self.class_labels = class_labels
        self.latent_dim = latent_dim
        self.encoder = encoder
        layers = []

        layer = nn.Linear(latent_dim, latent_dim)
        with torch.no_grad():
            nn.init.orthogonal_(layer.weight)
            # Get the number of features (output dimension)
            n_features, n_inputs = layer.weight.shape
            # Scale the orthogonal matrix
            scales = torch.linspace(1.0, 0.1, n_features)
            scaling_factor = 0.5  # To keep initial activations within a reasonable range for Softsign
            layer.weight.mul_(scales.unsqueeze(1) * scaling_factor)

        layers.append(layer)
        layers.append(nn.Softsign())

        self.polca_bottleneck = nn.Sequential(*layers)

        if class_labels is not None:
            # Binary, multi-class and multi-label-binary classification
            self.classifier = nn.Linear(latent_dim, len(class_labels), bias=False)

    def forward(self, x):
        z = self.encoder(x)
        z = self.polca_bottleneck(z)
        if self.class_labels is not None:
            return z, self.classifier(z)

        return z
