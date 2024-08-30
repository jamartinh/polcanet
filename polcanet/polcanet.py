from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from polcanet.utils import save_df_to_csv, save_figure


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
        a (torch.Tensor): The normalized axis.
        loss_names (dict): The dictionary of loss names.


    """

    def __init__(self, latent_dim, r=1., c=1., alpha=1e-2, beta=1e-2, gamma=1e-4, class_labels=None):
        """
        Initialize PolcaNetLoss with the provided parameters.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_labels = class_labels
        self.r = r  # reconstruction loss weight
        self.c = c  # classification loss weight

        pos = (torch.arange(latent_dim, dtype=torch.float32) ** 1.25) / latent_dim
        self.register_buffer('a', pos)  # normalized axis

        self.loss_names = {"loss": "Total Loss",
                           "rec": "Reconstruction Loss",
                           "ort": "Orthogonality Loss",
                           "com": "Center of Mass Loss",
                           "var": "Variance Reduction Loss",
                           "iort": "Instance Orthogonality Loss",
                           }
        if class_labels is not None:
            self.loss_names["class"] = "Classification Loss"
        else:
            self.c = 0

    def forward(self, z, r, x, yp=None, target=None):
        """
        Variables:
        z: latent representation
        r: reconstruction
        x: input data
        v: variance of latent representation

        Losses:
        L_rec: Reconstruction loss
        L_ort: Orthogonality loss
        L_com: Center of mass loss
        L_var: Variance regularization loss
        L_class: Classification loss

        """
        batch_size = x.shape[0]
        v = torch.var(z, dim=0)
        w = F.normalize(v, p=1.0, dim=0)

        L_rec = F.mse_loss(r, x) if self.r != 0 else 0
        L_ort = self.orthogonality_loss(z) if self.alpha != 0 and batch_size >= 16 else 0
        L_com = self.center_of_mass_loss(self.a, w) if self.beta != 0 and batch_size >= 16 else 0
        L_var = v.mean() if self.gamma != 0 else 0
        L_class = self.classification_loss(yp, target) if self.c != 0 and self.class_labels is not None else 0
        L_iort = self.instance_orthogonality_loss(z) if self.alpha != 0 and batch_size >= 16 else 0

        # Combine losses
        # Purpose: Balance all loss components for optimal latent representation
        # Method: Weighted sum of individual losses
        L = self.r * L_rec + self.c * L_class + self.alpha * L_ort + self.beta * L_com + self.gamma * L_var + 0.01*self.alpha * L_iort

        if self.class_labels is not None:
            return L, (L_rec, L_ort, L_com, L_var, L_class, L_iort)

        return L, (L_rec, L_ort, L_com, L_var, L_iort)

    @staticmethod
    def center_of_mass_loss(x, w):
        """
        Center of mass loss
        Purpose: Concentrate information in earlier latent dimensions
        Method: Minimize the weighted average of normalized variances, e.g., the center of mass.
        """
        # Compute and normalize variance and energy
        L_com = torch.mean(w * x, dim=0)
        return L_com

    @staticmethod
    def classification_loss(yp, target):
        # If class labels are provided, compute classification loss
        L_class = F.cross_entropy(yp, target)
        return L_class

    @staticmethod
    def orthogonality_loss(z, eps=1e-8):
        """ Orthogonality loss
        Purpose: Encourage latent dimensions to be uncorrelated
        Method: Penalize off-diagonal elements of the cosine similarity matrix
        """
        # Add a little additive noise to z
        z = z + 1e-6 * torch.randn_like(z)

        # Normalize z along the batch dimension (explicitly added)
        z_norm = F.normalize(z, p=2, dim=0)

        # Compute cosine similarity matrix
        S = torch.mm(z_norm.t(), z_norm)
        S = S.clamp(-1 + eps, 1 - eps)  # clamp to avoid NaNs and numerical instability

        idx0, idx1 = torch.triu_indices(S.shape[0], S.shape[1], offset=1)  # indices of triu w/o diagonal
        cos_sim = S[idx0, idx1]

        loss = torch.mean(cos_sim.square())
        if torch.isnan(loss):
            print("NAN")
            print("z shape:", z.shape)
            print("z_norm shape:", z_norm.shape)
            print("max min z_norm:", torch.max(z_norm), torch.min(z_norm))
            print("max min S:", torch.max(S), torch.min(S))
            print("max min S_triu:", torch.max(cos_sim), torch.min(cos_sim))
            raise ValueError("Orthogonality loss is NaN")

        return loss

    @staticmethod
    def instance_orthogonality_loss(z, eps=1e-8):
        """ Instance orthogonality loss
        Purpose: Encourage instances (row vectors) to be uncorrelated
        Method: Penalize off-diagonal elements of the instance cosine similarity matrix
        """
        # Add a little additive noise to z
        z = z + 1e-6 * torch.randn_like(z)

        # Normalize z along the feature dimension (dim=1)
        z_norm = F.normalize(z, p=2, dim=1)

        # Compute instance cosine similarity matrix
        S = torch.mm(z_norm, z_norm.t())

        # Apply clamp to avoid potential numerical instability
        S = S.clamp(-1 + eps, 1 - eps)

        # Extract upper triangular part (excluding diagonal)
        idx0, idx1 = torch.triu_indices(S.shape[0], S.shape[1], offset=1)
        cos_sim = S[idx0, idx1]

        # Compute loss
        loss = torch.mean(cos_sim.square())

        if torch.isnan(loss):
            print("NaN detected in instance orthogonality loss")
            print("z shape:", z.shape)
            print("z_norm shape:", z_norm.shape)
            print("max min z_norm:", torch.max(z_norm), torch.min(z_norm))
            print("max min S:", torch.max(S), torch.min(S))
            print("max min cos_sim:", torch.max(cos_sim), torch.min(cos_sim))
            raise ValueError("Instance orthogonality loss is NaN")

        return loss


class PolcaNet(nn.Module):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network encoder
    that is used for Principal Latent Components Analysis (Polca).

    Attributes:
        device (str): The device to run the encoder on ("cpu" or "cuda").
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        latent_dim (int): The number of latent dimensions.
        class_labels (list): The list of class labels.
        polca_loss (PolcaNetLoss): The loss function.
        loss_analyzer (LossConflictAnalyzer): The loss conflict analyzer.



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
                 r=1.,
                 c=1.,
                 alpha=1e-2,
                 beta=1e-2,
                 gamma=1e-4,
                 class_labels=None,
                 analyzer_rate=0.1,
                 ):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = class_labels

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

        self.polca_loss = PolcaNetLoss(latent_dim, r, c, alpha, beta, gamma, class_labels)
        self.loss_analyzer = LossConflictAnalyzer(self, loss_names=list(self.polca_loss.loss_names)[1:],
                                                  rate=analyzer_rate)

    def forward(self, x):
        z = self.encoder(x)
        if self.class_labels is not None:
            r = self.decoder(z[0])
            return (z[0], z[1]), r
        else:
            r = self.decoder(z)
        return z, r

    def predict(self, in_x, mask=None, n_components=None):
        z = self.encode(in_x)
        if n_components:
            z = z[:, :n_components]

        r = self.decode(z, mask=mask)
        return z, r

    def encode(self, in_x):
        self.encoder.eval()
        if isinstance(in_x, torch.Tensor):
            x = in_x.detach().clone().to(self.device)
        else:
            x = torch.tensor(in_x, dtype=torch.float32, device=self.device, requires_grad=False)

        with torch.no_grad():
            z = self.encoder(x)
            if self.class_labels is not None:
                return z[0].cpu().numpy()

        return z.cpu().numpy()

    def decode(self, z, mask=None):
        self.encoder.eval()

        # Convert z to tensor if it's not already
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)

        # Ensure z has the correct shape
        if z.shape[1] < self.latent_dim:
            difference = self.latent_dim - z.shape[1]
            padding = torch.zeros(z.shape[0], difference, device=self.device)
            z = torch.cat([z, padding], dim=1)

        # Apply mask if provided
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
            if len(mask) != self.latent_dim:
                mask = torch.cat([mask, torch.zeros(self.latent_dim - len(mask), device=self.device)])
            mask = mask.unsqueeze(0).expand(z.shape[0], -1)
            z = z * mask  # This effectively sets masked dimensions to zero

        with torch.no_grad():
            r = self.decoder(z)

        return r.cpu().numpy()

    def r_error(self, in_x, w=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, w)
        error = (in_x - r).reshape(in_x.shape[0], -1)
        return np.mean(error ** 2, axis=-1)

    def to_device(self, device):
        """Move the encoder to the given device."""
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def train_model(self, data, y=None, batch_size=None,
                    num_epochs=100, report_freq=10, lr=1e-3, verbose=1, fine_tune=False):
        """
        Train the model using the given data.
        Usage:
            In memory numpy or torch tensor inputs:
            >>> data = np.random.rand(100, 784)
            >>> latent_dim = 32
            >>> model = PolcaNet(encoder, decoder, latent_dim=latent_dim, alpha=1.0, beta=1.0, gamma=1.0, class_labels=None
            >>> model.to_device("cuda")
            >>> model.train_model(data, batch_size=256, num_epochs=10000, report_freq=10, lr=1e-3)

            Or using a data loader:
            >>> data_loader = DataLoader(data, batch_size=256, shuffle=True)
            >>> model.train_model(data_loader, num_epochs=10000, report_freq=10, lr=1e-3)

        """
        # Create the optimizer
        if not fine_tune:
            for param in self.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            self.loss_analyzer.enabled = True
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
            self.loss_analyzer.enabled = False



        if self.class_labels is not None and y is None:
            raise ValueError("Class labels are provided, but no target values are provided in train_model")

        if isinstance(data, (np.ndarray, torch.Tensor)):
            if batch_size is None:
                raise ValueError("batch_size must be provided when data is in memory")
            fitter = self.fitter_in_memory(optimizer, data, y=y, batch_size=batch_size, num_epochs=num_epochs)
        elif isinstance(data, DataLoader):
            fitter = self.fitter_data_loader(optimizer, data, num_epochs=num_epochs)
        else:
            fitter = None

        assert fitter is not None, "Data must be either a DataLoader or a numpy array or torch tensor"

        epoch_progress = tqdm(fitter,
                              desc="epoch", leave=True, total=num_epochs, mininterval=1.0, disable=bool(not verbose))
        metrics = {}
        epoch = 0
        try:
            for epoch, metrics in epoch_progress:
                if epoch % report_freq == 0:
                    epoch_progress.set_postfix(metrics)

            # pretty print final metrics stats
            if bool(verbose):
                print(f"Final metrics at epoch: {epoch + 1}")
                for k, v in metrics.items():
                    print(f"{self.polca_loss.loss_names[k]}: {v:.4g}")
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            return None

    def fitter_data_loader(self,optimizer, data_loader, num_epochs=100):
        self.train()

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            for batch in data_loader:
                x = batch[0].to(self.device)
                if self.class_labels is not None:
                    y = batch[1].to(self.device)
                    # assure that the target is an integer tensor
                    y = y.to(torch.int64)
                else:
                    y = None

                loss, aux_losses = self.learn_step(x, optimizer, y=y)
                losses["loss"].append(loss.item())
                for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                    losses[name].append(aux_losses[idx].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

    def fitter_in_memory(self, optimizer, data, y=None, batch_size=512, num_epochs=100):
        num_samples: int = len(data)
        self.train()
        torch_y = None

        if self.class_labels is None:
            y = None

        if not isinstance(data, torch.Tensor):
            torch_data = torch.tensor(data, dtype=torch.float32).to(self.device)
            if self.class_labels is not None:
                torch_y = torch.tensor(y, dtype=torch.int64).to(self.device)
        else:
            torch_data = data.to(self.device)
            if self.class_labels is not None:
                torch_y = y.to(self.device)

        num_full_batches = num_samples // batch_size
        adjusted_num_samples = num_full_batches * batch_size
        for epoch in range(num_epochs):
            losses = defaultdict(list)
            indices = torch.randperm(num_samples)[:adjusted_num_samples]
            for i in range(0, adjusted_num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                x = torch_data[batch_indices]
                if self.class_labels is not None:
                    targets = torch_y[batch_indices]
                else:
                    targets = None

                loss, aux_losses = self.learn_step(x, optimizer, y=targets)
                losses["loss"].append(loss.item())
                for aux_loss, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                    losses[name].append(aux_losses[aux_loss].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

    def learn_step(self, x, optimizer, y=None):
        z, r = self.forward(x)
        yp = z[1] if self.class_labels is not None else None
        z = z[0] if self.class_labels is not None else z
        loss, aux_losses = self.polca_loss(z, r, x, yp=yp, target=y)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)

        # Analyze loss conflicts
        if self.loss_analyzer is not None:
            self.loss_analyzer.analyze([l for l in aux_losses if not isinstance(l, (int, float))])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, aux_losses


class EncoderWrapper(nn.Module):
    """
    Wrapper class for an encoder module.
    A Softsign activation function is applied to the output of the encoder and there is an optional factor scale
    parameter that can be set to True to factor the scale of the latent vectors and return the unit vectors and
    the magnitudes as a tuple.
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

        # Apply orthogonal initialization
        nn.init.orthogonal_(layer.weight)
        # Apply the scaling to the weight matrix
        # with torch.no_grad():
        #     # Get the number of features (output dimension)
        #     n_features, n_inputs = layer.weight.shape
        #     # Scale the orthogonal matrix
        #     scales = torch.linspace(1.0, 0.1, n_features)
        #     scaling_factor = 0.5  # To keep initial activations within a reasonable range for Softsign
        #     layer.weight.mul_(scales.unsqueeze(1) * scaling_factor)

        layers.append(layer)
        layer = nn.Softsign()
        layers.append(layer)
        self.post_encoder = nn.Sequential(*layers)

        if class_labels is not None:
            # create a simple perceptron layer
            self.classifier = nn.Linear(latent_dim, len(class_labels))

    def forward(self, x):
        z = self.encoder(x)
        z = self.post_encoder(z)
        if self.class_labels is not None:
            return z, self.classifier(z)

        return z


class LossConflictAnalyzer:
    def __init__(self, model: torch.nn.Module, loss_names: List[str] = None, rate=0.1, conflict_threshold: float = 0.1):
        self.pairwise_similarities = {}
        self.pairwise_conflicts = {}
        self.pairwise_interactions = {}
        self.total_conflicts = 0
        self.total_interactions = 0
        self.model = model
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

    def analyze(self, losses: List[torch.Tensor]) -> None:
        if len(losses) != len(self.loss_names):
            raise ValueError("Number of losses must match number of loss names")

        # rate test
        if not self.enabled or np.random.rand() > self.rate:
            return

        grads = self._compute_grad(losses)
        self._analyze_conflicts(grads)

    def _compute_grad(self, losses: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        grads = []
        for loss in losses:
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            grad_dict = {name: param.grad.clone() for name, param in self.model.named_parameters() if
                         param.grad is not None}
            grads.append(grad_dict)
        return grads

    def _analyze_conflicts(self, grads: List[Dict[str, torch.Tensor]]) -> None:
        num_losses = len(grads)

        for (i, j) in combinations(range(num_losses), 2):
            conflict_key = self._get_conflict_key(i, j)
            g_i, g_j = grads[i], grads[j]

            similarity = self._compute_similarity(g_i, g_j)

            self.total_interactions += 1
            self.pairwise_interactions[conflict_key] = self.pairwise_interactions.get(conflict_key, 0) + 1
            self.pairwise_similarities[conflict_key] = self.pairwise_similarities.get(conflict_key, []) + [similarity]

            if similarity < -self.conflict_threshold:
                self.total_conflicts += 1
                self.pairwise_conflicts[conflict_key] = self.pairwise_conflicts.get(conflict_key, 0) + 1

    @staticmethod
    def _compute_similarity(grad1: Dict[str, torch.Tensor], grad2: Dict[str, torch.Tensor]) -> float:
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

    def _get_conflict_key(self, i: int, j: int) -> Tuple[str, str]:
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
        df = df.sort_values('avg_similarity').reset_index(drop=True)

        return report_dict, df

    def print_report(self) -> None:
        report_dict, df = self.report()
        print(f"Loss Interaction Analysis Report:")
        print(f"Total interactions: {report_dict['total_interactions']}")
        print(f"Total conflicts: {report_dict['total_conflicts']}")
        print(f"Overall conflict rate: {report_dict['overall_conflict_rate']:.4f}")
        print(f"\nPairwise Statistics (sorted by similarity):")

        def color_cells(val, column):
            if column == 'relationship':
                colors = {"Strongly Cooperative": "color: green", "Weakly Cooperative": "color: green",
                          "Weakly Conflicting": "color: coral", "Strongly Conflicting": "color: red"}
                return colors.get(val, "")
            elif column == 'avg_similarity':
                if val > self.conflict_threshold:
                    return 'color: green'
                elif val > 0:
                    return 'color: green'
                elif val > -self.conflict_threshold:
                    return 'color: coral'
                else:
                    return 'color: red'
            return ''

        styled_df = df.style.apply(lambda x: [color_cells(xi, col) for xi, col in zip(x, x.index)], axis=1).format(
            {'interactions': '{:.0f}', 'conflicts': '{:.0f}', 'conflict_rate': '{:.3f}', 'avg_similarity': '{:.3f}'})

        display(styled_df)
        save_df_to_csv(df, "loss_interaction_report.csv")

    def plot_correlation_matrix(self):
        _, df = self.report()

        # Create a square matrix of similarities
        matrix_size = len(self.loss_names)
        sim_matrix = np.ones((matrix_size, matrix_size))

        for _, row in df.iterrows():
            i = self.loss_names.index(row['loss1'])
            j = self.loss_names.index(row['loss2'])
            sim_matrix[i, j] = sim_matrix[j, i] = row['avg_similarity']

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool))

        # Set up the matplotlib figure
        fig, ax = plt.subplots()

        # Create colormap
        cmap = sns.diverging_palette(10, 220, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(sim_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5,
                    cbar_kws={"shrink": .5}, annot=True, xticklabels=self.loss_names, yticklabels=self.loss_names)

        # Add relationship annotations
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                if not mask[i, j]:
                    similarity = sim_matrix[i, j]
                    if similarity > self.conflict_threshold:
                        relationship = "SC"  # Strongly Cooperative
                    elif similarity > 0:
                        relationship = "WC"  # Weakly Cooperative
                    elif similarity > -self.conflict_threshold:
                        relationship = "WF"  # Weakly Conflicting
                    else:
                        relationship = "SF"  # Strongly Conflicting
                    ax.text(j + 0.5, i + 0.5, relationship, ha='center', va='center', color='black',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.title("Loss Interaction Matrix")
        plt.tight_layout()
        save_figure("loss_interaction.pdf")
        plt.show()
