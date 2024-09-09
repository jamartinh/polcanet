from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class LossConflictAnalyzer:
    def __init__(self, loss_names: List[str] = None, rate=0.1, conflict_threshold: float = 0.1):
        """
        Initializes the LossConflictAnalyzer class.

        Args:
            model (torch.nn.Module): The model to analyze.
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
        if len(losses) != len(self.loss_names):
            raise ValueError("Number of losses must match number of loss names")

        # rate test
        if not self.enabled or np.random.rand() > self.rate:
            return

        grads = self.compute_grad(model, losses)
        self.analyze_conflicts(grads)

    @staticmethod
    def compute_grad(model, losses: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        grads = []
        for loss in losses:
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

            similarity = self.compute_similarity(g_i, g_j)

            self.total_interactions += 1
            self.pairwise_interactions[conflict_key] = self.pairwise_interactions.get(conflict_key, 0) + 1
            self.pairwise_similarities[conflict_key] = self.pairwise_similarities.get(conflict_key, []) + [similarity]

            if similarity < -self.conflict_threshold:
                self.total_conflicts += 1
                self.pairwise_conflicts[conflict_key] = self.pairwise_conflicts.get(conflict_key, 0) + 1

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
        a (torch.Tensor): The normalized axis.
        loss_names (dict): The dictionary of loss names.


    """

    def __init__(self, latent_dim, r=1., c=1., alpha=1e-2, beta=1e-2, gamma=1e-4, class_labels=None, analyzer_rate=0.1):
        """
        Initialize PolcaNetLoss with the provided parameters.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.r = r  # reconstruction loss weight
        self.c = c if class_labels is not None else 0  # classification loss weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_labels = class_labels

        self.loss_names = {"loss": "Total Loss"}

        self.loss_names.update({
                "rec": "Reconstruction Loss" if self.r != 0 else None,
                "ort": "Orthogonality Loss" if self.alpha != 0 else None,
                "com": "Center of Mass Loss" if self.beta != 0 else None,
                "class": "Classification Loss" if self.class_labels is not None else None,
                "var": "Variance Reduction Loss" if self.gamma != 0 else None
        })
        self.loss_names = {k: v for k, v in self.loss_names.items() if v is not None}

        # precompute the normalized axis and store it as a buffer
        self.i = (torch.arange(latent_dim, dtype=torch.float32) ** 1.25) / latent_dim
        self.loss_analyzer = LossConflictAnalyzer(loss_names=list(self.loss_names)[1:], rate=analyzer_rate)

    def forward(self, z, r, x, yp=None, target=None):
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

        v = torch.var(z, dim=0)
        w = F.normalize(v, p=1.0, dim=0)

        l_rec = F.mse_loss(r, x) if self.r != 0 else 0
        l_ort = self.orthogonality_loss(z) if self.alpha != 0 else 0
        l_com = self.center_of_mass_loss(self.i, w) if self.beta != 0 else 0
        l_var = v.mean() if self.gamma != 0 else 0
        l_class = self.classification_loss(yp, target) if self.c != 0 else 0

        # Combine losses
        # Purpose: Balance all loss components
        # Method: Weighted sum of individual losses
        loss = self.r * l_rec + self.c * l_class + self.alpha * l_ort + self.beta * l_com + self.gamma * l_var

        # return a tuple with only the non-zero losses depending on the weights
        # dict of losses:
        loss_dict = {
                "rec": (l_rec, self.r),
                "ort": (l_ort, self.alpha),
                "com": (l_com, self.beta),
                "var": (l_var, self.gamma),
                "class": (l_class, self.c),
        }
        aux_losses = [l for l, weight in loss_dict.values() if weight != 0]

        return loss, aux_losses

    @staticmethod
    def center_of_mass_loss(i, w):
        """
        Center of mass loss
        Purpose: Concentrate information in earlier latent dimensions
        Method: Minimize the weighted average of normalized variances, e.g., the center of mass.
        """
        # Compute and normalize variance and energy
        L_com = torch.mean(i.to(w.device) * w, dim=0)  # + torch.mean(z * w, dim=0)
        return L_com

    @staticmethod
    def classification_loss(yp, target):
        """
        Calculate classification loss based on prediction and target shapes.
        Handles binary, multi-class, and multi-label classification.

        Args:
        yp (torch.Tensor): Model predictions (logits), shape (batch_size, num_classes)
        target (torch.Tensor): Ground truth labels
            - For binary/multi-class: shape (batch_size,) or (batch_size, 1)
            - For multi-label: shape (batch_size, num_classes) where num_classes > 1

        Returns:
        torch.Tensor: Calculated loss
        """

        if len(target.shape) == 1 or (len(target.shape) == 2 and target.shape[1] == 1):
            # Binary and multi-class classification
            l_class = F.cross_entropy(yp, target.view(-1).long())
        elif len(target.shape) == 2 and target.shape[1] > 1:
            # Multi-label binary classification
            l_class = F.binary_cross_entropy_with_logits(yp, target.float())
        else:
            raise ValueError("Unsupported target shape")

        return l_class

    @staticmethod
    def orthogonality_loss(z, eps=1e-8):
        """ Orthogonality loss
        Purpose: Encourage latent dimensions to be uncorrelated
        Method: Penalize off-diagonal elements of the cosine similarity matrix
        """
        # Add a little additive noise to z
        z = z + 1e-8 * torch.randn_like(z)
        # Normalize z along the batch dimension
        z_norm = F.normalize(z, p=2, dim=0)

        # Compute cosine similarity matrix
        s = torch.mm(z_norm.t(), z_norm)  # z_norm.t() @ z_norm = I
        s = s.clamp(-1 + eps, 1 - eps)  # clamp to avoid NaNs and numerical instability

        idx0, idx1 = torch.triu_indices(s.shape[0], s.shape[1], offset=1)  # indices of triu w/o diagonal
        cos_sim = s[idx0, idx1]

        loss = torch.mean(cos_sim.square())

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
        if z.size(1) < self.latent_dim:
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

    def rec_error(self, in_x, w=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, w)
        error = (in_x - r).reshape(in_x.shape[0], -1)
        return np.mean(error ** 2, axis=-1)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def train_model(self, data, y=None, batch_size=None,
                    num_epochs=100, report_freq=10, lr=1e-3, weight_decay=1e-2, verbose=1, optimizer=None):
        """
        Train the model using the given data.
        Usage:
            In memory numpy or torch tensor inputs:
            >>> data = np.random.rand(100, 784)
            >>> latent_dim = 32
            >>> model = PolcaNet(encoder, decoder, latent_dim=latent_dim, alpha=1.0, beta=1.0, gamma=1.0, class_labels=None
            >>> model.to("cuda")
            >>> model.train_model(data, batch_size=256, num_epochs=10000, report_freq=10, lr=1e-3)

            Or using a data loader:
            >>> data_loader = DataLoader(data, batch_size=256, shuffle=True)
            >>> model.train_model(data_loader, num_epochs=10000, report_freq=10, lr=1e-3)

        """
        # Create the optimizer if not provided
        optimizer = optimizer or torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

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

    def fitter_data_loader(self, optimizer, data_loader, num_epochs=100):
        self.train()
        if self.class_labels is None:
            y = None

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            for batch in data_loader:
                x = batch[0].to(self.device)
                if self.class_labels is not None:
                    y = batch[1].to(self.device)
                    # assure that the target is an integer tensor
                    y = y.to(torch.int64)

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
        # # L1 regularization factor
        lambda_l1 = 0.1
        l1_penalty = torch.cat([p.view(-1) for p in self.parameters() if p.requires_grad]).abs().mean()
        loss += lambda_l1 * l1_penalty
        # torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)

        # Analyze loss conflicts
        self.polca_loss.loss_analyzer.step(self, [l for l in aux_losses if not isinstance(l, (int, float))])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, aux_losses


class LatentSpaceConv(nn.Module):

    def __init__(self, n_features, conv_out_channels, kernel_size=5, padding=2):
        super(LatentSpaceConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=conv_out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.fc = nn.Linear(conv_out_channels * n_features, n_features)

    def forward(self, z):
        batch_dim, n_features = z.shape
        z_b = z.unsqueeze(1)  # Shape: (batch_dim, 1, n_features)
        conv_out = self.conv(z_b)  # Shape: (batch_dim, conv_out_channels, n_features)
        flattened = conv_out.view(batch_dim, -1)  # Shape: (batch_dim, conv_out_channels * n_features)
        output = self.fc(flattened)  # Shape: (batch_dim, n_features)
        return output


def legendre_polynomial_cache(n, x, cache):
    """
    Compute the n-th Legendre polynomial P_n(x) using a recurrence relation, with caching of previously computed polynomials.

    Parameters:
    - n (int): Degree of the Legendre polynomial.
    - x (torch.Tensor): The input tensor of shape (batch_size,).
    - cache (dict): A dictionary to store and retrieve previously computed Legendre polynomials.

    Returns:
    - torch.Tensor: The n-th Legendre polynomial evaluated at x.
    """
    if n in cache:
        return cache[n]  # Return cached polynomial if already computed

    if n == 1:
        P1 = x
        cache[1] = P1
        return P1
    elif n == 2:
        P2 = (3 * x ** 2 - 1) / 2
        cache[2] = P2
        return P2
    else:
        # Recurrence relation: P_n(x) = ((2n-1)xP_(n-1) - (n-1)P_(n-2)) / n
        Pn_1 = legendre_polynomial_cache(n - 1, x, cache)
        Pn_2 = legendre_polynomial_cache(n - 2, x, cache)
        Pn = ((2 * n - 1) * x * Pn_1 - (n - 1) * Pn_2) / n
        cache[n] = Pn  # Store the computed polynomial in cache
        return Pn


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
        for i in range(3):
            layer = nn.Linear(latent_dim, latent_dim)
            # Apply orthogonal initialization
            nn.init.orthogonal_(layer.weight)
            # Apply the scaling to the weight matrix
            with torch.no_grad():
                # Get the number of features (output dimension)
                n_features, n_inputs = layer.weight.shape
                # Scale the orthogonal matrix
                scales = torch.linspace(1.0, 0.1, n_features)
                scaling_factor = 0.5  # To keep initial activations within a reasonable range for Tanh
                layer.weight.mul_(scales.unsqueeze(1) * scaling_factor)
            layers.append(layer)

        layer = nn.Tanh()
        layers.append(layer)

        self.post_encoder = nn.Sequential(*layers)

        if class_labels is not None:
            # Binary, multi-class and multi-label-binary classification
            self.classifier = nn.Linear(latent_dim, len(class_labels), bias=False)

    def forward(self, x):
        z = self.encoder(x)
        z = self.post_encoder(z)
        if self.class_labels is not None:
            return z, self.classifier(z)

        return z
