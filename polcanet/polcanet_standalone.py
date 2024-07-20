from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from polcanet.polcanet_utils import FakeScaler


class PolcaNetLoss(nn.Module):
    """
    PolcaNetLoss is a class that extends PyTorch's Module class. It represents the loss function used for
    Principal Latent Components Analysis (Polca).
    """

    def __init__(self, latent_dim, alpha=1.0, beta=1.0, gamma=1.0, decay_rate=0.5):
        """
        Initialize PolcaNetLoss with the provided parameters.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        axis = torch.arange(0, latent_dim, dtype=torch.float32) ** 2
        self.register_buffer('a', axis / axis.shape[0])  # normalized axis

        target_decay = torch.exp(-decay_rate * torch.arange(latent_dim, dtype=torch.float32))
        self.register_buffer('ne', torch.exp(-0.1 * torch.arange(latent_dim, dtype=torch.float32)))
        self.register_buffer('t', F.normalize(target_decay, p=1.0, dim=0))  # normalized target decay

        self.register_buffer('I', torch.eye(latent_dim))  # identity matrix

    def forward(self, z, r, x):
        """
        Variables:
        z: latent representation
        r: reconstruction
        x: input data
        n: number of latent dimensions
        Z: normalized latent representation (L2 norm)
        S: cosine similarity matrix
        v: variance of latent representation
        w: normalized variance (L1 norm)

        Losses:
        L1: reconstruction loss
        L2: orthogonality loss
        L3: center of mass loss
        L4: exponential decay variance loss
        L: total combined loss
        """

        # Reconstruction loss
        # Purpose: Ensure the model can accurately reconstruct the input data
        # Method: Mean Squared Error between input and reconstruction
        L1 = F.mse_loss(r, x)

        # Orthogonality loss
        # Purpose: Encourage latent dimensions to be uncorrelated
        # Method: Penalize off-diagonal elements of the cosine similarity matrix
        if self.alpha != 0:
            # n = z.shape[1]
            Z = F.normalize(z, p=2, dim=0)
            S = torch.mm(Z.t(), Z * self.ne)
            idx0, idx1 = torch.triu_indices(S.shape[0], S.shape[0], offset=1)  # indices of triu w/o diagonal
            S_triu = S[idx0, idx1]
            L2 = torch.mean(S_triu ** 2)
        else:
            L2 = torch.tensor([0], dtype=torch.float32, device=x.device)

        # L2 = ((S ** 2) * (1 - self.I)).sum() / (n * (n - 1))

        # Compute and normalize variance
        v = torch.var(z, dim=0)
        w = F.normalize(v, p=1.0, dim=0)

        # Center of mass loss
        # Purpose: Concentrate information in earlier latent dimensions
        # Method: Minimize the weighted average of normalized variances, e.g., the center of mass.
        if self.beta != 0:
            L3 = torch.mean(w * self.a)
        else:
            L3 = torch.tensor([0], dtype=torch.float32, device=x.device)

        # Exponential decay variance loss
        # Purpose: Encourage an exponential decay of variances across latent dimensions
        # Method: Minimize the difference between normalized variances and a target exponential decay
        if self.gamma != 0:
            L4 = F.mse_loss(w, self.t)
        else:
            L4 = torch.tensor([0], dtype=torch.float32, device=x.device)

        # Combine losses
        # Purpose: Balance all loss components for optimal latent representation
        # Method: Weighted sum of individual losses
        L = L1 + self.alpha * L2 + self.beta * L3 + self.gamma * L4

        return L, (L1, L2, L3, L4)


class PolcaNet(nn.Module):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network encoder
    that is used for Principal Latent Components Analysis (Polca).

    Attributes:
        alpha (float): The weight for the cross-correlation loss.
        beta (float): The weight for the center of mass loss.
        gamma (float): The weight for the low variance loss.
        device (str): The device to run the encoder on ("cpu" or "cuda").
        encoder (examples.example_aencoders.BaseAutoEncoder): The base autoencoder encoder.
        scaler (examples.example_aencoders.MinMaxScalerTorch): The scaler used to normalize the data.
        std_metrics (torch.Tensor): The standard deviation of the latent space.
        mean_metrics (torch.Tensor): The mean of the latent space.
        r_mean_metrics (torch.Tensor): The mean of the reconstruction error.
        aux_loss_names (list): The names of the auxiliary losses.

    Methods:
        forward(x): Encodes the input, adds the mean to the latent space, and decodes the result.
        compute_cross_correlation_loss(latent): Computes the cross-correlation loss.
        compute_center_of_mass_loss(latent): Computes the center of mass loss.
        low_variance_loss(latent): Minimizes the variance in the batch.
        predict(in_x, w): Encodes and decodes the input.
        encode(in_x): Encodes the input and adds the mean to the latent space.
        decode(z, w): Decodes the latent space.
        update_metrics(in_x): Updates the metrics based on the input.
        inv_score(x, idx): Scores the energy of a given x and y.
        r_error(in_x, w): Computes the reconstruction error.
        score(in_x, w): Computes the reconstruction score.
        to_device(device): Moves the encoder to the given device.
        train_model(data, batch_size, num_epochs, report_freq, lr): Trains the encoder.
        fitter(data, batch_size, num_epochs, lr): Fits the encoder to the data.
        learn_on_batch(x, optimizer): Performs a learning step on a batch.
        compute_losses(z, r, x): Computes the losses.
    """

    def __init__(self, encoder, decoder, latent_dim, alpha=0.1, beta=0.1, gamma=0.01, scaler=None, device="cpu"):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device(device)

        self.encoder = encoder
        self.decoder = decoder

        if hasattr(self.encoder, "decoder"):
            # remove the attribute decoder from the encoder
            self.encoder.decoder = None
            del self.encoder.decoder

        if hasattr(self.decoder, "encoder"):
            # remove the attribute encoder from the decoder
            self.decoder.encoder = None
            del self.decoder.encoder

        self.scaler = scaler or FakeScaler()
        self.std_metrics = None
        self.mean_metrics = None
        self.r_mean_metrics = None

        # Loss names
        self.aux_loss_names = {
                "loss": "Total Loss",
                "rec": "Reconstruction Loss",
                "ort": "Orthogonality Loss",
                "com": "Center of Mass Loss",
                "var": "Variance Distribution Loss"

        }
        self.polcanet_loss = PolcaNetLoss(latent_dim, alpha, beta, gamma)

    def forward(self, x, mask=None):
        latent = self.encoder.encode(x)
        latent = latent
        reconstruction = self.decoder.decode(latent)
        return latent, reconstruction

    def predict(self, in_x, mask=None):
        z = self.encode(in_x)
        r = self.decode(z, mask=mask)
        return z, r

    def encode(self, in_x):
        self.encoder.eval()
        with torch.no_grad():
            if isinstance(in_x, torch.Tensor):
                x = in_x.detach().clone().to(self.device)
            else:
                x = torch.tensor(in_x, dtype=torch.float32, device=self.device, requires_grad=False)
            x = self.scaler.transform(x)
            latent = self.encoder.encode(x)
            # latent = self.middleware(latent)
            z = latent

        return z.detach().cpu().numpy()

    def decode(self, z, mask=None):
        self.encoder.eval()
        if mask is not None:
            if len(mask) != z.shape[1]:
                mask = np.array(mask)
                mask = np.concatenate([mask, np.zeros(z.shape[1] - len(mask))])

            z = np.where(mask == 0, self.mean_metrics, z)
        with torch.inference_mode():
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
            r = self.decoder.decode(z)
            r = self.scaler.inverse_transform(r)
            r = r.detach().cpu().numpy()
        return r

    def update_metrics(self, in_x):
        if isinstance(in_x, torch.Tensor):
            in_x = in_x.detach().cpu().numpy()
        z = self.encode(in_x)
        r = self.decode(z)
        self.r_mean_metrics = np.mean((in_x - r) ** 2)
        self.std_metrics = np.std(z, axis=0)
        self.mean_metrics = np.mean(z, axis=0)

    def r_error(self, in_x, w=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, w)
        error = (in_x - r).reshape(in_x.shape[0], -1)
        return np.mean(error ** 2, axis=-1)

    def score(self, in_x, w=None):
        """Compute the reconstruction score"""
        return np.abs(self.r_error(in_x, w) - self.r_mean_metrics)

    def to_device(self, device):
        """Move the encoder to the given device."""
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)
        self.scaler.to(device)
        return self

    def train_model(self, data, batch_size=512, num_epochs=100, report_freq=10, lr=1e-3):
        fitter = self.fitter(data,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             lr=lr,
                             )
        epoch_progress = tqdm(fitter, desc="epoch", leave=True, total=num_epochs, miniters=report_freq)
        metrics = {}
        epoch = 0
        for epoch, metrics in epoch_progress:
            if epoch % report_freq == 0:
                epoch_progress.set_postfix(metrics)

        # pretty print final metrics stats
        print(f"Final metrics at epoch: {epoch}")
        for k, v in metrics.items():
            print(f"{self.aux_loss_names[k]}: {v:.4g}")

    def fitter(self, data: np.ndarray | torch.Tensor, batch_size=512, num_epochs=100, lr=1e-3):
        num_samples: int = len(data)
        self.train()
        if not isinstance(data, torch.Tensor):
            torch_data = torch.tensor(data, dtype=torch.float32).to(self.device)
        else:
            torch_data = data.to(self.device)
        self.scaler.fit(torch_data)
        torch_data = self.scaler.transform(torch_data)
        self.scaler.to(self.device)

        # Create the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            indices = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                x = torch_data[batch_indices]
                loss, aux_losses = self.learn_step(x, optimizer)

                losses["loss"].append(loss.item())
                for aux_loss, name in enumerate(list(self.aux_loss_names)[1:]):
                    losses[name].append(aux_losses[aux_loss].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

        self.update_metrics(data)

    def learn_step(self, x, optimizer):
        z, r = self.forward(x)
        loss, aux_losses = self.polcanet_loss(z, r, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, aux_losses

    # @staticmethod
    # def cross_correlation_loss(latent):
    #     """Compute the cross correlation loss"""
    #     z = latent
    #     rnd = 1e-12 * torch.randn(size=z.shape, device=latent.device)
    #     x = z + rnd
    #     x_corr = torch.corrcoef(x.T)
    #     ll = x_corr.shape[0]
    #     idx = torch.triu_indices(ll, ll, offset=1)  # indices of triu w/o diagonal
    #     x_triu = x_corr[idx[0], idx[1]]
    #     corr_matrix_triu = x_triu
    #     corr_loss = torch.mean(corr_matrix_triu.pow(2))
    #     return corr_loss
    #
    # @staticmethod
    # def orthogonality_loss(latent: torch.Tensor):
    #     # Normalize each column (feature)
    #     normalized = nn.functional.normalize(latent, p=2, dim=0)
    #
    #     # Calculate pairwise cosine similarity
    #     cosine_sim = torch.mm(normalized.t(), normalized)
    #
    #     # Create a mask to ignore the diagonal (self-similarity)
    #     mask = torch.eye(cosine_sim.shape[0], device=cosine_sim.device)
    #
    #     # Square the cosine similarities to penalize both positive and negative correlations
    #     # and mask out the diagonal
    #     loss = (cosine_sim ** 2) * (1 - mask)
    #
    #     # Sum all pairwise losses and normalize by the number of pairs
    #     n = cosine_sim.shape[0]
    #     total_loss = loss.sum() / (n * (n - 1))
    #
    #     return total_loss
    #
    # @staticmethod
    # def center_of_mass_loss(latent):
    #     """Calculate center of mass loss"""
    #     axis = torch.arange(0, latent.shape[1], device=latent.device, dtype=torch.float32) ** 2
    #     std_latent = torch.var(latent, dim=0)
    #
    #     w = nn.functional.normalize(std_latent, p=1.0, dim=0)
    #     com = w * axis / axis.shape[0]  # weight the latent space
    #     loss = torch.mean(com)
    #     return loss
    #
    # @staticmethod
    # def exp_decay_var_loss(latent, decay_rate=0.5):
    #     """Encourage exponential decay of variances"""
    #     var_latent = torch.var(latent, dim=0)
    #
    #     # Create exponential decay target
    #     target_decay = torch.exp(-decay_rate * torch.arange(latent.shape[1], device=latent.device, dtype=torch.float32))
    #
    #     # Normalize both to sum to 1 for fair comparison
    #     var_latent_norm = var_latent / torch.sum(var_latent)
    #     target_decay_norm = target_decay / torch.sum(target_decay)
    #
    #     return torch.nn.functional.mse_loss(var_latent_norm, target_decay_norm)
    #
    # def compute_loss(self, z, r, x):
    #     # reconstruction loss
    #     l1 = nn.functional.mse_loss(r, x)
    #     # correlation loss
    #     # l21 = self.cross_correlation_loss(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32,
    #     #                                                                                  device=x.device)
    #     l2 = self.orthogonality_loss(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)
    #     # l2 = l21 + l22
    #     # ordering loss
    #     l3 = self.center_of_mass_loss(z) if self.beta != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)
    #     # low variance loss
    #     l4 = self.exp_decay_var_loss(z) if self.gamma != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)
    #
    #     # mean_loss = (torch.mean(z, dim=0) - self.bias).pow(2).sum()
    #     loss = l1 + self.alpha * l2 + self.beta * l3 + self.gamma * l4
    #     return loss, (l1, l2, l3, l4)
