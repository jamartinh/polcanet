from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from polcanet.polcanet_utils import EncoderWrapper


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

        pos = torch.arange(latent_dim, dtype=torch.float32) ** 2
        self.register_buffer('a', pos / latent_dim)  # normalized axis
        target_decay = torch.exp(-decay_rate * torch.arange(latent_dim, dtype=torch.float32))
        # self.register_buffer('ne', torch.exp(-0.5 * torch.arange(latent_dim, dtype=torch.float32)))
        self.register_buffer('t', F.normalize(target_decay, p=1.0, dim=0))  # normalized target decay
        # self.register_buffer('I', 1-torch.eye(latent_dim))  # identity matrix

        self.loss_names = {
                "loss": "Total Loss",
                "rec": "Reconstruction Loss",
                "ort": "Orthogonality Loss",
                "com": "Center of Mass Loss",
                "var": "Variance Distribution Loss"

        }

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
        e: energy (squared z**2) z being unit vectors

        Losses:
        L1: reconstruction loss
        L2: orthogonality loss
        L3: center of mass loss
        L4: exponential decay variance loss
        L: total combined loss
        """

        if isinstance(z, (tuple, list)):
            z = z[0] * z[1]
        # Reconstruction loss
        # Purpose: Ensure the model can accurately reconstruct the input data
        # Method: Mean Squared Error between input and reconstruction
        L_rec = F.mse_loss(r, x)

        # Orthogonality loss
        # Purpose: Encourage latent dimensions to be uncorrelated
        # Method: Penalize off-diagonal elements of the cosine similarity matrix
        device = z.device
        if self.alpha != 0:

            Z = F.normalize(z, p=2, dim=0)
            S = torch.mm(Z.t(), Z)
            # Set the correlation between the first two components to zero in the upper triangular part
            # S[0, 1] *= 10  # Only need to set this one, not the symmetric one
            idx0, idx1 = torch.triu_indices(S.shape[0], S.shape[1], offset=1)  # indices of triu w/o diagonal
            S_triu = S[idx0, idx1]
            L_ort = torch.mean(S_triu ** 2)


        else:
            L_ort = torch.tensor([0], dtype=torch.float32, device=x.device)

        # Compute and normalize variance
        v = torch.var(z, dim=0)
        w = F.normalize(v, p=1.0, dim=0)
        e = torch.mean(z**2, dim=0)
        e = F.normalize(e, p=1.0, dim=0)

        # Center of mass loss
        # Purpose: Concentrate information in earlier latent dimensions
        # Method: Minimize the weighted average of normalized variances, e.g., the center of mass.
        if self.beta != 0:
            # L_com = torch.mean(z ** 2 * self.a, dim=0).mean() + torch.mean(v * self.a)
            L_com = torch.mean((w + e) * self.a, dim=0)
        else:
            L_com = torch.tensor([0], dtype=torch.float32, device=x.device)

        # Exponential decay variance loss
        # Purpose: Encourage an exponential decay of variances across latent dimensions
        # Method: Minimize the difference between normalized variances and a target exponential decay
        if self.gamma != 0:
            L_var = F.mse_loss(w, self.t)
        else:
            L_var = torch.tensor([0], dtype=torch.float32, device=x.device)

        # Combine losses
        # Purpose: Balance all loss components for optimal latent representation
        # Method: Weighted sum of individual losses
        L = L_rec + self.alpha * L_ort + self.beta * L_com + self.gamma * L_var + v.mean()

        return L, (L_rec, L_ort, L_com, L_var)


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
        std_metrics (torch.Tensor): The standard deviation of the latent space.
        mean_metrics (torch.Tensor): The mean of the latent space.
        r_mean_metrics (torch.Tensor): The mean of the reconstruction error.

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

    def __init__(self, encoder, decoder, latent_dim, alpha=0.1, beta=0.1, gamma=0.01,
                 device="cpu", center=True, factor_scale=False):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device(device)

        self.encoder = EncoderWrapper(encoder, factor_scale) if center or factor_scale else encoder
        self.decoder = decoder

        if hasattr(self.encoder, "decoder"):
            # remove the attribute decoder from the encoder
            self.encoder.decoder = None
            del self.encoder.decoder

        if hasattr(self.decoder, "encoder"):
            # remove the attribute encoder from the decoder
            self.decoder.encoder = None
            del self.decoder.encoder

        # Define std_metrics, mean_metrics and as buffers
        self.register_buffer("std_metrics", torch.zeros(latent_dim))
        self.register_buffer("mean_metrics", torch.zeros(latent_dim))
        self.register_buffer("r_mean_metrics", torch.zeros(1))

        self.polcanet_loss = PolcaNetLoss(latent_dim, alpha, beta, gamma)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
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

            latent = self.encoder(x)
            z = latent

        return z.detach().cpu().numpy()

    def decode(self, z, mask=None):
        self.encoder.eval()
        if z.shape[1] < self.latent_dim:
            mask = np.concatenate([np.ones(z.shape[1]), np.zeros(self.latent_dim - z.shape[1])])

        if mask is not None:
            if len(mask) != z.shape[1]:
                mask = np.array(mask)
                mask = np.concatenate([mask, np.zeros(z.shape[1] - len(mask))])
            z = np.where(mask == 0, self.mean_metrics.cpu().numpy(), z)

        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
            r = self.decoder(z)
            r = r.detach().cpu().numpy()
        return r

    def update_metrics(self, z, r, x):
        # make a random update with probability 0.25
        if np.random.rand() < 0.5:
            x = x.detach()
            if isinstance(z, (tuple, list)):
                z = z[0] * z[1]
            z = z.detach()
            r = r.detach()
            self.r_mean_metrics += 0.01 * (torch.mean((x - r) ** 2) - self.r_mean_metrics)
            self.std_metrics += 0.01 * (torch.std(z, dim=0) - self.std_metrics)
            self.mean_metrics += 0.01 * (torch.mean(z, dim=0) - self.mean_metrics)

    def r_error(self, in_x, w=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, w)
        error = (in_x - r).reshape(in_x.shape[0], -1)
        return np.mean(error ** 2, axis=-1)

    def score(self, in_x, w=None):
        """Compute the reconstruction score"""
        return np.abs(self.r_error(in_x, w) - self.r_mean_metrics.cpu().numpy())

    def to_device(self, device):
        """Move the encoder to the given device."""
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def train_model(self, data, batch_size=None, num_epochs=100, report_freq=10, lr=1e-3, verbose=1):
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if batch_size is None:
                raise ValueError("batch_size must be provided when data is in memory")
            fitter = self.fitter_in_memory(data, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
        elif isinstance(data, torch.utils.data.DataLoader):
            fitter = self.fitter_data_loader(data, num_epochs=num_epochs, lr=lr)
        else:
            fitter = None

        assert fitter is not None, "Data loader must be either a DataLoader or a numpy array or torch tensor"
        epoch_progress = tqdm(fitter, desc="epoch", leave=True,
                              total=num_epochs, miniters=report_freq, disable=bool(not verbose))
        metrics = {}
        epoch = 0
        for epoch, metrics in epoch_progress:
            if epoch % report_freq == 0:
                epoch_progress.set_postfix(metrics)

        # pretty print final metrics stats
        if bool(verbose):
            print(f"Final metrics at epoch: {epoch + 1}")
            for k, v in metrics.items():
                print(f"{self.polcanet_loss.loss_names[k]}: {v:.4g}")

        return metrics["loss"]

    def fitter_data_loader(self, data_loader, num_epochs=100, lr=1e-3):
        self.train()
        # Create the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            for batch in data_loader:
                x = batch
                x = x.to(self.device)
                loss, aux_losses = self.learn_step(x, optimizer=optimizer)
                losses["loss"].append(loss.item())
                for idx, name in enumerate(list(self.polcanet_loss.loss_names)[1:]):
                    losses[name].append(aux_losses[idx].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

    def fitter_in_memory(self, data: np.ndarray | torch.Tensor, batch_size=512, num_epochs=100, lr=1e-3):
        num_samples: int = len(data)
        self.train()
        if not isinstance(data, torch.Tensor):
            torch_data = torch.tensor(data, dtype=torch.float32).to(self.device)
        else:
            torch_data = data.to(self.device)

        # Create the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            indices = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                x = torch_data[batch_indices]
                loss, aux_losses = self.learn_step(x, optimizer)

                losses["loss"].append(loss.item())
                for aux_loss, name in enumerate(list(self.polcanet_loss.loss_names)[1:]):
                    losses[name].append(aux_losses[aux_loss].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

    def learn_step(self, x, optimizer):
        z, r = self.forward(x)
        loss, aux_losses = self.polcanet_loss(z, r, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.update_metrics(z, r, x)
        return loss, aux_losses
