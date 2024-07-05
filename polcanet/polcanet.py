from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm


class FakeScaler:
    def __init__(self):
        pass

    @staticmethod
    def transform(x):
        return x

    @staticmethod
    def inverse_transform(x):
        return x

    @staticmethod
    def fit(data):
        pass


class PolcaNet(nn.Module):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network model
    that is used for Principal Latent Components Analysis (Polca).

    Attributes:
        alpha (float): The weight for the cross-correlation loss.
        beta (float): The weight for the center of mass loss.
        gamma (float): The weight for the low variance loss.
        device (str): The device to run the model on ("cpu" or "cuda").
        model (examples.example_aencoders.BaseAutoEncoder): The base autoencoder model.
        bias (nn.Parameter): A learnable constant added to the latent space.
        reconstruction_loss_fn (nn.MSELoss): The reconstruction loss function.
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
        to_device(device): Moves the model to the given device.
        train_model(data, batch_size, num_epochs, report_freq, lr): Trains the model.
        fitter(data, batch_size, num_epochs, lr): Fits the model to the data.
        learn_on_batch(x, optimizer): Performs a learning step on a batch.
        compute_losses(z, r, x): Computes the losses.
    """

    def __init__(self, model, alpha=0.1, beta=0.1, gamma=0.01, scaler=None, device="cpu"):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device(device)

        self.bias = None
        self.model = model

        self.reconstruction_loss_fn = nn.MSELoss()
        self.scaler = scaler or FakeScaler()
        self.std_metrics = None
        self.mean_metrics = None
        self.r_mean_metrics = None

        self.aux_loss_names = ["decode", "orth", "com", "var"]

    def forward(self, x, mask=None):
        latent = self.model.encode(x)
        if self.bias is None:
            self.bias = nn.Parameter(torch.zeros(latent.shape[1]))
            self.to(self.device)
        latent = latent + self.bias  # Add self.bias to latent
        reconstruction = self.model.decode(latent)
        return latent, reconstruction

    def predict(self, in_x, mask=None):
        z = self.encode(in_x)
        r = self.decode(z, mask=mask)
        return z, r

    def encode(self, in_x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(in_x, dtype=torch.float32, device=self.device)
            x = self.scaler.transform(x)
            latent = self.model.encode(x)
            if self.bias is None:
                self.bias = nn.Parameter(torch.zeros(latent.shape[1]))
                self.to(self.device)
            z = latent + self.bias

        return z.detach().cpu().numpy()

    def decode(self, z, mask=None):
        self.model.eval()
        if mask is not None:
            if len(mask) != self.bias.shape[0]:
                mask = np.array(mask)
                mask = np.concatenate([mask, np.zeros(self.latent_dim - len(mask))])

            z = np.where(mask == 0, self.bias.detach().cpu().numpy(), z)
        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
            r = self.model.decode(z)
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
        return np.mean((in_x - r) ** 2, axis=1)

    def score(self, in_x, w=None):
        """Compute the reconstruction score"""
        return np.abs(self.r_error(in_x, w) - self.r_mean_metrics)

    def to_device(self, device):
        """Move the model to the given device."""
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)
        self.model.to(device)
        return self

    def train_model(self, data, batch_size=512, num_epochs=100, report_freq=10, lr=1e-3):
        fitter = self.fitter(data, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
        epoch_progress = tqdm(fitter, desc="epoch", leave=True, total=num_epochs)
        metrics = {}
        epoch = 0
        for epoch, metrics in epoch_progress:
            if epoch % report_freq == 0:
                epoch_progress.set_postfix(metrics)

        # pretty print final metrics stats
        print(f"Final metrics at epoch: {epoch}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4g}")

    def fitter(self, data, batch_size=512, num_epochs=100, lr=1e-3):

        self.to(self.device)

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Create DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        with torch.no_grad():
            # Make some initializations
            _, _ = self.forward(data[0:2].to(self.device))

        # Create an optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            for batch in dataloader:
                x = batch[0].to(self.device)
                loss, aux_losses = self.learn_on_batch(x, optimizer)
                losses["loss"].append(loss.item())
                for aux_loss, name in enumerate(self.aux_loss_names):
                    losses[name].append(aux_losses[aux_loss].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

        self.update_metrics(data)

    def learn_on_batch(self, x, optimizer):
        z, r = self.forward(x)
        loss, aux_losses = self.compute_loss(z, r, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, aux_losses

    @staticmethod
    def cross_correlation_loss(latent):
        """Compute the cross correlation loss"""
        z = latent
        rnd = 1e-12 * torch.randn(size=z.shape, device=latent.device)
        x = z + rnd
        x_corr = torch.corrcoef(x.T)
        ll = x_corr.shape[0]
        idx = torch.triu_indices(ll, ll, offset=1)  # indices of triu w/o diagonal
        x_triu = x_corr[idx[0], idx[1]]
        corr_matrix_triu = x_triu
        corr_loss = torch.mean(corr_matrix_triu.pow(2))
        return corr_loss

    @staticmethod
    def orthogonality_loss(latent: torch.Tensor) -> torch.Tensor:
        """
        Compute the average pairwise orthogonality across all latent in a batch.

        Args:
        latent (torch.Tensor): Input tensor of shape (batch_dim, n_features)

        Returns:
        torch.Tensor: Average pairwise orthogonality measure for each batch
        """
        # Normalize the latent
        normalized_vectors = latent / torch.norm(latent, dim=1, keepdim=True)

        # Compute pairwise cosine similarities
        cosine_similarities = torch.matmul(normalized_vectors, normalized_vectors.transpose(0, 1))

        # Set diagonal to zero to exclude self-similarities
        mask = torch.eye(cosine_similarities.shape[0], device=latent.device).bool()
        cosine_similarities = cosine_similarities.masked_fill(mask, 0)

        # Compute orthogonality measure: |cos(theta)|
        orthogonality = torch.abs(cosine_similarities)

        # Compute average, excluding the diagonal zeros
        n = latent.shape[0]
        average_orthogonality = orthogonality.sum() / (n * (n - 1))

        return average_orthogonality

    @staticmethod
    def center_of_mass_loss(latent):
        """Calculate center of mass loss"""
        axis = torch.arange(0, latent.shape[1], device=latent.device, dtype=torch.float32) ** 2
        std_latent = torch.var(latent, dim=0)

        w = nn.functional.normalize(std_latent, p=1.0, dim=0)
        com = w * axis / axis.shape[0]  # weight the latent space
        loss = torch.mean(com)
        return loss

    @staticmethod
    def exp_decay_var_loss(latent, decay_rate=0.5):
        """Encourage exponential decay of variances"""
        var_latent = torch.var(latent, dim=0)

        # Create exponential decay target
        target_decay = torch.exp(-decay_rate * torch.arange(latent.shape[1], device=latent.device, dtype=torch.float32))

        # Normalize both to sum to 1 for fair comparison
        var_latent_norm = var_latent / torch.sum(var_latent)
        target_decay_norm = target_decay / torch.sum(target_decay)

        return torch.nn.functional.mse_loss(var_latent_norm, target_decay_norm)

    def compute_loss(self, z, r, x):
        # reconstruction loss
        l1 = self.reconstruction_loss_fn(r, x)
        # correlation loss
        # l2 = self.compute_cross_correlation_loss(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32,
        #                                                                                  device=x.device)
        l2 = self.orthogonality_loss(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)
        # ordering loss
        l3 = self.center_of_mass_loss(z) if self.beta != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)
        # low variance loss
        l4 = self.exp_decay_var_loss(z) if self.gamma != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)

        mean_loss = (torch.mean(z, dim=0) - self.bias).pow(2).sum()
        loss = l1 + self.alpha * l2 + self.beta * l3 + self.gamma * l4 + mean_loss

        return loss, (l1, l2, l3, l4)
