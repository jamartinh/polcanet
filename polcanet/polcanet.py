import math
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

    def to(self, device):
        return self


# Custom weight initialization function
def custom_weight_init(m):
    """
        Custom weight initialization for linear layers.

        This function initializes the weights of linear layers to account for non-uniform variance
        in input features. The input vector is assumed to have independent features with variance
        concentrated in the first components (from left to right). The initialization starts with
        Xavier initialization and applies custom scaling to emphasize the higher variance in the
        first components.

        Theory:
            Standard weight initialization techniques like Xavier or He initialization aim to preserve
            the variance of activations across layers. However, these methods assume uniform variance
            across input features, which may not be optimal when the input features have non-uniform
            variance.

            In cases where the input features have higher variance in the first components, a custom
            scaling approach can be used. This involves:
            1. Initializing weights using Xavier initialization.
            2. Applying a custom scaling factor based on the feature index, with larger values for
               weights corresponding to the first components.

            This approach ensures that the initialized weights reflect the variance distribution of the
            input features, potentially leading to better training dynamics and performance.

        Args:
            m (nn.Module): The module to initialize. This function only applies to nn.Linear layers.

        Example:
            >>> layer = nn.Linear(10, 5)
            >>> custom_weight_init(layer)
            >>> print(layer.weight)

        """
    if isinstance(m, nn.Linear):
        # Xavier initialization as base
        fan_in, fan_out = m.weight.size(1), m.weight.size(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))

        # Custom scaling based on feature index
        weight_shape = m.weight.shape
        custom_scale = torch.linspace(1.5, 0.5, weight_shape[1])  # Example scaling from 1.5 to 0.5
        custom_scale = custom_scale.view(1, -1).expand(weight_shape)

        # Initialize weights
        with torch.no_grad():
            m.weight.copy_(torch.randn(weight_shape) * std * custom_scale)

        # Bias initialization (optional, can be zero)
        if m.bias is not None:
            with torch.no_grad():
                m.bias.fill_(0.0)


class LinearDecoder(nn.Module):
    """
        A linear decoder module for an autoencoder.

        This class implements a versatile linear decoder that accepts a vector of latent features
        and outputs data in a specified shape. The decoder is fully linear (no nonlinearities) and
        can have multiple linear layers.

        Args:
            latent_dim (int): Dimension of the latent vector.
            output_shape (tuple): Desired shape of a single instance of the output data.
            num_layers (int): Number of linear layers in the decoder (default is 1).

        Attributes:
            latent_dim (int): Dimension of the latent vector.
            output_shape (tuple): Desired shape of a single instance of the output data.
            output_dim (int): Total number of elements in the output data, calculated as the product
                              of the dimensions in output_shape.
            decoder (nn.Sequential): Sequential container of linear layers.

        Methods:
            forward(x):
                Passes the input latent vector through the linear layers and reshapes the output to
                the specified output shape, including the batch dimension.

        Example:
            >>> latent_dim = 16
            >>> output_shape = (3, 32, 32)  # Shape of a 32x32 RGB image
            >>> num_layers = 2
            >>> decoder = LinearDecoder(latent_dim, output_shape, num_layers)
            >>> latent_vector = torch.randn((4, latent_dim))  # Batch of 4 latent vectors
            >>> output = decoder(latent_vector)
            >>> print(output.shape)
            torch.Size([4, 3, 32, 32])

        Example 2: Decoding to a vector shape
            >>> latent_dim = 8
            >>> output_shape = (20,)  # Shape of a vector with 20 elements
            >>> num_layers = 3
            >>> decoder = LinearDecoder(latent_dim, output_shape, num_layers)
            >>> latent_vector = torch.randn((5, latent_dim))  # Batch of 5 latent vectors
            >>> output = decoder(latent_vector)
            >>> print(output.shape)
            torch.Size([5, 20])
    """

    def __init__(self, latent_dim, output_shape, hidden_dim=256, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.output_dim = int(torch.prod(torch.tensor(output_shape)))

        layers = []
        input_dim = latent_dim
        for i in range(num_layers - 1):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim)
                # Apply custom weight initialization to the first layer only
                custom_weight_init(layer)
                layers.append(layer)
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x)
        return x.view(-1, *self.output_shape)

    def decode(self, z):
        reconstruction = self.forward(z)
        return reconstruction


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
        if hasattr(self.encoder, "decoder"):
            self.encoder.decoder = None

        # A middleware layers to add non-linearity to the latent space including batch norms
        # self.middleware = nn.Sequential(
        #     # nn.BatchNorm1d(latent_dim),
        #     # nn.Linear(latent_dim, 256),
        #     # nn.Mish(),
        #     # nn.BatchNorm1d(256),
        #     # nn.Linear(256, 256),
        #     # nn.Mish(),
        #     # nn.BatchNorm1d(256),
        #     # nn.Linear(256, 256),
        #     nn.Linear(latent_dim, latent_dim)
        # )

        self.decoder = decoder
        if hasattr(self.decoder, "encoder"):
            self.encoder.encoder = None

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

    def forward(self, x, mask=None):
        latent = self.encoder.encode(x)
        # latent = self.middleware(latent)
        latent = latent
        reconstruction = self.decoder.decode(latent)
        return latent, reconstruction

    def predict(self, in_x, mask=None):
        z = self.encode(in_x)
        r = self.decode(z, mask=mask)
        return z, r

    def encode(self, in_x):
        self.encoder.eval()
        with torch.inference_mode():
            x = torch.tensor(in_x, dtype=torch.float32, device=self.device)
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
        with torch.torch.inference_mode():
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
        return np.mean((in_x - r) ** 2, axis=1)

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
        # self.encoder.to(device)
        # self.decoder.to(device)
        # self.middleware.to(device)
        # self.bias = self.bias.to(device)
        self.scaler.to(device)
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
            print(f"{self.aux_loss_names[k]}: {v:.4g}")

    def fitter(self, data, batch_size=512, num_epochs=100, lr=1e-3):
        self.train()
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.scaler.to(self.device)

        # Create DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # Create an optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            for batch in dataloader:
                x = batch[0].to(self.device)
                loss, aux_losses = self.learn_on_batch(x, optimizer)
                losses["loss"].append(loss.item())
                for aux_loss, name in enumerate(list(self.aux_loss_names)[1:]):
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
        l1 = nn.functional.mse_loss(r, x)
        # correlation loss
        # l2 = self.compute_cross_correlation_loss(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32,
        #                                                                                  device=x.device)
        l2 = self.orthogonality_loss(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)
        # ordering loss
        l3 = self.center_of_mass_loss(z) if self.beta != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)
        # low variance loss
        l4 = self.exp_decay_var_loss(z) if self.gamma != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)

        #mean_loss = (torch.mean(z, dim=0) - self.bias).pow(2).sum()
        loss = l1 + self.alpha * l2 + self.beta * l3 + self.gamma * l4 #+ mean_loss

        return loss, (l1, l2, l3, l4)
