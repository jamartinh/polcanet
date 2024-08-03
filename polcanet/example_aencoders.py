import warnings

import numpy as np
import torch
from torch import nn as nn


class MinMaxScalerTorch:
    def __init__(self, min_val=None, max_val=None):
        """Initialize the parameters used by the scaler."""
        self.max_val = None
        self.min_val = None
        if max_val is not None and min_val is not None:
            self.min_val = min_val
            self.max_val = max_val

    def transform(self, act):
        """Normalize input data x using the scaler."""
        low, high = self.min_val, self.max_val
        scale = high - low
        eps = torch.finfo(torch.float32).eps
        scale[scale < eps] += eps
        act = (act - low) * 2.0 / scale - 1.0
        return act

    def inverse_transform(self, act):
        """Undo the normalization effect on x."""
        act = torch.clamp(act, -1.0, 1.0)
        low, high = self.min_val, self.max_val
        act = low + (high - low) * (act + 1.0) / 2.0
        return act

    def fit(self, data):
        self.min_val = torch.min(data, dim=0).values
        self.max_val = torch.max(data, dim=0).values

    def to(self, device):
        if self.min_val is not None and self.max_val is not None:
            self.min_val = self.min_val.to(device)
            self.max_val = self.max_val.to(device)


# Create a StandardScalerTorch receiving the data as input
class StandardScalerTorch:
    def __init__(self, data=None):
        """Initialize the parameters used by the scaler."""
        self.mean = None
        self.std = None
        if data is not None:
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0)

    def transform(self, act):
        """Normalize input data x using the scaler."""
        return (act - self.mean) / self.std

    def inverse_transform(self, act):
        """Undo the normalization effect on x."""
        return act * self.std + self.mean

    def fit(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def to(self, device):
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class DenseEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=3, act_fn=nn.Mish, first_layer_size: int = None,
                 hidden_size: int = None):
        super().__init__()
        self.latent_dim = latent_dim
        layers = []

        if isinstance(input_dim, (list, tuple)):
            input_dim = np.prod(input_dim)

        # Ensure first_layer_size and hidden_size are not both specified
        if first_layer_size is not None and hidden_size is not None:
            raise ValueError("first_layer_size and hidden_size cannot be defined both. Please specify only one.")

        # Determine the size of the first layer
        if hidden_size is not None:
            first_layer_size = hidden_size
        elif first_layer_size is None:
            first_layer_size = input_dim
        else:
            first_layer_size = first_layer_size

        # Calculate the step size if hidden_size is not specified
        step_size = (first_layer_size - latent_dim) / (num_layers + 1) if hidden_size is None else 0
        current_dim = first_layer_size

        # First layer
        layer = nn.Linear(input_dim, current_dim)
        torch.nn.init.orthogonal_(layer.weight)
        layers.append(layer)
        layers.append(act_fn())

        # Hidden layers
        for i in range(num_layers):
            next_dim = hidden_size if hidden_size is not None else int(current_dim - step_size)
            layer = nn.Linear(current_dim, next_dim)
            torch.nn.init.orthogonal_(layer.weight)
            layers.append(layer)
            layers.append(act_fn())
            current_dim = next_dim

        # Final layer to latent_dim
        layer = nn.Linear(current_dim, latent_dim)
        torch.nn.init.orthogonal_(layer.weight)
        layers.append(layer)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return z


class DenseDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, num_layers=3, act_fn=nn.Mish):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        layers = []
        if not isinstance(hidden_dim, (list, tuple)):
            hidden_dim = [hidden_dim] * num_layers
        elif len(hidden_dim) != num_layers:
            warnings.warn("The hidden_dim is an iterable, the length of hidden_dim must be equal to num_layers."
                          " Setting it to [hidden_dim] * num_layers.")
            num_layers = len(hidden_dim)

        assert len(hidden_dim) == num_layers, "The length of hidden_dim must be equal to num_layers."

        if isinstance(output_dim, (list, tuple)):
            output_dim = np.prod(output_dim)

        layer = nn.Linear(latent_dim, hidden_dim[0])
        # torch.nn.init.orthogonal_(layer.weight)
        layers.append(layer)
        layers.append(act_fn())
        for i in range(1, num_layers):
            layer = nn.Linear(hidden_dim[i - 1], hidden_dim[i])
            # torch.nn.init.orthogonal_(layer.weight)
            # if hidden_dim[i - 1] == hidden_dim[i]:
            #     layer = ResNet(layer)
            layers.append(layer)
            # layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(act_fn())

        layer = nn.Linear(hidden_dim[-1], output_dim)
        # torch.nn.init.orthogonal_(layer.weight)
        layers.append(layer)

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        reconstruction = self.decoder(z)
        return reconstruction.view(-1, *self.output_dim)


class DenseAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=3, act_fn=nn.Mish):
        super(DenseAutoEncoder, self).__init__()
        self.encoder = DenseEncoder(input_dim, latent_dim, hidden_dim, num_layers, act_fn)
        self.decoder = DenseDecoder(latent_dim, input_dim, hidden_dim, num_layers, act_fn)

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return z, reconstruction


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, seq_len, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, input_dim, seq_len, num_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, latent_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]  # Get the output of the last layer


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, seq_len, num_layers=2):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, output_dim, num_layers=num_layers, batch_first=True)
        self.output_dim = output_dim
        self.seq_len = seq_len

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm(x)
        return x


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for dimensionality reduction and feature extraction.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB images).
        latent_dim (int): Dimension of the latent space (output dimension).
        conv_dim (int): Dimension of the convolution (1 for 1D, 2 for 2D). Default is 2.
        num_layers (int): Number of convolutional layers. Default is 4.
        initial_channels (int): Number of channels for the first layer. Default is 16.
        growth_factor (int): Factor by which the number of channels grows in each subsequent layer. Default is 2.
        act_fn (nn.Module): Activation function to use. Default is nn.ReLU.
    """

    def __init__(self, input_channels: int, latent_dim: int, conv_dim: int = 2, num_layers: int = 4,
                 initial_channels: int = 16, growth_factor: int = 2, act_fn: nn.Module = nn.ReLU):
        super(ConvEncoder, self).__init__()

        self.conv_dim = conv_dim

        if conv_dim == 1:
            ConvLayer = nn.Conv1d
        elif conv_dim == 2:
            ConvLayer = nn.Conv2d
        else:
            raise ValueError("conv_dim must be 1 or 2")

        layers = []

        # Initial input channels
        current_channels = input_channels

        # Create convolutional layers
        for i in range(num_layers):
            next_channels = min(initial_channels * (growth_factor ** i), 512)
            layers.append(ConvLayer(current_channels, next_channels, kernel_size=3, stride=2, padding=1))
            layers.append(act_fn())
            current_channels = next_channels

        # Flatten layer
        layers.append(nn.Flatten())

        # Calculate the flattened size after convolutional layers
        dummy_input = torch.zeros(1, input_channels, 32, 32) if conv_dim == 2 else torch.zeros(1, input_channels, 32)
        flattened_output_size = nn.Sequential(*layers[:-1])(dummy_input).view(1, -1).size(1)

        # Linear layers
        layers.append(nn.Linear(flattened_output_size, latent_dim * 4))
        layers.append(act_fn())
        layers.append(nn.Linear(latent_dim * 4, latent_dim))

        # Define the encoder as a sequential model
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width) for 2D or (batch_size, channels, length) for 1D.

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, latent_dim).
        """
        if self.conv_dim == 1:
            x = x.permute(0, 2, 1)  # (batch, N, M) -> (batch, M, N)
        elif self.conv_dim == 2:
            if x.ndim == 3:  # if input is (batch, N, M)
                x = x.unsqueeze(1)  # (batch, N, M) -> (batch, 1, N, M)
        z = self.encoder(x)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, conv_dim=2, act_fn=None):
        super(ConvDecoder, self).__init__()
        self.conv_dim = conv_dim

        # Select appropriate Conv and ConvTranspose layers
        if conv_dim == 1:
            ConvTransposeLayer = nn.ConvTranspose1d
            UnflattenLayer = lambda input_shape: nn.Unflatten(1, (latent_dim, input_shape))
            self.input_channels = input_dim
            self.output_channels = input_dim
            self.flattened_size = 2
        elif conv_dim == 2:
            ConvTransposeLayer = nn.ConvTranspose2d
            UnflattenLayer = lambda input_shape: nn.Unflatten(1, (latent_dim, *input_shape))
            self.input_channels = 1
            self.output_channels = 1
            self.flattened_size = (2, 2)
        else:
            raise ValueError("conv_dim must be 1 or 2")

        act_fn = act_fn or nn.Identity
        self.decoder = nn.Sequential(UnflattenLayer(self.flattened_size),
                                     ConvTransposeLayer(latent_dim, 64, kernel_size=3, stride=2, padding=1,
                                                        output_padding=1),
                                     act_fn(),
                                     ConvTransposeLayer(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     act_fn(),
                                     ConvTransposeLayer(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     act_fn(),
                                     ConvTransposeLayer(16, self.output_channels, kernel_size=3, stride=2, padding=1,
                                                        output_padding=1))

    def forward(self, x):
        decoded = self.decoder(x)
        if self.conv_dim == 2:
            decoded = decoded.squeeze(1)  # (batch, 1, N, M) -> (batch, N, M)
        return decoded


class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, conv_dim=2, act_fn=nn.Mish):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(input_dim, latent_dim, conv_dim, act_fn=act_fn)
        self.decoder = ConvDecoder(latent_dim, input_dim, conv_dim, act_fn=act_fn)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def generate_2d_sinusoidal_data(N, M, num_samples):
    data = []
    for _ in range(num_samples):
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, M)
        xx, yy = np.meshgrid(x, y)

        # Random phase shifts for x and y directions
        phase_shift_x = np.random.uniform(0, 2 * np.pi)
        phase_shift_y = np.random.uniform(0, 2 * np.pi)

        # Random frequency multipliers for x and y directions
        freq_multiplier_x = np.random.uniform(0.5, 1.5)
        freq_multiplier_y = np.random.uniform(0.5, 1.5)

        # Generate sinusoidal data with random phase and frequency
        z = np.sin(2 * np.pi * freq_multiplier_x * xx + phase_shift_x) * np.cos(
            2 * np.pi * freq_multiplier_y * yy + phase_shift_y)
        data.append(z)

    return np.array(data).astype(np.float32)


def bent_function_image(N, M, a=1, b=1, c=1, d=1):
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(2 * np.pi * (a * X + b * Y)) + np.cos(2 * np.pi * (c * X - d * Y))
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    return Z_norm


def generate_bent_images(N, M, n_images, param_range=(0.1, 5)):
    """
    Generate multiple random bent function images.

    Parameters:
    n_images (int): Number of images to generate
    image_size (tuple): Size of each image as (height, width)
    param_range (tuple): Range for random parameters (min, max)

    Returns:
    numpy.ndarray: Array of shape (n_images, height, width) containing the bent function images
    """

    images = np.zeros((n_images, N, M))

    for i in range(n_images):
        # Generate random parameters
        a, b, c, d = np.random.uniform(*param_range, size=4)

        # Generate image
        images[i] = bent_function_image(N, M, a, b, c, d)

    return images
