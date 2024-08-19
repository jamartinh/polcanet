import numpy as np
import torch
from torch import nn as nn


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
    def __init__(self, latent_dim, output_dim, num_layers=3, act_fn=nn.Mish, last_layer_size: int = None,
                 hidden_size: int = None, output_act_fn=nn.Identity, bias=True):
        super().__init__()
        self.output_dim = output_dim
        layers = []

        if isinstance(output_dim, (list, tuple)):
            output_dim = np.prod(output_dim)

        # Ensure last_layer_size and hidden_size are not both specified
        if last_layer_size is not None and hidden_size is not None:
            raise ValueError("last_layer_size and hidden_size cannot be defined both. Please specify only one.")

        # Determine the size of the last layer
        if hidden_size is not None:
            last_layer_size = hidden_size
        elif last_layer_size is None:
            last_layer_size = output_dim
        else:
            last_layer_size = last_layer_size

        # Calculate the step size if hidden_size is not specified
        step_size = (last_layer_size - latent_dim) / (num_layers + 1) if hidden_size is None else 0
        current_dim = latent_dim

        # # First layer
        current_dim = current_dim + int(step_size) if hidden_size is None else hidden_size
        layer = nn.Linear(latent_dim, current_dim, bias=bias)
        torch.nn.init.orthogonal_(layer.weight)
        layers.append(layer)
        layers.append(act_fn())

        # Hidden layers
        for i in range(num_layers):
            next_dim = current_dim + int(step_size) if hidden_size is None else hidden_size
            layer = nn.Linear(current_dim, next_dim, bias=bias)
            torch.nn.init.orthogonal_(layer.weight)
            layers.append(layer)
            layers.append(act_fn())
            current_dim = next_dim

        # Final layer to output_dim
        layer = nn.Linear(current_dim, output_dim, bias=bias)
        torch.nn.init.orthogonal_(layer.weight)
        layers.append(layer)
        layers.append(output_act_fn())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x = self.decoder(z)
        x = x.view(x.size(0), *self.output_dim)  # Reshape to the original input shape
        return x


class DenseAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=3, act_fn=nn.Mish, output_act_fn=nn.Identity,
                 bias=True):
        super(DenseAutoEncoder, self).__init__()
        self.encoder = DenseEncoder(input_dim, latent_dim, num_layers, act_fn, hidden_size=hidden_dim)
        self.decoder = DenseDecoder(latent_dim, input_dim, num_layers, act_fn, hidden_size=hidden_dim,
                                    output_act_fn=output_act_fn, bias=bias)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


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
        act_fn: Activation function to use. Default is nn.ReLU.
    """

    def __init__(self, input_channels: int, latent_dim: int, conv_dim: int = 2, num_layers: int = 4,
                 initial_channels: int = 16, growth_factor: int = 2, act_fn=nn.ReLU):
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
            stride = 1 if i < 1 else 2
            layer = ConvLayer(current_channels, next_channels, kernel_size=3, stride=stride, padding=1)
            layers.append(layer)
            layers.append(act_fn())
            current_channels = next_channels

        # Flatten layer
        layers.append(nn.Flatten())

        # Calculate the flattened size after convolutional layers
        dummy_input = torch.zeros(1, input_channels, 32, 32) if conv_dim == 2 else torch.zeros(1, input_channels, 32)
        flattened_output_size = nn.Sequential(*layers[:-1])(dummy_input).view(1, -1).size(1)

        # Linear layers
        layer = nn.Linear(flattened_output_size, latent_dim * 4)
        layers.append(layer)
        layers.append(act_fn())
        layer = nn.Linear(latent_dim * 4, latent_dim)
        layers.append(layer)

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
    """
    Convolutional Decoder for reconstructing data from a latent space.

    Args:
        latent_dim (int): Dimension of the latent space (input dimension).
        output_channels (int): Number of output channels (e.g., 3 for RGB images).
        conv_dim (int): Dimension of the convolution (1 for 1D, 2 for 2D). Default is 2.
        num_layers (int): Number of transposed convolutional layers. Default is 4.
        initial_channels (int): Number of channels for the last layer before output. Default is 16.
        growth_factor (int): Factor by which the number of channels reduces in each subsequent layer. Default is 2.
        act_fn : Activation function to use. Default is nn.ReLU.
        output_act_fn: Activation function to apply to the output. Default is nn.Sigmoid for normalization purposes.
        final_output_size (int or tuple): Target output size for reconstruction (e.g., 32 or (32, 32)). Used to determine intermediate dimensions.
    """

    def __init__(self, latent_dim: int, output_channels: int, conv_dim: int = 2, num_layers: int = 4,
                 initial_channels: int = 16, growth_factor: int = 2, act_fn=nn.ReLU,
                 output_act_fn: nn.Module = nn.Sigmoid, final_output_size=(32, 32)):
        super(ConvDecoder, self).__init__()

        self.conv_dim = conv_dim

        if conv_dim == 1:
            ConvTransposeLayer = nn.ConvTranspose1d
        elif conv_dim == 2:
            ConvTransposeLayer = nn.ConvTranspose2d
        else:
            raise ValueError("conv_dim must be 1 or 2")

        layers = []

        # Calculate the flattened size after the encoder
        if isinstance(final_output_size, int):
            final_output_size = (final_output_size,)

        intermediate_size = [max(1, dim // (2 ** num_layers)) for dim in final_output_size]
        intermediate_channels = min(initial_channels * (growth_factor ** (num_layers - 1)), 512)
        flattened_intermediate_size = intermediate_channels * int(torch.prod(torch.tensor(intermediate_size)).item())
        print(
            f"Intermediate size: {intermediate_size}, intermediate channels: {intermediate_channels}, flattened size: {flattened_intermediate_size}")

        # Linear layers to go from latent space to intermediate representation
        layers.append(nn.Linear(latent_dim, latent_dim * 4))
        layers.append(act_fn())
        layers.append(nn.Linear(latent_dim * 4, flattened_intermediate_size))
        layers.append(act_fn())
        layers.append(nn.Unflatten(1, (intermediate_channels, *intermediate_size)))

        # Create transposed convolutional layers
        current_channels = intermediate_channels
        for i in range(num_layers - 1):
            next_channels = max(current_channels // growth_factor, output_channels)
            layers.append(ConvTransposeLayer(current_channels, next_channels, kernel_size=3, stride=2, padding=1,
                                             output_padding=1))
            layers.append(act_fn())
            current_channels = next_channels

        # Output layer
        layers.append(ConvTransposeLayer(current_channels, output_channels, kernel_size=3, stride=1, padding=1))
        layers.append(output_act_fn())

        # Define the decoder as a sequential model
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, channels, height, width) for 2D or (batch_size, channels, length) for 1D.
        """
        x_reconstructed = self.decoder(z)
        if self.conv_dim == 1:
            x_reconstructed = x_reconstructed.permute(0, 2, 1)  # (batch, M, N) -> (batch, N, M)
        return x_reconstructed


class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, conv_dim: int = 2, num_layers: int = 4,
                 initial_channels: int = 16, growth_factor: int = 2, act_fn=nn.ReLU,
                 output_act_fn: nn.Module = nn.Sigmoid, final_output_size=(32, 32)):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(input_channels, latent_dim, conv_dim, num_layers, initial_channels, growth_factor,
                                   act_fn)
        self.decoder = ConvDecoder(latent_dim, input_channels, conv_dim, num_layers, initial_channels, growth_factor,
                                   act_fn, output_act_fn, final_output_size)

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class VGG(nn.Module):
    def __init__(self, vgg_name, input_channels, latent_dim, act_fn=nn.ReLU):
        super(VGG, self).__init__()
        self.input_channels = input_channels
        self.cfg = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                              512, 512, 'M'], }
        layers = self._make_layers(self.cfg[vgg_name])
        layers.append(nn.Flatten())
        self.features = nn.Sequential(*layers)
        # Calculate the flattened size after convolutional layers
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        flattened_output_size = nn.Sequential(*self.features[:-1])(dummy_input).view(1, -1).size(
            1)  # Flatten the output
        self.dense_layers = nn.Sequential(nn.Linear(flattened_output_size, latent_dim), act_fn())

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return layers


def test_vgg():
    net = VGG('VGG11', 10, nn.ReLU)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


class LinearDecoder(nn.Module):
    """
        A linear decoder module for an autoencoder.

        This class implements a versatile linear decoder that accepts a vector of latent features
        and outputs data in a specified shape. The decoder is fully linear (no nonlinearities) and
        can have multiple linear layers.

        Args:
            latent_dim (int): Dimension of the latent vector.
            input_dim (tuple): Desired shape of a single instance of the output data.
            hidden_dim (int): Dimension of the hidden layers in the decoder (default is 256).
            num_layers (int): Number of linear layers in the decoder (default is 1).

        Attributes:
            latent_dim (int): Dimension of the latent vector.
            input_dim (tuple): Desired shape of a single instance of the output data.
            prod_input_dim (int): Total number of elements in the output data, calculated as the product
                              of the dimensions in input_dim.
            decoder (nn.Sequential): Sequential container of linear layers.

        Methods:
            forward(x):
                Passes the input latent vector through the linear layers and reshapes the output to
                the specified output shape, including the batch dimension.

        Example:
            >>> latent_dim = 16
            >>> input_dim = (3, 32, 32)  # Shape of a 32x32 RGB image
            >>> num_layers = 2
            >>> decoder = LinearDecoder(latent_dim, input_dim, num_layers)
            >>> latent_vector = torch.randn((4, latent_dim))  # Batch of 4 latent vectors
            >>> output = decoder(latent_vector)
            >>> print(output.shape)
            torch.Size([4, 3, 32, 32])

        Example 2: Decoding to a vector shape
            >>> latent_dim = 8
            >>> input_dim = (20,)  # Shape of a vector with 20 elements
            >>> num_layers = 3
            >>> decoder = LinearDecoder(latent_dim, input_dim, num_layers)
            >>> latent_vector = torch.randn((5, latent_dim))  # Batch of 5 latent vectors
            >>> output = decoder(latent_vector)
            >>> print(output.shape)
            torch.Size([5, 20])
    """

    def __init__(self, latent_dim, input_dim, hidden_dim=256, num_layers=1, act_fn=None, bias=False,
                 output_act_fn=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.prod_input_dim = int(np.prod(input_dim))

        layers = []
        input_dim = latent_dim
        for i in range(num_layers - 1):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim, bias=bias)
                torch.nn.init.orthogonal_(layer.weight)
                layers.append(layer)

            else:
                layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)
                layers.append(layer)

            if act_fn is not None:
                layers.append(act_fn())

        layer = nn.Linear(hidden_dim, self.prod_input_dim, bias=bias)
        layers.append(layer)
        if output_act_fn is not None:
            layers.append(output_act_fn())
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        if isinstance(z, (tuple, list)):
            z = z[0] * z[1]
        x = self.decoder(z)
        return x.view(-1, *self.input_dim)
