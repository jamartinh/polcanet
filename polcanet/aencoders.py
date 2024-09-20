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

        layers.append(nn.LayerNorm(input_dim))
        # First layer
        layer = nn.Linear(input_dim, current_dim)
        layers.append(layer)
        layers.append(act_fn())

        # Hidden layers
        for i in range(num_layers):
            next_dim = hidden_size if hidden_size is not None else int(current_dim - step_size)
            layer = nn.Linear(current_dim, next_dim)
            layers.append(layer)
            layers.append(act_fn())
            current_dim = next_dim

        # Final layer to latent_dim
        layer = nn.Linear(current_dim, latent_dim)
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
        layers.append(layer)
        layers.append(act_fn())

        # Hidden layers
        for i in range(num_layers):
            next_dim = current_dim + int(step_size) if hidden_size is None else hidden_size
            layer = nn.Linear(current_dim, next_dim, bias=bias)
            layers.append(layer)
            layers.append(act_fn())
            # layers.append(nn.Dropout(0.1))
            current_dim = next_dim

        # Final layer to output_dim
        layer = nn.Linear(current_dim, output_dim, bias=bias)

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


class oldLSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, seq_len, num_layers=2):
        super(oldLSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm(x)
        return x


import torch
import torch.nn as nn
import numpy as np


class LSTMDecoder(nn.Module):
    """
    An enhanced flexible LSTM decoder module for an autoencoder.
    This class implements a versatile LSTM decoder that accepts a vector of latent features
    and outputs data in a specified shape. The decoder uses LSTM layers (optionally with projection)
    followed by a linear layer to reshape the output to the desired dimensions.

    Args:
        latent_dim (int): Dimension of the latent vector.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        output_dim (tuple): Desired shape of a single instance of the output data.
        num_layers (int): Number of LSTM layers (default is 2).
        proj_size (int, optional): If specified, adds a projection layer to reduce LSTM output dimension.
        bias (bool): If False, then the layer does not use bias weights b_ih and b_hh. Default: True.

    Attributes:
        latent_dim (int): Dimension of the latent vector.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        output_dim (tuple): Desired shape of a single instance of the output data.
        prod_output_dim (int): Total number of elements in the output data.
        seq_len (int): Calculated length of the sequence to be generated by LSTM.
        lstm (nn.LSTM): LSTM layers.
        fc (nn.Linear): Final linear layer to reshape the output.

    Methods:
        forward(x):
            Passes the input latent vector through the LSTM layers and reshapes the output to
            the specified output shape, including the batch dimension.
    """

    def __init__(self, latent_dim, hidden_size, output_dim, num_layers=2, proj_size=None, bias=True):
        super(LSTMDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.prod_output_dim = int(np.prod(output_dim))
        self.proj_size = proj_size or 0

        # Determine the effective output size of LSTM (either hidden_size or proj_size)
        effective_hidden_size = proj_size if proj_size is not None else hidden_size

        # Calculate seq_len based on the output size and effective hidden size
        self.seq_len = max(1, self.prod_output_dim // effective_hidden_size)

        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            proj_size=self.proj_size, bias=bias)

        self.fc = nn.Linear(effective_hidden_size * self.seq_len, self.prod_output_dim,bias=bias)

    def forward(self, x):
        # Reshape input to (batch_size, seq_len, latent_dim)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Reshape LSTM output and pass through final linear layer
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        # Reshape to desired output dimensions
        return x.view(-1, *self.output_dim)

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
        size (int): Size of the input image. Default is 32.
    """

    def __init__(self, input_channels: int, latent_dim: int, conv_dim: int = 2, num_layers: int = 4,
                 initial_channels: int = 16, growth_factor: int = 2, act_fn=nn.ReLU, size=32):
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
        layers.append(nn.BatchNorm2d(input_channels))

        # Create convolutional layers
        for i in range(num_layers):
            next_channels = min(initial_channels * (growth_factor ** i), 512)
            if i == 0:
                stride = 1
                kernel_size = 5
            elif i == 1:
                stride = 2
                kernel_size = 3
            else:
                stride = 2
                kernel_size = 3

            layer = ConvLayer(current_channels, next_channels,
                              kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            layers.append(layer)
            layers.append(act_fn())
            current_channels = next_channels

        # Flatten layer
        layers.append(nn.Flatten())

        # Calculate the flattened size after convolutional layers
        dummy_input = torch.zeros(1, input_channels, size, size) if conv_dim == 2 else torch.zeros(1, input_channels,
                                                                                                   size)
        flattened_output_size = nn.Sequential(*layers[:-1])(dummy_input).view(1, -1).size(1)

        # Linear layers
        layer = nn.Linear(flattened_output_size, latent_dim * 4)
        layers.append(layer)
        layers.append(act_fn())
        layer = nn.Linear(latent_dim * 4, latent_dim * 2)
        layers.append(layer)
        layers.append(act_fn())
        layer = nn.Linear(latent_dim * 2, latent_dim)
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
    Convolutional Decoder that mirrors the ConvEncoder.

    Args:
        input_channels (int): Number of output channels (e.g., 3 for RGB images).
        latent_dim (int): Dimension of the latent space (input dimension).
        conv_dim (int): Dimension of the convolution (1 for 1D, 2 for 2D). Default is 2.
        num_layers (int): Number of convolutional layers. Default is 4.
        initial_channels (int): Number of channels for the first layer in the encoder. Default is 16.
        growth_factor (int): Factor by which the number of channels grows in each subsequent layer in the encoder. Default is 2.
        act_fn: Activation function to use. Default is nn.ReLU.
        size (int): The size of the input (e.g., 32 for 32x32 images).
    """

    def __init__(self, input_channels: int, latent_dim: int, conv_dim: int = 2, num_layers: int = 4,
                 initial_channels: int = 16, growth_factor: int = 2, act_fn=nn.ReLU, size=32, bias=True):
        super(ConvDecoder, self).__init__()

        self.conv_dim = conv_dim

        if conv_dim == 1:
            ConvLayer = nn.ConvTranspose1d
            BatchNormLayer = nn.BatchNorm1d
        elif conv_dim == 2:
            ConvLayer = nn.ConvTranspose2d
            BatchNormLayer = nn.BatchNorm2d
        else:
            raise ValueError("conv_dim must be 1 or 2")

        # Reconstruct the parameters from the encoder
        conv_in_channels = []
        conv_out_channels = []
        kernel_sizes = []
        strides = []
        paddings = []
        output_sizes = []

        current_channels = input_channels
        input_size = size

        for i in range(num_layers):
            next_channels = min(initial_channels * (growth_factor ** i), 1024)
            if i == 0:
                stride = 1
                kernel_size = 1
            elif i == 1:
                stride = 1
                kernel_size = 3
            else:
                stride = 2
                kernel_size = 3
            padding = kernel_size // 2
            conv_in_channels.append(current_channels)
            conv_out_channels.append(next_channels)
            kernel_sizes.append(kernel_size)
            strides.append(stride)
            paddings.append(padding)
            # Compute output size
            output_size = (input_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            output_sizes.append(output_size)
            input_size = output_size
            current_channels = next_channels

        # Save the final output size and channels
        self.flattened_output_size = current_channels * input_size * input_size if conv_dim == 2 else current_channels * input_size
        self.unflatten_shape = (current_channels, input_size, input_size) if conv_dim == 2 else (
        current_channels, input_size)

        # Linear layers
        layers = []
        layers.append(nn.Linear(latent_dim, latent_dim * 2, bias=bias))
        layers.append(act_fn())
        layers.append(nn.Linear(latent_dim * 2, latent_dim * 4, bias=bias))
        layers.append(act_fn())
        layers.append(nn.Linear(latent_dim * 4, self.flattened_output_size, bias=bias))

        self.linear_layers = nn.Sequential(*layers)

        # Unflatten layer
        self.unflatten = nn.Unflatten(1, self.unflatten_shape)

        # Reverse the lists for the decoder
        deconv_in_channels = conv_out_channels[::-1]
        deconv_out_channels = conv_in_channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        strides = strides[::-1]
        paddings = paddings[::-1]
        output_sizes = output_sizes[::-1]

        # Compute output paddings
        output_paddings = []
        input_size = self.unflatten_shape[-1]  # Start from last output size in encoder
        for i in range(num_layers):
            stride = strides[i]
            padding = paddings[i]
            kernel_size = kernel_sizes[i]
            if i < num_layers - 1:
                target_output_size = output_sizes[i + 1]
            else:
                target_output_size = size  # Final output size should match the input size
            expected_output_size = (input_size - 1) * stride - 2 * padding + kernel_size
            output_padding = target_output_size - expected_output_size
            output_paddings.append(output_padding)
            input_size = target_output_size

        # Build deconvolutional layers
        deconv_layers = []
        for i in range(num_layers):
            layer = ConvLayer(deconv_in_channels[i], deconv_out_channels[i],
                              kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i],
                              output_padding=output_paddings[i], bias=bias)
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            deconv_layers.append(layer)
            # Do not include batch norm and activation in the last layer
            if i != num_layers - 1:
                deconv_layers.append(act_fn())
        self.deconv_layers = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.linear_layers(z)
        x = self.unflatten(x)
        x = self.deconv_layers(x)
        if self.conv_dim == 1:
            x = x.permute(0, 2, 1)  # (batch, M, N) -> (batch, N, M)
        elif self.conv_dim == 2:
            if x.shape[1] == 1:  # if output is (batch, 1, N, M)
                x = x.squeeze(1)  # (batch, 1, N, M) -> (batch, N, M)
        return x


class VGG(nn.Module):
    def __init__(self, vgg_name, input_channels, latent_dim, act_fn=nn.ReLU, size=64):
        super(VGG, self).__init__()
        self.input_channels = input_channels
        self.cfg = {
                'VGG8': [32, 'M', 64, 'M', 128, 'M'],
                'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                          512, 512, 'M'], }
        layers = self._make_layers(self.cfg[vgg_name], act_fn)
        layers.append(nn.Flatten())
        self.features = nn.Sequential(*layers)
        # Calculate the flattened size after convolutional layers
        dummy_input = torch.zeros(1, input_channels, size, size)
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

    def _make_layers(self, cfg, act_fn=nn.ReLU):
        layers = []
        in_channels = self.input_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x),
                           act_fn()]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return layers


def test_vgg():
    net = VGG('VGG11', 10, nn.ReLU)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


class VGGDecoder(nn.Module):
    def __init__(self, vgg_name, output_channels, latent_dim, act_fn=nn.ReLU, size=64, output_activation=nn.Sigmoid):
        super(VGGDecoder, self).__init__()
        self.output_channels = output_channels
        self.cfg = {
                'VGG8': [32, 'M', 64, 'M', 128, 'M'],
                'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512,
                          512, 'M'],
        }

        # Reverse the configuration for decoding
        self.cfg[vgg_name] = self.cfg[vgg_name][::-1]

        # Calculate the starting size for the decoder
        self.start_size = size // (2 ** self.cfg[vgg_name].count('M'))

        # Find the number of channels in the deepest layer (excluding 'M')
        deepest_channels = next(x for x in self.cfg[vgg_name] if isinstance(x, int))

        self.dense_layers = nn.Sequential(
            nn.Linear(latent_dim, deepest_channels * self.start_size * self.start_size),
            act_fn()
        )

        self.features = self._make_layers(self.cfg[vgg_name], act_fn)

        # Add final layer to match input channels
        last_channels = next(x for x in reversed(self.cfg[vgg_name]) if isinstance(x, int))
        self.final_conv = nn.Conv2d(last_channels, output_channels, kernel_size=3, padding=1)
        self.final_activation = output_activation()

    def forward(self, x):
        out = self.dense_layers(x)
        first_channel = next(c for c in self.cfg[self.vgg_name] if isinstance(c, int))
        out = out.view(out.size(0), first_channel, self.start_size, self.start_size)
        out = self.features(out)
        out = self.final_conv(out)
        out = self.final_activation(out)
        return out

    def _make_layers(self, cfg, act_fn=nn.ReLU):
        layers = []
        in_channels = cfg[0]  # Start with the first channel number in the config
        for x in cfg[1:]:  # Skip the first element as it's used for in_channels
            if x == 'M':
                layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            else:
                layers += [
                        nn.ConvTranspose2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        act_fn()
                ]
                in_channels = x
        return nn.Sequential(*layers)


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

    """

    def __init__(self, latent_dim, input_dim, hidden_dim=256, num_layers=1, act_fn=None, bias=False,
                 output_act_fn=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.prod_input_dim = int(np.prod(input_dim))

        layers = []
        input_dim = latent_dim
        last_dim = input_dim
        for i in range(num_layers - 1):
            in_dim = input_dim if i == 0 else hidden_dim
            layer = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=bias)
            layers.append(layer)
            if act_fn is not None:
                layers.append(act_fn())
            last_dim = hidden_dim

        layer = nn.Linear(last_dim, self.prod_input_dim, bias=bias)
        layers.append(layer)
        if output_act_fn is not None:
            layers.append(output_act_fn())
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x = self.decoder(z)
        return x.view(-1, *self.input_dim)


class SimpleEmbeddedRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256, num_layers=3, act_fn=nn.GELU):
        super(SimpleEmbeddedRegressor, self).__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layer = nn.Linear(in_features=in_dim, out_features=out_dim)
            layers.append(layer)
            if i < num_layers - 1:
                layers.append(act_fn())
            in_dim = hidden_dim
        self.regressor = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.regressor(x)

    def evaluate_loss(self, x, y):
        yp = self(x)
        squared_error = (yp - y) ** 2
        normalized_error = (1 + torch.tanh(squared_error))/2.0
        return 1 - normalized_error.mean()  # Take mean over the batch

    def train_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

