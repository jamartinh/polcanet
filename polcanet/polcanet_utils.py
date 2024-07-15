import math

import torch
from torch import nn as nn


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


class MeanCentering(nn.Module):
    def __init__(self):
        super(MeanCentering, self).__init__()

    @staticmethod
    def forward(x):
        # Compute the mean of each batch
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        # Subtract the mean from the input
        x_centered = x - batch_mean
        return x_centered


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

    def __init__(self, latent_dim, input_dim, hidden_dim=256, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.prod_input_dim = int(torch.prod(torch.tensor(input_dim)))

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

            layers.append(MeanCentering())

        layers.append(nn.Linear(hidden_dim, self.prod_input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x = self.decoder(z)
        return x.view(-1, *self.input_dim)

    def decode(self, z):
        return self.forward(z)
