import math

import numpy as np
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

    def __init__(self, latent_dim, input_dim, hidden_dim=256, num_layers=1, act_fn=None, bias=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.prod_input_dim = int(np.prod(input_dim))
        # bias = True if act_fn is not None or bias else False

        layers = []
        input_dim = latent_dim
        for i in range(num_layers - 1):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim, bias=bias)
                # Apply custom weight initialization to the first layer only
                # custom_weight_init(layer)
                layers.append(layer)

            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))

            if act_fn is not None:
                layers.append(act_fn())

        layers.append(nn.Linear(hidden_dim, self.prod_input_dim, bias=bias))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        if isinstance(z, (tuple, list)):
            z = z[0] * z[1]
        x = self.decoder(z)
        return x.view(-1, *self.input_dim)


def dct_1d(x):
    """
    Compute the DCT-II (1D DCT) for each vector in a batch.

    Args:
    - x (torch.Tensor): Input tensor of shape (batch_size, n).

    Returns:
    - torch.Tensor: DCT-II transformed tensor of shape (batch_size, n).
    """
    N = x.size(-1)

    # Create the frequency coefficients
    k = torch.arange(N, dtype=x.dtype, device=x.device).view(1, -1)
    n = torch.arange(N, dtype=x.dtype, device=x.device).view(-1, 1)

    # Compute the DCT-II matrix
    dct_matrix = torch.cos(math.pi / N * (n + 0.5) * k)

    # Apply scaling for orthogonality
    dct_matrix[:, 0] *= 1.0 / math.sqrt(2.0)
    dct_matrix *= math.sqrt(2.0 / N)

    # Perform the matrix multiplication (batch_size, n) @ (n, n) -> (batch_size, n)
    return torch.matmul(x, dct_matrix)


class EncoderWrapper(nn.Module):
    """
    Wrapper class for an encoder module.
    A Softsign activation function is applied to the output of the encoder and there is an optional factor scale
    parameter that can be set to True to factor the scale of the latent vectors and return the unit vectors and
    the magnitudes as a tuple.
    """

    def __init__(self, encoder, factor_scale=False):
        """
        Initializes the encoder wrapper with the encoder module and the factor scale parameter.
        :param encoder: The encoder module.
        :param factor_scale: Whether to factor the scale of the latent vectors.
        """
        super().__init__()
        self.factor_scale = factor_scale
        if not factor_scale:
            self.encoder = nn.Sequential(encoder, nn.Tanh())
        else:
            self.encoder = encoder

    def forward(self, x):
        z = self.encoder(x)
        if self.factor_scale:
            # detect if model is in train
            if self.training:
                # Calculate the target increase with initial exponential function
                # target_increase = torch.exp(-1.0 * torch.arange(z.shape[1] - 1, -1, -1, dtype=torch.float32,device=z.device))
                # # Normalize the curve to start from 0
                # target_increase = target_increase - target_increase.min()
                # # Optionally, scale the curve to have a maximum of 1
                # target_increase = target_increase / target_increase.max()
                #
                # # inject uniform noise to z following pos distribution taking into z now has range of [-1,1]
                # z = z + z.mean(dim=0)*(torch.rand_like(z)-0.5) * target_increase
                pass

            # dct_1d(z)
            z = torch.nn.functional.softsign(z)

            return z

            #  extract z_unitary
            # Gaussian function: exp(-x^2)
            # gaussian = torch.exp(-z ** 2)
            #
            # # Scale to [-1, 1]
            # scaled_gaussian = 2 * gaussian - 1
            # return scaled_gaussian

        return z
