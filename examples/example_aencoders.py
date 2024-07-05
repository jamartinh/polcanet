import warnings

import torch
from torch import nn as nn


class MinMaxScalerTorch:
    def __init__(self, min_val=None, max_val=None):
        """Initialize the parameters used by the scaler."""
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


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class BaseAutoEncoder(nn.Module):
    """
    Base autoencoder neural principal latent components decomposition model.
    """

    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=3, act_fn=nn.GELU()):
        super().__init__()
        self.latent_dim = latent_dim
        layers_encoder = []
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * num_layers
        elif len(hidden_dim) != num_layers:
            warnings.warn("The hidden_dim is a list, the length of hidden_dim must be equal to num_layers."
                          " Setting it to [hidden_dim] * num_layers.")
            num_layers = len(hidden_dim)

        assert len(hidden_dim) == num_layers, "The length of hidden_dim must be equal to num_layers."

        layer = nn.Linear(input_dim, hidden_dim[0])
        torch.nn.init.orthogonal_(layer.weight)

        layers_encoder.append(layer)
        layers_encoder.append(act_fn)
        for i in range(1, num_layers):
            layer = nn.Linear(hidden_dim[i - 1], hidden_dim[i])
            torch.nn.init.orthogonal_(layer.weight)
            if hidden_dim[i - 1] == hidden_dim[i]:
                layer = ResNet(layer)
            layers_encoder.append(layer)
            layers_encoder.append(nn.LayerNorm(hidden_dim[i]))
            layers_encoder.append(act_fn)

        layer = nn.Linear(hidden_dim[-1], latent_dim)
        torch.nn.init.orthogonal_(layer.weight)
        layers_encoder.append(layer)

        self.encoder = nn.Sequential(*layers_encoder)

        layers_decoder = []
        reversed_hidden_dim = hidden_dim[::-1]
        layer = nn.Linear(latent_dim, reversed_hidden_dim[0])
        torch.nn.init.orthogonal_(layer.weight)

        layers_decoder.append(layer)
        #layers_decoder.append(act_fn)
        for i in range(1, num_layers):
            layer = nn.Linear(reversed_hidden_dim[i - 1], reversed_hidden_dim[i])
            torch.nn.init.orthogonal_(layer.weight)
            if reversed_hidden_dim[i - 1] == reversed_hidden_dim[i]:
                layer = ResNet(layer)
            layers_decoder.append(layer)
            layers_encoder.append(nn.LayerNorm(hidden_dim[i]))
            #layers_decoder.append(act_fn)

        layer = nn.Linear(reversed_hidden_dim[-1], input_dim)
        torch.nn.init.orthogonal_(layer.weight)
        layers_decoder.append(layer)
        self.decoder = nn.Sequential(*layers_decoder)

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.encode(x)
            reconstruction = self.decode(z)

        return z.detach().cpu().numpy(), reconstruction.detach().cpu().numpy()

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return z, reconstruction

    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction

    def encode(self, x):
        z = self.encoder(x)
        return z


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, seq_len, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, input_dim, seq_len, num_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


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


class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, conv_dim=2):
        super(ConvAutoencoder, self).__init__()
        self.conv_dim = conv_dim

        # Select appropriate Conv and ConvTranspose layers
        if conv_dim == 1:
            ConvLayer = nn.Conv1d
            ConvTransposeLayer = nn.ConvTranspose1d
            FlattenLayer = nn.Flatten
            UnflattenLayer = lambda input_shape: nn.Unflatten(1, (latent_dim, input_shape))
            self.input_channels = input_dim
            self.output_channels = input_dim
            self.flattened_size = 2
        elif conv_dim == 2:
            ConvLayer = nn.Conv2d
            ConvTransposeLayer = nn.ConvTranspose2d
            FlattenLayer = nn.Flatten
            UnflattenLayer = lambda input_shape: nn.Unflatten(1, (latent_dim, *input_shape))
            self.input_channels = 1
            self.output_channels = 1
            self.flattened_size = (2, 2)
        else:
            raise ValueError("conv_dim must be 1 or 2")

        self.encoder = nn.Sequential(
            ConvLayer(self.input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvLayer(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvLayer(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvLayer(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            FlattenLayer()
        )

        self.decoder = nn.Sequential(
            UnflattenLayer(self.flattened_size),
            ConvTransposeLayer(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.GELU(),
            ConvTransposeLayer(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.GELU(),
            ConvTransposeLayer(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.GELU(),
            ConvTransposeLayer(16, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def encode(self, x):
        if self.conv_dim == 1:
            x = x.permute(0, 2, 1)  # (batch, N, M) -> (batch, M, N)
        else:
            x = x.unsqueeze(1)  # (batch, N, M) -> (batch, 1, N, M)
        return self.encoder(x)

    def decode(self, z):
        decoded = self.decoder(z)
        if self.conv_dim == 2:
            decoded = decoded.squeeze(1)  # (batch, 1, N, M) -> (batch, N, M)
        return decoded


def autoencoder_factory(autoencoder_type, input_dim, latent_dim, hidden_dim=None, seq_len=None, num_layers=None,
                        act_fn=None):
    if autoencoder_type == "dense":
        return BaseAutoEncoder(input_dim, latent_dim, hidden_dim, num_layers, act_fn=act_fn)
    elif autoencoder_type == "lstm":
        if seq_len is None:
            raise ValueError("seq_len must be provided for LSTMAutoencoder.")
        return LSTMAutoencoder(input_dim, latent_dim, seq_len, num_layers)
    elif autoencoder_type == "conv1d":
        return ConvAutoencoder(input_dim, latent_dim, conv_dim=1)
    elif autoencoder_type == "conv2d":
        return ConvAutoencoder(input_dim, latent_dim, conv_dim=2)
    else:
        raise ValueError(f"Unknown autoencoder_type: {autoencoder_type}")
