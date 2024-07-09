from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from pythae.data import BaseDataset
from pythae.models import BaseAE, AEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder

from polcanet import LinearDecoder
from polcanet.polcanet_config import PolcaNetConfig


class LinearDecoderPythae(LinearDecoder, BaseDecoder):
    def __init__(self, args, hidden_dim, num_layers):
        BaseDecoder.__init__(self)
        LinearDecoder.__init__(self,
                               latent_dim=args.latent_dim,
                               input_dim=args.input_dim,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               )

    def forward(self, z: torch.Tensor):
        output = ModelOutput()
        x = self.decoder(z)
        x = x.view(-1, *self.input_dim)
        output["reconstruction"] = x
        return output


class PolcaNetPythae(BaseAE):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network encoder
    that is used for Principal Latent Components Analysis (Polca).
    The current implementation is for, and following, the Pythae library specification.

    Attributes:
        model_name (str): The name of the model.
        input_dim (int): The input dimension.
        latent_dim (int): The latent dimension.
        alpha (float): Orthogonality loss weight.
        beta (float): Variance center of mass loss weight
        gamma (float): Variance exponential distribution loss weight
        encoder (BaseEncoder): The encoder model.
        decoder (BaseDecoder): The decoder model.



    Methods:
        __init__(model_config: AEConfig | PolcaNetConfig, encoder: BaseEncoder, decoder: Optional[BaseDecoder] = None,
                    decoder_hidden_dim: Optional[int] = None, decoder_hidden_layers: Optional[int] = None):
        forward(inputs: BaseDataset | dict, **kwargs):
            Forward pass through the encoder.
        cross_correlation_loss(latent: torch.Tensor) -> torch.Tensor:
            Computes the cross-correlation loss.
        orthogonality_loss(latent: torch.Tensor) -> torch.Tensor:
            Computes the average pairwise orthogonality across all latent in a batch.
        center_of_mass_loss(latent: torch.Tensor) -> torch.Tensor:
            Computes the center of mass loss.
        exp_decay_var_loss(latent: torch.Tensor, decay_rate: float = 0.5) -> torch.Tensor:
            Encourages exponential decay of variances.
        loss_function(x: torch.Tensor, r: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor]:
            Computes the loss function for the model.

    """

    def __init__(self,
                 model_config: AEConfig | PolcaNetConfig,
                 encoder: BaseEncoder,
                 decoder: Optional[BaseDecoder] = None,
                 decoder_hidden_dim: Optional[int] = None,
                 decoder_hidden_layers: Optional[int] = None,
                 ):
        """
        Initializes PolcaNet with the provided parameters.
        decoder is optional, but if not provided, the hidden dimensions and layers must be provided to create a default
        linear decoder.
        """

        self.mean_metrics = None
        self.std_metrics = None
        self.r_mean_metrics = None
        if decoder is None:
            assert decoder_hidden_dim is not None and decoder_hidden_layers is not None, (
                    "Please provide the hidden dimensions (decoder_hidden_dim: int) and"
                    " layers (decoder_hidden_layers: int) for the default linear decoder."
            )
            decoder = LinearDecoderPythae(
                args=model_config,
                hidden_dim=decoder_hidden_dim,
                num_layers=decoder_hidden_layers
            )
        BaseAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "PolcaNet"
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.alpha = model_config.alpha
        self.beta = model_config.beta
        self.gamma = model_config.gamma

        self.encoder = encoder
        self.decoder = decoder

        # Loss names
        self.aux_loss_names = {
                "loss": "Total Loss",
                "rec": "Reconstruction Loss",
                "ort": "Orthogonality Loss",
                "com": "Center of Mass Loss",
                "var": "Variance Distribution Loss"

        }

    def forward(self, inputs: BaseDataset | dict, **kwargs):
        """
        Forward pass through the encoder.
        :param inputs:
        :param kwargs:
        :return:
        """
        x: torch.Tensor = inputs["data"]
        z = self.encoder(x)
        r = self.decoder(z)["reconstruction"]

        losses = self.loss_function(x, r, z)
        output = ModelOutput(
            loss=losses[0],
            rec_loss=losses[1],
            ort_loss=losses[2],
            com_loss=losses[3],
            var_loss=losses[4],
            recon_x=r,
            z=z,
        )
        return output

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
        """Calculate center of mass loss of latent vectors"""
        axis = torch.arange(0, latent.shape[1], device=latent.device, dtype=torch.float32) ** 2
        std_latent = torch.var(latent, dim=0)

        w = nn.functional.normalize(std_latent, p=1.0, dim=0)
        com = w * axis / axis.shape[0]  # weight the latent space
        loss = torch.mean(com)
        return loss

    @staticmethod
    def exp_decay_var_loss(latent, decay_rate=0.5):
        """Encourage exponential decay of variances along latent space"""
        var_latent = torch.var(latent, dim=0)

        # Create exponential decay target
        target_decay = torch.exp(-decay_rate * torch.arange(latent.shape[1], device=latent.device, dtype=torch.float32))

        # Normalize both to sum to 1 for fair comparison
        var_latent_norm = var_latent / torch.sum(var_latent)
        target_decay_norm = target_decay / torch.sum(target_decay)

        return torch.nn.functional.mse_loss(var_latent_norm, target_decay_norm)

    def loss_function(self, x, r, z):
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

        loss = l1 + self.alpha * l2 + self.beta * l3 + self.gamma * l4

        return loss, l1, l2, l3, l4

    def encode(self, in_x):
        self.encoder.eval()
        with torch.inference_mode():
            x = torch.tensor(in_x, dtype=torch.float32, device=self.device)
            x = self.scaler.transform(x)
            z = self.encoder.encode(x).detach().cpu().numpy()

        return z

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

    def update_metrics(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        z = self.encode(x)
        r = self.decode(z)
        self.r_mean_metrics = np.mean((x - r) ** 2)
        self.std_metrics = np.std(z, axis=0)
        self.mean_metrics = np.mean(z, axis=0)
