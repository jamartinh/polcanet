from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig
from pythae.models import BaseAEConfig


@dataclass
class PolcaNetConfig(BaseAEConfig):
    r"""
    PolcaNet model config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        alpha (float): Orthogonality loss weight. Default: 0.1
        beta (float): Variance center of mass loss weight. Default: 1.0
        gamma (float): Variance exponential distribution loss weight. Default: 1.0

    """

    input_dim: Union[Tuple[int, ...], None] = None
    latent_dim: int = None

    alpha: float = 0.1  # Orthogonality loss weight
    beta: float = 1.0  # Variance center of mass loss weight
    gamma: float = 1.0  # Variance exponential distribution loss weight

    uses_default_encoder: bool = False
    uses_default_decoder: bool = False


@dataclass
class EnvironmentConfig(BaseConfig):
    python_version: str = "3.12"
