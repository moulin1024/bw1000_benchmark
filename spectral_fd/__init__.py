"""Horizontal-spectral / vertical-finite-difference solvers."""

from .config import Poisson3DConfig, PoissonDiscretization
from .factory import JaxRuntimeContext, runtime_from_initialized_jax
from .presets import (
    DCU_ROCM,
    MN5_CUDA,
    Poisson3DPreset,
    available_poisson3d_presets,
    get_poisson3d_preset,
)
from .layouts import SlabDecomposition
from .solver import Poisson3DSolver

__all__ = [
    "DCU_ROCM",
    "MN5_CUDA",
    "Poisson3DConfig",
    "PoissonDiscretization",
    "Poisson3DPreset",
    "Poisson3DSolver",
    "JaxRuntimeContext",
    "SlabDecomposition",
    "available_poisson3d_presets",
    "get_poisson3d_preset",
    "runtime_from_initialized_jax",
]
