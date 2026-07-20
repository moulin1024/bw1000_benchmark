"""Compatibility facade for the distributed 3D pressure-Poisson API."""

from .config import (
    DataLayout,
    DType,
    InterfaceCollective,
    InterfaceSolver,
    PipelineExecution,
    Platform,
    Poisson3DConfig,
    PoissonDiscretization,
    PoissonMethod,
    TridiagonalSolver,
)
from .solver import Poisson3DSolver

__all__ = [
    "DataLayout",
    "DType",
    "InterfaceCollective",
    "InterfaceSolver",
    "PipelineExecution",
    "Platform",
    "Poisson3DConfig",
    "PoissonDiscretization",
    "Poisson3DSolver",
    "PoissonMethod",
    "TridiagonalSolver",
]
