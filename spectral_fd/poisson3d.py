"""Compatibility facade for the distributed 3D pressure-Poisson API."""

from .config import (
    DataLayout,
    DType,
    InterfaceCollective,
    InterfaceSolver,
    PipelineExecution,
    Platform,
    Poisson3DConfig,
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
    "Poisson3DSolver",
    "PoissonMethod",
    "TridiagonalSolver",
]
