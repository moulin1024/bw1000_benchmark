"""Configuration types for horizontal-spectral / vertical-FD solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DType = Literal["float32", "float64"]
TridiagonalSolver = Literal["pcr", "thomas"]
PoissonMethod = Literal["transpose", "spike", "spike-adaptive"]
InterfaceCollective = Literal["alltoall", "allgather"]
InterfaceSolver = Literal["selected-rows", "block-thomas", "dense"]
PipelineExecution = Literal["monolithic", "staged"]
DataLayout = Literal["xyz", "z-first"]
Platform = Literal["cpu", "cuda", "rocm"]


@dataclass(frozen=True, slots=True)
class Poisson3DConfig:
    """Configuration for a periodic-horizontal, wall-bounded Poisson solve."""

    nx: int = 1024
    ny: int = 1024
    nz: int = 128
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    dtype: DType = "float64"
    nyquist_filter: bool = True
    tridiag: TridiagonalSolver = "pcr"
    thomas_chunk: int = 1
    method: PoissonMethod = "transpose"
    spike_interface_collective: InterfaceCollective = "alltoall"
    spike_interface_solver: InterfaceSolver = "selected-rows"
    pipeline_execution: PipelineExecution = "monolithic"
    data_layout: DataLayout = "xyz"
    platform: Platform | None = None
    distributed: bool = False

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "Poisson3DConfig":
        """Construct a configuration from a named platform preset."""
        from .presets import get_poisson3d_preset

        return get_poisson3d_preset(name).create_config(**overrides)

    def validate(self) -> None:
        """Validate device-independent configuration constraints."""
        if min(self.nx, self.ny, self.nz) <= 0:
            raise ValueError("nx, ny, and nz must be positive")
        if self.nx % 2 or self.ny % 2:
            raise ValueError("nx and ny must be even")
        if min(self.lx, self.ly, self.lz) <= 0:
            raise ValueError("lx, ly, and lz must be positive")
        if self.thomas_chunk <= 0:
            raise ValueError("thomas_chunk must be positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"unsupported dtype: {self.dtype!r}")
        if self.tridiag not in ("pcr", "thomas"):
            raise ValueError(f"unsupported tridiagonal solver: {self.tridiag!r}")
        if self.method not in ("transpose", "spike", "spike-adaptive"):
            raise ValueError(f"unsupported Poisson method: {self.method!r}")
        if self.spike_interface_collective not in ("alltoall", "allgather"):
            raise ValueError(
                "spike_interface_collective must be 'alltoall' or 'allgather'"
            )
        if self.spike_interface_solver not in (
            "selected-rows",
            "block-thomas",
            "dense",
        ):
            raise ValueError(
                f"unsupported SPIKE interface solver: {self.spike_interface_solver!r}"
            )
        if self.pipeline_execution not in ("monolithic", "staged"):
            raise ValueError("pipeline_execution must be 'monolithic' or 'staged'")
        if self.data_layout not in ("xyz", "z-first"):
            raise ValueError("data_layout must be 'xyz' or 'z-first'")
        if self.platform not in (None, "cpu", "cuda", "rocm"):
            raise ValueError("platform must be None, 'cpu', 'cuda', or 'rocm'")
