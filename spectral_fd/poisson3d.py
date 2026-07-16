"""Public API for the distributed 3D pressure-Poisson solver.

The computational implementation currently lives in the original benchmark
module so that the first library extraction does not disturb JAX compilation
or collective boundaries.  External callers should depend only on the API in
this module; the implementation can then be migrated behind it incrementally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal


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
    """Configuration for a periodic-horizontal, wall-bounded Poisson solve.

    The current discretization is periodic and spectral in ``x`` and ``y``.
    The vertical operator uses second-order finite differences with a Neumann
    bottom boundary, rigid-lid top boundary, and a pinned zero-horizontal-
    wavenumber mode.
    """

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
                "unsupported SPIKE interface solver: "
                f"{self.spike_interface_solver!r}"
            )
        if self.pipeline_execution not in ("monolithic", "staged"):
            raise ValueError(
                "pipeline_execution must be 'monolithic' or 'staged'"
            )
        if self.data_layout not in ("xyz", "z-first"):
            raise ValueError("data_layout must be 'xyz' or 'z-first'")
        if self.platform not in (None, "cpu", "cuda", "rocm"):
            raise ValueError("platform must be None, 'cpu', 'cuda', or 'rocm'")

    def _as_legacy_namespace(self) -> argparse.Namespace:
        """Translate the public configuration to the current implementation."""
        self.validate()
        return argparse.Namespace(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            lx=self.lx,
            ly=self.ly,
            lz=self.lz,
            dtype=self.dtype,
            no_nyquist_filter=not self.nyquist_filter,
            tridiag=self.tridiag,
            thomas_chunk=self.thomas_chunk,
            method=self.method,
            spike_interface_collective=self.spike_interface_collective,
            spike_interface_solver=self.spike_interface_solver,
            platform=self.platform,
            distributed=self.distributed,
            pipeline_execution=self.pipeline_execution,
            data_layout=self.data_layout,
            # Benchmark-only fields retained while the implementation is
            # migrated out of poisson3d_distributed.py.
            warmup=0,
            samples=1,
            iterations=1,
            seed=0,
            mms=False,
            mms_kind="modes",
            skip_components=True,
        )


class Poisson3DSolver:
    """Callable distributed Poisson solver with precomputed vertical factors.

    Construction initializes the selected JAX backend and precomputes solver
    factors. ``solve`` accepts one local slab per local device:

    - ``z-first``: ``(local_devices, nz / global_devices, ny, nx)``
    - ``xyz``: ``(local_devices, nx, ny, nz / global_devices)``

    The input dtype must match ``config.dtype``. The returned JAX array has
    the same shape and distribution. The first call compiles the selected
    solve pipeline.
    """

    def __init__(self, config: Poisson3DConfig):
        config.validate()
        # Lazy import is important: the implementation sets JAX backend and
        # precision environment variables before importing JAX.
        from poisson3d_distributed import build_solver

        self.config = config
        self._engine = build_solver(config)

    @property
    def global_devices(self) -> int:
        return self._engine.global_devices

    @property
    def local_devices(self) -> int:
        return self._engine.local_devices

    @property
    def process_count(self) -> int:
        return self._engine.process_count

    @property
    def process_index(self) -> int:
        return self._engine.process_index

    @property
    def local_input_shape(self) -> tuple[int, ...]:
        """Expected shape, including the leading local-device dimension."""
        return self._engine.local_input_shape

    @property
    def global_input_shape(self) -> tuple[int, int, int]:
        """Global physical-array shape in the configured data layout."""
        return self._engine.global_input_shape

    def solve(
        self,
        rhs: Any,
        *,
        execution: PipelineExecution | None = None,
    ) -> Any:
        """Solve the pressure-Poisson equation for locally sharded ``rhs``."""
        return self._engine.solve(rhs, execution=execution)

    def residual(self, rhs: Any) -> Any:
        """Return the distributed maximum relative tridiagonal residual."""
        return self._engine.residual(rhs)
