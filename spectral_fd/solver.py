"""Public callable solver facade."""

from __future__ import annotations

from typing import Any

from .config import PipelineExecution, Poisson3DConfig


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

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "Poisson3DSolver":
        return cls(Poisson3DConfig.from_preset(name, **overrides))

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
