"""Public callable solver facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import PipelineExecution, Poisson3DConfig

if TYPE_CHECKING:
    from .factory import JaxRuntimeContext


class Poisson3DSolver:
    """Callable distributed Poisson solver with precomputed vertical factors.

    By default construction initializes the selected JAX backend and
    precomputes solver factors. Passing an application-owned
    :class:`JaxRuntimeContext` skips all runtime initialization. ``solve``
    accepts one local slab per local device:

    - ``z-first``: ``(local_devices, nz / global_devices, ny, nx)``
    - ``xyz``: ``(local_devices, nx, ny, nz / global_devices)``

    The input dtype must match ``config.dtype``. The returned JAX array has
    the same shape and distribution. The first call compiles the selected
    solve pipeline.
    """

    def __init__(
        self,
        config: Poisson3DConfig,
        *,
        runtime: JaxRuntimeContext | None = None,
    ):
        config.validate()
        # Lazy import is important: the implementation sets JAX backend and
        # precision environment variables before importing JAX.
        from .factory import build_poisson_solver, build_solver_engine

        self.config = config
        self._engine = (
            build_solver_engine(config)
            if runtime is None
            else build_poisson_solver(config, runtime).create_engine(config)
        )

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

    def implicit_solve(
        self,
        matvec,
        rhs: Any,
        *,
        symmetric: bool = True,
        transpose_solve=None,
    ) -> Any:
        """Apply this factorized solver through JAX implicit differentiation.

        ``matvec`` is the application's semantic operator (for WiRE-LES,
        ``divergence(gradient(x))``). The caller is responsible for ensuring
        that the configured spectral/FD rows represent that operator. A
        nonsymmetric operator must provide its own transpose solve.
        """

        if not symmetric and transpose_solve is None:
            raise ValueError(
                "transpose_solve is required when symmetric=False"
            )
        from jax import lax

        def solve(_matvec, value):
            return self.solve(value)

        return lax.custom_linear_solve(
            matvec,
            rhs,
            solve=solve,
            transpose_solve=transpose_solve,
            symmetric=symmetric,
        )
