"""Assembly of mapped Poisson solve stages and their operator bundles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from .config import PoissonMethod, TridiagonalSolver
from .transforms import ArrayLayoutOps


@dataclass(frozen=True, slots=True)
class PoissonPipelineStages:
    """Compiled-stage boundaries shared by the library and benchmark."""

    forward_fft: Callable
    transpose_z_to_y: Callable
    tridiagonal: Callable
    transpose_y_to_z: Callable
    inverse_fft: Callable
    spike: Callable
    solve_monolithic: Callable
    solve_staged: Callable
    residual_spike: Callable
    residual_transpose: Callable


@dataclass(frozen=True, slots=True)
class PipelineOperatorBundle:
    """Factors passed to solve stages plus rows retained for validation."""

    method: PoissonMethod
    solve_args: tuple[Any, ...]
    row_operators: tuple[Any, Any, Any] | None = None
    solver_operators: tuple[Any, Any, Any] | None = None
    keep: Any | None = None

    @property
    def is_spike(self) -> bool:
        return self.method != "transpose"

    def residual(self, pipeline: PoissonPipelineStages, rhs):
        if self.is_spike:
            return pipeline.residual_spike(rhs, *self.solve_args)
        assert self.row_operators is not None
        assert self.solver_operators is not None
        assert self.keep is not None
        return pipeline.residual_transpose(
            rhs,
            *self.row_operators,
            *self.solver_operators,
            self.keep,
        )

    def bind_residual(self, pipeline: PoissonPipelineStages) -> Callable:
        return partial(self.residual, pipeline)


def assemble_pipeline_operators(
    *,
    method: PoissonMethod,
    tridiagonal: TridiagonalSolver,
    token,
    build_spike_adaptive: Callable,
    build_spike: Callable,
    build_rows: Callable,
    build_thomas: Callable,
    build_pcr: Callable,
) -> PipelineOperatorBundle:
    """Run the selected setup builders and normalize their factor bundle."""
    if method == "spike-adaptive":
        solve_args = tuple(build_spike_adaptive(token))
        bundle = PipelineOperatorBundle(method=method, solve_args=solve_args)
    elif method == "spike":
        solve_args = tuple(build_spike(token))
        bundle = PipelineOperatorBundle(method=method, solve_args=solve_args)
    else:
        a, b, c, keep = build_rows(token)
        if tridiagonal == "thomas":
            inv_bet, gamma = build_thomas(a, b, c)
            solver_operators = (a, inv_bet, gamma)
        else:
            alpha, gamma, inv_b = build_pcr(a, b, c)
            solver_operators = (alpha, gamma, inv_b)
        bundle = PipelineOperatorBundle(
            method=method,
            solve_args=(*solver_operators, keep),
            row_operators=(a, b, c),
            solver_operators=solver_operators,
            keep=keep,
        )
    bundle.solve_args[0].block_until_ready()
    return bundle


def build_poisson_pipeline(
    *,
    jax: Any,
    jnp: Any,
    lax: Any,
    layout: ArrayLayoutOps,
    axis_name: str,
    device_count: int,
    nz: int,
    real_dtype: Any,
    is_spike: bool,
    keep_modes,
    tridiagonal_solve: Callable,
    spike_raw: Callable,
    spike_filtered: Callable,
    block_rows: Callable,
) -> PoissonPipelineStages:
    """Create mapped component, full-solve, and residual callables."""
    mapped = partial(jax.pmap, axis_name=axis_name)
    z_to_y = partial(
        layout.z_to_y,
        lax=lax,
        axis_name=axis_name,
        device_count=device_count,
    )
    y_to_z = partial(
        layout.y_to_z,
        lax=lax,
        axis_name=axis_name,
        device_count=device_count,
    )

    def make_rhs_column(rhs_hat_y):
        if layout.z_first:
            rhs_column = jnp.zeros(
                (nz + 1,) + rhs_hat_y.shape[1:],
                dtype=rhs_hat_y.dtype,
            )
            return rhs_column.at[1:nz].set(rhs_hat_y[: nz - 1])
        rhs_column = jnp.zeros(
            rhs_hat_y.shape[:-1] + (nz + 1,),
            dtype=rhs_hat_y.dtype,
        )
        return rhs_column.at[..., 1:nz].set(rhs_hat_y[..., : nz - 1])

    def tridiagonal_pressure(rhs_hat_y, operator1, operator2, operator3, keep):
        pressure_column = tridiagonal_solve(
            operator1,
            operator2,
            operator3,
            make_rhs_column(rhs_hat_y),
        )
        if layout.z_first:
            return pressure_column[1:] * keep.astype(pressure_column.dtype)
        return pressure_column[..., 1:] * keep.astype(pressure_column.dtype)

    def forward_fft_local(rhs):
        return layout.forward_fft_local(rhs)

    def transpose_z_to_y_local(spectral):
        return z_to_y(spectral)

    def tridiagonal_stage_local(spectral, operator1, operator2, operator3, keep):
        return tridiagonal_pressure(
            spectral,
            operator1,
            operator2,
            operator3,
            keep,
        )

    def transpose_y_to_z_local(spectral):
        return y_to_z(spectral)

    def inverse_fft_local(spectral):
        return layout.inverse_fft_local(spectral)

    def spike_stage_local(spectral, *operators):
        return spike_filtered(spectral, *operators)

    forward_fft = mapped(forward_fft_local)
    transpose_z_to_y = mapped(transpose_z_to_y_local)
    tridiagonal_stage = mapped(tridiagonal_stage_local)
    transpose_y_to_z = mapped(transpose_y_to_z_local)
    inverse_fft = mapped(inverse_fft_local)
    spike_stage = mapped(spike_stage_local)

    def solve_monolithic_local(rhs, *operators):
        spectral = layout.forward_fft_local(rhs)
        if is_spike:
            spectral = spike_filtered(spectral, *operators)
        else:
            spectral = z_to_y(spectral)
            spectral = tridiagonal_pressure(
                spectral,
                operators[0],
                operators[1],
                operators[2],
                operators[3],
            )
            spectral = y_to_z(spectral)
        return layout.inverse_fft_local(spectral)

    solve_monolithic = mapped(solve_monolithic_local)

    def solve_staged(rhs, *operators):
        """Chain component executables without intermediate synchronization."""
        spectral = forward_fft(rhs)
        if is_spike:
            spectral = spike_stage(spectral, *operators)
        else:
            spectral = transpose_z_to_y(spectral)
            spectral = tridiagonal_stage(
                spectral,
                operators[0],
                operators[1],
                operators[2],
                operators[3],
            )
            spectral = transpose_y_to_z(spectral)
        return inverse_fft(spectral)

    def residual_spike_local(rhs, *operators):
        spectral = layout.forward_fft_local(rhs)
        pressure, left_right, masked_rhs = spike_raw(spectral, *operators)
        device_index = lax.axis_index(axis_name)
        a_block, b_block, c_block = block_rows(device_index)
        complex_dtype = pressure.dtype
        if layout.z_first:
            left = layout.interface_to_mode(left_right[0])
            right = layout.interface_to_mode(left_right[1])
            pressure_down = jnp.concatenate(
                (left[None, ...], pressure[:-1]),
                axis=0,
            )
            pressure_up = jnp.concatenate(
                (pressure[1:], right[None, ...]),
                axis=0,
            )
        else:
            pressure_down = jnp.concatenate(
                (left_right[0][..., None], pressure[..., :-1]),
                axis=-1,
            )
            pressure_up = jnp.concatenate(
                (pressure[..., 1:], left_right[1][..., None]),
                axis=-1,
            )
        residual = (
            layout.z_broadcast(a_block) * pressure_down
            + b_block.astype(complex_dtype) * pressure
            + layout.z_broadcast(c_block) * pressure_up
            - masked_rhs
        )
        mask = layout.spectral_keep(keep_modes)
        numerator = lax.pmax(jnp.max(jnp.abs(residual) * mask), axis_name)
        denominator = lax.pmax(
            jnp.max(jnp.abs(masked_rhs) * mask),
            axis_name,
        )
        return numerator / jnp.maximum(
            denominator,
            jnp.finfo(real_dtype).tiny,
        )

    residual_spike = mapped(residual_spike_local)

    def residual_transpose_local(
        rhs,
        a,
        b,
        c,
        operator1,
        operator2,
        operator3,
        keep,
    ):
        spectral = layout.forward_fft_local(rhs)
        rhs_column = make_rhs_column(z_to_y(spectral))
        pressure_column = tridiagonal_solve(
            operator1,
            operator2,
            operator3,
            rhs_column,
        )
        complex_dtype = pressure_column.dtype
        if layout.z_first:
            pressure_down = jnp.pad(
                pressure_column[:-1],
                ((1, 0), (0, 0), (0, 0)),
            )
            pressure_up = jnp.pad(
                pressure_column[1:],
                ((0, 1), (0, 0), (0, 0)),
            )
        else:
            pressure_down = jnp.pad(
                pressure_column[..., :-1],
                ((0, 0), (0, 0), (1, 0)),
            )
            pressure_up = jnp.pad(
                pressure_column[..., 1:],
                ((0, 0), (0, 0), (0, 1)),
            )
        residual = (
            a.astype(complex_dtype) * pressure_down
            + b.astype(complex_dtype) * pressure_column
            + c.astype(complex_dtype) * pressure_up
            - rhs_column
        )
        mask = keep.astype(real_dtype)
        numerator = lax.pmax(jnp.max(jnp.abs(residual) * mask), axis_name)
        denominator = lax.pmax(
            jnp.max(jnp.abs(rhs_column) * mask),
            axis_name,
        )
        return numerator / jnp.maximum(
            denominator,
            jnp.finfo(real_dtype).tiny,
        )

    residual_transpose = mapped(residual_transpose_local)
    return PoissonPipelineStages(
        forward_fft=forward_fft,
        transpose_z_to_y=transpose_z_to_y,
        tridiagonal=tridiagonal_stage,
        transpose_y_to_z=transpose_y_to_z,
        inverse_fft=inverse_fft,
        spike=spike_stage,
        solve_monolithic=solve_monolithic,
        solve_staged=solve_staged,
        residual_spike=residual_spike,
        residual_transpose=residual_transpose,
    )
