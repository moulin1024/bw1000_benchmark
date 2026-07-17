"""Validation-field generation and error metrics for Poisson solves."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np

from .pipeline import PipelineOperatorBundle, PoissonPipelineStages
from .transforms import ArrayLayoutOps


MMS_TERMS = (
    (1.00, 2, 3, 1, "cos", "cos"),
    (0.70, 1, 0, 2, "sin", "cos"),
    (0.50, 0, 4, 5, "cos", "sin"),
    (0.40, 1, 1, 0, "cos", "cos"),
    (0.30, 3, 2, 3, "sin", "sin"),
)


def validate_mms_configuration(args) -> None:
    """Validate the fixed manufactured-mode set against the requested grid."""
    if not args.mms or args.mms_kind != "modes":
        return
    for _, mode_x, mode_y, mode_z, _, _ in MMS_TERMS:
        if mode_x >= args.nx // 2 or mode_y >= args.ny // 2:
            raise ValueError("--mms horizontal modes must stay below Nyquist")
        if mode_z > args.nz - 2:
            raise ValueError("--mms requires nz >= 8")


@dataclass(frozen=True, slots=True)
class ValidationFields:
    """Right-hand side and optional manufactured reference solution."""

    rhs: Any
    reference: Any | None


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Discrete residual and optional manufactured-solution error."""

    relative_residual: float
    mms_error: float | None


@dataclass(frozen=True, slots=True)
class PoissonValidationSuite:
    """Mapped generators and metrics bound to one solver decomposition."""

    jax: Any
    jnp: Any
    local_devices: int
    process_index: int
    make_random_rhs: Callable
    make_mode_fields: Callable
    make_broadband_fields: Callable
    relative_max_difference: Callable
    global_max: Callable

    def generate(self, *, seed: int, mms: bool, mms_kind: str) -> ValidationFields:
        """Generate and materialize the selected distributed validation fields."""
        if mms:
            token = np.zeros((self.local_devices,), dtype=np.float32)
            builder = (
                self.make_broadband_fields
                if mms_kind == "broadband"
                else self.make_mode_fields
            )
            rhs, reference = builder(token)
        else:
            first_device = self.process_index * self.local_devices
            device_ids = np.arange(
                first_device,
                first_device + self.local_devices,
                dtype=np.uint32,
            )
            base_key = self.jax.random.PRNGKey(seed)
            keys = self.jax.vmap(
                lambda index: self.jax.random.fold_in(base_key, index)
            )(self.jnp.asarray(device_ids))
            rhs = self.make_random_rhs(keys)
            reference = None
        rhs.block_until_ready()
        return ValidationFields(rhs=rhs, reference=reference)

    def evaluate(
        self,
        *,
        pipeline: PoissonPipelineStages,
        operators: PipelineOperatorBundle,
        rhs,
        output,
        reference,
    ) -> ValidationMetrics:
        """Evaluate the discrete solve residual and optional MMS error."""
        residual = operators.residual(pipeline, rhs)
        residual.block_until_ready()
        relative_residual = float(np.asarray(residual)[0])

        mms_error = None
        if reference is not None:
            difference = self.relative_max_difference(output, reference)
            difference.block_until_ready()
            mms_error = float(np.asarray(difference)[0])
        return ValidationMetrics(
            relative_residual=relative_residual,
            mms_error=mms_error,
        )


def build_poisson_validation(
    *,
    args,
    jax: Any,
    jnp: Any,
    lax: Any,
    layout: ArrayLayoutOps,
    axis_name: str,
    local_devices: int,
    process_index: int,
    nz_local: int,
    real_dtype: Any,
    kx,
    ky,
    keep,
    zero_tolerance: float,
    dz2: float,
) -> PoissonValidationSuite:
    """Build mapped RHS generators and validation collectives."""
    nx, ny, nz = args.nx, args.ny, args.nz
    mapped = partial(jax.pmap, axis_name=axis_name)
    z_first_layout = layout.z_first

    def make_random_rhs_local(key):
        shape = (nz_local, ny, nx) if z_first_layout else (nx, ny, nz_local)
        return jax.random.uniform(key, shape, dtype=real_dtype) - 0.5

    make_random_rhs = mapped(make_random_rhs_local)

    def make_mode_fields_local(_token):
        """Build an exact discrete manufactured pair on one z slab."""
        device_index = lax.axis_index(axis_name)
        x = jnp.arange(nx, dtype=real_dtype) * (args.lx / nx)
        y = jnp.arange(ny, dtype=real_dtype) * (args.ly / ny)
        level = device_index * nz_local + jnp.arange(nz_local, dtype=real_dtype)
        zeta = (level + 0.5) / (nz - 1)
        shape = (nz_local, ny, nx) if z_first_layout else (nx, ny, nz_local)
        pressure = jnp.zeros(shape, real_dtype)
        rhs = jnp.zeros_like(pressure)
        for amplitude, mode_x, mode_y, mode_z, trig_x, trig_y in MMS_TERMS:
            wave_x = 2.0 * math.pi * mode_x / args.lx
            wave_y = 2.0 * math.pi * mode_y / args.ly
            eigenvalue = (2.0 - 2.0 * math.cos(mode_z * math.pi / (nz - 1))) / dz2
            profile_x = jnp.cos(wave_x * x) if trig_x == "cos" else jnp.sin(wave_x * x)
            profile_y = jnp.cos(wave_y * y) if trig_y == "cos" else jnp.sin(wave_y * y)
            profile_z = jnp.cos(mode_z * jnp.pi * zeta)
            if z_first_layout:
                term = (
                    amplitude
                    * profile_z[:, None, None]
                    * profile_y[None, :, None]
                    * profile_x[None, None, :]
                )
            else:
                term = (
                    amplitude
                    * profile_x[:, None, None]
                    * profile_y[None, :, None]
                    * profile_z[None, None, :]
                )
            pressure = pressure + term
            rhs = rhs - (wave_x * wave_x + wave_y * wave_y + eigenvalue) * term
        return rhs, pressure

    make_mode_fields = mapped(make_mode_fields_local)

    def make_broadband_fields_local(_token):
        """Build a deterministic broad-spectrum exact discrete pair."""
        device_index = lax.axis_index(axis_name)
        wave_squared = kx[:, None] * kx[:, None] + ky[None, :] * ky[None, :]
        reference_wave = 4.0 * 2.0 * math.pi / args.lx
        envelope = (1.0 + wave_squared / (reference_wave * reference_wave)) ** (
            -2.0 / 3.0
        )
        shape_mask = (envelope * keep).astype(real_dtype)
        base_key = jax.random.PRNGKey(args.seed + 7)

        def level_spectrum(level):
            source = jnp.minimum(level, nz - 2)
            key = jax.random.fold_in(base_key, source)
            if z_first_layout:
                values = jax.random.uniform(key, (ny, nx), dtype=real_dtype) - 0.5
                return jnp.fft.rfftn(values, axes=(-2, -1)) * jnp.swapaxes(
                    shape_mask,
                    0,
                    1,
                )
            values = jax.random.uniform(key, (nx, ny), dtype=real_dtype) - 0.5
            return jnp.fft.rfftn(values, axes=(-1, -2)) * shape_mask

        base_level = device_index * nz_local
        levels = jnp.clip(
            base_level + jnp.arange(-1, nz_local + 1),
            0,
            nz - 1,
        )
        spectra_by_level = jax.vmap(level_spectrum)(levels)
        spectra = (
            spectra_by_level
            if z_first_layout
            else jnp.moveaxis(spectra_by_level, 0, -1)
        )

        if z_first_layout:
            middle = spectra[1:-1]
            below = spectra[:-2]
            above = spectra[2:]
        else:
            middle = spectra[..., 1:-1]
            below = spectra[..., :-2]
            above = spectra[..., 2:]
        global_levels = base_level + jnp.arange(nz_local)
        if z_first_layout:
            local_wave_squared = jnp.swapaxes(wave_squared, 0, 1)
            pinned = (global_levels == 0)[:, None, None] & (
                jnp.abs(local_wave_squared) < zero_tolerance
            )[None, ...]
        else:
            local_wave_squared = wave_squared
            pinned = (jnp.abs(wave_squared) < zero_tolerance)[..., None] & (
                global_levels == 0
            )[None, None, :]
        below = jnp.where(pinned, 0.0, below)
        if z_first_layout:
            rhs_spectrum = jnp.where(
                (global_levels <= nz - 2)[:, None, None],
                (below + above) / dz2
                + (-local_wave_squared[None, ...] - 2.0 / dz2) * middle,
                0.0,
            )
        else:
            rhs_spectrum = jnp.where(
                (global_levels <= nz - 2)[None, None, :],
                (below + above) / dz2
                + (-local_wave_squared[..., None] - 2.0 / dz2) * middle,
                0.0,
            )
        rhs = layout.inverse_fft_local(rhs_spectrum).astype(real_dtype)
        pressure = layout.inverse_fft_local(middle).astype(real_dtype)
        return rhs, pressure

    make_broadband_fields = mapped(make_broadband_fields_local)

    def relative_max_difference_local(left, right):
        numerator = lax.pmax(jnp.max(jnp.abs(left - right)), axis_name)
        denominator = lax.pmax(jnp.max(jnp.abs(right)), axis_name)
        return numerator / jnp.maximum(
            denominator,
            jnp.finfo(real_dtype).tiny,
        )

    relative_max_difference = mapped(relative_max_difference_local)

    def global_max_local(value):
        return lax.pmax(value, axis_name)

    global_max = mapped(global_max_local)
    return PoissonValidationSuite(
        jax=jax,
        jnp=jnp,
        local_devices=local_devices,
        process_index=process_index,
        make_random_rhs=make_random_rhs,
        make_mode_fields=make_mode_fields,
        make_broadband_fields=make_broadband_fields,
        relative_max_difference=relative_max_difference,
        global_max=global_max,
    )
