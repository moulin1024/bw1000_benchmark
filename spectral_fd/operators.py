"""Construction of horizontal symbols and vertical tridiagonal rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import DType
from .transforms import ArrayLayoutOps


@dataclass(frozen=True, slots=True)
class HorizontalSymbols:
    """Host-side wavenumbers and the spectral-mode retention mask."""

    kx: np.ndarray
    ky: np.ndarray
    keep: np.ndarray
    zero_tolerance: float


def build_horizontal_symbols(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    dtype: DType,
    nyquist_filter: bool,
) -> HorizontalSymbols:
    """Build periodic spectral symbols without importing or initializing JAX."""
    real_dtype = np.float32 if dtype == "float32" else np.float64
    kx = (2.0 * np.pi * np.fft.rfftfreq(nx, d=lx / nx)).astype(real_dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)).astype(real_dtype)

    # WIRELES treats the self-conjugate Nyquist modes as zero wavenumbers;
    # the separate keep mask controls whether those modes participate.
    kx[-1] = 0.0
    ky[ny // 2] = 0.0
    keep = np.ones((nx // 2 + 1, ny), dtype=real_dtype)
    if nyquist_filter:
        keep[-1, :] = 0.0
        keep[:, ny // 2] = 0.0

    return HorizontalSymbols(
        kx=kx,
        ky=ky,
        keep=keep,
        zero_tolerance=float(np.finfo(real_dtype).eps) * 128.0,
    )


def build_vertical_operator_rows(
    device_index,
    *,
    jnp: Any,
    lax: Any,
    layout: ArrayLayoutOps,
    kx,
    ky,
    keep,
    ny_local: int,
    nz: int,
    dz2: float,
    real_dtype: Any,
    zero_tolerance: float,
    discretization: str = "legacy-augmented",
):
    """Build one device's rows of the distributed vertical operator."""
    ky_local = lax.dynamic_slice_in_dim(
        ky,
        device_index * ny_local,
        ny_local,
    )
    keep_local = lax.dynamic_slice_in_dim(
        keep,
        device_index * ny_local,
        ny_local,
        axis=1,
    )
    k2 = kx[:, None] * kx[:, None] + ky_local[None, :] * ky_local[None, :]
    zero_k2 = jnp.abs(k2) < zero_tolerance
    nxh = kx.shape[0]
    one = jnp.asarray(1.0, real_dtype)

    if discretization == "cell-centered-compatible":
        if layout.z_first:
            shape = (nz, ny_local, nxh)
            k2_local = layout.interface_to_mode(k2)
            pinned_local = layout.interface_to_mode(zero_k2 | (keep_local == 0))
            a = jnp.zeros(shape, real_dtype)
            a = a.at[1:].set(1.0 / dz2)
            b = jnp.broadcast_to(
                -k2_local[None, ...] - 2.0 / dz2,
                shape,
            )
            b = b.at[0].add(1.0 / dz2)
            b = b.at[-1].add(1.0 / dz2)
            c = jnp.zeros(shape, real_dtype)
            c = c.at[:-1].set(1.0 / dz2)
            b = b.at[0].set(jnp.where(pinned_local, one, b[0]))
            c = c.at[0].set(jnp.where(pinned_local, 0.0 * one, c[0]))
            return a, b, c, layout.mode_broadcast(keep_local)

        shape = (nxh, ny_local, nz)
        pinned = zero_k2 | (keep_local == 0)
        a = jnp.zeros(shape, real_dtype)
        a = a.at[..., 1:].set(1.0 / dz2)
        b = jnp.broadcast_to(
            -k2[..., None] - 2.0 / dz2,
            shape,
        )
        b = b.at[..., 0].add(1.0 / dz2)
        b = b.at[..., -1].add(1.0 / dz2)
        c = jnp.zeros(shape, real_dtype)
        c = c.at[..., :-1].set(1.0 / dz2)
        b = b.at[..., 0].set(jnp.where(pinned, one, b[..., 0]))
        c = c.at[..., 0].set(jnp.where(pinned, 0.0 * one, c[..., 0]))
        return a, b, c, layout.mode_broadcast(keep_local)

    if discretization != "legacy-augmented":
        raise ValueError(f"unsupported discretization: {discretization!r}")

    if layout.z_first:
        shape = (nz + 1, ny_local, nxh)
        k2_local = layout.interface_to_mode(k2)
        zero_local = layout.interface_to_mode(zero_k2)
        a = jnp.zeros(shape, real_dtype)
        a = a.at[1:nz].set(1.0 / dz2)
        a = a.at[nz].set(-1.0)
        b = jnp.zeros(shape, real_dtype)
        b = b.at[0].set(jnp.where(zero_local, one, -one))
        b = b.at[1:nz].set(-k2_local[None, ...] - 2.0 / dz2)
        b = b.at[nz].set(1.0)
        c = jnp.zeros(shape, real_dtype)
        c = c.at[0].set(jnp.where(zero_local, 0.0 * one, one))
        c = c.at[1:nz].set(1.0 / dz2)
        return a, b, c, layout.mode_broadcast(keep_local)

    shape = (nxh, ny_local, nz + 1)
    a = jnp.zeros(shape, real_dtype)
    a = a.at[..., 1:nz].set(1.0 / dz2)
    a = a.at[..., nz].set(-1.0)
    b = jnp.zeros(shape, real_dtype)
    b = b.at[..., 0].set(jnp.where(zero_k2, one, -one))
    b = b.at[..., 1:nz].set(-k2[..., None] - 2.0 / dz2)
    b = b.at[..., nz].set(1.0)
    c = jnp.zeros(shape, real_dtype)
    c = c.at[..., 0].set(jnp.where(zero_k2, 0.0 * one, one))
    c = c.at[..., 1:nz].set(1.0 / dz2)
    return a, b, c, layout.mode_broadcast(keep_local)


def make_vertical_operator_builder(
    *,
    jnp: Any,
    lax: Any,
    layout: ArrayLayoutOps,
    axis_name: str,
    kx,
    ky,
    keep,
    ny_local: int,
    nz: int,
    dz2: float,
    real_dtype: Any,
    zero_tolerance: float,
    discretization: str = "legacy-augmented",
) -> Callable[[Any], tuple[Any, Any, Any, Any]]:
    """Bind static setup data to the function mapped across JAX devices."""

    def build(_token):
        return build_vertical_operator_rows(
            lax.axis_index(axis_name),
            jnp=jnp,
            lax=lax,
            layout=layout,
            kx=kx,
            ky=ky,
            keep=keep,
            ny_local=ny_local,
            nz=nz,
            dz2=dz2,
            real_dtype=real_dtype,
            zero_tolerance=zero_tolerance,
            discretization=discretization,
        )

    return build


def build_spike_block_rows(
    device_index,
    *,
    jnp: Any,
    layout: ArrayLayoutOps,
    kx,
    ky,
    block_size: int,
    nz: int,
    dz2: float,
    real_dtype: Any,
    zero_tolerance: float,
    discretization: str = "legacy-augmented",
):
    """Build the full-coupling tridiagonal rows owned by one SPIKE block."""
    inverse_dz2 = 1.0 / dz2
    k2 = kx[:, None] * kx[:, None] + ky[None, :] * ky[None, :]

    if discretization == "cell-centered-compatible":
        rows = device_index * block_size + jnp.arange(block_size)
        a = jnp.where(rows > 0, inverse_dz2, 0.0).astype(real_dtype)
        c = jnp.where(rows < nz - 1, inverse_dz2, 0.0).astype(real_dtype)
        shape = (
            (block_size, ky.shape[0], kx.shape[0])
            if layout.z_first
            else (kx.shape[0], ky.shape[0], block_size)
        )
        b = jnp.broadcast_to(
            -layout.mode_broadcast(k2) - 2.0 * inverse_dz2,
            shape,
        )
        lower_boundary = rows == 0
        upper_boundary = rows == nz - 1
        b = b + layout.z_broadcast(
            (lower_boundary | upper_boundary).astype(real_dtype) * inverse_dz2
        )

        # The Neumann zero mode is singular. Adding gamma*e0*e0^T with
        # gamma=2/dz^2 pins p[0]=0 for a compatible RHS without replacing a
        # physical PDE row, and keeps the first Thomas pivot nonzero. The
        # inverse transform later converts this to the public mean-zero gauge.
        zero_k2 = jnp.abs(k2) < zero_tolerance
        b = b + (
            layout.z_broadcast(lower_boundary.astype(real_dtype))
            * layout.mode_broadcast(zero_k2.astype(real_dtype))
            * (2.0 * inverse_dz2)
        )
        return a, b, c

    if discretization != "legacy-augmented":
        raise ValueError(f"unsupported discretization: {discretization!r}")

    rows = device_index * block_size + 1 + jnp.arange(block_size)
    interior = rows <= nz - 1
    a = jnp.where(interior, inverse_dz2, -1.0).astype(real_dtype)
    c = jnp.where(interior, inverse_dz2, 0.0).astype(real_dtype)
    if layout.z_first:
        b = jnp.where(
            interior[:, None, None],
            (-layout.interface_to_mode(k2)[None, ...] - 2.0 * inverse_dz2).astype(
                real_dtype
            ),
            jnp.asarray(1.0, real_dtype),
        )
    else:
        b = jnp.where(
            interior[None, None, :],
            (-k2[..., None] - 2.0 * inverse_dz2).astype(real_dtype),
            jnp.asarray(1.0, real_dtype),
        )
    return a, b, c
