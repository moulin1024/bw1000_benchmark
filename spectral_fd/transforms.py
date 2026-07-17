"""Array-layout conversions and local horizontal FFT operations."""

from __future__ import annotations

from typing import Any

from .config import DataLayout


class ArrayLayoutOps:
    """Static layout operations shared by setup and compiled solve paths.

    The array namespace is injected so importing :mod:`spectral_fd` remains
    independent of JAX initialization. In production it is ``jax.numpy``;
    tests can use NumPy to check the layout algebra directly.
    """

    __slots__ = ("data_layout", "nx", "ny", "xp")

    def __init__(
        self,
        *,
        data_layout: DataLayout,
        nx: int,
        ny: int,
        array_namespace: Any,
    ) -> None:
        if data_layout not in ("xyz", "z-first"):
            raise ValueError(f"unsupported data layout: {data_layout!r}")
        self.data_layout = data_layout
        self.nx = nx
        self.ny = ny
        self.xp = array_namespace

    @property
    def z_first(self) -> bool:
        return self.data_layout == "z-first"

    def move_z_first(self, values):
        return values if self.z_first else self.xp.moveaxis(values, -1, 0)

    def move_z_last(self, values):
        return values if self.z_first else self.xp.moveaxis(values, 0, -1)

    def z_first_value(self, values):
        return values[0] if self.z_first else values[..., 0]

    def z_last_value(self, values):
        return values[-1] if self.z_first else values[..., -1]

    def without_z_first(self, values):
        return values[1:] if self.z_first else values[..., 1:]

    def without_z_last(self, values):
        return values[:-1] if self.z_first else values[..., :-1]

    def prepend_z(self, first, tail_z):
        if self.z_first:
            return self.xp.concatenate((first[None, ...], tail_z), axis=0)
        return self.xp.concatenate(
            (first[..., None], self.move_z_last(tail_z)),
            axis=-1,
        )

    def append_z(self, prefix_z, last):
        if self.z_first:
            return self.xp.concatenate((prefix_z, last[None, ...]), axis=0)
        return self.xp.concatenate(
            (self.move_z_last(prefix_z), last[..., None]),
            axis=-1,
        )

    def z_broadcast(self, values):
        return values[:, None, None] if self.z_first else values[None, None, :]

    def mode_to_interface(self, values):
        """Convert a local horizontal plane to canonical ``(kx, ky)``."""
        return self.xp.swapaxes(values, -1, -2) if self.z_first else values

    def interface_to_mode(self, values):
        """Convert canonical ``(kx, ky)`` to the local horizontal order."""
        return self.xp.swapaxes(values, -1, -2) if self.z_first else values

    def mode_broadcast(self, values):
        local = self.interface_to_mode(values)
        return local[None, ...] if self.z_first else local[..., None]

    def spectral_keep(self, keep_modes):
        return self.mode_broadcast(keep_modes)

    def forward_fft_local(self, rhs):
        axes = (-2, -1) if self.z_first else (-2, -3)
        return self.xp.fft.rfftn(rhs, axes=axes)

    def inverse_fft_local(self, spectral):
        axes = (-2, -1) if self.z_first else (-2, -3)
        return self.xp.fft.irfftn(
            spectral,
            s=(self.ny, self.nx),
            axes=axes,
        )

    def z_to_y(self, spectral, *, lax: Any, axis_name: str, device_count: int):
        """Exchange a distributed z-slab into the vertical-solve y-slab."""
        if self.z_first:
            nz_local, ny, nxh = spectral.shape
            exchanged = spectral.reshape(
                nz_local,
                device_count,
                ny // device_count,
                nxh,
            )
            if device_count > 1:
                exchanged = lax.all_to_all(
                    exchanged,
                    axis_name,
                    split_axis=1,
                    concat_axis=0,
                    tiled=True,
                )
            return exchanged.reshape(
                nz_local * device_count,
                ny // device_count,
                nxh,
            )

        nxh, ny, nz_local = spectral.shape
        exchanged = spectral.reshape(
            nxh,
            device_count,
            ny // device_count,
            nz_local,
        )
        if device_count > 1:
            exchanged = lax.all_to_all(
                exchanged,
                axis_name,
                split_axis=1,
                concat_axis=3,
                tiled=True,
            )
        return exchanged.reshape(
            nxh,
            ny // device_count,
            nz_local * device_count,
        )

    def y_to_z(self, spectral, *, lax: Any, axis_name: str, device_count: int):
        """Exchange a vertical-solve y-slab back into the FFT z-slab."""
        if self.z_first:
            nz, ny_local, nxh = spectral.shape
            exchanged = spectral.reshape(
                device_count,
                nz // device_count,
                ny_local,
                nxh,
            )
            if device_count > 1:
                exchanged = lax.all_to_all(
                    exchanged,
                    axis_name,
                    split_axis=0,
                    concat_axis=2,
                    tiled=True,
                )
            return exchanged.reshape(
                nz // device_count,
                ny_local * device_count,
                nxh,
            )

        nxh, ny_local, nz = spectral.shape
        exchanged = spectral.reshape(
            nxh,
            ny_local,
            device_count,
            nz // device_count,
        )
        if device_count > 1:
            exchanged = lax.all_to_all(
                exchanged,
                axis_name,
                split_axis=2,
                concat_axis=1,
                tiled=True,
            )
        return exchanged.reshape(
            nxh,
            ny_local * device_count,
            nz // device_count,
        )

    def shift_z_down(self, values, distance: int, *, fill: float = 0.0):
        if self.z_first:
            pad = self.xp.full(
                (distance,) + values.shape[1:],
                fill,
                dtype=values.dtype,
            )
            return self.xp.concatenate((pad, values[:-distance]), axis=0)
        pad = self.xp.full(
            values.shape[:-1] + (distance,),
            fill,
            dtype=values.dtype,
        )
        return self.xp.concatenate((pad, values[..., :-distance]), axis=-1)

    def shift_z_up(self, values, distance: int, *, fill: float = 0.0):
        if self.z_first:
            pad = self.xp.full(
                (distance,) + values.shape[1:],
                fill,
                dtype=values.dtype,
            )
            return self.xp.concatenate((values[distance:], pad), axis=0)
        pad = self.xp.full(
            values.shape[:-1] + (distance,),
            fill,
            dtype=values.dtype,
        )
        return self.xp.concatenate((values[..., distance:], pad), axis=-1)
