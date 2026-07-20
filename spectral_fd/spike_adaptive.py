"""Adaptive PDD closure and low-wavenumber exact SPIKE correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DType
from .spike import SpikeInterfaceOps
from .spike_local import SpikeLocalBlockOps
from .transforms import ArrayLayoutOps


@dataclass(frozen=True, slots=True)
class AdaptiveModeBox:
    """Low-wavenumber box retaining the exact global interface solve."""

    ic_cut: int
    jc_cut: int
    global_modes: int


def compute_adaptive_mode_box(
    *,
    nxh: int,
    ny: int,
    block_size: int,
    dz: float,
    lx: float,
    ly: float,
    dtype: DType,
) -> AdaptiveModeBox:
    """Select modes whose cross-block spike decay is not negligible."""
    real_dtype = np.float32 if dtype == "float32" else np.float64
    truncation_tau = float(-np.log(np.finfo(real_dtype).eps)) + 4.0
    truncation_argument = truncation_tau / (2.0 * block_size)
    if truncation_argument < 30.0:
        kh_cut = (2.0 / dz) * math.sinh(truncation_argument)
        ic_cut = max(
            1,
            min(nxh, int(kh_cut * lx / (2.0 * math.pi)) + 2),
        )
        jc_cut = max(
            1,
            min(ny // 2, int(kh_cut * ly / (2.0 * math.pi)) + 2),
        )
    else:
        ic_cut = nxh
        jc_cut = ny // 2
    return AdaptiveModeBox(
        ic_cut=ic_cut,
        jc_cut=jc_cut,
        global_modes=ic_cut * 2 * jc_cut,
    )


class AdaptiveSpikeOps:
    """Build and apply neighbour PDD closure plus exact low-kh correction."""

    __slots__ = (
        "jnp",
        "lax",
        "layout",
        "local_blocks",
        "interface",
        "axis_name",
        "device_count",
        "kx",
        "ky",
        "real_dtype",
        "zero_tolerance",
        "tiny",
    )

    def __init__(
        self,
        *,
        jnp: Any,
        lax: Any,
        layout: ArrayLayoutOps,
        local_blocks: SpikeLocalBlockOps,
        interface: SpikeInterfaceOps,
        axis_name: str,
        device_count: int,
        kx,
        ky,
        dtype: DType,
        real_dtype: Any,
        zero_tolerance: float,
    ) -> None:
        self.jnp = jnp
        self.lax = lax
        self.layout = layout
        self.local_blocks = local_blocks
        self.interface = interface
        self.axis_name = axis_name
        self.device_count = device_count
        self.kx = kx
        self.ky = ky
        self.real_dtype = real_dtype
        self.zero_tolerance = zero_tolerance
        host_dtype = np.float32 if dtype == "float32" else np.float64
        self.tiny = float(np.finfo(host_dtype).tiny)

    def _previous_permutation(self):
        return [
            (index, (index + 1) % self.device_count)
            for index in range(self.device_count)
        ]

    def _next_permutation(self):
        return [
            (index, (index - 1) % self.device_count)
            for index in range(self.device_count)
        ]

    def build_factors(self, device_index):
        """Build local factors, PDD coefficients, and exact-box inverse."""
        operator1, operator2, operator3, left_spike, right_spike = (
            self.local_blocks.build(device_index)
        )
        w_first = self.layout.mode_to_interface(self.layout.z_first_value(left_spike))
        v_last = self.layout.mode_to_interface(self.layout.z_last_value(right_spike))
        v_last_previous = self.lax.ppermute(
            v_last,
            self.axis_name,
            self._previous_permutation(),
        )
        w_first_next = self.lax.ppermute(
            w_first,
            self.axis_name,
            self._next_permutation(),
        )
        tiny = self.jnp.asarray(self.tiny, self.real_dtype)
        previous_determinant = 1.0 - v_last_previous * w_first
        next_determinant = 1.0 - v_last * w_first_next
        inverse_previous = 1.0 / self.jnp.where(
            self.jnp.abs(previous_determinant) > tiny,
            previous_determinant,
            1.0,
        )
        inverse_next = 1.0 / self.jnp.where(
            self.jnp.abs(next_determinant) > tiny,
            next_determinant,
            1.0,
        )

        k2 = self.kx[:, None] * self.kx[:, None] + self.ky[None, :] * self.ky[None, :]
        if self.interface.cell_centered:
            bottom_ratio = self.jnp.zeros_like(k2, dtype=self.real_dtype)
        else:
            bottom_ratio = self.jnp.where(
                self.jnp.abs(k2) < self.zero_tolerance,
                0.0,
                -1.0,
            )
        bottom_denominator = 1.0 - w_first * bottom_ratio
        bottom_left_coefficient = -bottom_ratio / self.jnp.where(
            self.jnp.abs(bottom_denominator) > tiny,
            bottom_denominator,
            1.0,
        )

        endpoints = self.jnp.stack(
            (
                self.interface.box_slice(w_first),
                self.interface.box_slice(
                    self.layout.mode_to_interface(self.layout.z_last_value(left_spike))
                ),
                self.interface.box_slice(
                    self.layout.mode_to_interface(
                        self.layout.z_first_value(right_spike)
                    )
                ),
                self.interface.box_slice(v_last),
            )
        )
        gathered = self.interface.gather_block_scalars(endpoints)
        exact_inverse = self.jnp.linalg.inv(
            self.interface.assemble_matrix(
                gathered,
                self.interface.box_slice(k2),
            )
        )
        return (
            operator1,
            operator2,
            operator3,
            left_spike,
            right_spike,
            exact_inverse,
            inverse_previous,
            v_last_previous,
            inverse_next,
            w_first_next,
            bottom_left_coefficient,
        )

    def apply(
        self,
        rhs_hat_local,
        operator1,
        operator2,
        operator3,
        left_spike,
        right_spike,
        exact_inverse,
        inverse_previous,
        v_last_previous,
        inverse_next,
        w_first_next,
        bottom_left_coefficient,
        *,
        device_index,
    ):
        """Return unfiltered pressure, interface values, and masked RHS."""
        block_size = self.local_blocks.block_size
        if self.interface.cell_centered:
            rhs = rhs_hat_local
        else:
            z_mask = (self.jnp.arange(block_size) < block_size - 1) | (
                device_index != self.device_count - 1
            )
            rhs = self.jnp.where(
                self.layout.z_broadcast(z_mask),
                rhs_hat_local,
                0,
            )
        local_solution = self.local_blocks.solve(
            operator1,
            operator2,
            operator3,
            rhs,
        )
        first = self.layout.mode_to_interface(self.layout.z_first_value(local_solution))
        last = self.layout.mode_to_interface(self.layout.z_last_value(local_solution))

        previous_last = self.lax.ppermute(
            last,
            self.axis_name,
            self._previous_permutation(),
        )
        next_first = self.lax.ppermute(
            first,
            self.axis_name,
            self._next_permutation(),
        )
        left = self.jnp.where(
            device_index == 0,
            bottom_left_coefficient * first,
            inverse_previous * (previous_last - v_last_previous * first),
        )
        right = self.jnp.where(
            device_index == self.device_count - 1,
            self.jnp.zeros_like(last),
            inverse_next * (next_first - w_first_next * last),
        )

        gathered = self.interface.gather_block_scalars(
            self.jnp.stack(
                (
                    self.interface.box_slice(first),
                    self.interface.box_slice(last),
                )
            )
        )
        rhs_interface = self.jnp.transpose(gathered, (2, 3, 0, 1)).reshape(
            self.interface.ic_cut,
            2 * self.interface.jc_cut,
            2 * self.device_count,
        )
        if not self.interface.cell_centered:
            rhs_interface = self.jnp.concatenate(
                (self.jnp.zeros_like(rhs_interface[..., :1]), rhs_interface),
                axis=-1,
            )
        exact_solution = self.jnp.einsum(
            "...ij,...j->...i",
            exact_inverse,
            rhs_interface,
        )
        zero = self.jnp.zeros_like(exact_solution[..., :1])
        if self.interface.cell_centered:
            exact_solution = self.jnp.concatenate(
                (zero, exact_solution, zero),
                axis=-1,
            )
        else:
            exact_solution = self.jnp.concatenate(
                (exact_solution, zero),
                axis=-1,
            )
        left = self.interface.box_scatter(
            left,
            self.lax.dynamic_index_in_dim(
                exact_solution,
                2 * device_index,
                axis=-1,
                keepdims=False,
            ),
        )
        right = self.interface.box_scatter(
            right,
            self.lax.dynamic_index_in_dim(
                exact_solution,
                2 * device_index + 3,
                axis=-1,
                keepdims=False,
            ),
        )
        interface_values = self.jnp.stack((left, right))
        pressure = (
            local_solution
            - left_spike * self.layout.mode_broadcast(left)
            - right_spike * self.layout.mode_broadcast(right)
        )
        return pressure, interface_values, rhs
