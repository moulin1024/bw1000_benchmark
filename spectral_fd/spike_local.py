"""Local block factorization and spike-vector assembly."""

from __future__ import annotations

from typing import Any

from .operators import build_spike_block_rows
from .transforms import ArrayLayoutOps
from .tridiagonal import TridiagonalOps


class SpikeLocalBlockOps:
    """Build and solve the independent tridiagonal blocks used by SPIKE."""

    __slots__ = (
        "jnp",
        "layout",
        "tridiagonal",
        "kx",
        "ky",
        "block_size",
        "nz",
        "dz2",
        "real_dtype",
        "pcr_steps",
    )

    def __init__(
        self,
        *,
        jnp: Any,
        layout: ArrayLayoutOps,
        tridiagonal: TridiagonalOps,
        kx,
        ky,
        block_size: int,
        nz: int,
        dz2: float,
        real_dtype: Any,
    ) -> None:
        self.jnp = jnp
        self.layout = layout
        self.tridiagonal = tridiagonal
        self.kx = kx
        self.ky = ky
        self.block_size = block_size
        self.nz = nz
        self.dz2 = dz2
        self.real_dtype = real_dtype
        self.pcr_steps = max(1, (block_size - 1).bit_length())

    @property
    def description(self) -> str:
        if self.tridiagonal.method == "thomas":
            return (
                f"thomas ({self.block_size}-row forward/backward scans, "
                f"compact-a, chunk {self.tridiagonal.thomas_chunk})"
            )
        return f"pcr ({self.pcr_steps} steps)"

    def build_rows(self, device_index):
        """Build full-coupling rows owned by one device block."""
        return build_spike_block_rows(
            device_index,
            jnp=self.jnp,
            layout=self.layout,
            kx=self.kx,
            ky=self.ky,
            block_size=self.block_size,
            nz=self.nz,
            dz2=self.dz2,
            real_dtype=self.real_dtype,
        )

    def solve(self, operator1, operator2, operator3, rhs):
        return self.tridiagonal.solve(
            operator1,
            operator2,
            operator3,
            rhs,
            pcr_steps=self.pcr_steps,
        )

    def build(self, device_index):
        """Return local factors and the left/right spike response vectors."""
        a_block, b_block, c_block = self.build_rows(device_index)

        # Couplings leaving the block move into the reduced interface system.
        a_first = a_block[0]
        c_last = c_block[self.block_size - 1]
        a_local = a_block.at[0].set(0.0)
        c_local = c_block.at[self.block_size - 1].set(0.0)

        if self.tridiagonal.method == "thomas":
            # a/c depend only on the block row. Keeping a compact avoids
            # storing and streaming a full mode-by-z array.
            inv_bet, gamma = self.tridiagonal.thomas_factor_arrays(
                a_local,
                b_block,
                c_local,
            )
            local_operators = (a_local, inv_bet, gamma)
        else:
            a_input = self.jnp.broadcast_to(
                self.layout.z_broadcast(a_local),
                b_block.shape,
            )
            c_input = self.jnp.broadcast_to(
                self.layout.z_broadcast(c_local),
                b_block.shape,
            )
            local_operators = self.tridiagonal.pcr_factor_arrays(
                a_input,
                b_block,
                c_input,
                steps=self.pcr_steps,
            )

        first_basis = (
            self.jnp.zeros(
                (self.block_size,),
                self.real_dtype,
            )
            .at[0]
            .set(1.0)
        )
        last_basis = (
            self.jnp.zeros(
                (self.block_size,),
                self.real_dtype,
            )
            .at[self.block_size - 1]
            .set(1.0)
        )
        left_spike = self.solve(
            *local_operators,
            self.jnp.broadcast_to(
                self.layout.z_broadcast(a_first * first_basis),
                b_block.shape,
            ),
        )
        right_spike = self.solve(
            *local_operators,
            self.jnp.broadcast_to(
                self.layout.z_broadcast(c_last * last_basis),
                b_block.shape,
            ),
        )
        return *local_operators, left_spike, right_spike
