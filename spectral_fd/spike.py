"""Communication and reduced-interface solvers for SPIKE decomposition."""

from __future__ import annotations

from typing import Any

from .config import InterfaceCollective, InterfaceSolver


class SpikeInterfaceOps:
    """Bind SPIKE interface communication, factorization, and solve policy."""

    __slots__ = (
        "jnp",
        "lax",
        "axis_name",
        "device_count",
        "nxh",
        "ny",
        "nyq",
        "real_dtype",
        "zero_tolerance",
        "collective",
        "solver",
        "ic_cut",
        "jc_cut",
    )

    def __init__(
        self,
        *,
        jnp: Any,
        lax: Any,
        axis_name: str,
        device_count: int,
        nxh: int,
        ny: int,
        real_dtype: Any,
        zero_tolerance: float,
        collective: InterfaceCollective,
        solver: InterfaceSolver,
        ic_cut: int,
        jc_cut: int,
    ) -> None:
        if collective not in ("alltoall", "allgather"):
            raise ValueError(f"unsupported SPIKE collective: {collective!r}")
        if solver not in ("selected-rows", "block-thomas", "dense"):
            raise ValueError(f"unsupported SPIKE interface solver: {solver!r}")
        self.jnp = jnp
        self.lax = lax
        self.axis_name = axis_name
        self.device_count = device_count
        self.nxh = nxh
        self.ny = ny
        self.nyq = ny // device_count
        self.real_dtype = real_dtype
        self.zero_tolerance = zero_tolerance
        self.collective = collective
        self.solver = solver
        self.ic_cut = ic_cut
        self.jc_cut = jc_cut

    @property
    def reduced_size(self) -> int:
        return 2 * self.device_count + 1

    def box_slice(self, values):
        """Extract the low-|ky| columns of the low-kx adaptive box."""
        return self.jnp.concatenate(
            (
                values[: self.ic_cut, : self.jc_cut],
                values[: self.ic_cut, self.ny - self.jc_cut :],
            ),
            axis=1,
        )

    def box_scatter(self, full, small):
        full = full.at[: self.ic_cut, : self.jc_cut].set(small[:, : self.jc_cut])
        return full.at[
            : self.ic_cut,
            self.ny - self.jc_cut :,
        ].set(small[:, self.jc_cut :])

    def scalars_to_modes(self, stacked):
        """Move per-block scalars onto horizontal-mode owners."""
        scalar_count = stacked.shape[0]
        modes = stacked.reshape(
            scalar_count,
            self.nxh,
            self.device_count,
            self.nyq,
        )
        modes = self.jnp.moveaxis(modes, 2, 0)
        if self.device_count > 1:
            modes = self.lax.all_to_all(
                modes,
                self.axis_name,
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
        return modes

    def modes_to_scalars(self, modes):
        """Return per-destination mode values to their SPIKE blocks."""
        if self.device_count > 1:
            modes = self.lax.all_to_all(
                modes,
                self.axis_name,
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
        modes = self.jnp.moveaxis(modes, 0, 2)
        return modes.reshape(modes.shape[0], self.nxh, self.ny)

    def gather_block_scalars(self, stacked):
        """Replicate per-block scalars on every device."""
        if self.device_count > 1:
            return self.lax.all_gather(
                stacked,
                self.axis_name,
                axis=0,
                tiled=False,
            )
        return stacked[None, ...]

    def collect_block_scalars(self, stacked):
        """Apply the configured interface communication layout."""
        if self.collective == "allgather":
            return self.gather_block_scalars(stacked)
        return self.scalars_to_modes(stacked)

    def assemble_matrix(self, interface, k2):
        """Assemble the static ``(2P+1)``-row reduced interface matrix."""
        zero_k2 = self.jnp.abs(k2) < self.zero_tolerance
        one = self.jnp.asarray(1.0, self.real_dtype)
        matrix = self.jnp.zeros(
            k2.shape + (self.reduced_size, self.reduced_size),
            self.real_dtype,
        )
        matrix = matrix.at[..., 0, 0].set(self.jnp.where(zero_k2, one, -one))
        matrix = matrix.at[..., 0, 1].set(self.jnp.where(zero_k2, 0.0 * one, one))
        for block in range(self.device_count):
            alpha_row = 1 + 2 * block
            beta_row = 2 + 2 * block
            matrix = matrix.at[..., alpha_row, alpha_row].set(1.0)
            matrix = matrix.at[..., beta_row, beta_row].set(1.0)
            matrix = matrix.at[..., alpha_row, 2 * block].set(interface[block, 0])
            matrix = matrix.at[..., beta_row, 2 * block].set(interface[block, 1])
            if block < self.device_count - 1:
                matrix = matrix.at[..., alpha_row, 2 * block + 3].set(
                    interface[block, 2]
                )
                matrix = matrix.at[..., beta_row, 2 * block + 3].set(
                    interface[block, 3]
                )
        return matrix

    def build_block_factors(self, interface, k2):
        """Build exact structured block-Thomas interface factors."""
        bottom_coefficient = self.jnp.where(
            self.jnp.abs(k2) < self.zero_tolerance,
            0.0,
            1.0,
        ).astype(self.real_dtype)
        zero = self.jnp.zeros_like(bottom_coefficient)
        factors = []
        previous_c1 = zero
        for block in range(self.device_count):
            w_first, w_last, v_first, v_last = interface[block]
            if block == 0:
                pivot = 1.0 + w_first * bottom_coefficient
                g00 = 1.0 / pivot
                g10 = -(w_last * bottom_coefficient) * g00
                a0 = zero
                a1 = zero
            else:
                pivot = 1.0 - w_first * previous_c1
                g00 = 1.0 / pivot
                g10 = (w_last * previous_c1) * g00
                a0 = g00 * w_first
                a1 = g10 * w_first + w_last
            if block < self.device_count - 1:
                c0 = g00 * v_first
                c1 = g10 * v_first + v_last
            else:
                c0 = zero
                c1 = zero
            factors.append(self.jnp.stack((g00, g10, a0, a1, c0, c1)))
            previous_c1 = c1
        return self.jnp.stack(factors), bottom_coefficient

    def solve_blocks(self, factors, rhs_blocks, bottom_coefficient):
        """Apply the structured block-Thomas interface factors."""

        def forward(previous_beta, values):
            factor, rhs_block = values
            g00, g10, a0, a1 = factor[:4]
            rhs_first, rhs_last = rhs_block
            z0 = g00 * rhs_first - a0 * previous_beta
            z1 = g10 * rhs_first + rhs_last - a1 * previous_beta
            return z1, self.jnp.stack((z0, z1))

        initial = self.jnp.zeros_like(rhs_blocks[0, 0])
        _, z_blocks = self.lax.scan(
            forward,
            initial,
            (factors, rhs_blocks),
        )

        def backward(next_alpha, values):
            factor, z_block = values
            c0, c1 = factor[4], factor[5]
            alpha = z_block[0] - c0 * next_alpha
            beta = z_block[1] - c1 * next_alpha
            return alpha, self.jnp.stack((alpha, beta))

        _, reversed_blocks = self.lax.scan(
            backward,
            initial,
            (factors[::-1], z_blocks[::-1]),
        )
        blocks = reversed_blocks[::-1]
        left = self.jnp.concatenate(
            (
                (bottom_coefficient * blocks[0, 0])[None, ...],
                blocks[:-1, 1],
            ),
            axis=0,
        )
        right = self.jnp.concatenate(
            (blocks[1:, 0], self.jnp.zeros_like(blocks[:1, 0])),
            axis=0,
        )
        return self.jnp.stack((left, right), axis=1)

    def build_selected_response(self, factors, bottom_coefficient, selectors):
        """Build selected response rows via the transposed structured solve."""
        left_bar = selectors[:, :, 0]
        right_bar = selectors[:, :, 1]
        zero_block = self.jnp.zeros_like(left_bar[:, :1])
        x0_bar = self.jnp.concatenate(
            (
                bottom_coefficient[None, None, ...] * left_bar[:, :1],
                right_bar[:, :-1],
            ),
            axis=1,
        )
        x1_bar = self.jnp.concatenate((left_bar[:, 1:], zero_block), axis=1)

        def backward_transpose(next_x0_bar, values):
            factor, direct_x0_bar, direct_x1_bar = values
            c0, c1 = factor[4], factor[5]
            total_x0_bar = direct_x0_bar + next_x0_bar
            z0_bar = total_x0_bar
            z1_bar = direct_x1_bar
            following_x0_bar = -c0 * z0_bar - c1 * z1_bar
            return following_x0_bar, self.jnp.stack(
                (z0_bar, z1_bar),
                axis=1,
            )

        initial = self.jnp.zeros_like(x0_bar[:, 0])
        _, z_bar_scan = self.lax.scan(
            backward_transpose,
            initial,
            (
                factors,
                self.jnp.swapaxes(x0_bar, 0, 1),
                self.jnp.swapaxes(x1_bar, 0, 1),
            ),
        )
        z_bar = self.jnp.swapaxes(z_bar_scan, 0, 1)

        def forward_transpose(previous_z1_bar, values):
            factor, direct_z_bar = values
            g00, g10, a0, a1 = factor[:4]
            z0_bar = direct_z_bar[:, 0]
            z1_bar = direct_z_bar[:, 1] + previous_z1_bar
            rhs_first_bar = g00 * z0_bar + g10 * z1_bar
            rhs_last_bar = z1_bar
            preceding_z1_bar = -a0 * z0_bar - a1 * z1_bar
            return preceding_z1_bar, self.jnp.stack(
                (rhs_first_bar, rhs_last_bar),
                axis=1,
            )

        _, rhs_bar_reversed = self.lax.scan(
            forward_transpose,
            initial,
            (factors[::-1], self.jnp.swapaxes(z_bar, 0, 1)[::-1]),
        )
        return self.jnp.swapaxes(rhs_bar_reversed[::-1], 0, 1)

    def build_operator(self, interface, k2, *, device_index):
        """Build the configured dense, block, or selected-row operator."""
        if self.solver == "dense":
            return (
                self.jnp.linalg.inv(self.assemble_matrix(interface, k2)),
                self.jnp.asarray(0.0, self.real_dtype),
            )

        block_factors, bottom_coefficient = self.build_block_factors(
            interface,
            k2,
        )
        if self.solver == "block-thomas":
            return block_factors, bottom_coefficient

        selector_count = 2 if self.collective == "allgather" else 2 * self.device_count
        selector_basis = self.jnp.eye(
            2 * self.device_count,
            dtype=self.real_dtype,
        )
        if self.collective == "allgather":
            selector_basis = self.lax.dynamic_slice_in_dim(
                selector_basis,
                2 * device_index,
                selector_count,
                axis=0,
            )
        selectors = self.jnp.broadcast_to(
            selector_basis.reshape(
                selector_count,
                self.device_count,
                2,
                1,
                1,
            ),
            (selector_count, self.device_count, 2) + k2.shape,
        )
        response = self.build_selected_response(
            block_factors,
            bottom_coefficient,
            selectors,
        )
        if self.collective == "alltoall":
            response = response.reshape(
                self.device_count,
                2,
                self.device_count,
                2,
                self.nxh,
                self.nyq,
            )
        return response, self.jnp.asarray(0.0, self.real_dtype)

    def apply_operator(
        self,
        operator,
        interface_rhs,
        bottom_coefficient,
        *,
        device_index,
    ):
        """Solve the configured interface representation for ``(L, R)``."""
        if self.solver == "dense":
            interface_ny = interface_rhs.shape[-1]
            rhs = self.jnp.transpose(interface_rhs, (2, 3, 0, 1)).reshape(
                self.nxh,
                interface_ny,
                2 * self.device_count,
            )
            rhs = self.jnp.concatenate(
                (self.jnp.zeros_like(rhs[..., :1]), rhs),
                axis=-1,
            )
            solution = self.jnp.einsum("...ij,...j->...i", operator, rhs)
            solution = self.jnp.concatenate(
                (solution, self.jnp.zeros_like(solution[..., :1])),
                axis=-1,
            )
            if self.collective == "allgather":
                left = self.lax.dynamic_index_in_dim(
                    solution,
                    2 * device_index,
                    axis=-1,
                    keepdims=False,
                )
                right = self.lax.dynamic_index_in_dim(
                    solution,
                    2 * device_index + 3,
                    axis=-1,
                    keepdims=False,
                )
                return self.jnp.stack((left, right))
            outputs = self.jnp.stack(
                [
                    self.jnp.stack(
                        (
                            solution[..., 2 * block],
                            solution[..., 2 * block + 3],
                        )
                    )
                    for block in range(self.device_count)
                ]
            )
            return self.modes_to_scalars(outputs)

        if self.solver == "block-thomas":
            outputs = self.solve_blocks(
                operator,
                interface_rhs,
                bottom_coefficient,
            )
            if self.collective == "allgather":
                return self.lax.dynamic_index_in_dim(
                    outputs,
                    device_index,
                    axis=0,
                    keepdims=False,
                )
            return self.modes_to_scalars(outputs)

        if self.collective == "allgather":
            return self.jnp.einsum(
                "rpqij,pqij->rij",
                operator,
                interface_rhs,
            )
        outputs = self.jnp.einsum(
            "dspqij,pqij->dsij",
            operator,
            interface_rhs,
        )
        return self.modes_to_scalars(outputs)
