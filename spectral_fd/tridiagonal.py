"""PCR and Thomas factorization/solve kernels for vertical systems."""

from __future__ import annotations

from typing import Any

from .config import TridiagonalSolver
from .transforms import ArrayLayoutOps


class TridiagonalOps:
    """Bind a vertical layout and solver policy to reusable JAX kernels."""

    __slots__ = ("jnp", "lax", "layout", "method", "thomas_chunk")

    def __init__(
        self,
        *,
        jnp: Any,
        lax: Any,
        layout: ArrayLayoutOps,
        method: TridiagonalSolver,
        thomas_chunk: int,
    ) -> None:
        if method not in ("pcr", "thomas"):
            raise ValueError(f"unsupported tridiagonal solver: {method!r}")
        if thomas_chunk <= 0:
            raise ValueError("thomas_chunk must be positive")
        self.jnp = jnp
        self.lax = lax
        self.layout = layout
        self.method = method
        self.thomas_chunk = thomas_chunk

    def thomas_factor_arrays(self, a, b, c):
        """Precompute the Thomas LU sweep factors."""

        def factor_step(carry, rows):
            inv_bet_previous, c_previous = carry
            a_row, b_row, c_row = rows
            gamma_row = c_previous * inv_bet_previous
            inv_bet_row = 1.0 / (b_row - a_row * gamma_row)
            return (inv_bet_row, c_row), (inv_bet_row, gamma_row)

        inv_bet_first = 1.0 / self.layout.z_first_value(b)
        _, (inv_bet_tail, gamma_tail) = self.lax.scan(
            factor_step,
            (inv_bet_first, self.layout.z_first_value(c)),
            (
                self.layout.move_z_first(self.layout.without_z_first(a)),
                self.layout.move_z_first(self.layout.without_z_first(b)),
                self.layout.move_z_first(self.layout.without_z_first(c)),
            ),
        )
        inv_bet = self.layout.prepend_z(inv_bet_first, inv_bet_tail)
        gamma = self.layout.prepend_z(
            self.jnp.zeros_like(inv_bet_first),
            gamma_tail,
        )
        return inv_bet, gamma

    def pcr_factor_arrays(self, a, b, c, *, steps: int):
        """Precompute parallel-cyclic-reduction elimination factors."""
        alphas, gammas = [], []
        for index in range(steps):
            distance = 1 << index
            alpha = -a / self.layout.shift_z_down(b, distance, fill=1.0)
            gamma = -c / self.layout.shift_z_up(b, distance, fill=1.0)
            b = (
                b
                + alpha * self.layout.shift_z_down(c, distance)
                + gamma * self.layout.shift_z_up(a, distance)
            )
            a = alpha * self.layout.shift_z_down(a, distance)
            c = gamma * self.layout.shift_z_up(c, distance)
            alphas.append(alpha)
            gammas.append(gamma)
        return self.jnp.stack(alphas), self.jnp.stack(gammas), 1.0 / b

    def pcr_apply(self, alphas, gammas, inv_b, rhs, *, steps: int):
        """Apply precomputed PCR factors to one or more right-hand sides."""
        result = rhs
        for index in range(steps):
            distance = 1 << index
            result = (
                result
                + alphas[index] * self.layout.shift_z_down(result, distance)
                + gammas[index] * self.layout.shift_z_up(result, distance)
            )
        return result * inv_b

    def thomas_solve_scalar_scan(self, a, inv_bet, gamma, rhs):
        """Thomas forward/backward substitution with one row per scan step."""
        u_first = self.layout.z_first_value(rhs) * self.layout.z_first_value(inv_bet)

        def forward(previous, values):
            a_row, inv_bet_row, rhs_row = values
            current = (rhs_row - a_row * previous) * inv_bet_row
            return current, current

        _, u_tail_z = self.lax.scan(
            forward,
            u_first,
            (
                self.layout.move_z_first(self.layout.without_z_first(a)),
                self.layout.move_z_first(self.layout.without_z_first(inv_bet)),
                self.layout.move_z_first(self.layout.without_z_first(rhs)),
            ),
        )
        u_forward = self.layout.prepend_z(u_first, u_tail_z)

        def backward(next_value, values):
            forward_value, gamma_next = values
            current = forward_value - gamma_next * next_value
            return current, current

        _, prefix_reversed_z = self.lax.scan(
            backward,
            self.layout.z_last_value(u_forward),
            (
                self.layout.move_z_first(self.layout.without_z_last(u_forward))[::-1],
                self.layout.move_z_first(self.layout.without_z_first(gamma))[::-1],
            ),
        )
        return self.layout.append_z(
            prefix_reversed_z[::-1],
            self.layout.z_last_value(u_forward),
        )

    def _chunked_scan_rows(self, initial, row_arrays, step):
        """Scan rows while statically unrolling each outer scan chunk."""
        row_count = row_arrays[0].shape[0]
        chunk = min(self.thomas_chunk, row_count)
        chunk_count = row_count // chunk
        full_count = chunk_count * chunk
        carry = initial
        pieces = []

        if chunk_count:
            chunked_arrays = tuple(
                values[:full_count].reshape((chunk_count, chunk) + values.shape[1:])
                for values in row_arrays
            )

            def scan_chunk(previous, chunks):
                outputs = []
                current = previous
                for row in range(chunk):
                    current = step(
                        current,
                        tuple(values[row] for values in chunks),
                    )
                    outputs.append(current)
                return current, self.jnp.stack(outputs)

            carry, full_output = self.lax.scan(
                scan_chunk,
                carry,
                chunked_arrays,
            )
            pieces.append(full_output.reshape((full_count,) + full_output.shape[2:]))

        if full_count < row_count:
            tail = []
            for row in range(full_count, row_count):
                carry = step(
                    carry,
                    tuple(values[row] for values in row_arrays),
                )
                tail.append(carry)
            pieces.append(self.jnp.stack(tail))

        if len(pieces) == 1:
            return pieces[0]
        return self.jnp.concatenate(pieces, axis=0)

    def thomas_solve_chunked(self, a, inv_bet, gamma, rhs):
        """Thomas solve with dependent z rows fused per outer scan body."""
        a_z = self.layout.move_z_first(a)
        inv_bet_z = self.layout.move_z_first(inv_bet)
        rhs_z = self.layout.move_z_first(rhs)
        zero = self.jnp.zeros_like(rhs_z[0])

        def forward_step(previous, values):
            a_row, inv_bet_row, rhs_row = values
            return (rhs_row - a_row * previous) * inv_bet_row

        u_forward_z = self._chunked_scan_rows(
            zero,
            (a_z, inv_bet_z, rhs_z),
            forward_step,
        )
        gamma_z = self.layout.move_z_first(gamma)
        gamma_next_z = self.jnp.concatenate(
            (gamma_z[1:], self.jnp.zeros_like(gamma_z[:1])),
            axis=0,
        )

        def backward_step(next_value, values):
            forward_value, gamma_next = values
            return forward_value - gamma_next * next_value

        reversed_z = self._chunked_scan_rows(
            zero,
            (u_forward_z[::-1], gamma_next_z[::-1]),
            backward_step,
        )
        return self.layout.move_z_last(reversed_z[::-1])

    def thomas_solve(self, a, inv_bet, gamma, rhs):
        if self.thomas_chunk == 1:
            return self.thomas_solve_scalar_scan(a, inv_bet, gamma, rhs)
        return self.thomas_solve_chunked(a, inv_bet, gamma, rhs)

    def solve(self, operator1, operator2, operator3, rhs, *, pcr_steps: int):
        """Apply the configured factor representation to ``rhs``."""
        if self.method == "thomas":
            return self.thomas_solve(operator1, operator2, operator3, rhs)
        return self.pcr_apply(
            operator1,
            operator2,
            operator3,
            rhs,
            steps=pcr_steps,
        )
