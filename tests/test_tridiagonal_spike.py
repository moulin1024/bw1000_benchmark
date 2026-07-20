from __future__ import annotations

import unittest

import numpy as np

from spectral_fd.spike import SpikeInterfaceOps
from spectral_fd.transforms import ArrayLayoutOps
from spectral_fd.tridiagonal import TridiagonalOps

try:
    import jax.numpy as jnp
    from jax import lax
except ImportError:  # pragma: no cover - platform packages can omit JAX
    jnp = None
    lax = None


@unittest.skipIf(jnp is None, "JAX is not installed")
class TridiagonalOpsTests(unittest.TestCase):
    def _system(self, layout_name: str):
        row_count = 6
        rhs_z = np.arange(1, row_count * 6 + 1, dtype=np.float32).reshape(
            row_count,
            2,
            3,
        )
        a_z = np.full_like(rhs_z, -1.0)
        b_z = np.full_like(rhs_z, 4.0)
        c_z = np.full_like(rhs_z, -1.0)
        a_z[0] = 0.0
        c_z[-1] = 0.0
        if layout_name == "z-first":
            arrays = (a_z, b_z, c_z, rhs_z)
        else:
            arrays = tuple(
                np.moveaxis(values, 0, -1) for values in (a_z, b_z, c_z, rhs_z)
            )

        matrix = np.diag(np.full(row_count, 4.0, dtype=np.float32))
        matrix += np.diag(np.full(row_count - 1, -1.0, dtype=np.float32), -1)
        matrix += np.diag(np.full(row_count - 1, -1.0, dtype=np.float32), 1)
        expected = np.linalg.solve(matrix, rhs_z.reshape(row_count, -1)).reshape(
            rhs_z.shape
        )
        layout = ArrayLayoutOps(
            data_layout=layout_name,
            nx=8,
            ny=6,
            array_namespace=jnp,
        )
        return layout, tuple(jnp.asarray(values) for values in arrays), expected

    def test_thomas_scalar_and_chunked_match_dense_reference(self) -> None:
        for layout_name in ("z-first", "xyz"):
            for chunk in (1, 2):
                with self.subTest(layout=layout_name, chunk=chunk):
                    layout, (a, b, c, rhs), expected = self._system(layout_name)
                    solver = TridiagonalOps(
                        jnp=jnp,
                        lax=lax,
                        layout=layout,
                        method="thomas",
                        thomas_chunk=chunk,
                    )
                    inv_bet, gamma = solver.thomas_factor_arrays(a, b, c)
                    result = solver.thomas_solve(a, inv_bet, gamma, rhs)
                    result_z = np.asarray(layout.move_z_first(result))

                    np.testing.assert_allclose(result_z, expected, rtol=2e-6, atol=2e-6)

    def test_pcr_matches_dense_reference_for_both_layouts(self) -> None:
        for layout_name in ("z-first", "xyz"):
            with self.subTest(layout=layout_name):
                layout, (a, b, c, rhs), expected = self._system(layout_name)
                solver = TridiagonalOps(
                    jnp=jnp,
                    lax=lax,
                    layout=layout,
                    method="pcr",
                    thomas_chunk=1,
                )
                steps = 3
                factors = solver.pcr_factor_arrays(a, b, c, steps=steps)
                result = solver.pcr_apply(*factors, rhs, steps=steps)
                result_z = np.asarray(layout.move_z_first(result))

                np.testing.assert_allclose(result_z, expected, rtol=2e-6, atol=2e-6)


@unittest.skipIf(jnp is None, "JAX is not installed")
class SpikeInterfaceOpsTests(unittest.TestCase):
    def _ops(
        self,
        solver: str,
        discretization: str = "legacy-augmented",
    ) -> SpikeInterfaceOps:
        return SpikeInterfaceOps(
            jnp=jnp,
            lax=lax,
            axis_name="devices",
            device_count=2,
            nxh=1,
            ny=2,
            real_dtype=jnp.float32,
            zero_tolerance=1e-6,
            collective="allgather",
            solver=solver,
            ic_cut=1,
            jc_cut=1,
            discretization=discretization,
        )

    def test_structured_block_solve_matches_dense_interface_matrix(self) -> None:
        ops = self._ops("block-thomas")
        interface = jnp.asarray(
            [
                [[[-0.10]], [[-0.04]], [[-0.03]], [[-0.08]]],
                [[[-0.07]], [[-0.02]], [[0.00]], [[0.00]]],
            ],
            dtype=jnp.float32,
        )
        k2 = jnp.asarray([[2.0]], dtype=jnp.float32)
        rhs_blocks = jnp.asarray(
            [[[[0.3]], [[-0.2]]], [[[0.4]], [[0.1]]]],
            dtype=jnp.float32,
        )
        factors, bottom = ops.build_block_factors(interface, k2)
        structured = np.asarray(ops.solve_blocks(factors, rhs_blocks, bottom))

        matrix = np.asarray(ops.assemble_matrix(interface, k2))[0, 0]
        rhs = np.concatenate(([0.0], np.asarray(rhs_blocks)[:, :, 0, 0].reshape(-1)))
        solution = np.concatenate((np.linalg.solve(matrix, rhs), [0.0]))
        expected = np.asarray(
            [
                [[solution[0]], [solution[3]]],
                [[solution[2]], [solution[5]]],
            ],
            dtype=np.float32,
        )[..., None]

        np.testing.assert_allclose(structured, expected, rtol=2e-6, atol=2e-6)

    def test_dense_block_and_selected_rows_agree_for_owned_rows(self) -> None:
        interface = jnp.asarray(
            [
                [[[-0.10]], [[-0.04]], [[-0.03]], [[-0.08]]],
                [[[-0.07]], [[-0.02]], [[0.00]], [[0.00]]],
            ],
            dtype=jnp.float32,
        )
        rhs = jnp.asarray(
            [[[[0.3]], [[-0.2]]], [[[0.4]], [[0.1]]]],
            dtype=jnp.float32,
        )
        k2 = jnp.asarray([[2.0]], dtype=jnp.float32)
        results = []
        for solver_name in ("dense", "block-thomas", "selected-rows"):
            ops = self._ops(solver_name)
            operator, bottom = ops.build_operator(interface, k2, device_index=0)
            results.append(
                np.asarray(
                    ops.apply_operator(
                        operator,
                        rhs,
                        bottom,
                        device_index=0,
                    )
                )
            )

        np.testing.assert_allclose(results[1], results[0], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(results[2], results[0], rtol=2e-6, atol=2e-6)

    def test_compatible_interface_is_standard_two_endpoint_system(self) -> None:
        ops = self._ops("dense", "cell-centered-compatible")
        interface = jnp.asarray(
            [
                [[[-0.10]], [[-0.04]], [[-0.03]], [[-0.08]]],
                [[[-0.07]], [[-0.02]], [[0.00]], [[0.00]]],
            ],
            dtype=jnp.float32,
        )
        matrix = np.asarray(
            ops.assemble_matrix(interface, jnp.asarray([[0.0]], jnp.float32))
        )[0, 0]
        expected = np.asarray(
            [
                [1.0, 0.0, -0.03, 0.0],
                [0.0, 1.0, -0.08, 0.0],
                [0.0, -0.07, 1.0, 0.0],
                [0.0, -0.02, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.assertEqual(ops.reduced_size, 4)
        np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1.0e-7)

    def test_compatible_dense_block_and_selected_rows_agree(self) -> None:
        interface = jnp.asarray(
            [
                [[[-0.10]], [[-0.04]], [[-0.03]], [[-0.08]]],
                [[[-0.07]], [[-0.02]], [[0.00]], [[0.00]]],
            ],
            dtype=jnp.float32,
        )
        rhs = jnp.asarray(
            [[[[0.3]], [[-0.2]]], [[[0.4]], [[0.1]]]],
            dtype=jnp.float32,
        )
        k2 = jnp.asarray([[0.0]], dtype=jnp.float32)
        results = []
        for solver_name in ("dense", "block-thomas", "selected-rows"):
            ops = self._ops(solver_name, "cell-centered-compatible")
            operator, bottom = ops.build_operator(interface, k2, device_index=0)
            results.append(
                np.asarray(
                    ops.apply_operator(
                        operator,
                        rhs,
                        bottom,
                        device_index=0,
                    )
                )
            )

        np.testing.assert_allclose(results[1], results[0], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(results[2], results[0], rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
