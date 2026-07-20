from __future__ import annotations

import unittest

import numpy as np

from spectral_fd.spike_adaptive import compute_adaptive_mode_box
from spectral_fd.spike_local import SpikeLocalBlockOps
from spectral_fd.transforms import ArrayLayoutOps
from spectral_fd.tridiagonal import TridiagonalOps

try:
    import jax.numpy as jnp
    from jax import lax
except ImportError:  # pragma: no cover - platform packages can omit JAX
    jnp = None
    lax = None


class AdaptiveModeBoxTests(unittest.TestCase):
    def test_cutoff_is_bounded_and_counts_selected_modes(self) -> None:
        box = compute_adaptive_mode_box(
            nxh=513,
            ny=1024,
            block_size=128,
            dz=1.0 / 1024.0,
            lx=1.0,
            ly=1.0,
            dtype="float64",
        )

        self.assertGreaterEqual(box.ic_cut, 1)
        self.assertLessEqual(box.ic_cut, 513)
        self.assertGreaterEqual(box.jc_cut, 1)
        self.assertLessEqual(box.jc_cut, 512)
        self.assertEqual(box.global_modes, box.ic_cut * 2 * box.jc_cut)
        self.assertLess(box.global_modes, 513 * 1024)

    def test_short_blocks_keep_the_full_mode_box(self) -> None:
        box = compute_adaptive_mode_box(
            nxh=17,
            ny=32,
            block_size=1,
            dz=1.0 / 32.0,
            lx=1.0,
            ly=1.0,
            dtype="float64",
        )

        self.assertEqual((box.ic_cut, box.jc_cut), (17, 16))
        self.assertEqual(box.global_modes, 17 * 32)


@unittest.skipIf(jnp is None, "JAX is not installed")
class SpikeLocalBlockOpsTests(unittest.TestCase):
    def test_local_factors_and_spike_vectors_match_dense_solves(self) -> None:
        rng = np.random.default_rng(13)
        for layout_name in ("z-first", "xyz"):
            for method in ("pcr", "thomas"):
                with self.subTest(layout=layout_name, method=method):
                    layout = ArrayLayoutOps(
                        data_layout=layout_name,
                        nx=4,
                        ny=2,
                        array_namespace=jnp,
                    )
                    tridiagonal = TridiagonalOps(
                        jnp=jnp,
                        lax=lax,
                        layout=layout,
                        method=method,
                        thomas_chunk=2,
                    )
                    local = SpikeLocalBlockOps(
                        jnp=jnp,
                        layout=layout,
                        tridiagonal=tridiagonal,
                        kx=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
                        ky=jnp.asarray([0.5, 1.5], dtype=jnp.float32),
                        block_size=4,
                        nz=8,
                        dz2=0.25,
                        real_dtype=jnp.float32,
                        zero_tolerance=1.0e-6,
                    )
                    a_block, b_block, c_block = local.build_rows(0)
                    operator1, operator2, operator3, left_spike, right_spike = (
                        local.build(0)
                    )
                    rhs_z = rng.standard_normal((4, 2, 2)).astype(np.float32)
                    rhs = (
                        jnp.asarray(rhs_z)
                        if layout.z_first
                        else jnp.asarray(np.moveaxis(rhs_z, 0, -1))
                    )
                    result = local.solve(
                        operator1,
                        operator2,
                        operator3,
                        rhs,
                    )

                    a = np.asarray(a_block).copy()
                    c = np.asarray(c_block).copy()
                    a_first = a[0]
                    c_last = c[-1]
                    a[0] = 0.0
                    c[-1] = 0.0
                    b_z = np.asarray(layout.move_z_first(b_block))
                    expected = np.empty_like(rhs_z)
                    expected_left = np.empty_like(rhs_z)
                    expected_right = np.empty_like(rhs_z)
                    for ky_index in range(2):
                        for kx_index in range(2):
                            matrix = np.diag(b_z[:, ky_index, kx_index])
                            matrix += np.diag(a[1:], -1)
                            matrix += np.diag(c[:-1], 1)
                            expected[:, ky_index, kx_index] = np.linalg.solve(
                                matrix,
                                rhs_z[:, ky_index, kx_index],
                            )
                            left_rhs = np.zeros(4, dtype=np.float32)
                            left_rhs[0] = a_first
                            expected_left[:, ky_index, kx_index] = np.linalg.solve(
                                matrix,
                                left_rhs,
                            )
                            right_rhs = np.zeros(4, dtype=np.float32)
                            right_rhs[-1] = c_last
                            expected_right[:, ky_index, kx_index] = np.linalg.solve(
                                matrix,
                                right_rhs,
                            )

                    np.testing.assert_allclose(
                        np.asarray(layout.move_z_first(result)),
                        expected,
                        rtol=3e-6,
                        atol=3e-6,
                    )
                    np.testing.assert_allclose(
                        np.asarray(layout.move_z_first(left_spike)),
                        expected_left,
                        rtol=3e-6,
                        atol=3e-6,
                    )
                    np.testing.assert_allclose(
                        np.asarray(layout.move_z_first(right_spike)),
                        expected_right,
                        rtol=3e-6,
                        atol=3e-6,
                    )


if __name__ == "__main__":
    unittest.main()
