from __future__ import annotations

import unittest

import numpy as np

from spectral_fd.operators import build_horizontal_symbols
from spectral_fd.transforms import ArrayLayoutOps


class ArrayLayoutOpsTests(unittest.TestCase):
    def test_local_fft_round_trip_for_both_layouts(self) -> None:
        rng = np.random.default_rng(7)
        z_first = rng.standard_normal((3, 6, 8))

        for name, physical in (
            ("z-first", z_first),
            ("xyz", np.transpose(z_first, (2, 1, 0))),
        ):
            with self.subTest(layout=name):
                layout = ArrayLayoutOps(
                    data_layout=name,
                    nx=8,
                    ny=6,
                    array_namespace=np,
                )
                spectral = layout.forward_fft_local(physical)
                restored = layout.inverse_fft_local(spectral)

                self.assertEqual(restored.shape, physical.shape)
                np.testing.assert_allclose(restored, physical, atol=1e-12)

    def test_vertical_axis_conversions_are_inverse_for_xyz(self) -> None:
        layout = ArrayLayoutOps(
            data_layout="xyz",
            nx=8,
            ny=6,
            array_namespace=np,
        )
        values = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        z_last = layout.move_z_last(values)

        np.testing.assert_array_equal(layout.move_z_first(z_last), values)
        np.testing.assert_array_equal(layout.z_first_value(z_last), values[0])
        np.testing.assert_array_equal(layout.z_last_value(z_last), values[-1])

    def test_mode_conversion_and_broadcast_for_z_first(self) -> None:
        layout = ArrayLayoutOps(
            data_layout="z-first",
            nx=8,
            ny=6,
            array_namespace=np,
        )
        canonical = np.arange(5 * 6).reshape(5, 6)
        local = layout.interface_to_mode(canonical)

        self.assertEqual(local.shape, (6, 5))
        np.testing.assert_array_equal(layout.mode_to_interface(local), canonical)
        self.assertEqual(layout.mode_broadcast(canonical).shape, (1, 6, 5))

    def test_z_shifts_follow_each_layout_axis(self) -> None:
        canonical = np.arange(4 * 2 * 3).reshape(4, 2, 3)
        for name, values in (
            ("z-first", canonical),
            ("xyz", np.moveaxis(canonical, 0, -1)),
        ):
            with self.subTest(layout=name):
                layout = ArrayLayoutOps(
                    data_layout=name,
                    nx=8,
                    ny=6,
                    array_namespace=np,
                )
                down = layout.move_z_first(layout.shift_z_down(values, 1))
                up = layout.move_z_first(layout.shift_z_up(values, 1))

                np.testing.assert_array_equal(down[0], 0)
                np.testing.assert_array_equal(down[1:], canonical[:-1])
                np.testing.assert_array_equal(up[:-1], canonical[1:])
                np.testing.assert_array_equal(up[-1], 0)

    def test_single_device_slab_exchange_round_trip(self) -> None:
        rng = np.random.default_rng(11)
        for name, spectral in (
            ("z-first", rng.standard_normal((3, 6, 5))),
            ("xyz", rng.standard_normal((5, 6, 3))),
        ):
            with self.subTest(layout=name):
                layout = ArrayLayoutOps(
                    data_layout=name,
                    nx=8,
                    ny=6,
                    array_namespace=np,
                )
                y_slab = layout.z_to_y(
                    spectral,
                    lax=None,
                    axis_name="devices",
                    device_count=1,
                )
                restored = layout.y_to_z(
                    y_slab,
                    lax=None,
                    axis_name="devices",
                    device_count=1,
                )

                np.testing.assert_array_equal(restored, spectral)


class HorizontalSymbolsTests(unittest.TestCase):
    def test_nyquist_filter_and_zero_wavenumbers(self) -> None:
        symbols = build_horizontal_symbols(
            nx=8,
            ny=6,
            lx=2.0,
            ly=3.0,
            dtype="float64",
            nyquist_filter=True,
        )

        self.assertEqual(symbols.kx.shape, (5,))
        self.assertEqual(symbols.ky.shape, (6,))
        self.assertEqual(symbols.keep.shape, (5, 6))
        self.assertEqual(symbols.kx[-1], 0.0)
        self.assertEqual(symbols.ky[3], 0.0)
        self.assertTrue(np.all(symbols.keep[-1] == 0.0))
        self.assertTrue(np.all(symbols.keep[:, 3] == 0.0))

    def test_nyquist_modes_can_be_retained(self) -> None:
        symbols = build_horizontal_symbols(
            nx=8,
            ny=6,
            lx=1.0,
            ly=1.0,
            dtype="float32",
            nyquist_filter=False,
        )

        self.assertEqual(symbols.kx.dtype, np.dtype("float32"))
        self.assertTrue(np.all(symbols.keep == 1.0))


if __name__ == "__main__":
    unittest.main()
