from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spectral_fd import Poisson3DConfig, Poisson3DSolver


class Poisson3DConfigTests(unittest.TestCase):
    def test_defaults_translate_to_legacy_arguments(self) -> None:
        config = Poisson3DConfig()
        args = config._as_legacy_namespace()

        self.assertEqual((args.nx, args.ny, args.nz), (1024, 1024, 128))
        self.assertFalse(args.no_nyquist_filter)
        self.assertFalse(args.mms)
        self.assertTrue(args.skip_components)

    def test_nyquist_filter_translation(self) -> None:
        args = Poisson3DConfig(nyquist_filter=False)._as_legacy_namespace()
        self.assertTrue(args.no_nyquist_filter)

    def test_rejects_odd_horizontal_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "nx and ny must be even"):
            Poisson3DConfig(nx=33).validate()

    def test_rejects_nonpositive_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "lx, ly, and lz"):
            Poisson3DConfig(lz=0.0).validate()

    def test_solver_proxies_public_engine(self) -> None:
        engine = SimpleNamespace(
            global_devices=8,
            local_devices=1,
            process_count=8,
            process_index=3,
            local_input_shape=(1, 16, 128, 128),
            global_input_shape=(128, 128, 128),
            solve=lambda rhs, execution=None: ("solve", rhs, execution),
            residual=lambda rhs: ("residual", rhs),
        )
        config = Poisson3DConfig(
            nx=128,
            ny=128,
            nz=128,
            data_layout="z-first",
        )

        with patch("poisson3d_distributed.build_solver", return_value=engine):
            solver = Poisson3DSolver(config)

        self.assertEqual(solver.global_devices, 8)
        self.assertEqual(solver.local_input_shape, (1, 16, 128, 128))
        self.assertEqual(
            solver.solve("rhs", execution="staged"),
            ("solve", "rhs", "staged"),
        )
        self.assertEqual(solver.residual("rhs"), ("residual", "rhs"))


if __name__ == "__main__":
    unittest.main()
