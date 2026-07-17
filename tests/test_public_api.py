from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spectral_fd import Poisson3DConfig, Poisson3DSolver
from spectral_fd.cli import build_parser
from spectral_fd.driver import solver_config_from_options


class Poisson3DConfigTests(unittest.TestCase):
    def test_cli_defaults_translate_to_public_config(self) -> None:
        args = build_parser().parse_args([])
        config = solver_config_from_options(args)

        self.assertEqual((config.nx, config.ny, config.nz), (1024, 1024, 128))
        self.assertTrue(config.nyquist_filter)
        self.assertEqual(config.method, "transpose")

    def test_cli_nyquist_filter_translation(self) -> None:
        args = build_parser().parse_args(["--no-nyquist-filter"])
        self.assertFalse(solver_config_from_options(args).nyquist_filter)

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

        with patch("spectral_fd.factory.build_solver_engine", return_value=engine):
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
