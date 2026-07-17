from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from spectral_fd import Poisson3DConfig
from spectral_fd.cli import build_parser, main
from spectral_fd.runtime import (
    configure_jax_environment,
    initialize_jax_distributed,
    local_device_id,
)


class Poisson3DCLITests(unittest.TestCase):
    def test_parser_solver_defaults_match_public_config(self) -> None:
        args = build_parser().parse_args([])
        config = Poisson3DConfig()

        self.assertEqual((args.nx, args.ny, args.nz), (config.nx, config.ny, config.nz))
        self.assertEqual(args.tridiag, config.tridiag)
        self.assertEqual(args.method, config.method)
        self.assertEqual(args.pipeline_execution, config.pipeline_execution)
        self.assertEqual(args.data_layout, config.data_layout)

    def test_package_cli_delegates_parsed_options_to_driver(self) -> None:
        with patch("spectral_fd.driver.run_benchmark", return_value=17) as run:
            result = main(["--nx", "64", "--ny", "96", "--skip-components"])

        self.assertEqual(result, 17)
        options = run.call_args.args[0]
        self.assertEqual((options.nx, options.ny), (64, 96))
        self.assertTrue(options.skip_components)

    def test_parser_accepts_structured_report_paths(self) -> None:
        args = build_parser().parse_args(
            ["--report-json", "result.json", "--report-csv", "result.csv"]
        )
        self.assertEqual(args.report_json, "result.json")
        self.assertEqual(args.report_csv, "result.csv")

    def test_legacy_script_is_only_a_package_cli_forwarder(self) -> None:
        import poisson3d_distributed

        with patch("spectral_fd.cli.main", return_value=23) as package_main:
            result = poisson3d_distributed.main(["--nx", "32"])

        self.assertEqual(result, 23)
        package_main.assert_called_once_with(["--nx", "32"])


class JaxRuntimeTests(unittest.TestCase):
    def test_configure_environment_preserves_explicit_preallocation(self) -> None:
        environ = {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}

        configure_jax_environment(
            platform="rocm",
            dtype="float64",
            environ=environ,
        )

        self.assertEqual(environ["JAX_PLATFORMS"], "rocm")
        self.assertEqual(environ["JAX_ENABLE_X64"], "true")
        self.assertEqual(environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "true")

    def test_local_device_uses_zero_for_single_visible_device(self) -> None:
        environ = {"ROCR_VISIBLE_DEVICES": "5", "SLURM_LOCALID": "3"}
        self.assertEqual(local_device_id(environ), 0)

    def test_local_device_uses_slurm_rank_for_node_visible_devices(self) -> None:
        environ = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_LOCALID": "2"}
        self.assertEqual(local_device_id(environ), 2)

    def test_distributed_initialization_is_idempotent(self) -> None:
        distributed = SimpleNamespace(
            is_initialized=Mock(return_value=False),
            initialize=Mock(),
        )
        jax_module = SimpleNamespace(distributed=distributed)

        initialize_jax_distributed(
            jax_module,
            enabled=True,
            environ={"SLURM_LOCALID": "4"},
        )

        distributed.initialize.assert_called_once_with(local_device_ids=[4])

        distributed.is_initialized.return_value = True
        initialize_jax_distributed(jax_module, enabled=True)
        distributed.initialize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
