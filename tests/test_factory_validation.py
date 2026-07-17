from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from spectral_fd.factory import JaxRuntimeContext, PoissonSolverAssembly
from spectral_fd.validation import (
    PoissonValidationSuite,
    validate_mms_configuration,
)


class ReadyArray:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)
        self.waits = 0

    def block_until_ready(self):
        self.waits += 1
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


class ValidationConfigurationTests(unittest.TestCase):
    def test_mode_mms_rejects_grid_below_fixed_modes(self) -> None:
        args = SimpleNamespace(mms=True, mms_kind="modes", nx=8, ny=8, nz=8)
        with self.assertRaisesRegex(ValueError, "below Nyquist"):
            validate_mms_configuration(args)

    def test_broadband_mms_does_not_use_fixed_mode_limits(self) -> None:
        args = SimpleNamespace(mms=True, mms_kind="broadband", nx=4, ny=4, nz=4)
        validate_mms_configuration(args)

    def test_validation_suite_combines_residual_and_mms_error(self) -> None:
        residual = ReadyArray([2.0e-6])
        difference = ReadyArray([3.0e-5])
        operators = SimpleNamespace(residual=Mock(return_value=residual))
        relative_max_difference = Mock(return_value=difference)
        suite = PoissonValidationSuite(
            jax=None,
            jnp=None,
            local_devices=1,
            process_index=0,
            make_random_rhs=Mock(),
            make_mode_fields=Mock(),
            make_broadband_fields=Mock(),
            relative_max_difference=relative_max_difference,
            global_max=Mock(),
        )

        metrics = suite.evaluate(
            pipeline="pipeline",
            operators=operators,
            rhs="rhs",
            output="output",
            reference="reference",
        )

        self.assertAlmostEqual(metrics.relative_residual, 2.0e-6)
        self.assertAlmostEqual(metrics.mms_error, 3.0e-5)
        operators.residual.assert_called_once_with("pipeline", "rhs")
        relative_max_difference.assert_called_once_with("output", "reference")
        self.assertEqual((residual.waits, difference.waits), (1, 1))


class SolverAssemblyTests(unittest.TestCase):
    def test_assembly_projects_engine_and_benchmark_metadata(self) -> None:
        runtime = JaxRuntimeContext(
            jax=SimpleNamespace(__version__="test-jax"),
            jnp=None,
            lax=None,
            global_devices=4,
            local_devices=2,
            process_count=2,
            process_index=1,
            backend="cpu",
        )
        decomposition = SimpleNamespace(
            nx=32,
            ny=16,
            nz=8,
            nxh=17,
            ny_local=4,
            nz_local=2,
            local_physical_shape=Mock(return_value=(2, 2, 16, 32)),
            global_physical_shape=Mock(return_value=(8, 16, 32)),
        )
        residual = Mock()
        operators = SimpleNamespace(
            solve_args=("factors",),
            bind_residual=Mock(return_value=residual),
        )
        pipeline = SimpleNamespace(
            solve_monolithic=Mock(),
            solve_staged=Mock(),
        )
        assembly = PoissonSolverAssembly(
            runtime=runtime,
            decomposition=decomposition,
            layout=SimpleNamespace(z_first=True),
            real_dtype="float64",
            real_itemsize=8,
            physical_bytes=4096,
            spectral_bytes=8192,
            axis_name="devices",
            pcr_steps=4,
            block_rows=2,
            spike_local_description="thomas",
            reduced_size=9,
            interface_ny=4,
            adaptive_global_modes=11,
            kx="kx",
            ky="ky",
            keep="keep",
            zero_tolerance=1.0e-12,
            dz2=0.01,
            pipeline=pipeline,
            operators=operators,
        )
        config = SimpleNamespace(
            data_layout="z-first",
            dtype="float64",
            pipeline_execution="monolithic",
        )

        engine = assembly.create_engine(config)
        context = assembly.benchmark_context()

        self.assertEqual(engine.local_input_shape, (2, 2, 16, 32))
        self.assertEqual(engine.global_input_shape, (8, 16, 32))
        self.assertEqual(context.jax_version, "test-jax")
        self.assertEqual(context.global_devices, 4)
        self.assertEqual(context.adaptive_global_modes, 11)
        operators.bind_residual.assert_called_once_with(pipeline)


if __name__ == "__main__":
    unittest.main()
