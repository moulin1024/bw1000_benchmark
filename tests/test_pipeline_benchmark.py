from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from spectral_fd.benchmark import DistributedBenchmarkRunner, human_bytes, percentile
from spectral_fd.pipeline import assemble_pipeline_operators


class ReadyArray:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)
        self.waits = 0

    def block_until_ready(self):
        self.waits += 1
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


class PipelineOperatorBundleTests(unittest.TestCase):
    def test_transpose_bundle_keeps_rows_and_binds_residual(self) -> None:
        a = ReadyArray([1.0])
        b = object()
        c = object()
        keep = object()
        inverse_beta = object()
        gamma = object()
        build_rows = Mock(return_value=(a, b, c, keep))
        build_thomas = Mock(return_value=(inverse_beta, gamma))

        bundle = assemble_pipeline_operators(
            method="transpose",
            tridiagonal="thomas",
            token="token",
            build_spike_adaptive=Mock(),
            build_spike=Mock(),
            build_rows=build_rows,
            build_thomas=build_thomas,
            build_pcr=Mock(),
        )

        self.assertEqual(bundle.solve_args, (a, inverse_beta, gamma, keep))
        self.assertEqual(bundle.row_operators, (a, b, c))
        self.assertEqual(a.waits, 1)
        residual_transpose = Mock(return_value="residual")
        pipeline = SimpleNamespace(residual_transpose=residual_transpose)
        self.assertEqual(bundle.bind_residual(pipeline)("rhs"), "residual")
        residual_transpose.assert_called_once_with(
            "rhs",
            a,
            b,
            c,
            a,
            inverse_beta,
            gamma,
            keep,
        )

    def test_spike_bundle_dispatches_to_spike_residual(self) -> None:
        factor = ReadyArray([1.0])
        interface = object()
        bundle = assemble_pipeline_operators(
            method="spike",
            tridiagonal="pcr",
            token="token",
            build_spike_adaptive=Mock(),
            build_spike=Mock(return_value=(factor, interface)),
            build_rows=Mock(),
            build_thomas=Mock(),
            build_pcr=Mock(),
        )
        residual_spike = Mock(return_value="residual")
        pipeline = SimpleNamespace(residual_spike=residual_spike)

        self.assertEqual(bundle.residual(pipeline, "rhs"), "residual")
        residual_spike.assert_called_once_with("rhs", factor, interface)
        self.assertEqual(factor.waits, 1)


class DistributedBenchmarkTests(unittest.TestCase):
    def test_helpers_preserve_original_reporting_units(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.75), 2.5)
        self.assertEqual(human_bytes(1536), "1.50 KiB")

    def test_runner_applies_warmup_samples_and_iterations(self) -> None:
        calls = 0

        def global_max(values):
            return ReadyArray(values)

        def mapped_function(value):
            nonlocal calls
            calls += 1
            return ReadyArray([value])

        runner = DistributedBenchmarkRunner(
            global_max=global_max,
            local_devices=1,
            process_index=1,
            warmup=1,
            samples=2,
            iterations=3,
        )
        output, first_call, times = runner.measure(mapped_function, 4.0)

        self.assertEqual(calls, 8)
        self.assertEqual(np.asarray(output)[0], 4.0)
        self.assertGreaterEqual(first_call, 0.0)
        self.assertEqual(len(times), 2)
        timing = runner.report("test", first_call, times)
        self.assertEqual(timing.median_seconds, np.median(times))
        self.assertEqual(timing.samples_seconds, tuple(times))


if __name__ == "__main__":
    unittest.main()
