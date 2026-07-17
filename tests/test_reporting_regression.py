from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from spectral_fd.benchmark import (
    OperationTiming,
    PipelineBenchmarkResult,
    PoissonBenchmarkContext,
)
from spectral_fd.regression import full_numerical_matrix, smoke_numerical_matrix
from spectral_fd.reporting import build_benchmark_record, write_benchmark_reports


class StructuredReportingTests(unittest.TestCase):
    def _fixture(self):
        args = SimpleNamespace(
            lx=1.0,
            ly=2.0,
            lz=3.0,
            dtype="float64",
            method="spike",
            tridiag="thomas",
            thomas_chunk=4,
            data_layout="z-first",
            pipeline_execution="staged",
            no_nyquist_filter=False,
            spike_interface_collective="allgather",
            spike_interface_solver="selected-rows",
            mms=True,
            mms_kind="broadband",
            seed=7,
            warmup=2,
            samples=3,
            iterations=4,
        )
        context = PoissonBenchmarkContext(
            jax_version="test-jax",
            backend="gpu",
            process_count=1,
            process_index=0,
            global_devices=2,
            local_devices=2,
            nx=16,
            ny=16,
            nz=8,
            nxh=9,
            ny_local=8,
            nz_local=4,
            real_itemsize=8,
            physical_bytes=16384,
            spectral_bytes=18432,
            z_first_layout=True,
            pcr_steps=4,
            block_rows=4,
            spike_local_description="thomas",
            reduced_size=5,
            interface_ny=8,
            adaptive_global_modes=12,
        )
        full = OperationTiming(
            first_call_seconds=0.2,
            samples_seconds=(0.01, 0.012, 0.011),
            median_seconds=0.011,
            best_seconds=0.01,
            mean_seconds=0.011,
            p95_seconds=0.0119,
            first_call_note="compile+run",
        )
        component = OperationTiming(
            first_call_seconds=0.1,
            samples_seconds=(0.002, 0.003, 0.0025),
            median_seconds=0.0025,
            best_seconds=0.002,
            mean_seconds=0.0025,
            p95_seconds=0.00295,
            first_call_note="compile+run",
        )
        result = PipelineBenchmarkResult(
            output=None,
            full_timing=full,
            component_timings={"spike": component},
        )
        return args, context, result

    def test_json_record_keeps_samples_and_derived_metrics(self) -> None:
        args, context, result = self._fixture()
        record = build_benchmark_record(
            args,
            context=context,
            result=result,
            relative_residual=1.0e-12,
            mms_error=2.0e-11,
        )

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(
            record["timing"]["full"]["samples_seconds"], [0.01, 0.012, 0.011]
        )
        self.assertAlmostEqual(record["performance"]["solves_per_second"], 1 / 0.011)
        self.assertEqual(record["communication"]["interface_collective"], "allgather")

    def test_json_and_csv_writers_create_parseable_files(self) -> None:
        args, context, result = self._fixture()
        record = build_benchmark_record(
            args,
            context=context,
            result=result,
            relative_residual=1.0e-12,
            mms_error=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "nested" / "result.json"
            csv_path = Path(directory) / "nested" / "result.csv"
            write_benchmark_reports(
                record,
                json_path=str(json_path),
                csv_path=str(csv_path),
            )

            self.assertEqual(json.loads(json_path.read_text())["schema_version"], 1)
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["config_method"], "spike")
            self.assertEqual(rows[0]["component_spike_median_seconds"], "0.0025")


class NumericalRegressionMatrixTests(unittest.TestCase):
    def test_full_matrix_is_the_expected_cartesian_product(self) -> None:
        matrix = full_numerical_matrix()
        self.assertEqual(len(matrix), 144)
        self.assertEqual({case.devices for case in matrix}, {1, 2, 4})
        self.assertEqual({case.dtype for case in matrix}, {"float32", "float64"})
        self.assertEqual(
            {case.method for case in matrix},
            {"transpose", "spike", "spike-adaptive"},
        )
        self.assertEqual({case.tridiagonal for case in matrix}, {"pcr", "thomas"})
        self.assertEqual({case.layout for case in matrix}, {"xyz", "z-first"})
        self.assertEqual(
            {case.execution for case in matrix},
            {"monolithic", "staged"},
        )

    def test_smoke_matrix_covers_every_parameter_value(self) -> None:
        smoke = smoke_numerical_matrix()
        self.assertEqual(len(smoke), 9)
        for field in (
            "devices",
            "dtype",
            "method",
            "tridiagonal",
            "layout",
            "execution",
        ):
            self.assertEqual(
                {getattr(case, field) for case in smoke},
                {getattr(case, field) for case in full_numerical_matrix()},
            )


if __name__ == "__main__":
    unittest.main()
