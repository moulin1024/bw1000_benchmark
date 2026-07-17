from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from spectral_fd.compare import (
    BenchmarkSnapshot,
    compare_benchmarks,
    load_benchmark_records,
    main,
    render_markdown,
)


def make_snapshot(
    seconds: float,
    *,
    method: str = "transpose",
    source: str = "record",
) -> BenchmarkSnapshot:
    return BenchmarkSnapshot(
        source=source,
        configuration={
            "nx": 16,
            "ny": 16,
            "nz": 8,
            "lx": 1.0,
            "ly": 1.0,
            "lz": 1.0,
            "dtype": "float64",
            "method": method,
            "tridiagonal_solver": "pcr",
            "thomas_chunk": 1,
            "data_layout": "z-first",
            "pipeline_execution": "monolithic",
            "nyquist_filter": True,
            "spike_interface_collective": "alltoall",
            "spike_interface_solver": "selected-rows",
            "rhs": "random",
            "seed": 0,
        },
        runtime={"backend": "gpu", "global_devices": 2},
        full_median_seconds=seconds,
        component_medians={"rfft2": seconds / 4},
        relative_residual=1.0e-12,
        mms_error=None,
    )


def json_record(seconds: float) -> dict:
    snapshot = make_snapshot(seconds)
    return {
        "schema_version": 1,
        "configuration": snapshot.configuration,
        "runtime": snapshot.runtime,
        "timing": {
            "full": {"median_seconds": seconds},
            "components": {"rfft2": {"median_seconds": seconds / 4}},
        },
        "validation": {
            "relative_residual": snapshot.relative_residual,
            "mms_relative_max_error": None,
        },
    }


class BenchmarkComparisonTests(unittest.TestCase):
    def test_threshold_detects_runtime_regression(self) -> None:
        baseline = (make_snapshot(1.0, source="baseline"),)
        candidate = (make_snapshot(1.06, source="candidate"),)

        comparison = compare_benchmarks(
            baseline,
            candidate,
            max_regression_percent=5.0,
        )

        self.assertEqual(comparison["matched"], 1)
        self.assertEqual(comparison["regressions"], 1)
        self.assertAlmostEqual(comparison["comparisons"][0]["change_percent"], 6.0)
        markdown = render_markdown(comparison)
        self.assertIn("REGRESSION", markdown)
        self.assertIn("Component median changes", markdown)

    def test_missing_and_added_configurations_are_reported(self) -> None:
        baseline = (make_snapshot(1.0, method="transpose"),)
        candidate = (make_snapshot(1.0, method="spike"),)
        comparison = compare_benchmarks(
            baseline,
            candidate,
            max_regression_percent=5.0,
        )

        self.assertEqual(comparison["matched"], 0)
        self.assertEqual(len(comparison["missing"]), 1)
        self.assertEqual(len(comparison["added"]), 1)

    def test_json_and_csv_inputs_match_the_same_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "baseline.json"
            csv_path = root / "candidate.csv"
            json_path.write_text(json.dumps(json_record(1.0)), encoding="utf-8")
            snapshot = make_snapshot(0.95)
            row = {
                **{
                    f"config_{key}": value
                    for key, value in snapshot.configuration.items()
                },
                **{f"runtime_{key}": value for key, value in snapshot.runtime.items()},
                "full_median_seconds": snapshot.full_median_seconds,
                "component_rfft2_median_seconds": 0.2,
                "validation_relative_residual": snapshot.relative_residual,
                "validation_mms_relative_max_error": "",
            }
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)

            comparison = compare_benchmarks(
                load_benchmark_records(str(json_path)),
                load_benchmark_records(str(csv_path)),
                max_regression_percent=5.0,
            )

        self.assertEqual(comparison["matched"], 1)
        self.assertAlmostEqual(comparison["comparisons"][0]["change_percent"], -5.0)

    def test_cli_exit_code_and_output_files_follow_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            markdown = root / "comparison.md"
            output_json = root / "comparison.json"
            baseline.write_text(json.dumps(json_record(1.0)), encoding="utf-8")
            candidate.write_text(json.dumps(json_record(1.10)), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        str(baseline),
                        str(candidate),
                        "--max-regression-percent",
                        "5",
                        "--markdown",
                        str(markdown),
                        "--json",
                        str(output_json),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("REGRESSION", markdown.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(output_json.read_text())["regressions"], 1)


if __name__ == "__main__":
    unittest.main()
