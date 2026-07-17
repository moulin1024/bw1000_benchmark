"""Compare structured benchmark reports and detect performance regressions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIGURATION_MATCH_FIELDS = (
    "nx",
    "ny",
    "nz",
    "lx",
    "ly",
    "lz",
    "dtype",
    "method",
    "tridiagonal_solver",
    "thomas_chunk",
    "data_layout",
    "pipeline_execution",
    "nyquist_filter",
    "spike_interface_collective",
    "spike_interface_solver",
    "rhs",
    "seed",
)
RUNTIME_MATCH_FIELDS = ("backend", "global_devices")


@dataclass(frozen=True, slots=True)
class BenchmarkSnapshot:
    """Normalized subset of one JSON or CSV benchmark record."""

    source: str
    configuration: dict[str, Any]
    runtime: dict[str, Any]
    full_median_seconds: float
    component_medians: dict[str, float]
    relative_residual: float | None
    mms_error: float | None

    @property
    def match_key(self) -> tuple[Any, ...]:
        return tuple(
            self.configuration.get(field) for field in CONFIGURATION_MATCH_FIELDS
        ) + tuple(self.runtime.get(field) for field in RUNTIME_MATCH_FIELDS)

    @property
    def label(self) -> str:
        config = self.configuration
        return (
            f"{config.get('nx')}x{config.get('ny')}x{config.get('nz')} "
            f"{config.get('dtype')} {config.get('method')}/"
            f"{config.get('tridiagonal_solver')} {config.get('data_layout')} "
            f"{config.get('pipeline_execution')} d{self.runtime.get('global_devices')}"
        )


def _coerce_csv_value(value: str) -> Any:
    if value == "" or value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _snapshot_from_json(record: dict[str, Any], source: str) -> BenchmarkSnapshot:
    try:
        timing = record["timing"]
        validation = record["validation"]
        return BenchmarkSnapshot(
            source=source,
            configuration=dict(record["configuration"]),
            runtime=dict(record["runtime"]),
            full_median_seconds=float(timing["full"]["median_seconds"]),
            component_medians={
                name: float(values["median_seconds"])
                for name, values in timing.get("components", {}).items()
            },
            relative_residual=validation.get("relative_residual"),
            mms_error=validation.get("mms_relative_max_error"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid benchmark JSON record in {source}: {exc}") from exc


def _snapshot_from_csv(row: dict[str, str], source: str) -> BenchmarkSnapshot:
    values = {key: _coerce_csv_value(value) for key, value in row.items()}
    configuration = {
        key.removeprefix("config_"): value
        for key, value in values.items()
        if key.startswith("config_")
    }
    runtime = {
        key.removeprefix("runtime_"): value
        for key, value in values.items()
        if key.startswith("runtime_")
    }
    components = {
        key.removeprefix("component_").removesuffix("_median_seconds"): float(value)
        for key, value in values.items()
        if key.startswith("component_")
        and key.endswith("_median_seconds")
        and value is not None
    }
    try:
        return BenchmarkSnapshot(
            source=source,
            configuration=configuration,
            runtime=runtime,
            full_median_seconds=float(values["full_median_seconds"]),
            component_medians=components,
            relative_residual=values.get("validation_relative_residual"),
            mms_error=values.get("validation_mms_relative_max_error"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid benchmark CSV row in {source}: {exc}") from exc


def _load_json(path: Path) -> list[BenchmarkSnapshot]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "configuration" in payload:
        records = [payload]
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"{path} does not contain benchmark record(s)")
    return [
        _snapshot_from_json(record, f"{path}#{index}")
        for index, record in enumerate(records)
    ]


def _load_csv(path: Path) -> list[BenchmarkSnapshot]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        _snapshot_from_csv(row, f"{path}#{index}") for index, row in enumerate(rows)
    ]


def load_benchmark_records(path: str) -> tuple[BenchmarkSnapshot, ...]:
    """Load a JSON/CSV file or recursively load all reports in a directory."""
    source = Path(path)
    if source.is_dir():
        files = sorted(
            file
            for file in source.rglob("*")
            if file.is_file() and file.suffix.lower() in (".json", ".csv")
        )
    elif source.is_file():
        files = [source]
    else:
        raise ValueError(f"benchmark report path does not exist: {source}")
    records: list[BenchmarkSnapshot] = []
    for file in files:
        if file.suffix.lower() == ".json":
            records.extend(_load_json(file))
        else:
            records.extend(_load_csv(file))
    if not records:
        raise ValueError(f"no benchmark records found in {source}")
    return tuple(records)


def _index_records(
    records: tuple[BenchmarkSnapshot, ...],
    *,
    side: str,
) -> dict[tuple[Any, ...], BenchmarkSnapshot]:
    indexed = {}
    for record in records:
        if record.match_key in indexed:
            first = indexed[record.match_key]
            raise ValueError(
                f"duplicate {side} configuration in {first.source} and {record.source}"
            )
        indexed[record.match_key] = record
    return indexed


def compare_benchmarks(
    baseline: tuple[BenchmarkSnapshot, ...],
    candidate: tuple[BenchmarkSnapshot, ...],
    *,
    max_regression_percent: float,
) -> dict[str, Any]:
    """Match configurations and compare full/component median runtimes."""
    baseline_by_key = _index_records(baseline, side="baseline")
    candidate_by_key = _index_records(candidate, side="candidate")
    matched_keys = sorted(
        baseline_by_key.keys() & candidate_by_key.keys(),
        key=lambda key: tuple(str(value) for value in key),
    )
    comparisons = []
    for key in matched_keys:
        before = baseline_by_key[key]
        after = candidate_by_key[key]
        change_percent = (
            after.full_median_seconds / before.full_median_seconds - 1.0
        ) * 100.0
        component_changes = {
            name: (after.component_medians[name] / before.component_medians[name] - 1.0)
            * 100.0
            for name in sorted(
                before.component_medians.keys() & after.component_medians.keys()
            )
        }
        comparisons.append(
            {
                "label": before.label,
                "match_key": list(key),
                "baseline_source": before.source,
                "candidate_source": after.source,
                "baseline_seconds": before.full_median_seconds,
                "candidate_seconds": after.full_median_seconds,
                "change_percent": change_percent,
                "regression": change_percent > max_regression_percent,
                "component_change_percent": component_changes,
                "baseline_relative_residual": before.relative_residual,
                "candidate_relative_residual": after.relative_residual,
                "baseline_mms_error": before.mms_error,
                "candidate_mms_error": after.mms_error,
            }
        )
    missing = [
        baseline_by_key[key].label
        for key in baseline_by_key.keys() - candidate_by_key.keys()
    ]
    added = [
        candidate_by_key[key].label
        for key in candidate_by_key.keys() - baseline_by_key.keys()
    ]
    regressions = sum(item["regression"] for item in comparisons)
    return {
        "schema_version": 1,
        "max_regression_percent": max_regression_percent,
        "matched": len(comparisons),
        "regressions": regressions,
        "missing": sorted(missing),
        "added": sorted(added),
        "comparisons": comparisons,
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    """Render a compact Markdown performance comparison."""
    lines = [
        "# Benchmark comparison",
        "",
        f"Regression threshold: **{comparison['max_regression_percent']:.2f}%**",
        "",
        "| Configuration | Baseline (ms) | Candidate (ms) | Change | Status |",
        "|---|---:|---:|---:|:---:|",
    ]
    for item in comparison["comparisons"]:
        status = "REGRESSION" if item["regression"] else "OK"
        lines.append(
            f"| {item['label']} | {item['baseline_seconds'] * 1e3:.3f} | "
            f"{item['candidate_seconds'] * 1e3:.3f} | "
            f"{item['change_percent']:+.2f}% | {status} |"
        )
    if not comparison["comparisons"]:
        lines.append("| _No matching configurations_ | — | — | — | — |")
    component_rows = [
        (item["label"], component, change)
        for item in comparison["comparisons"]
        for component, change in item["component_change_percent"].items()
    ]
    if component_rows:
        lines.extend(
            (
                "",
                "## Component median changes",
                "",
                "| Configuration | Component | Change |",
                "|---|---|---:|",
            )
        )
        lines.extend(
            f"| {label} | {component} | {change:+.2f}% |"
            for label, component, change in component_rows
        )
    if comparison["missing"]:
        lines.extend(("", "## Missing candidate configurations", ""))
        lines.extend(f"- {label}" for label in comparison["missing"])
    if comparison["added"]:
        lines.extend(("", "## New candidate configurations", ""))
        lines.extend(f"- {label}" for label in comparison["added"])
    lines.extend(
        (
            "",
            f"Matched: {comparison['matched']}; regressions: "
            f"{comparison['regressions']}; missing: {len(comparison['missing'])}.",
        )
    )
    return "\n".join(lines) + "\n"


def _write_text(path: str, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="baseline JSON/CSV file or directory")
    parser.add_argument("candidate", help="candidate JSON/CSV file or directory")
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=5.0,
        help="fail when full median runtime increases by more than this percentage",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="do not fail when a baseline configuration is absent from the candidate",
    )
    parser.add_argument("--markdown", help="write the Markdown summary to this path")
    parser.add_argument("--json", dest="json_output", help="write comparison JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_regression_percent < 0:
        raise SystemExit("--max-regression-percent must be nonnegative")
    try:
        baseline = load_benchmark_records(args.baseline)
        candidate = load_benchmark_records(args.candidate)
        comparison = compare_benchmarks(
            baseline,
            candidate,
            max_regression_percent=args.max_regression_percent,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    markdown = render_markdown(comparison)
    print(markdown, end="")
    if args.markdown:
        _write_text(args.markdown, markdown)
    if args.json_output:
        _write_text(
            args.json_output,
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        )
    failed = comparison["regressions"] > 0 or (
        bool(comparison["missing"]) and not args.allow_missing
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
