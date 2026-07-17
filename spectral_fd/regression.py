"""Subprocess-isolated numerical regression matrix for Poisson solvers."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class NumericalRegressionCase:
    """One point in the numerical solver compatibility matrix."""

    index: int
    devices: int
    dtype: Literal["float32", "float64"]
    method: Literal["transpose", "spike", "spike-adaptive"]
    tridiagonal: Literal["pcr", "thomas"]
    layout: Literal["xyz", "z-first"]
    execution: Literal["monolithic", "staged"]

    @property
    def name(self) -> str:
        return (
            f"d{self.devices}-{self.dtype}-{self.method}-{self.tridiagonal}-"
            f"{self.layout}-{self.execution}"
        )


def full_numerical_matrix() -> tuple[NumericalRegressionCase, ...]:
    """Return the full 144-case Cartesian compatibility matrix."""
    values = itertools.product(
        (1, 2, 4),
        ("float32", "float64"),
        ("transpose", "spike", "spike-adaptive"),
        ("pcr", "thomas"),
        ("xyz", "z-first"),
        ("monolithic", "staged"),
    )
    return tuple(
        NumericalRegressionCase(
            index=index,
            devices=devices,
            dtype=dtype,
            method=method,
            tridiagonal=tridiagonal,
            layout=layout,
            execution=execution,
        )
        for index, (
            devices,
            dtype,
            method,
            tridiagonal,
            layout,
            execution,
        ) in enumerate(values)
    )


def smoke_numerical_matrix() -> tuple[NumericalRegressionCase, ...]:
    """Return a nine-case covering matrix spanning every parameter value."""
    selected = (
        (1, "float32", "transpose", "pcr", "xyz", "monolithic"),
        (1, "float64", "spike", "thomas", "z-first", "staged"),
        (1, "float32", "spike-adaptive", "pcr", "z-first", "monolithic"),
        (2, "float64", "transpose", "thomas", "z-first", "staged"),
        (2, "float32", "spike", "pcr", "xyz", "monolithic"),
        (2, "float64", "spike-adaptive", "thomas", "xyz", "staged"),
        (4, "float32", "transpose", "pcr", "z-first", "staged"),
        (4, "float64", "spike", "thomas", "xyz", "monolithic"),
        (4, "float32", "spike-adaptive", "pcr", "xyz", "staged"),
    )
    lookup = {
        (
            case.devices,
            case.dtype,
            case.method,
            case.tridiagonal,
            case.layout,
            case.execution,
        ): case
        for case in full_numerical_matrix()
    }
    return tuple(lookup[parameters] for parameters in selected)


def _run_worker(cases: list[NumericalRegressionCase]) -> int:
    import numpy as np

    from .config import Poisson3DConfig
    from .solver import Poisson3DSolver

    if not cases:
        print(json.dumps({"results": []}))
        return 0
    expected_devices = cases[0].devices
    results = []
    for case in cases:
        try:
            config = Poisson3DConfig(
                nx=8,
                ny=8,
                nz=8,
                dtype=case.dtype,
                method=case.method,
                tridiag=case.tridiagonal,
                data_layout=case.layout,
                pipeline_execution=case.execution,
                platform="cpu",
                spike_interface_collective="allgather",
            )
            solver = Poisson3DSolver(config)
            if solver.global_devices != expected_devices:
                raise RuntimeError(
                    f"expected {expected_devices} devices, got {solver.global_devices}"
                )
            import jax.numpy as jnp

            values = np.random.default_rng(case.index + 17).standard_normal(
                solver.local_input_shape
            )
            rhs = jnp.asarray(values, dtype=getattr(jnp, case.dtype))
            output = solver.solve(rhs, execution=case.execution)
            output.block_until_ready()
            residual = float(np.asarray(solver.residual(rhs))[0])
            tolerance = 2.0e-5 if case.dtype == "float32" else 2.0e-11
            passed = (
                bool(np.isfinite(np.asarray(output)).all()) and residual < tolerance
            )
            results.append(
                {
                    **asdict(case),
                    "name": case.name,
                    "residual": residual,
                    "tolerance": tolerance,
                    "passed": passed,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - matrix records every failure
            results.append(
                {
                    **asdict(case),
                    "name": case.name,
                    "residual": None,
                    "tolerance": None,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    print(json.dumps({"results": results}))
    return 0 if all(result["passed"] for result in results) else 1


def _worker_command(cases: list[NumericalRegressionCase]) -> list[str]:
    indices = ",".join(str(case.index) for case in cases)
    return [
        sys.executable,
        "-m",
        "spectral_fd.regression",
        "--worker",
        "--case-indices",
        indices,
    ]


def run_matrix(
    cases: tuple[NumericalRegressionCase, ...],
    *,
    output_path: str | None,
) -> int:
    """Run cases in dtype/device-isolated subprocess groups."""
    grouped: dict[tuple[int, str], list[NumericalRegressionCase]] = {}
    for case in cases:
        grouped.setdefault((case.devices, case.dtype), []).append(case)

    results: list[dict] = []
    for (devices, _dtype), group in sorted(grouped.items()):
        environment = os.environ.copy()
        environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
        process = subprocess.run(
            _worker_command(group),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError:
            print(process.stdout, end="", file=sys.stderr)
            print(process.stderr, end="", file=sys.stderr)
            return 1
        results.extend(payload["results"])
        if process.returncode and process.stderr:
            print(process.stderr, end="", file=sys.stderr)

    results.sort(key=lambda result: result["index"])
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        detail = (
            f"residual={result['residual']:.3e}"
            if result["residual"] is not None
            else result["error"]
        )
        print(f"{status:4} {result['name']} {detail}")
    summary = {
        "schema_version": 1,
        "case_count": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "results": results,
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"matrix: {summary['passed']}/{summary['case_count']} passed, "
        f"{summary['failed']} failed"
    )
    return 0 if summary["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the nine-case covering matrix instead of all 144 cases",
    )
    parser.add_argument("--devices", nargs="+", type=int, choices=(1, 2, 4))
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float64"))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", help="write detailed matrix results as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-indices", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    matrix = full_numerical_matrix()
    if args.worker:
        indices = {int(value) for value in args.case_indices.split(",") if value}
        return _run_worker([case for case in matrix if case.index in indices])

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard index must be in [0, shard count), with count > 0")
    selected = smoke_numerical_matrix() if args.quick else matrix
    if args.devices:
        selected = tuple(case for case in selected if case.devices in args.devices)
    if args.dtypes:
        selected = tuple(case for case in selected if case.dtype in args.dtypes)
    selected = tuple(
        case
        for position, case in enumerate(selected)
        if position % args.shard_count == args.shard_index
    )
    return run_matrix(selected, output_path=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
