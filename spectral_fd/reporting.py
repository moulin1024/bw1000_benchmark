"""Stable JSON and CSV serialization for Poisson benchmark results."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any


BENCHMARK_SCHEMA_VERSION = 1
COMPONENT_NAMES = ("rfft2", "z_to_y", "tridiag", "y_to_z", "spike", "irfft2")


def _timing_record(timing) -> dict[str, Any]:
    return {
        "first_call_seconds": timing.first_call_seconds,
        "first_call_note": timing.first_call_note,
        "samples_seconds": list(timing.samples_seconds),
        "median_seconds": timing.median_seconds,
        "best_seconds": timing.best_seconds,
        "mean_seconds": timing.mean_seconds,
        "p95_seconds": timing.p95_seconds,
    }


def _communication_record(args, context) -> dict[str, Any]:
    remote_payload = (
        context.spectral_bytes * (context.global_devices - 1) / context.global_devices
    )
    record: dict[str, Any] = {
        "remote_payload_per_exchange_bytes": remote_payload,
        "transpose_payload_per_solve_bytes": 2 * remote_payload,
        "interface_collective": None,
        "interface_collectives_per_solve": 0,
        "interface_payload_per_solve_bytes": 0,
        "interface_factor_storage_per_device_bytes": None,
    }
    if args.method == "transpose":
        return record

    endpoint_bytes = 2 * context.nxh * context.ny * 2 * context.real_itemsize
    if context.global_devices == 1:
        interface_payload = 0
        interface_collectives = 0
        interface_collective = "none"
    elif args.method == "spike-adaptive":
        box_bytes = 2 * context.adaptive_global_modes * 2 * context.real_itemsize
        interface_payload = endpoint_bytes + box_bytes * (context.global_devices - 1)
        interface_collectives = 3
        interface_collective = "ppermute+low-kh-allgather"
    elif args.spike_interface_collective == "allgather":
        interface_payload = endpoint_bytes * (context.global_devices - 1)
        interface_collectives = 1
        interface_collective = "allgather"
    else:
        interface_payload = (
            2 * endpoint_bytes * (context.global_devices - 1) / context.global_devices
        )
        interface_collectives = 2
        interface_collective = "alltoall"

    factor_storage = None
    if args.method == "spike":
        interface_modes = context.nxh * (
            context.ny
            if args.spike_interface_collective == "allgather"
            else context.interface_ny
        )
        if args.spike_interface_solver == "selected-rows":
            factors_per_mode = (
                4 * context.global_devices
                if args.spike_interface_collective == "allgather"
                else 4 * context.global_devices * context.global_devices
            )
        elif args.spike_interface_solver == "block-thomas":
            factors_per_mode = 6 * context.global_devices + 1
        else:
            factors_per_mode = context.reduced_size * context.reduced_size
        factor_storage = interface_modes * factors_per_mode * context.real_itemsize
    record.update(
        {
            "interface_collective": interface_collective,
            "interface_collectives_per_solve": interface_collectives,
            "interface_payload_per_solve_bytes": interface_payload,
            "interface_factor_storage_per_device_bytes": factor_storage,
        }
    )
    return record


def build_benchmark_record(
    args,
    *,
    context,
    result,
    relative_residual: float,
    mms_error: float | None,
) -> dict[str, Any]:
    """Build one versioned, JSON-serializable benchmark record."""
    component_seconds = result.component_seconds
    component_sum = sum(component_seconds.values()) if component_seconds else None
    full_seconds = result.full_seconds
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "configuration": {
            "nx": context.nx,
            "ny": context.ny,
            "nz": context.nz,
            "lx": args.lx,
            "ly": args.ly,
            "lz": args.lz,
            "dtype": args.dtype,
            "method": args.method,
            "tridiagonal_solver": args.tridiag,
            "thomas_chunk": args.thomas_chunk,
            "data_layout": args.data_layout,
            "pipeline_execution": args.pipeline_execution,
            "nyquist_filter": not args.no_nyquist_filter,
            "spike_interface_collective": args.spike_interface_collective,
            "spike_interface_solver": args.spike_interface_solver,
            "rhs": f"mms-{args.mms_kind}" if args.mms else "random",
            "seed": args.seed,
        },
        "runtime": {
            "python": platform.python_version(),
            "jax": context.jax_version,
            "backend": context.backend,
            "process_count": context.process_count,
            "process_index": context.process_index,
            "global_devices": context.global_devices,
            "local_devices": context.local_devices,
        },
        "sampling": {
            "warmup": args.warmup,
            "samples": args.samples,
            "iterations": args.iterations,
        },
        "timing": {
            "full": _timing_record(result.full_timing),
            "components": {
                name: _timing_record(timing)
                for name, timing in result.component_timings.items()
            },
        },
        "validation": {
            "relative_residual": relative_residual,
            "mms_relative_max_error": mms_error,
        },
        "performance": {
            "solves_per_second": 1.0 / full_seconds,
            "grid_points_per_second": context.nx
            * context.ny
            * context.nz
            / full_seconds,
            "component_sum_seconds": component_sum,
            "full_minus_component_sum_seconds": (
                full_seconds - component_sum if component_sum is not None else None
            ),
        },
        "storage": {
            "physical_bytes": context.physical_bytes,
            "spectral_bytes": context.spectral_bytes,
        },
        "communication": _communication_record(args, context),
    }


def _csv_record(record: dict[str, Any]) -> dict[str, Any]:
    configuration = record["configuration"]
    runtime = record["runtime"]
    sampling = record["sampling"]
    timing = record["timing"]
    validation = record["validation"]
    performance = record["performance"]
    storage = record["storage"]
    communication = record["communication"]
    flat = {
        "schema_version": record["schema_version"],
        **{f"config_{key}": value for key, value in configuration.items()},
        **{f"runtime_{key}": value for key, value in runtime.items()},
        **{f"sampling_{key}": value for key, value in sampling.items()},
        **{f"full_{key}": value for key, value in timing["full"].items()},
        **{f"validation_{key}": value for key, value in validation.items()},
        **{f"performance_{key}": value for key, value in performance.items()},
        **{f"storage_{key}": value for key, value in storage.items()},
        **{f"communication_{key}": value for key, value in communication.items()},
    }
    flat.pop("full_samples_seconds")
    for component in COMPONENT_NAMES:
        component_timing = timing["components"].get(component)
        flat[f"component_{component}_median_seconds"] = (
            component_timing["median_seconds"] if component_timing else None
        )
    return flat


def write_benchmark_reports(
    record: dict[str, Any],
    *,
    json_path: str | None,
    csv_path: str | None,
) -> None:
    """Write requested structured reports, creating parent directories."""
    if json_path:
        destination = Path(json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if csv_path:
        destination = Path(csv_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        flat = _csv_record(record)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(flat))
            writer.writeheader()
            writer.writerow(flat)
