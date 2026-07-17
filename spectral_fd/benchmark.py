"""Distributed timing and reporting for the Poisson pipeline."""

from __future__ import annotations

import gc
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .pipeline import PipelineOperatorBundle, PoissonPipelineStages


def percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def human_bytes(size: float) -> str:
    """Format a byte count using binary units."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class PoissonBenchmarkContext:
    """Derived runtime and storage facts used by benchmark reports."""

    jax_version: str
    backend: str
    process_count: int
    process_index: int
    global_devices: int
    local_devices: int
    nx: int
    ny: int
    nz: int
    nxh: int
    ny_local: int
    nz_local: int
    real_itemsize: int
    physical_bytes: int
    spectral_bytes: int
    z_first_layout: bool
    pcr_steps: int
    block_rows: int
    spike_local_description: str
    reduced_size: int
    interface_ny: int
    adaptive_global_modes: int

    @property
    def is_root(self) -> bool:
        return self.process_index == 0


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkResult:
    """Output and timing statistics from one benchmark suite."""

    output: Any
    full_seconds: float
    component_seconds: dict[str, float]


class DistributedBenchmarkRunner:
    """Measure mapped calls using a global maximum across all devices."""

    def __init__(
        self,
        *,
        global_max: Callable,
        local_devices: int,
        process_index: int,
        warmup: int,
        samples: int,
        iterations: int,
    ) -> None:
        self._global_max = global_max
        self._local_devices = local_devices
        self._process_index = process_index
        self._warmup = warmup
        self._samples = samples
        self._iterations = iterations
        self._token = np.zeros((local_devices,), dtype=np.float32)
        self._global_max(self._token).block_until_ready()

    def synchronize(self) -> None:
        self._global_max(self._token).block_until_ready()

    def _global_max_seconds(self, value: float) -> float:
        values = np.full((self._local_devices,), value, dtype=np.float32)
        result = self._global_max(values)
        result.block_until_ready()
        return float(np.asarray(result)[0])

    def measure(self, function: Callable, *arguments) -> tuple[Any, float, list[float]]:
        """Compile, warm up, and measure a mapped function."""
        self.synchronize()
        started = time.perf_counter()
        output = function(*arguments)
        output.block_until_ready()
        first_call = self._global_max_seconds(time.perf_counter() - started)

        for _ in range(self._warmup):
            output = function(*arguments)
            output.block_until_ready()

        times = []
        for _ in range(self._samples):
            self.synchronize()
            started = time.perf_counter()
            for _ in range(self._iterations):
                output = function(*arguments)
                output.block_until_ready()
            elapsed = (time.perf_counter() - started) / self._iterations
            times.append(self._global_max_seconds(elapsed))
        return output, first_call, times

    def report(
        self,
        label: str,
        first_call: float,
        times: list[float],
        first_call_note: str = "compile+run",
    ) -> float:
        """Print one timing table and return its median."""
        median_seconds = statistics.median(times)
        if self._process_index == 0:
            print(f"\n{label}")
            print(f"  first call ({first_call_note}): {first_call * 1e3:.3f} ms")
            print(f"  median                 : {median_seconds * 1e3:.3f} ms")
            print(f"  best                   : {min(times) * 1e3:.3f} ms")
            print(
                f"  mean / p95             : {statistics.mean(times) * 1e3:.3f} / "
                f"{percentile(times, 0.95) * 1e3:.3f} ms"
            )
        return median_seconds


def print_benchmark_configuration(args, context: PoissonBenchmarkContext) -> None:
    """Print the selected discretization, layout, and solver configuration."""
    if not context.is_root:
        return

    print("Distributed 3D pressure-Poisson benchmark (spectral xy + FD z)")
    print(f"Python                  : {sys.version.split()[0]}")
    print(f"JAX                     : {context.jax_version}")
    print(f"Backend                 : {context.backend}")
    print(f"Processes               : {context.process_count}")
    print(
        f"Global / local GPUs     : {context.global_devices} / {context.local_devices}"
    )
    print(f"Grid                    : {context.nx} x {context.ny} x {context.nz}")
    if context.z_first_layout:
        print("Data layout             : z-first")
        print(
            f"Physical z slab / GPU   : {context.nz_local} x {context.ny} x "
            f"{context.nx}  (z, y, x)"
        )
        print(
            f"Spectral z slab / GPU   : {context.nz_local} x {context.ny} x "
            f"{context.nxh}  (z, y, kx)"
        )
        print(
            f"Spectral y slab / GPU   : {context.nz} x {context.ny_local} x "
            f"{context.nxh}  (z, y, kx)"
        )
    else:
        print("Data layout             : xyz")
        print(
            f"Spectral layout         : {context.nxh} x {context.ny} x "
            f"{context.nz}  (Fortran, kx halved)"
        )
        print(
            f"Physical z slab / GPU   : {context.nx} x {context.ny} x "
            f"{context.nz_local}"
        )
        print(
            f"Spectral y slab / GPU   : {context.nxh} x {context.ny_local} x "
            f"{context.nz}"
        )
    print(f"Dtype                   : {args.dtype}")
    print(f"Nyquist filter          : {not args.no_nyquist_filter}")
    if args.method == "spike-adaptive":
        global_percentage = (
            100.0 * context.adaptive_global_modes / (context.nxh * context.ny)
        )
        solver_description = (
            f"spike-adaptive (blocks of {context.block_rows}, local "
            f"{context.spike_local_description}, PDD closure + exact "
            f"{context.reduced_size}-row solve on {context.adaptive_global_modes} "
            f"low-kh modes = {global_percentage:.2f}%)"
        )
    elif args.method == "spike":
        solver_description = (
            f"spike (blocks of {context.block_rows}, local "
            f"{context.spike_local_description}, {context.reduced_size}-row "
            f"interface, {args.spike_interface_collective}, "
            f"{args.spike_interface_solver})"
        )
    elif args.tridiag == "pcr":
        solver_description = f"pcr ({context.pcr_steps} steps)"
    else:
        solver_description = f"thomas (scan, chunk {args.thomas_chunk})"
    print(f"Method                  : {args.method}")
    print(f"Tridiagonal solver      : {solver_description}")
    print(f"Pipeline execution      : {args.pipeline_execution}")
    rhs_description = (
        f"manufactured solution (MMS, {args.mms_kind})" if args.mms else "random"
    )
    print(f"RHS                     : {rhs_description}")
    print(
        f"Physical / spectral data: {human_bytes(context.physical_bytes)} / "
        f"{human_bytes(context.spectral_bytes)}"
    )
    print(
        f"Warmup / timing         : {args.warmup} / {args.samples} x {args.iterations}"
    )


def run_pipeline_benchmark(
    args,
    *,
    context: PoissonBenchmarkContext,
    runner: DistributedBenchmarkRunner,
    pipeline: PoissonPipelineStages,
    operators: PipelineOperatorBundle,
    rhs,
) -> PipelineBenchmarkResult:
    """Measure requested component stages and the complete solve."""
    component_times: dict[str, float] = {}
    if not args.skip_components and operators.is_spike:
        rhs_hat, first_call, times = runner.measure(pipeline.forward_fft, rhs)
        component_times["rfft2"] = runner.report(
            "Forward rfft2 (local)", first_call, times
        )

        pressure_hat, first_call, times = runner.measure(
            pipeline.spike,
            rhs_hat,
            *operators.solve_args,
        )
        component_times["spike"] = runner.report(
            f"{args.method} vertical solve (local "
            f"{context.spike_local_description} + "
            f"{context.reduced_size}-row interface)",
            first_call,
            times,
        )
        del rhs_hat
        gc.collect()

        _, first_call, times = runner.measure(pipeline.inverse_fft, pressure_hat)
        component_times["irfft2"] = runner.report(
            "Inverse irfft2 (local)", first_call, times
        )
        del pressure_hat
        gc.collect()
    elif not args.skip_components:
        rhs_hat, first_call, times = runner.measure(pipeline.forward_fft, rhs)
        component_times["rfft2"] = runner.report(
            "Forward rfft2 (local)", first_call, times
        )

        rhs_hat_y, first_call, times = runner.measure(
            pipeline.transpose_z_to_y,
            rhs_hat,
        )
        component_times["z_to_y"] = runner.report(
            "z-slab -> y-slab all-to-all"
            if context.global_devices > 1
            else "z-slab -> y-slab (local reshape)",
            first_call,
            times,
        )
        del rhs_hat
        gc.collect()

        tridiagonal_label = (
            f"Tridiagonal solve (PCR, {context.pcr_steps} steps)"
            if args.tridiag == "pcr"
            else f"Tridiagonal solve (Thomas scans, chunk {args.thomas_chunk})"
        )
        pressure_hat_y, first_call, times = runner.measure(
            pipeline.tridiagonal,
            rhs_hat_y,
            *operators.solve_args,
        )
        component_times["tridiag"] = runner.report(
            tridiagonal_label,
            first_call,
            times,
        )
        del rhs_hat_y
        gc.collect()

        pressure_hat, first_call, times = runner.measure(
            pipeline.transpose_y_to_z,
            pressure_hat_y,
        )
        component_times["y_to_z"] = runner.report(
            "y-slab -> z-slab all-to-all"
            if context.global_devices > 1
            else "y-slab -> z-slab (local reshape)",
            first_call,
            times,
        )
        del pressure_hat_y
        gc.collect()

        _, first_call, times = runner.measure(pipeline.inverse_fft, pressure_hat)
        component_times["irfft2"] = runner.report(
            "Inverse irfft2 (local)", first_call, times
        )
        del pressure_hat
        gc.collect()

    full_solver = (
        pipeline.solve_staged
        if args.pipeline_execution == "staged"
        else pipeline.solve_monolithic
    )
    output, first_call, times = runner.measure(
        full_solver,
        rhs,
        *operators.solve_args,
    )
    first_call_note = (
        "stages cached"
        if args.pipeline_execution == "staged" and not args.skip_components
        else "compile+run"
    )
    full_seconds = runner.report(
        f"Complete pressure-Poisson solve ({args.pipeline_execution})",
        first_call,
        times,
        first_call_note,
    )
    return PipelineBenchmarkResult(
        output=output,
        full_seconds=full_seconds,
        component_seconds=component_times,
    )


def print_benchmark_summary(
    args,
    *,
    context: PoissonBenchmarkContext,
    result: PipelineBenchmarkResult,
    relative_residual: float,
    mms_error: float | None,
) -> None:
    """Print throughput, validation, communication, and storage estimates."""
    if not context.is_root:
        return

    full_seconds = result.full_seconds
    component_times = result.component_seconds
    remote_payload = (
        context.spectral_bytes * (context.global_devices - 1) / context.global_devices
    )
    print("\nSummary")
    print(f"  data layout                   : {args.data_layout}")
    print(f"  pipeline execution            : {args.pipeline_execution}")
    if args.tridiag == "thomas":
        print(f"  Thomas chunk                  : {args.thomas_chunk}")
    print(f"  solves/s                     : {1.0 / full_seconds:,.3f}")
    print(
        "  grid throughput              : "
        f"{context.nx * context.ny * context.nz / full_seconds / 1e9:,.3f} "
        "Gpoints/s"
    )
    print(f"  tridiagonal relative residual: {relative_residual:.3e}")
    if mms_error is not None:
        print(f"  MMS relative max error       : {mms_error:.3e}")
    if component_times:
        component_sum = sum(component_times.values())
        print(f"  sum of measured stages       : {component_sum * 1e3:.3f} ms")
        print(
            "  full minus isolated stages   : "
            f"{(full_seconds - component_sum) * 1e3:+.3f} ms"
        )
        if "z_to_y" in component_times:
            exchange_seconds = component_times["z_to_y"] + component_times["y_to_z"]
            print(
                "  exchange fraction of stages  : "
                f"{exchange_seconds / component_sum * 100:.2f}%"
            )
            if context.global_devices > 1:
                bandwidth = 2.0 * remote_payload / exchange_seconds / 2**30
                print(f"  remote payload per exchange  : {human_bytes(remote_payload)}")
                print(f"  aggregate one-way bandwidth  : {bandwidth:,.3f} GiB/s")
            else:
                print("  remote payload per exchange  : 0 B (single GPU)")
    if args.method == "transpose":
        return

    endpoint_bytes = 2 * context.nxh * context.ny * 2 * context.real_itemsize
    if context.global_devices == 1:
        interface_payload = 0
        interface_collectives = 0
        interface_description = "none (single GPU)"
    elif args.method == "spike-adaptive":
        box_bytes = 2 * context.adaptive_global_modes * 2 * context.real_itemsize
        interface_payload = endpoint_bytes + box_bytes * (context.global_devices - 1)
        interface_collectives = 3
        interface_description = "2 ppermute + low-kh all-gather"
    elif args.spike_interface_collective == "allgather":
        interface_payload = endpoint_bytes * (context.global_devices - 1)
        interface_collectives = 1
        interface_description = "allgather"
    else:
        interface_payload = (
            2 * endpoint_bytes * (context.global_devices - 1) / context.global_devices
        )
        interface_collectives = 2
        interface_description = "alltoall"
    print(f"  interface collective         : {interface_description}")
    if args.method == "spike":
        print(f"  interface solver             : {args.spike_interface_solver}")
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
        factor_bytes = interface_modes * factors_per_mode * context.real_itemsize
        print(f"  interface factor storage/GPU : {human_bytes(factor_bytes)}")
    print(f"  interface collectives/solve  : {interface_collectives}")
    print(f"  interface payload per solve  : {human_bytes(interface_payload)}")
    print(f"  transpose payload per solve  : {human_bytes(2 * remote_payload)}")
