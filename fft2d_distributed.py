#!/usr/bin/env python3
"""Pure-JAX distributed 2D complex FFT benchmark.

The global N x N array uses a 1D slab decomposition across P GPUs. The
default optimized layout keeps both local FFTs on the contiguous last axis:

  row slabs (N/P, N)
    -> local FFT on axis 1
    -> local transpose + lax.all_to_all
    -> spectral slabs (N/P, N)
    -> local FFT on axis 1

No MPI/FFT Python package is required. XLA uses the accelerator collective
backend (NCCL on CUDA, RCCL-compatible collectives on ROCm).

Batched workloads (--batch B fields of N x N) support two distributions:
  --decomp slab   every field's rows split across GPUs; one all-to-all whose
                  messages are B times larger than the single-field case
  --decomp batch  whole fields per GPU (B % P == 0): both FFT axes local,
                  zero communication, embarrassingly parallel

Exchange experiments (A/B against the defaults; profile decides):
  --transpose pairs  moves the local transpose data as (rows, cols, 2) reals in
                     two coalesced passes instead of one 16-byte-complex
                     transpose; wins if the backend's complex transpose kernel
                     is the bottleneck
  --chunks K         issues the exchange as K transpose/all-to-all/column-FFT
                     chains so a chunk's collective can overlap other chunks'
                     FFT and copy work (requires XLA async collectives /
                     latency-hiding scheduler to materialize). The spectral ky
                     rows become block-cyclic with stripes of N/(P*K) rows; the
                     inverse mirrors this, but downstream k-space bookkeeping
                     must account for it.

Single host, multiple visible GPUs:
  python jax_distributed_fft2d_benchmark.py --size 16384

Single-GPU strong-scaling baseline (same algorithm and spectral layout):
  python jax_distributed_fft2d_benchmark.py --size 16384 --pipeline compare

Slurm, one process per GPU (example):
  srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 \
    --cpu-bind=cores \
    python jax_distributed_fft2d_benchmark.py --distributed --size 32768
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from functools import partial

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a slab-decomposed pure-JAX distributed 2D FFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=8192, help="global square transform size N")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="number of independent N x N fields transformed per call",
    )
    parser.add_argument(
        "--decomp",
        choices=("slab", "batch"),
        default="slab",
        help=(
            "slab splits every field's rows across GPUs (all-to-all exchange); "
            "batch gives each GPU whole fields, so both FFT axes are local and "
            "no communication happens (requires batch %% GPUs == 0)"
        ),
    )
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--warmup", type=int, default=3, help="warm-up executions per operation")
    parser.add_argument("--samples", type=int, default=10, help="timing samples per operation")
    parser.add_argument("--iterations", type=int, default=5, help="synchronized calls per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layout",
        choices=("optimized", "strided"),
        default="optimized",
        help="optimized uses two contiguous FFTs; strided reproduces the original baseline",
    )
    parser.add_argument(
        "--transpose",
        choices=("native", "pairs"),
        default="native",
        help=(
            "local transpose strategy for the optimized layout; pairs moves the "
            "data as (rows, cols, 2) reals instead of 16-byte complex elements"
        ),
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help=(
            "split the optimized-layout exchange into this many overlappable "
            "transpose/all-to-all/FFT chains; > 1 makes the spectral ky "
            "distribution block-cyclic"
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=("fused", "staged", "compare"),
        default="fused",
        help=(
            "fused uses one pmap executable; staged puts an executable/layout boundary "
            "after the row FFT; compare times both and summarizes the faster one"
        ),
    )
    parser.add_argument(
        "--platform",
        choices=("cuda", "rocm"),
        help="require a specific JAX backend; otherwise use JAX_PLATFORMS/default",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="call jax.distributed.initialize(); use with one process per GPU",
    )
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="measure only the complete distributed FFT",
    )
    return parser


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def main() -> int:
    args = build_parser().parse_args()
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if args.warmup < 0 or args.samples <= 0 or args.iterations <= 0:
        raise SystemExit("warmup must be nonnegative; samples and iterations must be positive")
    if args.chunks < 1:
        raise SystemExit("--chunks must be positive")
    if args.batch < 1:
        raise SystemExit("--batch must be positive")
    if args.chunks > 1 and (
        args.layout != "optimized" or args.batch > 1 or args.decomp == "batch"
    ):
        raise SystemExit("--chunks > 1 requires --layout optimized, --batch 1, --decomp slab")
    if args.batch > 1 and args.decomp == "slab" and args.layout != "optimized":
        raise SystemExit("--batch > 1 with slab decomposition requires --layout optimized")
    if args.decomp == "batch":
        # Whole fields per GPU: both FFT axes are local and there is no
        # exchange stage to time in isolation.
        args.skip_components = True

    # These must be set before JAX initializes a backend.
    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform
    if args.dtype == "complex128":
        os.environ["JAX_ENABLE_X64"] = "true"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from jax import lax

    if args.distributed:
        # One Slurm task per GPU, with two visibility conventions:
        #  - per-task GRES binding: the task sees exactly one GPU (local
        #    device 0), but cgroup isolation can break cross-process RCCL IPC
        #    ("invalid device pointer") because peer devices are hidden;
        #  - job-level GRES (preferred): every task sees all node GPUs and
        #    must claim the one matching its node-local rank.
        visible = next(
            (
                os.environ[k]
                for k in (
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                )
                if os.environ.get(k)
            ),
            None,
        )
        if visible is not None and len(visible.split(",")) == 1:
            local_device = 0
        else:
            local_device = int(os.environ.get("SLURM_LOCALID", "0"))
        jax.distributed.initialize(local_device_ids=[local_device])

    global_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_count = jax.process_count()
    process_index = jax.process_index()

    if jax.default_backend() != "gpu":
        raise SystemExit(f"GPU backend required; got {jax.default_backend()!r}")
    if args.size % global_devices:
        raise SystemExit(
            f"global size {args.size} must be divisible by global GPU count {global_devices}"
        )
    if args.size % (args.chunks * global_devices):
        raise SystemExit(
            f"global size {args.size} must be divisible by chunks * GPU count = "
            f"{args.chunks * global_devices}"
        )
    if args.decomp == "batch" and args.batch % global_devices:
        raise SystemExit(
            f"--batch {args.batch} must be divisible by GPU count {global_devices} "
            "for --decomp batch"
        )

    n = args.size
    p = global_devices
    local_n = n // p
    batch = args.batch
    local_layers = batch // p if args.decomp == "batch" else batch
    complex_dtype = jnp.complex64 if args.dtype == "complex64" else jnp.complex128
    real_dtype = jnp.float32 if args.dtype == "complex64" else jnp.float64
    itemsize = 8 if args.dtype == "complex64" else 16
    global_bytes = batch * n * n * itemsize
    local_bytes = global_bytes // p
    axis_name = "fft_devices"

    # pmap spans local devices on each process; its collectives span every
    # participating device across all processes.
    mapped = partial(jax.pmap, axis_name=axis_name)

    def local_transpose(x):
        # Swap the two spatial (trailing) axes; leading batch axes untouched.
        if args.transpose == "pairs":
            # Move the data as 8-byte reals: a transposed write of the
            # (..., rows, cols, 2) view plus a coalesced recombine, instead of
            # one transpose over 16-byte complex elements.
            r = jnp.stack((x.real, x.imag), axis=-1)
            r = jnp.swapaxes(r, -3, -2)
            return lax.complex(r[..., 0], r[..., 1])
        return jnp.swapaxes(x, -1, -2)

    def optimized_exchange(x, fft_each):
        """(local x, global ky) row slab -> (ky, global x) spectral slab.

        With --chunks K > 1 the slab is exchanged as K independent
        transpose + all-to-all (+ column FFT) chains, so the scheduler can
        overlap one chunk's collective with other chunks' local work. Each
        chunk carries 1/K of the ky range, making the resulting ky
        distribution block-cyclic with stripes of local_n / K rows;
        optimized_inverse mirrors that ordering.
        """
        maybe_fft = (lambda v: jnp.fft.fft(v, axis=-1)) if fft_each else (lambda v: v)

        def exchange(piece):
            # (..., local x, ky block) -> (..., ky block, local x), then
            # exchange ky slabs and concatenate x slabs:
            # (..., ky stripe, global x). Axes are trailing-relative so a
            # leading batch dimension passes through untouched.
            piece = local_transpose(piece)
            if p > 1:
                piece = lax.all_to_all(
                    piece,
                    axis_name,
                    split_axis=piece.ndim - 2,
                    concat_axis=piece.ndim - 1,
                    tiled=True,
                )
            return maybe_fft(piece)

        if args.chunks == 1:
            return exchange(x)
        cols = n // args.chunks
        return jnp.concatenate(
            [exchange(x[:, c * cols : (c + 1) * cols]) for c in range(args.chunks)],
            axis=0,
        )

    def optimized_inverse(x):
        """Mirror of optimized_exchange: ifft columns, exchange back, ifft rows."""

        def exchange(piece):
            piece = jnp.fft.ifft(piece, axis=-1)
            if p > 1:
                piece = lax.all_to_all(
                    piece,
                    axis_name,
                    split_axis=piece.ndim - 1,
                    concat_axis=piece.ndim - 2,
                    tiled=True,
                )
            return local_transpose(piece)

        if args.chunks == 1:
            x = exchange(x)
        else:
            rows = local_n // args.chunks
            x = jnp.concatenate(
                [exchange(x[c * rows : (c + 1) * rows]) for c in range(args.chunks)],
                axis=1,
            )
        return jnp.fft.ifft(x, axis=-1)

    if args.decomp == "batch":
        local_shape = (local_layers, n, n)
    elif batch > 1:
        local_shape = (batch, local_n, n)
    else:
        local_shape = (local_n, n)

    @mapped
    def make_input(key):
        # One random block per GPU, created directly on-device. A single random
        # real array is used to limit initialization workspace.
        r = jax.random.uniform(key, local_shape, dtype=real_dtype)
        return (r + jnp.asarray(0.5j, dtype=complex_dtype) * r).astype(complex_dtype)

    @mapped
    def fft_rows(x):
        return jnp.fft.fft(x, axis=-1)

    @mapped
    def row_to_column_slabs(x):
        if args.layout == "optimized":
            return optimized_exchange(x, fft_each=False)
        # Original baseline: (local x, global ky) -> (global x, local ky).
        if p == 1:
            return x
        return lax.all_to_all(
            x,
            axis_name,
            split_axis=1,
            concat_axis=0,
            tiled=True,
        )

    @mapped
    def fft_columns(x):
        axis = -1 if args.layout == "optimized" else 0
        return jnp.fft.fft(x, axis=axis)

    @mapped
    def distributed_fft2d(x):
        if args.decomp == "batch":
            # Whole fields are local: one 2D FFT plan, no collective.
            return jnp.fft.fft2(x)
        x = jnp.fft.fft(x, axis=-1)
        if args.layout == "optimized":
            return optimized_exchange(x, fft_each=True)
        if p > 1:
            x = lax.all_to_all(
                x,
                axis_name,
                split_axis=1,
                concat_axis=0,
                tiled=True,
            )
        return jnp.fft.fft(x, axis=0)

    @mapped
    def exchange_and_fft_columns(x):
        """Second half of the staged pipeline.

        Keeping this in a separate executable makes the row-FFT output an
        explicit device-array boundary.  That can prevent a fused XLA layout
        assignment from adding a hidden copy around the collective/second FFT.
        Whether it wins is backend and size dependent, hence --pipeline compare.
        """
        if args.decomp == "batch":
            # Second local FFT axis; compare mode then contrasts one fused 2D
            # plan against two 1D passes with a layout boundary between them.
            return jnp.fft.fft(x, axis=-2)
        if args.layout == "optimized":
            return optimized_exchange(x, fft_each=True)
        if p > 1:
            x = lax.all_to_all(
                x,
                axis_name,
                split_axis=1,
                concat_axis=0,
                tiled=True,
            )
        return jnp.fft.fft(x, axis=0)

    def distributed_fft2d_staged(x):
        # Do not block between calls: the device dependency orders the two
        # executables, while benchmark() synchronizes only the final result.
        return exchange_and_fft_columns(fft_rows(x))

    @mapped
    def distributed_ifft2d(x):
        if args.decomp == "batch":
            return jnp.fft.ifft2(x)
        if args.layout == "optimized":
            return optimized_inverse(x)
        x = jnp.fft.ifft(x, axis=0)
        if p > 1:
            x = lax.all_to_all(
                x,
                axis_name,
                split_axis=0,
                concat_axis=1,
                tiled=True,
            )
        return jnp.fft.ifft(x, axis=1)

    @mapped
    def global_max_token(x):
        return lax.pmax(x, axis_name)

    @mapped
    def relative_max_error(a, b):
        local_error = jnp.max(jnp.abs(a - b))
        local_scale = jnp.max(jnp.abs(b))
        error = lax.pmax(local_error, axis_name)
        scale = lax.pmax(local_scale, axis_name)
        return error / jnp.maximum(scale, jnp.finfo(real_dtype).tiny)

    # Global device IDs are process-major in multi-process pmap programs.
    first_id = process_index * local_devices
    device_ids = np.arange(first_id, first_id + local_devices, dtype=np.uint32)
    base_key = jax.random.PRNGKey(args.seed)
    keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(jnp.asarray(device_ids))
    x = make_input(keys)
    x.block_until_ready()

    token = np.zeros((local_devices,), dtype=np.float32)
    global_max_token(token).block_until_ready()  # compile the timing collective

    def sync_all_processes() -> None:
        global_max_token(token).block_until_ready()

    def global_max_seconds(value: float) -> float:
        values = np.full((local_devices,), value, dtype=np.float32)
        result = global_max_token(values)
        result.block_until_ready()
        return float(np.asarray(result)[0])

    def benchmark(fn, arg):
        sync_all_processes()
        t0 = time.perf_counter()
        out = fn(arg)
        out.block_until_ready()
        compile_and_run = global_max_seconds(time.perf_counter() - t0)

        for _ in range(args.warmup):
            out = fn(arg)
            out.block_until_ready()

        times = []
        for _ in range(args.samples):
            sync_all_processes()
            t0 = time.perf_counter()
            for _ in range(args.iterations):
                # Synchronize each call to prevent a large queue of full-size
                # output buffers from exhausting device memory.
                out = fn(arg)
                out.block_until_ready()
            elapsed = (time.perf_counter() - t0) / args.iterations
            times.append(global_max_seconds(elapsed))
        return out, compile_and_run, times

    def report(label: str, compile_s: float, times: list[float]) -> float:
        median_s = statistics.median(times)
        if process_index == 0:
            print(f"\n{label}")
            print(f"  first call (compile+run): {compile_s * 1e3:.3f} ms")
            print(f"  median                 : {median_s * 1e3:.3f} ms")
            print(f"  best                   : {min(times) * 1e3:.3f} ms")
            print(
                f"  mean / p95             : {statistics.mean(times) * 1e3:.3f} / "
                f"{percentile(times, 0.95) * 1e3:.3f} ms"
            )
        return median_s

    if process_index == 0:
        print("Pure-JAX distributed 2D FFT benchmark")
        print(f"Python                  : {sys.version.split()[0]}")
        print(f"JAX                     : {jax.__version__}")
        print(f"Backend                 : {jax.default_backend()}")
        print(f"Processes               : {process_count}")
        print(f"Global / local GPUs     : {global_devices} / {local_devices}")
        print(f"Global shape            : {n} x {n}")
        if batch > 1:
            print(f"Fields per transform    : {batch}")
        print(f"Decomposition           : {args.decomp}")
        if args.decomp == "batch":
            print(f"Fields per GPU          : {local_layers} x {n} x {n}  (all-local, no exchange)")
        elif args.layout == "optimized":
            slab = f"{local_n} x {n}" if batch == 1 else f"{batch} x {local_n} x {n}"
            ky_layout = "local ky" if args.chunks == 1 else "block-cyclic ky"
            print(f"Input row slab / GPU    : {slab}")
            print(f"Output spectral slab/GPU: {slab}  ({ky_layout}, global kx)")
        else:
            print(f"Input row slab / GPU    : {local_n} x {n}")
            print(f"Output column slab / GPU: {n} x {local_n}")
        print(f"Dtype                   : {args.dtype}")
        print(f"Layout                  : {args.layout}")
        print(f"Local transpose         : {args.transpose}")
        print(f"Exchange chunks         : {args.chunks}")
        print(f"Pipeline                : {args.pipeline}")
        print(f"Global / local data     : {human_bytes(global_bytes)} / {human_bytes(local_bytes)}")
        print(f"Warmup / timing         : {args.warmup} / {args.samples} x {args.iterations}")

    component_times = {}
    if not args.skip_components:
        row_frequency, compile_s, times = benchmark(fft_rows, x)
        component_times["row_fft"] = report("Local row FFT", compile_s, times)

        exchange_label = (
            "Local transpose"
            if p == 1 and args.layout == "optimized"
            else "Identity exchange"
            if p == 1
            else "Local transpose + all-to-all"
            if args.layout == "optimized"
            else "All-to-all slab transpose"
        )
        if args.chunks > 1:
            exchange_label += f" ({args.chunks} chunks)"
        column_frequency, compile_s, times = benchmark(row_to_column_slabs, row_frequency)
        component_times["all_to_all"] = report(exchange_label, compile_s, times)

        del row_frequency
        gc.collect()

        stage_output, compile_s, times = benchmark(fft_columns, column_frequency)
        component_times["column_fft"] = report("Local column FFT", compile_s, times)

        del column_frequency, stage_output
        gc.collect()

    pipeline_results = {}
    if args.pipeline in ("fused", "compare"):
        spectrum_fused, compile_s, times = benchmark(distributed_fft2d, x)
        pipeline_results["fused"] = (
            spectrum_fused,
            report("Complete distributed 2D FFT (fused)", compile_s, times),
        )
    if args.pipeline in ("staged", "compare"):
        spectrum_staged, compile_s, times = benchmark(distributed_fft2d_staged, x)
        pipeline_results["staged"] = (
            spectrum_staged,
            report("Complete distributed 2D FFT (staged layout boundary)", compile_s, times),
        )

    selected_pipeline = min(pipeline_results, key=lambda name: pipeline_results[name][1])
    spectrum, full_s = pipeline_results[selected_pipeline]
    # compare mode temporarily owns two full-size spectra. Drop the unselected
    # result before validation to keep peak memory close to the fused mode.
    if selected_pipeline != "fused" and "spectrum_fused" in locals():
        del spectrum_fused
    if selected_pipeline != "staged" and "spectrum_staged" in locals():
        del spectrum_staged
    del pipeline_results
    gc.collect()

    # Validate forward followed by inverse, excluding validation from timings.
    reconstructed = distributed_ifft2d(spectrum)
    reconstructed.block_until_ready()
    error = relative_max_error(reconstructed, x)
    error.block_until_ready()
    relative_error = float(np.asarray(error)[0])

    if process_index == 0:
        estimated_flops = 5.0 * batch * n * n * math.log2(n * n)
        remote_payload = 0.0 if args.decomp == "batch" else global_bytes * (p - 1) / p
        print("\nSummary")
        print(f"  global transforms/s          : {1.0 / full_s:,.3f}")
        print(f"  estimated FFT rate           : {estimated_flops / full_s / 1e9:,.3f} GFLOP/s")
        print(f"  selected pipeline            : {selected_pipeline}")
        print(f"  round-trip relative max error: {relative_error:.3e}")
        if component_times:
            component_sum = sum(component_times.values())
            comm_fraction = component_times["all_to_all"] / component_sum
            print(f"  sum of measured stages       : {component_sum * 1e3:.3f} ms")
            print(f"  isolated exchange fraction   : {comm_fraction * 100:.2f}%")
            print(f"  full minus isolated stages   : {(full_s - component_sum) * 1e3:+.3f} ms")
            if p > 1:
                one_way_bw = remote_payload / component_times["all_to_all"] / 2**30
                print(f"  remote payload per transpose : {human_bytes(int(remote_payload))}")
                print(f"  aggregate one-way bandwidth  : {one_way_bw:,.3f} GiB/s")
                print(f"  aggregate send+recv bandwidth: {2 * one_way_bw:,.3f} GiB/s")
            else:
                print("  remote payload per transpose : 0 B (single GPU)")

    # Ensure process 0 does not exit while peers are still printing/finishing.
    sync_all_processes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())