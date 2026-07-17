"""Command-line interface for the distributed Poisson benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import Poisson3DConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the package-owned benchmark argument parser."""
    defaults = Poisson3DConfig()
    parser = argparse.ArgumentParser(
        description="Benchmark a slab-decomposed spectral/FD 3D pressure-Poisson solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nx", type=int, default=defaults.nx, help="periodic points in x"
    )
    parser.add_argument(
        "--ny", type=int, default=defaults.ny, help="periodic points in y"
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=defaults.nz,
        help="vertical levels (wall-bounded)",
    )
    parser.add_argument(
        "--lx", type=float, default=defaults.lx, help="domain length in x"
    )
    parser.add_argument(
        "--ly", type=float, default=defaults.ly, help="domain length in y"
    )
    parser.add_argument("--lz", type=float, default=defaults.lz, help="domain height")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default=defaults.dtype,
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warm-up executions per operation"
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="timing samples per operation"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="synchronized calls per sample"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-nyquist-filter",
        action="store_true",
        help="keep Nyquist modes instead of zeroing them (WIRELES default filters)",
    )
    parser.add_argument(
        "--tridiag",
        choices=("pcr", "thomas"),
        default=defaults.tridiag,
        help=(
            "vertical solver: pcr runs ceil(log2(nz+1)) vectorized cyclic-"
            "reduction steps (few large kernels; stores 2*steps+1 factor "
            "arrays instead of thomas's 3); thomas runs 2(nz+1) sequential "
            "scan steps (tiny kernels, launch-latency bound on GPUs)"
        ),
    )
    parser.add_argument(
        "--thomas-chunk",
        type=int,
        default=defaults.thomas_chunk,
        help=(
            "Thomas runtime scan rows statically unrolled inside each outer "
            "scan iteration; 1 is the original row-at-a-time baseline, while "
            "8/16/32 reduce scan dispatches without adding full-field storage"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("transpose", "spike", "spike-adaptive"),
        default=defaults.method,
        help=(
            "transpose moves ky slabs through two full-field all-to-alls "
            "around the vertical solve; spike keeps the z-slab layout, solves "
            "each GPU's row block locally (PCR or Thomas, precomputed spike vectors) "
            "and couples blocks through a prefactored (2P+1)-row interface "
            "system, exchanging only 2 scalars per mode each way; "
            "spike-adaptive additionally truncates the interface system per "
            "mode: diagonally dominant modes (coupling decay exp(-2m asinh(kh "
            "dz/2)) below machine precision) use a neighbour-only PDD closure "
            "(ppermute), only the low-kh box keeps the exact global solve"
        ),
    )
    parser.add_argument(
        "--spike-interface-collective",
        choices=("alltoall", "allgather"),
        default=defaults.spike_interface_collective,
        help=(
            "SPIKE interface exchange: alltoall shards horizontal modes and "
            "uses two compact exchanges per solve; allgather replicates the "
            "small interface solve on every GPU and uses one exchange, avoiding "
            "the NCCL all-to-all path"
        ),
    )
    parser.add_argument(
        "--spike-interface-solver",
        choices=("selected-rows", "block-thomas", "dense"),
        default=defaults.spike_interface_solver,
        help=(
            "plain-SPIKE reduced interface solver: selected-rows uses the exact "
            "2x2 block factorization at setup to precompute only the response "
            "rows consumed by each GPU, then applies one parallel contraction; "
            "block-thomas applies the factors with runtime scans; dense stores "
            "and applies the full inverse as a reference path"
        ),
    )
    parser.add_argument(
        "--mms",
        action="store_true",
        help=(
            "replace the random RHS with a manufactured solution; the solver "
            "must reproduce it to roundoff"
        ),
    )
    parser.add_argument(
        "--mms-kind",
        choices=("modes", "broadband"),
        default="modes",
        help=(
            "modes = a few exact discrete eigenmodes; broadband = random-phase "
            "Kolmogorov-like spectrum over every resolved mode with the exact "
            "discrete-operator rhs (exercises the full spectrum, including the "
            "modes near the spike-adaptive truncation cutoff)"
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
        "--pipeline-execution",
        choices=("monolithic", "staged"),
        default=defaults.pipeline_execution,
        help=(
            "complete-solve execution: monolithic compiles FFT, vertical solve, "
            "and inverse FFT into one mapped executable; staged dispatches the "
            "separately compiled component executables asynchronously and "
            "synchronizes only after the inverse FFT"
        ),
    )
    parser.add_argument(
        "--data-layout",
        choices=("xyz", "z-first"),
        default=defaults.data_layout,
        help=(
            "local field layout: xyz stores physical (x,y,z) and spectral "
            "(kx,y,z); z-first stores physical (z,y,x) and spectral (z,y,kx), "
            "making x/y FFTs contiguous and the Thomas scan axis leading "
            "without any full-field transpose"
        ),
    )
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="measure only the complete solve",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark through the package CLI."""
    args = build_parser().parse_args(argv)

    # Keep JAX and the numerical implementation lazy so ``--help`` and parser
    # tests do not initialize a backend.
    from poisson3d_distributed import _run

    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
