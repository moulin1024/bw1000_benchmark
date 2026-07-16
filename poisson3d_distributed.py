#!/usr/bin/env python3
"""Distributed 3D pressure-Poisson benchmark (channel / ABL configuration).

Horizontal directions are periodic and spectrally discretized (rfft2);
the vertical direction is wall-bounded and discretized with second-order
finite differences, giving one tridiagonal system per horizontal mode.
The discretization mirrors WIRELES_GPU (wireles_jax/pressure_sharded.py):
Neumann bottom row, rigid-lid top row, k^2 = 0 mode pinned at the bottom,
optional Nyquist mode filtering, and precomputed PCR or Thomas factors so
the timed solve applies only the selected tridiagonal solver. Both the
original xyz/Fortran-compatible spectral layout and a native z-first layout
are available.

Pipeline (z-slab base layout, nz/P levels per GPU):

  rhs (nx, ny, nz/P)   physical z slab
    -> rfft2 (local)              (nx/2+1, ny, nz/P)
    -> all-to-all                 (nx/2+1, ny/P, nz)   y slab
    -> tridiagonal solve per mode (nx/2+1, ny/P, nz)
    -> all-to-all                 (nx/2+1, ny, nz/P)   z slab
    -> irfft2 (local)             (nx, ny, nz/P)

The transpose method communicates two full spectral fields per solve. The
SPIKE method exchanges only interface scalars, using either two compact
all-to-alls or one all-gather followed by a replicated interface solve.
Validation: relative residual of the discrete tridiagonal equations
(A p_col - rhs_col) per mode, evaluated on unfiltered modes.

Single host, multiple visible GPUs:
  python poisson3d_distributed.py --nx 1024 --ny 1024 --nz 128

Slurm, one process per GPU:
  srun --ntasks=8 --cpu-bind=cores python poisson3d_distributed.py --distributed ...
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
        description="Benchmark a slab-decomposed spectral/FD 3D pressure-Poisson solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=1024, help="periodic points in x")
    parser.add_argument("--ny", type=int, default=1024, help="periodic points in y")
    parser.add_argument("--nz", type=int, default=128, help="vertical levels (wall-bounded)")
    parser.add_argument("--lx", type=float, default=1.0, help="domain length in x")
    parser.add_argument("--ly", type=float, default=1.0, help="domain length in y")
    parser.add_argument("--lz", type=float, default=1.0, help="domain height")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=3, help="warm-up executions per operation")
    parser.add_argument("--samples", type=int, default=10, help="timing samples per operation")
    parser.add_argument("--iterations", type=int, default=5, help="synchronized calls per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-nyquist-filter",
        action="store_true",
        help="keep Nyquist modes instead of zeroing them (WIRELES default filters)",
    )
    parser.add_argument(
        "--tridiag",
        choices=("pcr", "thomas"),
        default="pcr",
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
        default=1,
        help=(
            "Thomas runtime scan rows statically unrolled inside each outer "
            "scan iteration; 1 is the original row-at-a-time baseline, while "
            "8/16/32 reduce scan dispatches without adding full-field storage"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("transpose", "spike", "spike-adaptive"),
        default="transpose",
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
        default="alltoall",
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
        default="selected-rows",
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
        default="monolithic",
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
        default="xyz",
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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def human_bytes(n: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


class _SolverEngine:
    """Internal bridge between the legacy implementation and public API."""

    def __init__(
        self,
        *,
        config,
        global_devices,
        local_devices,
        process_count,
        process_index,
        local_input_shape,
        global_input_shape,
        solve_monolithic,
        solve_staged,
        residual,
        pipeline_ops,
    ):
        self.config = config
        self.global_devices = global_devices
        self.local_devices = local_devices
        self.process_count = process_count
        self.process_index = process_index
        self.local_input_shape = local_input_shape
        self.global_input_shape = global_input_shape
        self._solve_monolithic = solve_monolithic
        self._solve_staged = solve_staged
        self._residual = residual
        self._pipeline_ops = pipeline_ops

    def _validate_rhs_shape(self, rhs) -> None:
        try:
            shape = tuple(rhs.shape)
            dtype = np.dtype(rhs.dtype)
        except AttributeError as exc:
            raise TypeError("rhs must be an array with shape and dtype") from exc
        if shape != self.local_input_shape:
            raise ValueError(
                f"rhs shape {shape} does not match the expected local shape "
                f"{self.local_input_shape}"
            )
        expected_dtype = np.dtype(self.config.dtype)
        if dtype != expected_dtype:
            raise TypeError(
                f"rhs dtype {dtype} does not match configured dtype "
                f"{expected_dtype}"
            )

    def solve(self, rhs, *, execution=None):
        self._validate_rhs_shape(rhs)
        selected_execution = execution or self.config.pipeline_execution
        if selected_execution == "staged":
            solve = self._solve_staged
        elif selected_execution == "monolithic":
            solve = self._solve_monolithic
        else:
            raise ValueError("execution must be 'monolithic' or 'staged'")
        return solve(rhs, *self._pipeline_ops)

    def residual(self, rhs):
        self._validate_rhs_shape(rhs)
        return self._residual(rhs)


def _run(args, *, library_mode: bool = False):
    def fail(message: str, *, runtime: bool = False):
        if library_mode:
            error = RuntimeError if runtime else ValueError
            raise error(message)
        raise SystemExit(message)

    if min(args.nx, args.ny, args.nz) <= 0:
        fail("--nx/--ny/--nz must be positive")
    if args.nx % 2 or args.ny % 2:
        fail("--nx and --ny must be even")
    if (
        not library_mode
        and (args.warmup < 0 or args.samples <= 0 or args.iterations <= 0)
    ):
        fail("warmup must be nonnegative; samples and iterations must be positive")
    if args.thomas_chunk <= 0:
        fail("--thomas-chunk must be positive")

    # Manufactured solution: sum of separable terms
    #   A * trig_x(2 pi mx x / lx) * trig_y(2 pi my y / ly) * cos(mz pi zeta)
    # with zeta = (j - 1/2) / (nz - 1) at pressure node j. The vertical
    # profiles are exact discrete eigenfunctions of the FD stencil (symmetric
    # about both walls, so the one-sided Neumann rows hold exactly), hence the
    # discrete solver must return the manufactured field to roundoff.
    mms_terms = (
        (1.00, 2, 3, 1, "cos", "cos"),
        (0.70, 1, 0, 2, "sin", "cos"),
        (0.50, 0, 4, 5, "cos", "sin"),
        (0.40, 1, 1, 0, "cos", "cos"),
        (0.30, 3, 2, 3, "sin", "sin"),
    )
    if args.mms and args.mms_kind == "modes":
        for _, mx, my, mz, _, _ in mms_terms:
            if mx >= args.nx // 2 or my >= args.ny // 2:
                fail("--mms horizontal modes must stay below Nyquist")
            if mz > args.nz - 2:
                fail("--mms requires nz >= 8")

    # These must be set before JAX initializes a backend.
    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform
    if args.dtype == "float64":
        os.environ["JAX_ENABLE_X64"] = "true"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from jax import lax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    if args.distributed and not jax.distributed.is_initialized():
        # One Slurm task per GPU, same conventions as fft2d_distributed.py:
        # per-task GRES binding shows one local device; job-level GRES shows
        # all node devices and the task claims its node-local rank.
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

    if not library_mode and jax.default_backend() != "gpu":
        fail(
            f"GPU backend required; got {jax.default_backend()!r}",
            runtime=True,
        )

    nx, ny, nz = args.nx, args.ny, args.nz
    p = global_devices
    if ny % p:
        fail(f"ny={ny} must be divisible by GPU count {p}")
    if nz % p:
        fail(f"nz={nz} must be divisible by GPU count {p}")

    is_spike = args.method != "transpose"
    if is_spike and nz // p < 2:
        fail(f"SPIKE methods need nz/GPUs >= 2; got {nz // p}")

    nxh = nx // 2 + 1
    ny_local = ny // p
    nz_local = nz // p
    z_first_layout = args.data_layout == "z-first"
    real_dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    real_itemsize = 4 if args.dtype == "float32" else 8
    dz = args.lz / nz
    dz2 = dz * dz

    physical_bytes = nx * ny * nz * real_itemsize
    spectral_bytes = nxh * ny * nz * 2 * real_itemsize
    axis_name = "poisson_devices"
    mapped = partial(jax.pmap, axis_name=axis_name)

    # Horizontal wavenumbers, Fortran spectral layout (kx halved axis), with
    # Nyquist wavenumbers zeroed exactly as in WIRELES.
    np_real = np.float32 if args.dtype == "float32" else np.float64
    kx_np = (2.0 * np.pi * np.fft.rfftfreq(nx, d=args.lx / nx)).astype(np_real)
    ky_np = (2.0 * np.pi * np.fft.fftfreq(ny, d=args.ly / ny)).astype(np_real)
    kx_np[-1] = 0.0
    ky_np[ny // 2] = 0.0
    keep_np = np.ones((nxh, ny), dtype=bool)
    if not args.no_nyquist_filter:
        keep_np[-1, :] = False
        keep_np[:, ny // 2] = False
    kx_all = jnp.asarray(kx_np)
    ky_all = jnp.asarray(ky_np)
    keep_all = jnp.asarray(keep_np.astype(np_real))
    eps128 = float(np.finfo(np_real).eps) * 128.0

    def _move_z_first(q):
        return q if z_first_layout else jnp.moveaxis(q, -1, 0)

    def _move_z_last(q):
        return q if z_first_layout else jnp.moveaxis(q, 0, -1)

    def _z_first(q):
        return q[0] if z_first_layout else q[..., 0]

    def _z_last(q):
        return q[-1] if z_first_layout else q[..., -1]

    def _z_without_first(q):
        return q[1:] if z_first_layout else q[..., 1:]

    def _z_without_last(q):
        return q[:-1] if z_first_layout else q[..., :-1]

    def _prepend_z(first, tail_z):
        if z_first_layout:
            return jnp.concatenate((first[None, ...], tail_z), axis=0)
        return jnp.concatenate((first[..., None], _move_z_last(tail_z)), axis=-1)

    def _append_z(prefix_z, last):
        if z_first_layout:
            return jnp.concatenate((prefix_z, last[None, ...]), axis=0)
        return jnp.concatenate((_move_z_last(prefix_z), last[..., None]), axis=-1)

    def _z_broadcast(values):
        return values[:, None, None] if z_first_layout else values[None, None, :]

    def _mode_to_interface(values):
        """Local horizontal plane -> canonical interface order (kx, ky)."""
        return jnp.swapaxes(values, -1, -2) if z_first_layout else values

    def _interface_to_mode(values):
        """Canonical interface order (kx, ky) -> local horizontal plane."""
        return jnp.swapaxes(values, -1, -2) if z_first_layout else values

    def _mode_broadcast(values):
        local = _interface_to_mode(values)
        return local[None, ...] if z_first_layout else local[..., None]

    def _spectral_keep():
        return _mode_broadcast(keep_all)

    def _forward_fft_local(rhs):
        axes = (-2, -1) if z_first_layout else (-2, -3)
        return jnp.fft.rfftn(rhs, axes=axes)

    def _inverse_fft_local(h):
        axes = (-2, -1) if z_first_layout else (-2, -3)
        return jnp.fft.irfftn(h, s=(ny, nx), axes=axes)

    @mapped
    def build_operators(_token):
        """Per-device tridiagonal rows for the local ky slab.

        Rows follow wireles_jax._pressure_tridiag_fortran_layout: Neumann
        bottom (p1 - p0 = 0), rigid-lid top (p_nz - p_{nz-1} = 0), interior
        (p_{j-1} - 2 p_j + p_{j+1}) / dz^2 - k^2 p_j, and the k^2 = 0 mode
        pinned via p0 = 0.
        """
        j = lax.axis_index(axis_name)
        ky_l = lax.dynamic_slice_in_dim(ky_all, j * ny_local, ny_local)
        keep_l = lax.dynamic_slice_in_dim(keep_all, j * ny_local, ny_local, axis=1)
        k2 = kx_all[:, None] * kx_all[:, None] + ky_l[None, :] * ky_l[None, :]
        zero_k2 = jnp.abs(k2) < eps128

        one = jnp.asarray(1.0, real_dtype)
        if z_first_layout:
            shape = (nz + 1, ny_local, nxh)
            k2_local = jnp.swapaxes(k2, 0, 1)
            zero_local = jnp.swapaxes(zero_k2, 0, 1)
            a = jnp.zeros(shape, real_dtype)
            a = a.at[1:nz].set(1.0 / dz2)
            a = a.at[nz].set(-1.0)
            b = jnp.zeros(shape, real_dtype)
            b = b.at[0].set(jnp.where(zero_local, one, -one))
            b = b.at[1:nz].set(-k2_local[None, ...] - 2.0 / dz2)
            b = b.at[nz].set(1.0)
            c = jnp.zeros(shape, real_dtype)
            c = c.at[0].set(jnp.where(zero_local, 0.0 * one, one))
            c = c.at[1:nz].set(1.0 / dz2)
            return a, b, c, jnp.swapaxes(keep_l, 0, 1)[None, ...]

        shape = (nxh, ny_local, nz + 1)
        a = jnp.zeros(shape, real_dtype)
        a = a.at[..., 1:nz].set(1.0 / dz2)
        a = a.at[..., nz].set(-1.0)
        b = jnp.zeros(shape, real_dtype)
        b = b.at[..., 0].set(jnp.where(zero_k2, one, -one))
        b = b.at[..., 1:nz].set(-k2[..., None] - 2.0 / dz2)
        b = b.at[..., nz].set(1.0)
        c = jnp.zeros(shape, real_dtype)
        c = c.at[..., 0].set(jnp.where(zero_k2, 0.0 * one, one))
        c = c.at[..., 1:nz].set(1.0 / dz2)
        return a, b, c, keep_l[..., None]

    def thomas_factor_arrays(a, b, c):
        """Precompute the Thomas (LU) sweep factors, as in wireles_jax."""

        def factor_step(carry, rows):
            inv_bet_prev, c_prev = carry
            a_j, b_j, c_j = rows
            gam_j = c_prev * inv_bet_prev
            inv_bet_j = 1.0 / (b_j - a_j * gam_j)
            return (inv_bet_j, c_j), (inv_bet_j, gam_j)

        inv_bet0 = 1.0 / _z_first(b)
        _, (inv_bet_tail, gam_tail) = lax.scan(
            factor_step,
            (inv_bet0, _z_first(c)),
            (
                _move_z_first(_z_without_first(a)),
                _move_z_first(_z_without_first(b)),
                _move_z_first(_z_without_first(c)),
            ),
        )
        inv_bet = _prepend_z(inv_bet0, inv_bet_tail)
        gam = _prepend_z(jnp.zeros_like(inv_bet0), gam_tail)
        return inv_bet, gam

    @mapped
    def build_thomas_factors(a, b, c):
        return thomas_factor_arrays(a, b, c)

    # Parallel cyclic reduction. Each step with stride s eliminates the
    # couplings a_i, c_i against rows i -+ s; after ceil(log2(nz+1)) steps the
    # systems are diagonal. The row reduction is RHS-independent, so the
    # per-step elimination factors alpha/gamma are precomputed and the timed
    # solve is one fused update per step:
    #   d <- d + alpha_k * d[i-s] + gamma_k * d[i+s]
    # Out-of-range neighbours are identity rows: b filled with 1, the rest
    # with 0 (the PCR invariant a_i = 0 for i < s makes those terms vanish).
    pcr_steps = nz.bit_length()  # ceil(log2(nz + 1))

    def _shift_dn(v, s, fill=0.0):
        if z_first_layout:
            pad = jnp.full((s,) + v.shape[1:], fill, dtype=v.dtype)
            return jnp.concatenate((pad, v[:-s]), axis=0)
        pad = jnp.full(v.shape[:-1] + (s,), fill, dtype=v.dtype)
        return jnp.concatenate((pad, v[..., :-s]), axis=-1)

    def _shift_up(v, s, fill=0.0):
        if z_first_layout:
            pad = jnp.full((s,) + v.shape[1:], fill, dtype=v.dtype)
            return jnp.concatenate((v[s:], pad), axis=0)
        pad = jnp.full(v.shape[:-1] + (s,), fill, dtype=v.dtype)
        return jnp.concatenate((v[..., s:], pad), axis=-1)

    def pcr_factor_arrays(a, b, c, steps):
        alphas, gammas = [], []
        for k in range(steps):
            s = 1 << k
            alpha = -a / _shift_dn(b, s, fill=1.0)
            gamma = -c / _shift_up(b, s, fill=1.0)
            b = b + alpha * _shift_dn(c, s) + gamma * _shift_up(a, s)
            a = alpha * _shift_dn(a, s)
            c = gamma * _shift_up(c, s)
            alphas.append(alpha)
            gammas.append(gamma)
        return jnp.stack(alphas), jnp.stack(gammas), 1.0 / b

    def pcr_apply(alphas, gammas, inv_b, rhs, steps):
        d = rhs
        for k in range(steps):
            s = 1 << k
            d = d + alphas[k] * _shift_dn(d, s) + gammas[k] * _shift_up(d, s)
        return d * inv_b

    @mapped
    def build_pcr_factors(a, b, c):
        return pcr_factor_arrays(a, b, c, pcr_steps)

    def pcr_solve(alphas, gammas, inv_b, rhs):
        return pcr_apply(alphas, gammas, inv_b, rhs, pcr_steps)

    def tridiag_solve_op(o1, o2, o3, rhs_col):
        """Dispatch on the selected solver; (o1, o2, o3) are its factors."""
        if args.tridiag == "thomas":
            return solve_tridiag(o1, o2, o3, rhs_col)  # (a, inv_bet, gam)
        return pcr_solve(o1, o2, o3, rhs_col)  # (alphas, gammas, inv_b)

    # ------------------------------------------------------------------
    # SPIKE (substructuring) vertical solve in the z-slab layout.
    #
    # Global system rows 0..nz per mode. Row 0 (bottom Neumann / k^2 = 0 pin)
    # joins the interface system; GPU k owns rows [k m + 1, (k+1) m] with
    # m = nz/P, which aligns exactly with its physical rhs slab (row j reads
    # rhs level j - 1) and with the output p levels. Per solve:
    #   y   = A_k^-1 d           local PCR or Thomas, precomputed factors
    #   exchange 2 interface scalars per mode (y[0], y[m-1]) <- all the comm
    #   u   = M^-1 rhs_u         prefactored (2P+1)-row interface system,
    #                            unknowns (x0, alpha_0, beta_0, ..., beta_{P-1})
    #   x   = y - w L - v R      precomputed spike vectors w = A_k^-1 a_1 e_0,
    #                            v = A_k^-1 c_m e_{m-1}; L = u[2k], R = u[2k+3]
    # ------------------------------------------------------------------
    m_blk = nz_local
    spike_steps = max(1, (m_blk - 1).bit_length())
    spike_local_desc = (
        f"thomas ({m_blk}-row forward/backward scans, compact-a, "
        f"chunk {args.thomas_chunk})"
        if args.tridiag == "thomas"
        else f"pcr ({spike_steps} steps)"
    )
    reduced_n = 2 * p + 1
    nyq = ny // p

    # spike-adaptive truncation: the spike of a mode with horizontal
    # wavenumber kh attenuates by exp(-2 m asinh(kh dz / 2)) across one block.
    # Modes attenuated below machine precision (with margin) close with the
    # neighbour-only PDD 2x2 systems; the low-kh box keeps the exact global
    # interface solve. The residual check measures the truncation error.
    trunc_tau = float(-np.log(np.finfo(np_real).eps)) + 4.0
    trunc_arg = trunc_tau / (2.0 * m_blk)
    kh_cut = (2.0 / dz) * math.sinh(trunc_arg) if trunc_arg < 30.0 else float("inf")
    if math.isfinite(kh_cut):
        ic_cut = max(1, min(nxh, int(kh_cut * args.lx / (2.0 * math.pi)) + 2))
        jc_cut = max(1, min(ny // 2, int(kh_cut * args.ly / (2.0 * math.pi)) + 2))
    else:
        ic_cut, jc_cut = nxh, ny // 2
    global_modes = ic_cut * 2 * jc_cut

    def box_slice(h):
        """(nxh, ny) -> (ic_cut, 2 jc_cut): the low-|ky| columns of low-kx rows."""
        return jnp.concatenate(
            (h[:ic_cut, :jc_cut], h[:ic_cut, ny - jc_cut :]), axis=1
        )

    def box_scatter(full, small):
        full = full.at[:ic_cut, :jc_cut].set(small[:, :jc_cut])
        return full.at[:ic_cut, ny - jc_cut :].set(small[:, jc_cut:])

    def scalars_to_modes(stacked):
        """(S, nxh, ny) per-block scalars -> (P, S, nxh, nyq) on mode owners."""
        s_count = stacked.shape[0]
        t = stacked.reshape(s_count, nxh, p, nyq)
        t = jnp.moveaxis(t, 2, 0)
        if p > 1:
            t = lax.all_to_all(t, axis_name, split_axis=0, concat_axis=0, tiled=True)
        return t

    def modes_to_scalars(t):
        """(P, S, nxh, nyq) per-destination values -> (S, nxh, ny) on blocks."""
        if p > 1:
            t = lax.all_to_all(t, axis_name, split_axis=0, concat_axis=0, tiled=True)
        t = jnp.moveaxis(t, 0, 2)  # (S, nxh, P, nyq), axis 2 = ky chunk
        return t.reshape(t.shape[0], nxh, ny)

    def gather_block_scalars(stacked):
        """(S, nxh, ny) per block -> (P, S, nxh, ny) replicated on every GPU."""
        if p > 1:
            return lax.all_gather(stacked, axis_name, axis=0, tiled=False)
        return stacked[None, ...]

    def block_rows_abc(dev):
        """Tridiagonal rows owned by this block, full couplings included."""
        rows = dev * m_blk + 1 + jnp.arange(m_blk)
        interior = rows <= nz - 1  # row nz is the rigid-lid top row
        a_blk = jnp.where(interior, 1.0 / dz2, -1.0).astype(real_dtype)
        c_blk = jnp.where(interior, 1.0 / dz2, 0.0).astype(real_dtype)
        k2 = kx_all[:, None] * kx_all[:, None] + ky_all[None, :] * ky_all[None, :]
        if z_first_layout:
            b_blk = jnp.where(
                interior[:, None, None],
                (-jnp.swapaxes(k2, 0, 1)[None, ...] - 2.0 / dz2).astype(real_dtype),
                jnp.asarray(1.0, real_dtype),
            )
        else:
            b_blk = jnp.where(
                interior[None, None, :],
                (-k2[..., None] - 2.0 / dz2).astype(real_dtype),
                jnp.asarray(1.0, real_dtype),
            )
        return a_blk, b_blk, c_blk

    def assemble_interface_matrix(iface, k2_q):
        """Static (2P+1)-row interface matrix from spike endpoints.

        iface is (P, 4, ...) holding each block's (w[0], w[m-1], v[0], v[m-1])
        for the modes described by k2_q; unknown ordering is
        (x0, alpha_0, beta_0, ..., alpha_{P-1}, beta_{P-1}).
        """
        zero_k2 = jnp.abs(k2_q) < eps128
        one = jnp.asarray(1.0, real_dtype)
        matrix = jnp.zeros(k2_q.shape + (reduced_n, reduced_n), real_dtype)
        matrix = matrix.at[..., 0, 0].set(jnp.where(zero_k2, one, -one))
        matrix = matrix.at[..., 0, 1].set(jnp.where(zero_k2, 0.0 * one, one))
        for k in range(p):
            row_a, row_b = 1 + 2 * k, 2 + 2 * k
            matrix = matrix.at[..., row_a, row_a].set(1.0)
            matrix = matrix.at[..., row_b, row_b].set(1.0)
            matrix = matrix.at[..., row_a, 2 * k].set(iface[k, 0])
            matrix = matrix.at[..., row_b, 2 * k].set(iface[k, 1])
            if k < p - 1:
                matrix = matrix.at[..., row_a, 2 * k + 3].set(iface[k, 2])
                matrix = matrix.at[..., row_b, 2 * k + 3].set(iface[k, 3])
        return matrix

    def build_interface_block_factors(iface, k2_q):
        """Exact 2x2 block-Thomas factors for the reduced SPIKE system.

        After eliminating x0 with the row-0 boundary equation, block k has
        unknowns (alpha_k, beta_k). Its lower block has only a beta_{k-1}
        column and its upper block only an alpha_{k+1} column. The modified
        diagonal inverse therefore remains lower triangular:

            G_k = [[g00, 0], [g10, 1]].

        Per block and mode we store (g00, g10), the two entries of G_k L_k,
        and the two entries of G_k U_k: six real values instead of a dense
        (2P+1)^2 inverse.
        """
        bottom_coef = jnp.where(jnp.abs(k2_q) < eps128, 0.0, 1.0).astype(real_dtype)
        zero = jnp.zeros_like(bottom_coef)
        factors = []
        c1_prev = zero
        for k in range(p):
            w0, wm, v0, vm = iface[k]
            if k == 0:
                pivot = 1.0 + w0 * bottom_coef
                g00 = 1.0 / pivot
                g10 = -(wm * bottom_coef) * g00
                a0 = zero
                a1 = zero
            else:
                pivot = 1.0 - w0 * c1_prev
                g00 = 1.0 / pivot
                g10 = (wm * c1_prev) * g00
                a0 = g00 * w0
                a1 = g10 * w0 + wm
            if k < p - 1:
                c0 = g00 * v0
                c1 = g10 * v0 + vm
            else:
                c0 = zero
                c1 = zero
            factors.append(jnp.stack((g00, g10, a0, a1, c0, c1)))
            c1_prev = c1
        return jnp.stack(factors), bottom_coef

    def solve_interface_blocks(factors, rhs_blocks, bottom_coef):
        """Solve all exact reduced interface systems from precomputed factors.

        factors:   (P, 6, ...)
        rhs_blocks:(P, 2, ...) holding (y_k[0], y_k[m-1])
        returns:   (P, 2, ...) holding each block's (L_k, R_k)
        """

        def forward(prev_beta, values):
            factor, rhs_block = values
            g00, g10, a0, a1 = factor[:4]
            f0, f1 = rhs_block
            z0 = g00 * f0 - a0 * prev_beta
            z1 = g10 * f0 + f1 - a1 * prev_beta
            return z1, jnp.stack((z0, z1))

        initial = jnp.zeros_like(rhs_blocks[0, 0])
        _, z_blocks = lax.scan(forward, initial, (factors, rhs_blocks))

        def backward(next_alpha, values):
            factor, z_block = values
            c0, c1 = factor[4], factor[5]
            x0 = z_block[0] - c0 * next_alpha
            x1 = z_block[1] - c1 * next_alpha
            return x0, jnp.stack((x0, x1))

        _, x_reversed = lax.scan(
            backward,
            initial,
            (factors[::-1], z_blocks[::-1]),
        )
        x_blocks = x_reversed[::-1]
        left = jnp.concatenate(
            ((bottom_coef * x_blocks[0, 0])[None, ...], x_blocks[:-1, 1]),
            axis=0,
        )
        right = jnp.concatenate(
            (x_blocks[1:, 0], jnp.zeros_like(x_blocks[:1, 0])),
            axis=0,
        )
        return jnp.stack((left, right), axis=1)

    def build_selected_interface_response(factors, bottom_coef, selectors):
        """Apply the transpose structured solve to interface selection vectors.

        ``solve_interface_blocks`` is a linear map from the 2P endpoint RHS
        values to the P pairs (L_k, R_k).  For selection vectors E, this
        routine evaluates

            response = E S = (S^T E^T)^T

        by transposing the two block-Thomas substitutions analytically.
        ``selectors`` has shape (R, P, 2, ...), and the returned response has
        the same shape.  Only R=2 rows are needed per GPU in the all-gather
        layout.  No dense (2P+1)-square inverse is formed, even temporarily.
        """
        left_bar = selectors[:, :, 0]
        right_bar = selectors[:, :, 1]
        zero_block = jnp.zeros_like(left_bar[:, :1])

        # Transpose of the final conversion from block unknowns
        # (alpha_k, beta_k) to the neighbour values (L_k, R_k).
        x0_bar = jnp.concatenate(
            (
                bottom_coef[None, None, ...] * left_bar[:, :1],
                right_bar[:, :-1],
            ),
            axis=1,
        )
        x1_bar = jnp.concatenate((left_bar[:, 1:], zero_block), axis=1)

        # Transpose of the backward substitution.  The original substitution
        # runs from P-1 to 0, so its transpose runs from 0 to P-1.
        def backward_transpose(next_x0_bar, values):
            factor, direct_x0_bar, direct_x1_bar = values
            c0, c1 = factor[4], factor[5]
            total_x0_bar = direct_x0_bar + next_x0_bar
            z0_bar = total_x0_bar
            z1_bar = direct_x1_bar
            following_x0_bar = -c0 * z0_bar - c1 * z1_bar
            return following_x0_bar, jnp.stack((z0_bar, z1_bar), axis=1)

        initial = jnp.zeros_like(x0_bar[:, 0])
        _, z_bar_scan = lax.scan(
            backward_transpose,
            initial,
            (
                factors,
                jnp.swapaxes(x0_bar, 0, 1),
                jnp.swapaxes(x1_bar, 0, 1),
            ),
        )
        z_bar = jnp.swapaxes(z_bar_scan, 0, 1)

        # Transpose of the forward substitution.  The original substitution
        # runs from 0 to P-1, so its transpose runs from P-1 to 0.
        def forward_transpose(previous_z1_bar, values):
            factor, direct_z_bar = values
            g00, g10, a0, a1 = factor[:4]
            z0_bar = direct_z_bar[:, 0]
            z1_bar = direct_z_bar[:, 1] + previous_z1_bar
            f0_bar = g00 * z0_bar + g10 * z1_bar
            f1_bar = z1_bar
            preceding_z1_bar = -a0 * z0_bar - a1 * z1_bar
            return preceding_z1_bar, jnp.stack((f0_bar, f1_bar), axis=1)

        _, rhs_bar_reversed = lax.scan(
            forward_transpose,
            initial,
            (factors[::-1], jnp.swapaxes(z_bar, 0, 1)[::-1]),
        )
        return jnp.swapaxes(rhs_bar_reversed[::-1], 0, 1)

    def build_local_block(dev):
        """Selected local factors and spike vectors for this block."""
        a_blk, b_blk, c_blk = block_rows_abc(dev)
        # The couplings that leave the block move to the interface system, so
        # the local matrix zeroes them; their values feed the spike RHS.
        a_first = a_blk[0]
        c_last = c_blk[m_blk - 1]
        a_local = a_blk.at[0].set(0.0)
        c_local = c_blk.at[m_blk - 1].set(0.0)
        if args.tridiag == "thomas":
            # a/c depend only on the block row, not on the horizontal mode.
            # Keep a as a compact m-vector and broadcast it inside the Thomas
            # scan instead of storing/streaming a full mode-by-z array.
            inv_bet, gam = thomas_factor_arrays(a_local, b_blk, c_local)
            local_ops = (a_local, inv_bet, gam)
        else:
            a_in = jnp.broadcast_to(_z_broadcast(a_local), b_blk.shape)
            c_in = jnp.broadcast_to(_z_broadcast(c_local), b_blk.shape)
            alphas, gammas, inv_b = pcr_factor_arrays(a_in, b_blk, c_in, spike_steps)
            local_ops = (alphas, gammas, inv_b)
        e_first = jnp.zeros((m_blk,), real_dtype).at[0].set(1.0)
        e_last = jnp.zeros((m_blk,), real_dtype).at[m_blk - 1].set(1.0)
        w = local_block_solve(
            *local_ops, jnp.broadcast_to(_z_broadcast(a_first * e_first), b_blk.shape)
        )
        v = local_block_solve(
            *local_ops, jnp.broadcast_to(_z_broadcast(c_last * e_last), b_blk.shape)
        )
        return *local_ops, w, v

    @mapped
    def build_spike_adaptive_factors(_token):
        dev = lax.axis_index(axis_name)
        local_o1, local_o2, local_o3, w, v = build_local_block(dev)
        w0 = _mode_to_interface(_z_first(w))
        vm = _mode_to_interface(_z_last(v))
        # Static neighbour spike endpoints for the PDD 2x2 closures:
        # beta_{k-1} pairs with alpha_k across each interface once the far
        # spike endpoints (attenuated below machine precision) are dropped.
        vm_prev = lax.ppermute(vm, axis_name, [(i, (i + 1) % p) for i in range(p)])
        w0_next = lax.ppermute(w0, axis_name, [(i, (i - 1) % p) for i in range(p)])
        tiny = jnp.asarray(np.finfo(np_real).tiny, real_dtype)
        det_prev = 1.0 - vm_prev * w0
        det_next = 1.0 - vm * w0_next
        inv_det_prev = 1.0 / jnp.where(jnp.abs(det_prev) > tiny, det_prev, 1.0)
        inv_det_next = 1.0 / jnp.where(jnp.abs(det_next) > tiny, det_next, 1.0)
        # Bottom closure (used on block 0 only): the row-0 equation gives
        # x0 = -(c0/b0) alpha_0, so alpha_0 (1 - w0 c0/b0) = y_0[0].
        k2 = kx_all[:, None] * kx_all[:, None] + ky_all[None, :] * ky_all[None, :]
        ratio0 = jnp.where(jnp.abs(k2) < eps128, 0.0, -1.0)  # c0 / b0
        denom0 = 1.0 - w0 * ratio0
        l0_coef = -ratio0 / jnp.where(jnp.abs(denom0) > tiny, denom0, 1.0)

        # Exact interface solve retained only for the low-kh box.
        endpoints = jnp.stack(
            (
                box_slice(w0),
                box_slice(_mode_to_interface(_z_last(w))),
                box_slice(_mode_to_interface(_z_first(v))),
                box_slice(vm),
            )
        )
        iface = gather_block_scalars(endpoints)  # (P, 4, ic_cut, 2 jc_cut)
        minv_small = jnp.linalg.inv(assemble_interface_matrix(iface, box_slice(k2)))
        return (
            local_o1,
            local_o2,
            local_o3,
            w,
            v,
            minv_small,
            inv_det_prev,
            vm_prev,
            inv_det_next,
            w0_next,
            l0_coef,
        )

    @mapped
    def build_spike_factors(_token):
        dev = lax.axis_index(axis_name)
        local_o1, local_o2, local_o3, w, v = build_local_block(dev)

        # Build the static interface operator. The all-to-all path shards ky
        # modes across GPUs; the all-gather path replicates all modes so the
        # timed solve needs only one mature NCCL all-gather collective.
        spike_endpoints = jnp.stack(
            (
                _mode_to_interface(_z_first(w)),
                _mode_to_interface(_z_last(w)),
                _mode_to_interface(_z_first(v)),
                _mode_to_interface(_z_last(v)),
            )
        )
        if args.spike_interface_collective == "allgather":
            iface = gather_block_scalars(spike_endpoints)  # (P, 4, nxh, ny)
            ky_q = ky_all
            interface_ny = ny
        else:
            iface = scalars_to_modes(spike_endpoints)  # (P, 4, nxh, nyq)
            ky_q = lax.dynamic_slice_in_dim(ky_all, dev * nyq, nyq)
            interface_ny = nyq
        k2_q = kx_all[:, None] * kx_all[:, None] + ky_q[None, :] * ky_q[None, :]
        del interface_ny  # shape comes from k2_q
        if args.spike_interface_solver == "dense":
            interface_op = jnp.linalg.inv(assemble_interface_matrix(iface, k2_q))
            bottom_coef = jnp.asarray(0.0, real_dtype)
        else:
            block_factors, bottom_coef = build_interface_block_factors(iface, k2_q)
            if args.spike_interface_solver == "block-thomas":
                interface_op = block_factors
            else:
                # The all-gather layout replicates every mode, so this GPU
                # stores only its own (L_d, R_d) response rows.  The all-to-all
                # layout owns 1/P of the modes and must produce rows for all
                # destination GPUs before the return exchange.
                selector_count = 2 if args.spike_interface_collective == "allgather" else 2 * p
                selector_basis = jnp.eye(2 * p, dtype=real_dtype)
                if args.spike_interface_collective == "allgather":
                    selector_basis = lax.dynamic_slice_in_dim(
                        selector_basis, 2 * dev, selector_count, axis=0
                    )
                selectors = jnp.broadcast_to(
                    selector_basis.reshape(selector_count, p, 2, 1, 1),
                    (selector_count, p, 2) + k2_q.shape,
                )
                interface_op = build_selected_interface_response(
                    block_factors, bottom_coef, selectors
                )
                if args.spike_interface_collective == "alltoall":
                    interface_op = interface_op.reshape(
                        p, 2, p, 2, nxh, nyq
                    )
                bottom_coef = jnp.asarray(0.0, real_dtype)
        return local_o1, local_o2, local_o3, w, v, interface_op, bottom_coef

    def spike_pressure_raw(
        rhs_hat_local,
        local_o1,
        local_o2,
        local_o3,
        w,
        v,
        interface_op,
        bottom_coef,
    ):
        """Returns (x unfiltered, (L, R) neighbour values, masked rhs)."""
        dev = lax.axis_index(axis_name)
        # Row nz is a BC row: the last physical rhs level is unused.
        zmask = (jnp.arange(m_blk) < m_blk - 1) | (dev != p - 1)
        d = jnp.where(_z_broadcast(zmask), rhs_hat_local, 0)
        y = local_block_solve(local_o1, local_o2, local_o3, d)
        block_endpoints = jnp.stack(
            (
                _mode_to_interface(_z_first(y)),
                _mode_to_interface(_z_last(y)),
            )
        )
        if args.spike_interface_collective == "allgather":
            iface = gather_block_scalars(block_endpoints)  # (P, 2, nxh, ny)
            interface_ny = ny
        else:
            iface = scalars_to_modes(block_endpoints)  # (P, 2, nxh, nyq)
            interface_ny = nyq
        if args.spike_interface_solver == "dense":
            rhs_u = jnp.transpose(iface, (2, 3, 0, 1)).reshape(
                nxh, interface_ny, 2 * p
            )
            rhs_u = jnp.concatenate((jnp.zeros_like(rhs_u[..., :1]), rhs_u), axis=-1)
            u = jnp.einsum("...ij,...j->...i", interface_op, rhs_u)
            u = jnp.concatenate(
                (u, jnp.zeros_like(u[..., :1])), axis=-1
            )  # R_{P-1} = 0
            if args.spike_interface_collective == "allgather":
                left = lax.dynamic_index_in_dim(u, 2 * dev, axis=-1, keepdims=False)
                right = lax.dynamic_index_in_dim(u, 2 * dev + 3, axis=-1, keepdims=False)
                left_right = jnp.stack((left, right))  # (2, nxh, ny)
            else:
                outs = jnp.stack(
                    [jnp.stack((u[..., 2 * k], u[..., 2 * k + 3])) for k in range(p)]
                )  # (P, 2, nxh, nyq)
                left_right = modes_to_scalars(outs)  # (2, nxh, ny)
        elif args.spike_interface_solver == "block-thomas":
            outs = solve_interface_blocks(interface_op, iface, bottom_coef)
            if args.spike_interface_collective == "allgather":
                left_right = lax.dynamic_index_in_dim(
                    outs, dev, axis=0, keepdims=False
                )  # (2, nxh, ny)
            else:
                left_right = modes_to_scalars(outs)  # (2, nxh, ny)
        elif args.spike_interface_collective == "allgather":
            left_right = jnp.einsum(
                "rpqij,pqij->rij", interface_op, iface
            )  # (2, nxh, ny)
        else:
            outs = jnp.einsum(
                "dspqij,pqij->dsij", interface_op, iface
            )  # (P, 2, nxh, nyq)
            left_right = modes_to_scalars(outs)  # (2, nxh, ny)
        x = y - w * _mode_broadcast(left_right[0]) - v * _mode_broadcast(left_right[1])
        return x, left_right, d

    def spike_adaptive_pressure_raw(
        rhs_hat_local,
        local_o1,
        local_o2,
        local_o3,
        w,
        v,
        minv_small,
        inv_det_prev,
        vm_prev,
        inv_det_next,
        w0_next,
        l0_coef,
    ):
        """Adaptive SPIKE: PDD closure everywhere, exact solve on the low-kh box."""
        dev = lax.axis_index(axis_name)
        zmask = (jnp.arange(m_blk) < m_blk - 1) | (dev != p - 1)
        d = jnp.where(_z_broadcast(zmask), rhs_hat_local, 0)
        y = local_block_solve(local_o1, local_o2, local_o3, d)
        y0 = _mode_to_interface(_z_first(y))
        ym = _mode_to_interface(_z_last(y))

        # Neighbour-only PDD closure (valid for diagonally dominant modes).
        y_prev_m = lax.ppermute(ym, axis_name, [(i, (i + 1) % p) for i in range(p)])
        y_next_0 = lax.ppermute(y0, axis_name, [(i, (i - 1) % p) for i in range(p)])
        left = jnp.where(
            dev == 0,
            l0_coef * y0,
            inv_det_prev * (y_prev_m - vm_prev * y0),
        )
        right = jnp.where(
            dev == p - 1,
            jnp.zeros_like(ym),
            inv_det_next * (y_next_0 - w0_next * ym),
        )

        # Exact global interface solve for the low-kh box only.
        gathered = gather_block_scalars(jnp.stack((box_slice(y0), box_slice(ym))))
        rhs_u = jnp.transpose(gathered, (2, 3, 0, 1)).reshape(ic_cut, 2 * jc_cut, 2 * p)
        rhs_u = jnp.concatenate((jnp.zeros_like(rhs_u[..., :1]), rhs_u), axis=-1)
        u = jnp.einsum("...ij,...j->...i", minv_small, rhs_u)
        u = jnp.concatenate((u, jnp.zeros_like(u[..., :1])), axis=-1)  # R_{P-1} = 0
        left = box_scatter(left, lax.dynamic_index_in_dim(u, 2 * dev, axis=-1, keepdims=False))
        right = box_scatter(
            right, lax.dynamic_index_in_dim(u, 2 * dev + 3, axis=-1, keepdims=False)
        )

        left_right = jnp.stack((left, right))
        x = y - w * _mode_broadcast(left) - v * _mode_broadcast(right)
        return x, left_right, d

    def spike_raw_dispatch(rhs_hat_local, *ops):
        if args.method == "spike-adaptive":
            return spike_adaptive_pressure_raw(rhs_hat_local, *ops)
        return spike_pressure_raw(rhs_hat_local, *ops)

    def spike_pressure(rhs_hat_local, *ops):
        x, _, _ = spike_raw_dispatch(rhs_hat_local, *ops)
        return x * _spectral_keep().astype(x.dtype)

    def solve_tridiag_scalar_scan(a, inv_bet, gam, rhs):
        """Thomas forward/backward substitution, verbatim from wireles_jax."""
        u0 = _z_first(rhs) * _z_first(inv_bet)

        def forward(carry, values):
            u_prev = carry
            a_j, inv_bet_j, rhs_j = values
            u_j = (rhs_j - a_j * u_prev) * inv_bet_j
            return u_j, u_j

        _, u_tail_z = lax.scan(
            forward,
            u0,
            (
                _move_z_first(_z_without_first(a)),
                _move_z_first(_z_without_first(inv_bet)),
                _move_z_first(_z_without_first(rhs)),
            ),
        )
        u_forward = _prepend_z(u0, u_tail_z)

        def backward(next_u, values):
            u_j_forward, gam_next = values
            u_j = u_j_forward - gam_next * next_u
            return u_j, u_j

        _, u_prefix_rev_z = lax.scan(
            backward,
            _z_last(u_forward),
            (
                _move_z_first(_z_without_last(u_forward))[::-1],
                _move_z_first(_z_without_first(gam))[::-1],
            ),
        )
        return _append_z(u_prefix_rev_z[::-1], _z_last(u_forward))

    def _chunked_scan_rows(initial, row_arrays, step):
        """Scan leading-axis rows, statically unrolling rows inside each chunk."""
        n_rows = row_arrays[0].shape[0]
        chunk = min(args.thomas_chunk, n_rows)
        n_chunks = n_rows // chunk
        n_full = n_chunks * chunk
        carry = initial
        pieces = []

        if n_chunks:
            chunked_arrays = tuple(
                values[:n_full].reshape(
                    (n_chunks, chunk) + values.shape[1:]
                )
                for values in row_arrays
            )

            def scan_chunk(previous, chunks):
                outputs = []
                current = previous
                for j in range(chunk):
                    current = step(current, tuple(values[j] for values in chunks))
                    outputs.append(current)
                return current, jnp.stack(outputs)

            carry, full_output = lax.scan(scan_chunk, carry, chunked_arrays)
            pieces.append(full_output.reshape((n_full,) + full_output.shape[2:]))

        if n_full < n_rows:
            tail = []
            for j in range(n_full, n_rows):
                carry = step(carry, tuple(values[j] for values in row_arrays))
                tail.append(carry)
            pieces.append(jnp.stack(tail))

        return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)

    def solve_tridiag_chunked(a, inv_bet, gam, rhs):
        """Thomas solve with several dependent z rows fused per outer scan body."""
        a_z = _move_z_first(a)
        inv_bet_z = _move_z_first(inv_bet)
        rhs_z = _move_z_first(rhs)
        zero = jnp.zeros_like(rhs_z[0])

        def forward_step(previous, values):
            a_j, inv_bet_j, rhs_j = values
            return (rhs_j - a_j * previous) * inv_bet_j

        u_forward_z = _chunked_scan_rows(
            zero,
            (a_z, inv_bet_z, rhs_z),
            forward_step,
        )

        gam_z = _move_z_first(gam)
        gam_next_z = jnp.concatenate(
            (gam_z[1:], jnp.zeros_like(gam_z[:1])),
            axis=0,
        )

        def backward_step(next_u, values):
            u_j_forward, gam_next = values
            return u_j_forward - gam_next * next_u

        x_reversed_z = _chunked_scan_rows(
            zero,
            (u_forward_z[::-1], gam_next_z[::-1]),
            backward_step,
        )
        return _move_z_last(x_reversed_z[::-1])

    def solve_tridiag(a, inv_bet, gam, rhs):
        if args.thomas_chunk == 1:
            return solve_tridiag_scalar_scan(a, inv_bet, gam, rhs)
        return solve_tridiag_chunked(a, inv_bet, gam, rhs)

    def local_block_solve(o1, o2, o3, rhs):
        """Apply the selected SPIKE-local tridiagonal solver."""
        if args.tridiag == "thomas":
            return solve_tridiag(o1, o2, o3, rhs)  # (a, inv_bet, gam)
        return pcr_apply(o1, o2, o3, rhs, spike_steps)  # (alphas, gammas, inv_b)

    def z_to_y(h):
        """Distributed z-slab -> y-slab layout exchange."""
        if z_first_layout:
            nzl, ny_, nxh_ = h.shape
            h = h.reshape(nzl, p, ny_ // p, nxh_)
            if p > 1:
                h = lax.all_to_all(h, axis_name, split_axis=1, concat_axis=0, tiled=True)
            return h.reshape(nzl * p, ny_ // p, nxh_)

        nxh_, ny_, nzl = h.shape
        h = h.reshape(nxh_, p, ny_ // p, nzl)
        if p > 1:
            h = lax.all_to_all(h, axis_name, split_axis=1, concat_axis=3, tiled=True)
        return h.reshape(nxh_, ny_ // p, nzl * p)

    def y_to_z(h):
        """Distributed y-slab -> z-slab layout exchange."""
        if z_first_layout:
            nz_, nyl, nxh_ = h.shape
            h = h.reshape(p, nz_ // p, nyl, nxh_)
            if p > 1:
                h = lax.all_to_all(h, axis_name, split_axis=0, concat_axis=2, tiled=True)
            return h.reshape(nz_ // p, nyl * p, nxh_)

        nxh_, nyl, nz_ = h.shape
        h = h.reshape(nxh_, nyl, p, nz_ // p)
        if p > 1:
            h = lax.all_to_all(h, axis_name, split_axis=2, concat_axis=1, tiled=True)
        return h.reshape(nxh_, nyl * p, nz_ // p)

    def make_rhs_col(rhs_hat_y):
        if z_first_layout:
            rhs_col = jnp.zeros(
                (nz + 1,) + rhs_hat_y.shape[1:], dtype=rhs_hat_y.dtype
            )
            return rhs_col.at[1:nz].set(rhs_hat_y[: nz - 1])
        rhs_col = jnp.zeros(rhs_hat_y.shape[:-1] + (nz + 1,), dtype=rhs_hat_y.dtype)
        return rhs_col.at[..., 1:nz].set(rhs_hat_y[..., : nz - 1])

    def tridiag_pressure(rhs_hat_y, o1, o2, o3, keep):
        p_col = tridiag_solve_op(o1, o2, o3, make_rhs_col(rhs_hat_y))
        if z_first_layout:
            return p_col[1:] * keep.astype(p_col.dtype)
        return p_col[..., 1:] * keep.astype(p_col.dtype)

    @mapped
    def forward_fft(rhs):
        return _forward_fft_local(rhs)

    @mapped
    def transpose_z_to_y(h):
        return z_to_y(h)

    @mapped
    def tridiag_stage(h, o1, o2, o3, keep):
        return tridiag_pressure(h, o1, o2, o3, keep)

    @mapped
    def transpose_y_to_z(h):
        return y_to_z(h)

    @mapped
    def inverse_fft(h):
        return _inverse_fft_local(h)

    @mapped
    def spike_stage(h, *ops):
        return spike_pressure(h, *ops)

    @mapped
    def solve_full(rhs, *ops):
        h = _forward_fft_local(rhs)
        if is_spike:
            h = spike_pressure(h, *ops)
        else:
            h = z_to_y(h)
            h = tridiag_pressure(h, ops[0], ops[1], ops[2], ops[3])
            h = y_to_z(h)
        return _inverse_fft_local(h)

    def solve_full_staged(rhs, *ops):
        """Complete solve as asynchronously chained component executables.

        There is deliberately no block_until_ready between stages. Device
        dependencies preserve their order, while the benchmark synchronizes
        only on the final physical-space result.
        """
        h = forward_fft(rhs)
        if is_spike:
            h = spike_stage(h, *ops)
        else:
            h = transpose_z_to_y(h)
            h = tridiag_stage(h, ops[0], ops[1], ops[2], ops[3])
            h = transpose_y_to_z(h)
        return inverse_fft(h)

    @mapped
    def solve_residual_spike(rhs, *ops):
        """Residual of the block rows, neighbour values taken from (L, R)."""
        h = _forward_fft_local(rhs)
        x, left_right, d = spike_raw_dispatch(h, *ops)
        dev = lax.axis_index(axis_name)
        a_blk, b_blk, c_blk = block_rows_abc(dev)
        cd = x.dtype
        if z_first_layout:
            left = _interface_to_mode(left_right[0])
            right = _interface_to_mode(left_right[1])
            x_dn = jnp.concatenate((left[None, ...], x[:-1]), axis=0)
            x_up = jnp.concatenate((x[1:], right[None, ...]), axis=0)
        else:
            x_dn = jnp.concatenate((left_right[0][..., None], x[..., :-1]), axis=-1)
            x_up = jnp.concatenate((x[..., 1:], left_right[1][..., None]), axis=-1)
        residual = (
            _z_broadcast(a_blk) * x_dn
            + b_blk.astype(cd) * x
            + _z_broadcast(c_blk) * x_up
            - d
        )
        mask = _spectral_keep()
        num = lax.pmax(jnp.max(jnp.abs(residual) * mask), axis_name)
        den = lax.pmax(jnp.max(jnp.abs(d) * mask), axis_name)
        return num / jnp.maximum(den, jnp.finfo(real_dtype).tiny)

    @mapped
    def solve_residual(rhs, a, b, c, o1, o2, o3, keep):
        """Relative residual of A p_col = rhs_col over unfiltered modes."""
        h = _forward_fft_local(rhs)
        rhs_col = make_rhs_col(z_to_y(h))
        p_col = tridiag_solve_op(o1, o2, o3, rhs_col)
        cd = p_col.dtype
        if z_first_layout:
            p_dn = jnp.pad(p_col[:-1], ((1, 0), (0, 0), (0, 0)))
            p_up = jnp.pad(p_col[1:], ((0, 1), (0, 0), (0, 0)))
        else:
            p_dn = jnp.pad(p_col[..., :-1], ((0, 0), (0, 0), (1, 0)))
            p_up = jnp.pad(p_col[..., 1:], ((0, 0), (0, 0), (0, 1)))
        residual = (
            a.astype(cd) * p_dn + b.astype(cd) * p_col + c.astype(cd) * p_up - rhs_col
        )
        mask = keep.astype(real_dtype)
        num = lax.pmax(jnp.max(jnp.abs(residual) * mask), axis_name)
        den = lax.pmax(jnp.max(jnp.abs(rhs_col) * mask), axis_name)
        return num / jnp.maximum(den, jnp.finfo(real_dtype).tiny)

    @mapped
    def make_rhs(key):
        shape = (
            (nz_local, ny, nx)
            if z_first_layout
            else (nx, ny, nz_local)
        )
        return jax.random.uniform(key, shape, dtype=real_dtype) - 0.5

    @mapped
    def make_mms_fields(_token):
        """Manufactured (rhs, p) pair on this device's z slab.

        Interior row j of the tridiagonal system reads physical rhs level
        j - 1, and output level jj holds p_col[jj + 1]; both therefore sample
        the vertical profile at zeta = (jj + 1/2) / (nz - 1). The discrete
        eigenvalue of cos(mz pi zeta) under the interior stencil is
        lambda = (2 - 2 cos(mz pi / (nz - 1))) / dz^2.
        """
        d = lax.axis_index(axis_name)
        x = jnp.arange(nx, dtype=real_dtype) * (args.lx / nx)
        y = jnp.arange(ny, dtype=real_dtype) * (args.ly / ny)
        jj = d * nz_local + jnp.arange(nz_local, dtype=real_dtype)
        zeta = (jj + 0.5) / (nz - 1)
        shape = (
            (nz_local, ny, nx)
            if z_first_layout
            else (nx, ny, nz_local)
        )
        p_ref = jnp.zeros(shape, real_dtype)
        rhs_ref = jnp.zeros_like(p_ref)
        for amp, mx, my, mz, trig_x, trig_y in mms_terms:
            kx_v = 2.0 * math.pi * mx / args.lx
            ky_v = 2.0 * math.pi * my / args.ly
            lam = (2.0 - 2.0 * math.cos(mz * math.pi / (nz - 1))) / dz2
            fx = jnp.cos(kx_v * x) if trig_x == "cos" else jnp.sin(kx_v * x)
            fy = jnp.cos(ky_v * y) if trig_y == "cos" else jnp.sin(ky_v * y)
            fz = jnp.cos(mz * jnp.pi * zeta)
            if z_first_layout:
                term = (
                    amp
                    * fz[:, None, None]
                    * fy[None, :, None]
                    * fx[None, None, :]
                )
            else:
                term = (
                    amp
                    * fx[:, None, None]
                    * fy[None, :, None]
                    * fz[None, None, :]
                )
            p_ref = p_ref + term
            rhs_ref = rhs_ref - (kx_v * kx_v + ky_v * ky_v + lam) * term
        return rhs_ref, p_ref

    @mapped
    def make_broadband_fields(_token):
        """Broad-spectrum manufactured pair on this device's z slab.

        The solution is built from one random-phase field per z level
        (deterministic per-level keys, so adjacent slabs agree without
        communication), shaped by a Kolmogorov-like envelope over every
        resolved horizontal mode. The rhs is the exact discrete operator
        applied in spectral space; BC rows hold by construction: level nz-1
        aliases level nz-2 (rigid lid), level -1 aliases level 0 (Neumann),
        and the k^2 = 0 column pins p_col[0] = 0.
        """
        dev = lax.axis_index(axis_name)
        k2 = kx_all[:, None] * kx_all[:, None] + ky_all[None, :] * ky_all[None, :]
        k_ref = 4.0 * 2.0 * math.pi / args.lx
        envelope = (1.0 + k2 / (k_ref * k_ref)) ** (-2.0 / 3.0)
        shape_mask = (envelope * keep_all).astype(real_dtype)
        base_key = jax.random.PRNGKey(args.seed + 7)

        def level_hat(jj):
            src = jnp.minimum(jj, nz - 2)  # rigid lid: level nz-1 copies nz-2
            key = jax.random.fold_in(base_key, src)
            if z_first_layout:
                q = jax.random.uniform(key, (ny, nx), dtype=real_dtype) - 0.5
                return jnp.fft.rfftn(q, axes=(-2, -1)) * jnp.swapaxes(
                    shape_mask, 0, 1
                )
            q = jax.random.uniform(key, (nx, ny), dtype=real_dtype) - 0.5
            return jnp.fft.rfftn(q, axes=(-1, -2)) * shape_mask

        base = dev * nz_local
        jjs = jnp.clip(base + jnp.arange(-1, nz_local + 1), 0, nz - 1)
        qh_levels = jax.vmap(level_hat)(jjs)
        qh = qh_levels if z_first_layout else jnp.moveaxis(qh_levels, 0, -1)

        if z_first_layout:
            mid = qh[1:-1]  # solution levels base .. base+nzl-1
            dn = qh[:-2]    # levels base-1 .. (level -1 clipped to 0 = Neumann)
            up = qh[2:]     # levels base+1 ..
        else:
            mid = qh[..., 1:-1]
            dn = qh[..., :-2]
            up = qh[..., 2:]
        ell = base + jnp.arange(nz_local)  # global level indices
        # The pinned k^2 = 0 mode has p_col[0] = 0 instead of the Neumann copy.
        if z_first_layout:
            k2_local = jnp.swapaxes(k2, 0, 1)
            pinned = (ell == 0)[:, None, None] & (
                jnp.abs(k2_local) < eps128
            )[None, ...]
        else:
            k2_local = k2
            pinned = (jnp.abs(k2) < eps128)[..., None] & (ell == 0)[None, None, :]
        dn = jnp.where(pinned, 0.0, dn)
        # Interior rows 1..nz-1 read levels ell <= nz-2; the last level feeds
        # the top BC row and is unused by the solver.
        if z_first_layout:
            rhs_hat = jnp.where(
                (ell <= nz - 2)[:, None, None],
                (dn + up) / dz2
                + (-k2_local[None, ...] - 2.0 / dz2) * mid,
                0.0,
            )
        else:
            rhs_hat = jnp.where(
                (ell <= nz - 2)[None, None, :],
                (dn + up) / dz2 + (-k2_local[..., None] - 2.0 / dz2) * mid,
                0.0,
            )
        rhs_ref = _inverse_fft_local(rhs_hat).astype(real_dtype)
        p_ref = _inverse_fft_local(mid).astype(real_dtype)
        return rhs_ref, p_ref

    @mapped
    def relative_max_difference(a_arr, b_arr):
        num = lax.pmax(jnp.max(jnp.abs(a_arr - b_arr)), axis_name)
        den = lax.pmax(jnp.max(jnp.abs(b_arr)), axis_name)
        return num / jnp.maximum(den, jnp.finfo(real_dtype).tiny)

    @mapped
    def global_max_token(x):
        return lax.pmax(x, axis_name)

    # Precompute the vertical and interface operators before entering either
    # the public-library path or the benchmark-only data generation/reporting
    # path. Keeping the same mapped builders preserves the original compiled
    # representation and collective boundaries.
    op_token = np.zeros((local_devices,), dtype=np.float32)
    if args.method == "spike-adaptive":
        pipeline_ops = tuple(build_spike_adaptive_factors(op_token))
    elif args.method == "spike":
        pipeline_ops = tuple(build_spike_factors(op_token))
    else:
        a_op, b_op, c_op, keep_op = build_operators(op_token)
        if args.tridiag == "thomas":
            inv_bet_op, gam_op = build_thomas_factors(a_op, b_op, c_op)
            solver_ops = (a_op, inv_bet_op, gam_op)
        else:
            alphas_op, gammas_op, inv_b_op = build_pcr_factors(a_op, b_op, c_op)
            solver_ops = (alphas_op, gammas_op, inv_b_op)
        pipeline_ops = (*solver_ops, keep_op)
    pipeline_ops[0].block_until_ready()

    if library_mode:
        if is_spike:
            def residual_bound(rhs):
                return solve_residual_spike(rhs, *pipeline_ops)
        else:
            def residual_bound(rhs):
                return solve_residual(
                    rhs,
                    a_op,
                    b_op,
                    c_op,
                    *solver_ops,
                    keep_op,
                )

        local_input_shape = (
            (local_devices, nz_local, ny, nx)
            if z_first_layout
            else (local_devices, nx, ny, nz_local)
        )
        global_input_shape = (
            (nz, ny, nx) if z_first_layout else (nx, ny, nz)
        )
        return _SolverEngine(
            config=args,
            global_devices=global_devices,
            local_devices=local_devices,
            process_count=process_count,
            process_index=process_index,
            local_input_shape=local_input_shape,
            global_input_shape=global_input_shape,
            solve_monolithic=solve_full,
            solve_staged=solve_full_staged,
            residual=residual_bound,
            pipeline_ops=pipeline_ops,
        )

    p_mms = None
    if args.mms:
        mms_token = np.zeros((local_devices,), dtype=np.float32)
        mms_builder = (
            make_broadband_fields if args.mms_kind == "broadband" else make_mms_fields
        )
        rhs, p_mms = mms_builder(mms_token)
    else:
        # Global device IDs are process-major in multi-process pmap programs.
        first_id = process_index * local_devices
        device_ids = np.arange(first_id, first_id + local_devices, dtype=np.uint32)
        base_key = jax.random.PRNGKey(args.seed)
        keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(jnp.asarray(device_ids))
        rhs = make_rhs(keys)
    rhs.block_until_ready()

    token = np.zeros((local_devices,), dtype=np.float32)
    global_max_token(token).block_until_ready()  # compile the timing collective

    def sync_all_processes() -> None:
        global_max_token(token).block_until_ready()

    def global_max_seconds(value: float) -> float:
        values = np.full((local_devices,), value, dtype=np.float32)
        result = global_max_token(values)
        result.block_until_ready()
        return float(np.asarray(result)[0])

    def benchmark(fn, *fn_args):
        sync_all_processes()
        t0 = time.perf_counter()
        out = fn(*fn_args)
        out.block_until_ready()
        compile_and_run = global_max_seconds(time.perf_counter() - t0)

        for _ in range(args.warmup):
            out = fn(*fn_args)
            out.block_until_ready()

        times = []
        for _ in range(args.samples):
            sync_all_processes()
            t0 = time.perf_counter()
            for _ in range(args.iterations):
                out = fn(*fn_args)
                out.block_until_ready()
            elapsed = (time.perf_counter() - t0) / args.iterations
            times.append(global_max_seconds(elapsed))
        return out, compile_and_run, times

    def report(
        label: str,
        compile_s: float,
        times: list[float],
        first_call_note: str = "compile+run",
    ) -> float:
        median_s = statistics.median(times)
        if process_index == 0:
            print(f"\n{label}")
            print(f"  first call ({first_call_note}): {compile_s * 1e3:.3f} ms")
            print(f"  median                 : {median_s * 1e3:.3f} ms")
            print(f"  best                   : {min(times) * 1e3:.3f} ms")
            print(
                f"  mean / p95             : {statistics.mean(times) * 1e3:.3f} / "
                f"{percentile(times, 0.95) * 1e3:.3f} ms"
            )
        return median_s

    if process_index == 0:
        print("Distributed 3D pressure-Poisson benchmark (spectral xy + FD z)")
        print(f"Python                  : {sys.version.split()[0]}")
        print(f"JAX                     : {jax.__version__}")
        print(f"Backend                 : {jax.default_backend()}")
        print(f"Processes               : {process_count}")
        print(f"Global / local GPUs     : {global_devices} / {local_devices}")
        print(f"Grid                    : {nx} x {ny} x {nz}")
        if z_first_layout:
            print(f"Data layout             : z-first")
            print(f"Physical z slab / GPU   : {nz_local} x {ny} x {nx}  (z, y, x)")
            print(f"Spectral z slab / GPU   : {nz_local} x {ny} x {nxh}  (z, y, kx)")
            print(f"Spectral y slab / GPU   : {nz} x {ny_local} x {nxh}  (z, y, kx)")
        else:
            print(f"Data layout             : xyz")
            print(f"Spectral layout         : {nxh} x {ny} x {nz}  (Fortran, kx halved)")
            print(f"Physical z slab / GPU   : {nx} x {ny} x {nz_local}")
            print(f"Spectral y slab / GPU   : {nxh} x {ny_local} x {nz}")
        print(f"Dtype                   : {args.dtype}")
        print(f"Nyquist filter          : {not args.no_nyquist_filter}")
        if args.method == "spike-adaptive":
            global_pct = 100.0 * global_modes / (nxh * ny)
            solver_desc = (
                f"spike-adaptive (blocks of {m_blk}, local {spike_local_desc}, "
                f"PDD closure + exact {reduced_n}-row solve on "
                f"{global_modes} low-kh modes = {global_pct:.2f}%)"
            )
        elif args.method == "spike":
            solver_desc = (
                f"spike (blocks of {m_blk}, local {spike_local_desc}, "
                f"{reduced_n}-row interface, {args.spike_interface_collective}, "
                f"{args.spike_interface_solver})"
            )
        elif args.tridiag == "pcr":
            solver_desc = f"pcr ({pcr_steps} steps)"
        else:
            solver_desc = f"thomas (scan, chunk {args.thomas_chunk})"
        print(f"Method                  : {args.method}")
        print(f"Tridiagonal solver      : {solver_desc}")
        print(f"Pipeline execution      : {args.pipeline_execution}")
        rhs_desc = f"manufactured solution (MMS, {args.mms_kind})" if args.mms else "random"
        print(f"RHS                     : {rhs_desc}")
        print(f"Physical / spectral data: {human_bytes(physical_bytes)} / {human_bytes(spectral_bytes)}")
        print(f"Warmup / timing         : {args.warmup} / {args.samples} x {args.iterations}")

    component_times = {}
    if not args.skip_components and is_spike:
        rhs_hat, compile_s, times = benchmark(forward_fft, rhs)
        component_times["rfft2"] = report("Forward rfft2 (local)", compile_s, times)

        p_hat_z, compile_s, times = benchmark(spike_stage, rhs_hat, *pipeline_ops)
        component_times["spike"] = report(
            f"{args.method} vertical solve (local {spike_local_desc} + "
            f"{reduced_n}-row interface)",
            compile_s,
            times,
        )
        del rhs_hat
        gc.collect()

        _, compile_s, times = benchmark(inverse_fft, p_hat_z)
        component_times["irfft2"] = report("Inverse irfft2 (local)", compile_s, times)
        del p_hat_z
        gc.collect()
    elif not args.skip_components:
        rhs_hat, compile_s, times = benchmark(forward_fft, rhs)
        component_times["rfft2"] = report("Forward rfft2 (local)", compile_s, times)

        hat_y, compile_s, times = benchmark(transpose_z_to_y, rhs_hat)
        component_times["z_to_y"] = report(
            "z-slab -> y-slab all-to-all" if p > 1 else "z-slab -> y-slab (local reshape)",
            compile_s,
            times,
        )
        del rhs_hat
        gc.collect()

        tridiag_label = (
            f"Tridiagonal solve (PCR, {pcr_steps} steps)"
            if args.tridiag == "pcr"
            else f"Tridiagonal solve (Thomas scans, chunk {args.thomas_chunk})"
        )
        p_hat_y, compile_s, times = benchmark(tridiag_stage, hat_y, *solver_ops, keep_op)
        component_times["tridiag"] = report(tridiag_label, compile_s, times)
        del hat_y
        gc.collect()

        p_hat_z, compile_s, times = benchmark(transpose_y_to_z, p_hat_y)
        component_times["y_to_z"] = report(
            "y-slab -> z-slab all-to-all" if p > 1 else "y-slab -> z-slab (local reshape)",
            compile_s,
            times,
        )
        del p_hat_y
        gc.collect()

        _, compile_s, times = benchmark(inverse_fft, p_hat_z)
        component_times["irfft2"] = report("Inverse irfft2 (local)", compile_s, times)
        del p_hat_z
        gc.collect()

    full_solver = (
        solve_full_staged
        if args.pipeline_execution == "staged"
        else solve_full
    )
    p_out, compile_s, times = benchmark(full_solver, rhs, *pipeline_ops)
    first_call_note = (
        "stages cached"
        if args.pipeline_execution == "staged" and not args.skip_components
        else "compile+run"
    )
    full_s = report(
        f"Complete pressure-Poisson solve ({args.pipeline_execution})",
        compile_s,
        times,
        first_call_note,
    )

    if is_spike:
        error = solve_residual_spike(rhs, *pipeline_ops)
    else:
        error = solve_residual(rhs, a_op, b_op, c_op, *solver_ops, keep_op)
    error.block_until_ready()
    relative_residual = float(np.asarray(error)[0])

    mms_error = None
    if args.mms:
        diff = relative_max_difference(p_out, p_mms)
        diff.block_until_ready()
        mms_error = float(np.asarray(diff)[0])

    if process_index == 0:
        remote_payload = spectral_bytes * (p - 1) / p
        print("\nSummary")
        print(f"  data layout                   : {args.data_layout}")
        print(f"  pipeline execution            : {args.pipeline_execution}")
        if args.tridiag == "thomas":
            print(f"  Thomas chunk                  : {args.thomas_chunk}")
        print(f"  solves/s                     : {1.0 / full_s:,.3f}")
        print(f"  grid throughput              : {nx * ny * nz / full_s / 1e9:,.3f} Gpoints/s")
        print(f"  tridiagonal relative residual: {relative_residual:.3e}")
        if mms_error is not None:
            print(f"  MMS relative max error       : {mms_error:.3e}")
        if component_times:
            component_sum = sum(component_times.values())
            print(f"  sum of measured stages       : {component_sum * 1e3:.3f} ms")
            print(f"  full minus isolated stages   : {(full_s - component_sum) * 1e3:+.3f} ms")
            if "z_to_y" in component_times:
                exchange_s = component_times["z_to_y"] + component_times["y_to_z"]
                print(f"  exchange fraction of stages  : {exchange_s / component_sum * 100:.2f}%")
                if p > 1:
                    bw = 2.0 * remote_payload / exchange_s / 2**30
                    print(f"  remote payload per exchange  : {human_bytes(remote_payload)}")
                    print(f"  aggregate one-way bandwidth  : {bw:,.3f} GiB/s")
                else:
                    print("  remote payload per exchange  : 0 B (single GPU)")
        if is_spike:
            endpoint_bytes = 2 * nxh * ny * 2 * real_itemsize
            if p == 1:
                iface_payload = 0
                iface_collectives = 0
                iface_desc = "none (single GPU)"
            elif args.method == "spike-adaptive":
                # 2 neighbour ppermutes (full endpoint fields) + 1 tiny
                # all-gather over the low-kh box.
                box_bytes = 2 * global_modes * 2 * real_itemsize
                iface_payload = endpoint_bytes + box_bytes * (p - 1)
                iface_collectives = 3
                iface_desc = "2 ppermute + low-kh all-gather"
            elif args.spike_interface_collective == "allgather":
                iface_payload = endpoint_bytes * (p - 1)
                iface_collectives = 1
                iface_desc = "allgather"
            else:
                iface_payload = 2 * endpoint_bytes * (p - 1) / p
                iface_collectives = 2
                iface_desc = "alltoall"
            print(f"  interface collective         : {iface_desc}")
            if args.method == "spike":
                print(f"  interface solver             : {args.spike_interface_solver}")
                interface_modes = nxh * (
                    ny if args.spike_interface_collective == "allgather" else nyq
                )
                if args.spike_interface_solver == "selected-rows":
                    # All-gather stores two response rows for every local mode.
                    # All-to-all stores 2P rows for a 1/P mode shard, giving
                    # the same aggregate bytes per GPU.
                    factors_per_mode = (
                        4 * p
                        if args.spike_interface_collective == "allgather"
                        else 4 * p * p
                    )
                elif args.spike_interface_solver == "block-thomas":
                    factors_per_mode = 6 * p + 1
                else:
                    factors_per_mode = reduced_n * reduced_n
                print(
                    "  interface factor storage/GPU : "
                    f"{human_bytes(interface_modes * factors_per_mode * real_itemsize)}"
                )
            print(f"  interface collectives/solve  : {iface_collectives}")
            print(f"  interface payload per solve  : {human_bytes(iface_payload)}")
            print(f"  transpose payload per solve  : {human_bytes(2 * remote_payload)}")

    sync_all_processes()
    return 0


def build_solver(config):
    """Build the reusable solver engine used by :mod:`spectral_fd`.

    ``config`` is intentionally duck-typed here so the benchmark module does
    not need to import the public package during command-line execution.
    """
    try:
        args = config._as_legacy_namespace()
    except AttributeError as exc:
        raise TypeError(
            "config must be a spectral_fd.poisson3d.Poisson3DConfig"
        ) from exc
    return _run(args, library_mode=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
