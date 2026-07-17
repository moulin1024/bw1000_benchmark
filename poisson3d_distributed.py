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

import math
from functools import partial

import numpy as np


def build_parser():
    """Compatibility wrapper for the package-owned CLI parser."""
    from spectral_fd.cli import build_parser as build_package_parser

    return build_package_parser()


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
    if not library_mode and (
        args.warmup < 0 or args.samples <= 0 or args.iterations <= 0
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
    from spectral_fd.runtime import (
        configure_jax_environment,
        initialize_jax_distributed,
    )

    configure_jax_environment(platform=args.platform, dtype=args.dtype)

    import jax
    import jax.numpy as jnp
    from jax import lax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    initialize_jax_distributed(jax, enabled=args.distributed)

    global_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_count = jax.process_count()
    process_index = jax.process_index()

    if not library_mode and jax.default_backend() != "gpu":
        fail(
            f"GPU backend required; got {jax.default_backend()!r}",
            runtime=True,
        )

    from spectral_fd.layouts import SlabDecomposition
    from spectral_fd.benchmark import (
        DistributedBenchmarkRunner,
        PoissonBenchmarkContext,
        print_benchmark_configuration,
        print_benchmark_summary,
        run_pipeline_benchmark,
    )
    from spectral_fd.operators import (
        build_horizontal_symbols,
        make_vertical_operator_builder,
    )
    from spectral_fd.pipeline import (
        assemble_pipeline_operators,
        build_poisson_pipeline,
    )
    from spectral_fd.spike import SpikeInterfaceOps
    from spectral_fd.spike_adaptive import (
        AdaptiveSpikeOps,
        compute_adaptive_mode_box,
    )
    from spectral_fd.spike_local import SpikeLocalBlockOps
    from spectral_fd.transforms import ArrayLayoutOps
    from spectral_fd.tridiagonal import TridiagonalOps

    nx, ny, nz = args.nx, args.ny, args.nz
    decomposition = SlabDecomposition(
        nx=nx,
        ny=ny,
        nz=nz,
        global_devices=global_devices,
        local_devices=local_devices,
        method=args.method,
    )
    try:
        decomposition.validate()
    except ValueError as exc:
        fail(str(exc))

    p = decomposition.global_devices
    is_spike = args.method != "transpose"
    nxh = decomposition.nxh
    ny_local = decomposition.ny_local
    nz_local = decomposition.nz_local
    layout = ArrayLayoutOps(
        data_layout=args.data_layout,
        nx=nx,
        ny=ny,
        array_namespace=jnp,
    )
    z_first_layout = layout.z_first
    real_dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    real_itemsize = 4 if args.dtype == "float32" else 8
    dz = args.lz / nz
    dz2 = dz * dz

    physical_bytes = nx * ny * nz * real_itemsize
    spectral_bytes = nxh * ny * nz * 2 * real_itemsize
    axis_name = "poisson_devices"
    mapped = partial(jax.pmap, axis_name=axis_name)

    symbols = build_horizontal_symbols(
        nx=nx,
        ny=ny,
        lx=args.lx,
        ly=args.ly,
        dtype=args.dtype,
        nyquist_filter=not args.no_nyquist_filter,
    )
    kx_all = jnp.asarray(symbols.kx)
    ky_all = jnp.asarray(symbols.ky)
    keep_all = jnp.asarray(symbols.keep)
    eps128 = symbols.zero_tolerance

    # Short aliases keep the numerical kernels readable while their layout
    # policy lives in one independently tested object.
    _z_first = layout.z_first_value
    _z_last = layout.z_last_value
    _z_broadcast = layout.z_broadcast
    _mode_to_interface = layout.mode_to_interface
    _interface_to_mode = layout.interface_to_mode
    _mode_broadcast = layout.mode_broadcast
    _spectral_keep = partial(layout.spectral_keep, keep_all)
    _forward_fft_local = layout.forward_fft_local
    _inverse_fft_local = layout.inverse_fft_local

    build_operators = mapped(
        make_vertical_operator_builder(
            jnp=jnp,
            lax=lax,
            layout=layout,
            axis_name=axis_name,
            kx=kx_all,
            ky=ky_all,
            keep=keep_all,
            ny_local=ny_local,
            nz=nz,
            dz2=dz2,
            real_dtype=real_dtype,
            zero_tolerance=eps128,
        )
    )

    # Parallel cyclic reduction. Each step with stride s eliminates the
    # couplings a_i, c_i against rows i -+ s; after ceil(log2(nz+1)) steps the
    # systems are diagonal. The row reduction is RHS-independent, so the
    # per-step elimination factors alpha/gamma are precomputed and the timed
    # solve is one fused update per step:
    #   d <- d + alpha_k * d[i-s] + gamma_k * d[i+s]
    # Out-of-range neighbours are identity rows: b filled with 1, the rest
    # with 0 (the PCR invariant a_i = 0 for i < s makes those terms vanish).
    pcr_steps = nz.bit_length()  # ceil(log2(nz + 1))
    tridiagonal = TridiagonalOps(
        jnp=jnp,
        lax=lax,
        layout=layout,
        method=args.tridiag,
        thomas_chunk=args.thomas_chunk,
    )
    build_thomas_factors = mapped(tridiagonal.thomas_factor_arrays)
    build_pcr_factors = mapped(partial(tridiagonal.pcr_factor_arrays, steps=pcr_steps))
    tridiag_solve_op = partial(tridiagonal.solve, pcr_steps=pcr_steps)

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
    spike_local = SpikeLocalBlockOps(
        jnp=jnp,
        layout=layout,
        tridiagonal=tridiagonal,
        kx=kx_all,
        ky=ky_all,
        block_size=m_blk,
        nz=nz,
        dz2=dz2,
        real_dtype=real_dtype,
    )
    spike_local_desc = spike_local.description
    reduced_n = 2 * p + 1
    nyq = ny // p

    adaptive_box = compute_adaptive_mode_box(
        nxh=nxh,
        ny=ny,
        block_size=m_blk,
        dz=dz,
        lx=args.lx,
        ly=args.ly,
        dtype=args.dtype,
    )
    global_modes = adaptive_box.global_modes
    spike_interface = SpikeInterfaceOps(
        jnp=jnp,
        lax=lax,
        axis_name=axis_name,
        device_count=p,
        nxh=nxh,
        ny=ny,
        real_dtype=real_dtype,
        zero_tolerance=eps128,
        collective=args.spike_interface_collective,
        solver=args.spike_interface_solver,
        ic_cut=adaptive_box.ic_cut,
        jc_cut=adaptive_box.jc_cut,
    )
    adaptive_spike = AdaptiveSpikeOps(
        jnp=jnp,
        lax=lax,
        layout=layout,
        local_blocks=spike_local,
        interface=spike_interface,
        axis_name=axis_name,
        device_count=p,
        kx=kx_all,
        ky=ky_all,
        dtype=args.dtype,
        real_dtype=real_dtype,
        zero_tolerance=eps128,
    )
    local_block_solve = spike_local.solve
    block_rows_abc = spike_local.build_rows

    @mapped
    def build_spike_adaptive_factors(_token):
        return adaptive_spike.build_factors(lax.axis_index(axis_name))

    @mapped
    def build_spike_factors(_token):
        dev = lax.axis_index(axis_name)
        local_o1, local_o2, local_o3, w, v = spike_local.build(dev)

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
        iface = spike_interface.collect_block_scalars(spike_endpoints)
        if args.spike_interface_collective == "allgather":
            ky_q = ky_all
        else:
            ky_q = lax.dynamic_slice_in_dim(ky_all, dev * nyq, nyq)
        k2_q = kx_all[:, None] * kx_all[:, None] + ky_q[None, :] * ky_q[None, :]
        interface_op, bottom_coef = spike_interface.build_operator(
            iface,
            k2_q,
            device_index=dev,
        )
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
        iface = spike_interface.collect_block_scalars(block_endpoints)
        left_right = spike_interface.apply_operator(
            interface_op,
            iface,
            bottom_coef,
            device_index=dev,
        )
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
        return adaptive_spike.apply(
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
            device_index=lax.axis_index(axis_name),
        )

    def spike_raw_dispatch(rhs_hat_local, *ops):
        if args.method == "spike-adaptive":
            return spike_adaptive_pressure_raw(rhs_hat_local, *ops)
        return spike_pressure_raw(rhs_hat_local, *ops)

    def spike_pressure(rhs_hat_local, *ops):
        x, _, _ = spike_raw_dispatch(rhs_hat_local, *ops)
        return x * _spectral_keep().astype(x.dtype)

    pipeline = build_poisson_pipeline(
        jax=jax,
        jnp=jnp,
        lax=lax,
        layout=layout,
        axis_name=axis_name,
        device_count=p,
        nz=nz,
        real_dtype=real_dtype,
        is_spike=is_spike,
        keep_modes=keep_all,
        tridiagonal_solve=tridiag_solve_op,
        spike_raw=spike_raw_dispatch,
        spike_filtered=spike_pressure,
        block_rows=block_rows_abc,
    )

    @mapped
    def make_rhs(key):
        shape = (nz_local, ny, nx) if z_first_layout else (nx, ny, nz_local)
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
        shape = (nz_local, ny, nx) if z_first_layout else (nx, ny, nz_local)
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
                term = amp * fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
            else:
                term = amp * fx[:, None, None] * fy[None, :, None] * fz[None, None, :]
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
                return jnp.fft.rfftn(q, axes=(-2, -1)) * jnp.swapaxes(shape_mask, 0, 1)
            q = jax.random.uniform(key, (nx, ny), dtype=real_dtype) - 0.5
            return jnp.fft.rfftn(q, axes=(-1, -2)) * shape_mask

        base = dev * nz_local
        jjs = jnp.clip(base + jnp.arange(-1, nz_local + 1), 0, nz - 1)
        qh_levels = jax.vmap(level_hat)(jjs)
        qh = qh_levels if z_first_layout else jnp.moveaxis(qh_levels, 0, -1)

        if z_first_layout:
            mid = qh[1:-1]  # solution levels base .. base+nzl-1
            dn = qh[:-2]  # levels base-1 .. (level -1 clipped to 0 = Neumann)
            up = qh[2:]  # levels base+1 ..
        else:
            mid = qh[..., 1:-1]
            dn = qh[..., :-2]
            up = qh[..., 2:]
        ell = base + jnp.arange(nz_local)  # global level indices
        # The pinned k^2 = 0 mode has p_col[0] = 0 instead of the Neumann copy.
        if z_first_layout:
            k2_local = jnp.swapaxes(k2, 0, 1)
            pinned = (ell == 0)[:, None, None] & (jnp.abs(k2_local) < eps128)[None, ...]
        else:
            k2_local = k2
            pinned = (jnp.abs(k2) < eps128)[..., None] & (ell == 0)[None, None, :]
        dn = jnp.where(pinned, 0.0, dn)
        # Interior rows 1..nz-1 read levels ell <= nz-2; the last level feeds
        # the top BC row and is unused by the solver.
        if z_first_layout:
            rhs_hat = jnp.where(
                (ell <= nz - 2)[:, None, None],
                (dn + up) / dz2 + (-k2_local[None, ...] - 2.0 / dz2) * mid,
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
    operator_bundle = assemble_pipeline_operators(
        method=args.method,
        tridiagonal=args.tridiag,
        token=op_token,
        build_spike_adaptive=build_spike_adaptive_factors,
        build_spike=build_spike_factors,
        build_rows=build_operators,
        build_thomas=build_thomas_factors,
        build_pcr=build_pcr_factors,
    )
    pipeline_ops = operator_bundle.solve_args

    if library_mode:
        from spectral_fd._engine import Poisson3DEngine

        return Poisson3DEngine(
            config=args,
            global_devices=global_devices,
            local_devices=local_devices,
            process_count=process_count,
            process_index=process_index,
            local_input_shape=decomposition.local_physical_shape(args.data_layout),
            global_input_shape=decomposition.global_physical_shape(args.data_layout),
            solve_monolithic=pipeline.solve_monolithic,
            solve_staged=pipeline.solve_staged,
            residual=operator_bundle.bind_residual(pipeline),
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
        keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(
            jnp.asarray(device_ids)
        )
        rhs = make_rhs(keys)
    rhs.block_until_ready()

    benchmark_context = PoissonBenchmarkContext(
        jax_version=jax.__version__,
        backend=jax.default_backend(),
        process_count=process_count,
        process_index=process_index,
        global_devices=global_devices,
        local_devices=local_devices,
        nx=nx,
        ny=ny,
        nz=nz,
        nxh=nxh,
        ny_local=ny_local,
        nz_local=nz_local,
        real_itemsize=real_itemsize,
        physical_bytes=physical_bytes,
        spectral_bytes=spectral_bytes,
        z_first_layout=z_first_layout,
        pcr_steps=pcr_steps,
        block_rows=m_blk,
        spike_local_description=spike_local_desc,
        reduced_size=reduced_n,
        interface_ny=nyq,
        adaptive_global_modes=global_modes,
    )
    benchmark_runner = DistributedBenchmarkRunner(
        global_max=global_max_token,
        local_devices=local_devices,
        process_index=process_index,
        warmup=args.warmup,
        samples=args.samples,
        iterations=args.iterations,
    )

    print_benchmark_configuration(args, benchmark_context)

    benchmark_result = run_pipeline_benchmark(
        args,
        context=benchmark_context,
        runner=benchmark_runner,
        pipeline=pipeline,
        operators=operator_bundle,
        rhs=rhs,
    )
    p_out = benchmark_result.output

    error = operator_bundle.residual(pipeline, rhs)
    error.block_until_ready()
    relative_residual = float(np.asarray(error)[0])

    mms_error = None
    if args.mms:
        diff = relative_max_difference(p_out, p_mms)
        diff.block_until_ready()
        mms_error = float(np.asarray(diff)[0])

    print_benchmark_summary(
        args,
        context=benchmark_context,
        result=benchmark_result,
        relative_residual=relative_residual,
        mms_error=mms_error,
    )
    benchmark_runner.synchronize()
    return 0


def build_solver(config):
    """Build the reusable solver engine used by :mod:`spectral_fd`.

    The adapter stays outside the public configuration model so benchmark-only
    fields do not leak into :class:`spectral_fd.Poisson3DConfig`.
    """
    from spectral_fd._compat import config_to_run_options

    args = config_to_run_options(config)
    return _run(args, library_mode=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
