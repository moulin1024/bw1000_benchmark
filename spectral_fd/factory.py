"""Runtime initialization and assembly of distributed Poisson solvers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class JaxRuntimeContext:
    """Initialized JAX modules and distributed-process metadata."""

    jax: Any
    jnp: Any
    lax: Any
    global_devices: int
    local_devices: int
    process_count: int
    process_index: int
    backend: str


def initialize_jax_runtime(args) -> JaxRuntimeContext:
    """Configure JAX before import and initialize its distributed runtime."""
    from .runtime import configure_jax_environment, initialize_jax_distributed

    configure_jax_environment(platform=args.platform, dtype=args.dtype)

    import jax
    import jax.numpy as jnp
    from jax import lax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    initialize_jax_distributed(jax, enabled=args.distributed)
    return JaxRuntimeContext(
        jax=jax,
        jnp=jnp,
        lax=lax,
        global_devices=jax.device_count(),
        local_devices=jax.local_device_count(),
        process_count=jax.process_count(),
        process_index=jax.process_index(),
        backend=jax.default_backend(),
    )


@dataclass(frozen=True, slots=True)
class PoissonSolverAssembly:
    """A complete pipeline, its factors, and derived runtime metadata."""

    runtime: JaxRuntimeContext
    decomposition: Any
    layout: Any
    real_dtype: Any
    real_itemsize: int
    physical_bytes: int
    spectral_bytes: int
    axis_name: str
    pcr_steps: int
    block_rows: int
    spike_local_description: str
    reduced_size: int
    interface_ny: int
    adaptive_global_modes: int
    kx: Any
    ky: Any
    keep: Any
    zero_tolerance: float
    dz2: float
    pipeline: Any
    operators: Any

    def create_engine(self, config):
        """Bind this assembly into the callable public solver engine."""
        from ._engine import Poisson3DEngine

        return Poisson3DEngine(
            config=config,
            global_devices=self.runtime.global_devices,
            local_devices=self.runtime.local_devices,
            process_count=self.runtime.process_count,
            process_index=self.runtime.process_index,
            local_input_shape=self.decomposition.local_physical_shape(
                config.data_layout
            ),
            global_input_shape=self.decomposition.global_physical_shape(
                config.data_layout
            ),
            solve_monolithic=self.pipeline.solve_monolithic,
            solve_staged=self.pipeline.solve_staged,
            residual=self.operators.bind_residual(self.pipeline),
            pipeline_ops=self.operators.solve_args,
        )

    def create_validation(self, args):
        """Build validation generators with the assembly's layout and symbols."""
        from .validation import build_poisson_validation

        return build_poisson_validation(
            args=args,
            jax=self.runtime.jax,
            jnp=self.runtime.jnp,
            lax=self.runtime.lax,
            layout=self.layout,
            axis_name=self.axis_name,
            local_devices=self.runtime.local_devices,
            process_index=self.runtime.process_index,
            nz_local=self.decomposition.nz_local,
            real_dtype=self.real_dtype,
            kx=self.kx,
            ky=self.ky,
            keep=self.keep,
            zero_tolerance=self.zero_tolerance,
            dz2=self.dz2,
        )

    def benchmark_context(self):
        """Project assembly metadata into the reporting context."""
        from .benchmark import PoissonBenchmarkContext

        return PoissonBenchmarkContext(
            jax_version=self.runtime.jax.__version__,
            backend=self.runtime.backend,
            process_count=self.runtime.process_count,
            process_index=self.runtime.process_index,
            global_devices=self.runtime.global_devices,
            local_devices=self.runtime.local_devices,
            nx=self.decomposition.nx,
            ny=self.decomposition.ny,
            nz=self.decomposition.nz,
            nxh=self.decomposition.nxh,
            ny_local=self.decomposition.ny_local,
            nz_local=self.decomposition.nz_local,
            real_itemsize=self.real_itemsize,
            physical_bytes=self.physical_bytes,
            spectral_bytes=self.spectral_bytes,
            z_first_layout=self.layout.z_first,
            pcr_steps=self.pcr_steps,
            block_rows=self.block_rows,
            spike_local_description=self.spike_local_description,
            reduced_size=self.reduced_size,
            interface_ny=self.interface_ny,
            adaptive_global_modes=self.adaptive_global_modes,
        )


def build_poisson_solver(config, runtime: JaxRuntimeContext) -> PoissonSolverAssembly:
    """Assemble decomposition, numerical components, pipeline, and factors."""
    from .layouts import SlabDecomposition
    from .operators import build_horizontal_symbols, make_vertical_operator_builder
    from .pipeline import assemble_pipeline_operators, build_poisson_pipeline
    from .spike import SpikeInterfaceOps
    from .spike_adaptive import AdaptiveSpikeOps, compute_adaptive_mode_box
    from .spike_local import SpikeLocalBlockOps
    from .transforms import ArrayLayoutOps
    from .tridiagonal import TridiagonalOps

    args = config
    jax, jnp, lax = runtime.jax, runtime.jnp, runtime.lax
    nx, ny, nz = args.nx, args.ny, args.nz
    decomposition = SlabDecomposition(
        nx=nx,
        ny=ny,
        nz=nz,
        global_devices=runtime.global_devices,
        local_devices=runtime.local_devices,
        method=args.method,
    )
    decomposition.validate()

    device_count = decomposition.global_devices
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
    real_dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    real_itemsize = 4 if args.dtype == "float32" else 8
    spacing_z = args.lz / nz
    dz2 = spacing_z * spacing_z
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
        nyquist_filter=args.nyquist_filter,
    )
    kx = jnp.asarray(symbols.kx)
    ky = jnp.asarray(symbols.ky)
    keep = jnp.asarray(symbols.keep)
    zero_tolerance = symbols.zero_tolerance

    z_first_value = layout.z_first_value
    z_last_value = layout.z_last_value
    z_broadcast = layout.z_broadcast
    mode_to_interface = layout.mode_to_interface
    mode_broadcast = layout.mode_broadcast
    spectral_keep = partial(layout.spectral_keep, keep)

    build_operators = mapped(
        make_vertical_operator_builder(
            jnp=jnp,
            lax=lax,
            layout=layout,
            axis_name=axis_name,
            kx=kx,
            ky=ky,
            keep=keep,
            ny_local=ny_local,
            nz=nz,
            dz2=dz2,
            real_dtype=real_dtype,
            zero_tolerance=zero_tolerance,
        )
    )

    pcr_steps = nz.bit_length()
    tridiagonal = TridiagonalOps(
        jnp=jnp,
        lax=lax,
        layout=layout,
        method=args.tridiag,
        thomas_chunk=args.thomas_chunk,
    )
    build_thomas_factors = mapped(tridiagonal.thomas_factor_arrays)
    build_pcr_factors = mapped(partial(tridiagonal.pcr_factor_arrays, steps=pcr_steps))
    tridiagonal_solve = partial(tridiagonal.solve, pcr_steps=pcr_steps)

    block_rows = nz_local
    spike_local = SpikeLocalBlockOps(
        jnp=jnp,
        layout=layout,
        tridiagonal=tridiagonal,
        kx=kx,
        ky=ky,
        block_size=block_rows,
        nz=nz,
        dz2=dz2,
        real_dtype=real_dtype,
    )
    reduced_size = 2 * device_count + 1
    interface_ny = ny // device_count
    adaptive_box = compute_adaptive_mode_box(
        nxh=nxh,
        ny=ny,
        block_size=block_rows,
        dz=spacing_z,
        lx=args.lx,
        ly=args.ly,
        dtype=args.dtype,
    )
    spike_interface = SpikeInterfaceOps(
        jnp=jnp,
        lax=lax,
        axis_name=axis_name,
        device_count=device_count,
        nxh=nxh,
        ny=ny,
        real_dtype=real_dtype,
        zero_tolerance=zero_tolerance,
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
        device_count=device_count,
        kx=kx,
        ky=ky,
        dtype=args.dtype,
        real_dtype=real_dtype,
        zero_tolerance=zero_tolerance,
    )
    local_block_solve = spike_local.solve
    build_block_rows = spike_local.build_rows

    def build_spike_adaptive_factors_local(_token):
        return adaptive_spike.build_factors(lax.axis_index(axis_name))

    build_spike_adaptive_factors = mapped(build_spike_adaptive_factors_local)

    def build_spike_factors_local(_token):
        device_index = lax.axis_index(axis_name)
        local_operator1, local_operator2, local_operator3, spike_w, spike_v = (
            spike_local.build(device_index)
        )
        spike_endpoints = jnp.stack(
            (
                mode_to_interface(z_first_value(spike_w)),
                mode_to_interface(z_last_value(spike_w)),
                mode_to_interface(z_first_value(spike_v)),
                mode_to_interface(z_last_value(spike_v)),
            )
        )
        interface_values = spike_interface.collect_block_scalars(spike_endpoints)
        if args.spike_interface_collective == "allgather":
            interface_ky = ky
        else:
            interface_ky = lax.dynamic_slice_in_dim(
                ky,
                device_index * interface_ny,
                interface_ny,
            )
        interface_k2 = (
            kx[:, None] * kx[:, None] + interface_ky[None, :] * interface_ky[None, :]
        )
        interface_operator, bottom_coefficient = spike_interface.build_operator(
            interface_values,
            interface_k2,
            device_index=device_index,
        )
        return (
            local_operator1,
            local_operator2,
            local_operator3,
            spike_w,
            spike_v,
            interface_operator,
            bottom_coefficient,
        )

    build_spike_factors = mapped(build_spike_factors_local)

    def spike_pressure_raw(
        rhs_spectrum,
        local_operator1,
        local_operator2,
        local_operator3,
        spike_w,
        spike_v,
        interface_operator,
        bottom_coefficient,
    ):
        device_index = lax.axis_index(axis_name)
        row_mask = (jnp.arange(block_rows) < block_rows - 1) | (
            device_index != device_count - 1
        )
        masked_rhs = jnp.where(z_broadcast(row_mask), rhs_spectrum, 0)
        local_solution = local_block_solve(
            local_operator1,
            local_operator2,
            local_operator3,
            masked_rhs,
        )
        block_endpoints = jnp.stack(
            (
                mode_to_interface(z_first_value(local_solution)),
                mode_to_interface(z_last_value(local_solution)),
            )
        )
        interface_values = spike_interface.collect_block_scalars(block_endpoints)
        left_right = spike_interface.apply_operator(
            interface_operator,
            interface_values,
            bottom_coefficient,
            device_index=device_index,
        )
        solution = (
            local_solution
            - spike_w * mode_broadcast(left_right[0])
            - spike_v * mode_broadcast(left_right[1])
        )
        return solution, left_right, masked_rhs

    def spike_adaptive_pressure_raw(rhs_spectrum, *operators):
        return adaptive_spike.apply(
            rhs_spectrum,
            *operators,
            device_index=lax.axis_index(axis_name),
        )

    def spike_raw_dispatch(rhs_spectrum, *operators):
        if args.method == "spike-adaptive":
            return spike_adaptive_pressure_raw(rhs_spectrum, *operators)
        return spike_pressure_raw(rhs_spectrum, *operators)

    def spike_pressure(rhs_spectrum, *operators):
        solution, _, _ = spike_raw_dispatch(rhs_spectrum, *operators)
        return solution * spectral_keep().astype(solution.dtype)

    pipeline = build_poisson_pipeline(
        jax=jax,
        jnp=jnp,
        lax=lax,
        layout=layout,
        axis_name=axis_name,
        device_count=device_count,
        nz=nz,
        real_dtype=real_dtype,
        is_spike=is_spike,
        keep_modes=keep,
        tridiagonal_solve=tridiagonal_solve,
        spike_raw=spike_raw_dispatch,
        spike_filtered=spike_pressure,
        block_rows=build_block_rows,
    )

    setup_token = np.zeros((runtime.local_devices,), dtype=np.float32)
    operators = assemble_pipeline_operators(
        method=args.method,
        tridiagonal=args.tridiag,
        token=setup_token,
        build_spike_adaptive=build_spike_adaptive_factors,
        build_spike=build_spike_factors,
        build_rows=build_operators,
        build_thomas=build_thomas_factors,
        build_pcr=build_pcr_factors,
    )
    return PoissonSolverAssembly(
        runtime=runtime,
        decomposition=decomposition,
        layout=layout,
        real_dtype=real_dtype,
        real_itemsize=real_itemsize,
        physical_bytes=physical_bytes,
        spectral_bytes=spectral_bytes,
        axis_name=axis_name,
        pcr_steps=pcr_steps,
        block_rows=block_rows,
        spike_local_description=spike_local.description,
        reduced_size=reduced_size,
        interface_ny=interface_ny,
        adaptive_global_modes=adaptive_box.global_modes,
        kx=kx,
        ky=ky,
        keep=keep,
        zero_tolerance=zero_tolerance,
        dz2=dz2,
        pipeline=pipeline,
        operators=operators,
    )


def build_solver_engine(config):
    """Build a public solver engine directly from ``Poisson3DConfig``."""
    from .config import Poisson3DConfig

    if not isinstance(config, Poisson3DConfig):
        raise TypeError("config must be a spectral_fd.Poisson3DConfig")
    config.validate()
    runtime = initialize_jax_runtime(config)
    assembly = build_poisson_solver(config, runtime)
    return assembly.create_engine(config)
