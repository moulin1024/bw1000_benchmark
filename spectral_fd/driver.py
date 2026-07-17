"""Package-owned orchestration for the Poisson benchmark application."""

from __future__ import annotations

from .config import Poisson3DConfig


def solver_config_from_options(args) -> Poisson3DConfig:
    """Project benchmark CLI options into the public solver configuration."""
    return Poisson3DConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        dtype=args.dtype,
        nyquist_filter=not args.no_nyquist_filter,
        tridiag=args.tridiag,
        thomas_chunk=args.thomas_chunk,
        method=args.method,
        spike_interface_collective=args.spike_interface_collective,
        spike_interface_solver=args.spike_interface_solver,
        pipeline_execution=args.pipeline_execution,
        data_layout=args.data_layout,
        platform=args.platform,
        distributed=args.distributed,
    )


def run_benchmark(args) -> int:
    """Assemble, validate, time, and report one benchmark configuration."""
    if args.warmup < 0 or args.samples <= 0 or args.iterations <= 0:
        raise SystemExit(
            "warmup must be nonnegative; samples and iterations must be positive"
        )

    config = solver_config_from_options(args)
    try:
        config.validate()
        from .validation import validate_mms_configuration

        validate_mms_configuration(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    from .factory import build_poisson_solver, initialize_jax_runtime

    runtime = initialize_jax_runtime(config)
    if runtime.backend != "gpu":
        raise SystemExit(f"GPU backend required; got {runtime.backend!r}")
    try:
        assembly = build_poisson_solver(config, runtime)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    from .benchmark import (
        DistributedBenchmarkRunner,
        print_benchmark_configuration,
        print_benchmark_summary,
        run_pipeline_benchmark,
    )

    validation = assembly.create_validation(args)
    fields = validation.generate(
        seed=args.seed,
        mms=args.mms,
        mms_kind=args.mms_kind,
    )
    benchmark_context = assembly.benchmark_context()
    benchmark_runner = DistributedBenchmarkRunner(
        global_max=validation.global_max,
        local_devices=runtime.local_devices,
        process_index=runtime.process_index,
        warmup=args.warmup,
        samples=args.samples,
        iterations=args.iterations,
    )

    print_benchmark_configuration(args, benchmark_context)
    benchmark_result = run_pipeline_benchmark(
        args,
        context=benchmark_context,
        runner=benchmark_runner,
        pipeline=assembly.pipeline,
        operators=assembly.operators,
        rhs=fields.rhs,
    )
    metrics = validation.evaluate(
        pipeline=assembly.pipeline,
        operators=assembly.operators,
        rhs=fields.rhs,
        output=benchmark_result.output,
        reference=fields.reference,
    )
    print_benchmark_summary(
        args,
        context=benchmark_context,
        result=benchmark_result,
        relative_residual=metrics.relative_residual,
        mms_error=metrics.mms_error,
    )
    benchmark_runner.synchronize()
    if benchmark_context.is_root and (args.report_json or args.report_csv):
        from .reporting import build_benchmark_record, write_benchmark_reports

        record = build_benchmark_record(
            args,
            context=benchmark_context,
            result=benchmark_result,
            relative_residual=metrics.relative_residual,
            mms_error=metrics.mms_error,
        )
        write_benchmark_reports(
            record,
            json_path=args.report_json,
            csv_path=args.report_csv,
        )
    return 0
