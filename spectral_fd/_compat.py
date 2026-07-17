"""Adapters between the public API and the current benchmark core."""

from __future__ import annotations

from types import SimpleNamespace

from .config import Poisson3DConfig


def config_to_run_options(config: Poisson3DConfig) -> SimpleNamespace:
    """Translate public solver configuration to the core's flat options."""
    if not isinstance(config, Poisson3DConfig):
        raise TypeError("config must be a spectral_fd.Poisson3DConfig")
    config.validate()
    return SimpleNamespace(
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        lx=config.lx,
        ly=config.ly,
        lz=config.lz,
        dtype=config.dtype,
        no_nyquist_filter=not config.nyquist_filter,
        tridiag=config.tridiag,
        thomas_chunk=config.thomas_chunk,
        method=config.method,
        spike_interface_collective=config.spike_interface_collective,
        spike_interface_solver=config.spike_interface_solver,
        platform=config.platform,
        distributed=config.distributed,
        pipeline_execution=config.pipeline_execution,
        data_layout=config.data_layout,
        # Benchmark-only fields are inert when ``_run`` is in library mode.
        warmup=0,
        samples=1,
        iterations=1,
        seed=0,
        mms=False,
        mms_kind="modes",
        skip_components=True,
    )
