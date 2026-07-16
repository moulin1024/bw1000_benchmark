"""Minimal external-call example for the distributed Poisson solver."""

from __future__ import annotations

import numpy as np

from spectral_fd import Poisson3DConfig, Poisson3DSolver


config = Poisson3DConfig(
    nx=128,
    ny=128,
    nz=128,
    dtype="float64",
    method="spike",
    tridiag="thomas",
    thomas_chunk=16,
    spike_interface_collective="allgather",
    spike_interface_solver="selected-rows",
    pipeline_execution="staged",
    data_layout="z-first",
)
solver = Poisson3DSolver(config)

# External applications may construct this array directly on JAX devices.
# NumPy is used here only to make the required local shape explicit.
rhs = np.zeros(solver.local_input_shape, dtype=config.dtype)
pressure = solver.solve(rhs)
pressure.block_until_ready()

print("local pressure shape:", pressure.shape)
