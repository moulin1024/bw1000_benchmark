"""Minimal external-call example for the distributed Poisson solver."""

from __future__ import annotations

import numpy as np

from spectral_fd import Poisson3DConfig, Poisson3DSolver


config = Poisson3DConfig.from_preset(
    "dcu-rocm",
    nx=128,
    ny=128,
    nz=128,
    # CPU is useful for this small standalone example. Production DCU runs
    # retain the preset's platform="rocm".
    platform="cpu",
)
solver = Poisson3DSolver(config)

# External applications may construct this array directly on JAX devices.
# NumPy is used here only to make the required local shape explicit.
rhs = np.zeros(solver.local_input_shape, dtype=config.dtype)
pressure = solver.solve(rhs)
pressure.block_until_ready()

print("local pressure shape:", pressure.shape)
