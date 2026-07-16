# spectral-fd

`spectral-fd` is evolving the original `poisson3d_distributed.py` benchmark
into a reusable library for problems that are spectral in the periodic
horizontal directions and finite-difference discretized in the vertical
direction.

The first public API keeps the benchmark's existing JAX `pmap` distribution
and compilation boundaries intact:

```python
from spectral_fd import Poisson3DConfig, Poisson3DSolver

config = Poisson3DConfig(
    nx=1024,
    ny=1024,
    nz=1024,
    method="spike",
    tridiag="thomas",
    thomas_chunk=16,
    spike_interface_collective="allgather",
    spike_interface_solver="selected-rows",
    pipeline_execution="staged",
    data_layout="z-first",
)
solver = Poisson3DSolver(config)

# z-first local shape:
# (local_devices, nz / global_devices, ny, nx)
assert rhs.shape == solver.local_input_shape
assert str(rhs.dtype) == config.dtype
pressure = solver.solve(rhs)
pressure.block_until_ready()
```

For multi-process execution, initialize one solver per process with
`distributed=True`, using the same Slurm launch conventions as the benchmark.
The library currently expects the application to own data distribution; it
does not perform implicit host-side global gather or scatter.

The public API also accepts `platform="cpu"` for small development and CI
regression cases. The benchmark CLI remains GPU-only.

## Current boundary

- Public configuration and callable solver live in `spectral_fd`.
- The tested numerical kernels remain in `poisson3d_distributed.py`.
- The benchmark CLI and library API share the same factor builders and solve
  callables, avoiding a second implementation.

This boundary is intentional. Subsequent refactors can move layouts, FFT
transposes, vertical operators, and SPIKE interface solvers into package
modules without changing external application code.
