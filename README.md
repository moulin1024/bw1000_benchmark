# spectral-fd

`spectral-fd` is evolving the original `poisson3d_distributed.py` benchmark
into a reusable library for problems that are spectral in the periodic
horizontal directions and finite-difference discretized in the vertical
direction.

The first public API keeps the benchmark's existing JAX `pmap` distribution
and compilation boundaries intact:

```python
from spectral_fd import Poisson3DConfig, Poisson3DSolver

config = Poisson3DConfig.from_preset("dcu-rocm")
solver = Poisson3DSolver(config)

# z-first local shape:
# (local_devices, nz / global_devices, ny, nx)
assert rhs.shape == solver.local_input_shape
assert str(rhs.dtype) == config.dtype
pressure = solver.solve(rhs)
pressure.block_until_ready()
```

Measured platform configurations are available as named presets:

```python
mn5_config = Poisson3DConfig.from_preset("mn5-cuda")
dcu_config = Poisson3DConfig.from_preset("dcu-rocm")
```

The presets intentionally differ in Thomas scan chunking:

| Preset | Platform | Devices | Thomas chunk | Interface |
|---|---|---:|---:|---|
| `mn5-cuda` | CUDA | 4 | 1 | AllGather + selected rows |
| `dcu-rocm` | ROCm | 8 | 16 | AllGather + selected rows |

The matching Slurm drivers load these presets by default:

```bash
sbatch --export=ALL,PRESET=mn5-cuda submit_mn5.sh
sbatch --export=ALL,PRESET=dcu-rocm poisson_scaling.sh
```

Every preset field remains overridable through the existing environment
variables. Use `PRESET=none` to restore the scripts' legacy fallback defaults.

An installed package also exposes the benchmark as a console command; the
legacy script remains available for existing launch files:

```bash
spectral-fd-poisson --nx 1024 --ny 1024 --nz 1024 --method spike
python poisson3d_distributed.py --nx 1024 --ny 1024 --nz 1024 --method spike
```

For multi-process execution, initialize one solver per process with
`distributed=True`, using the same Slurm launch conventions as the benchmark.
The library currently expects the application to own data distribution; it
does not perform implicit host-side global gather or scatter.

The public API also accepts `platform="cpu"` for small development and CI
regression cases. The benchmark CLI remains GPU-only.

## Package structure

- `spectral_fd.config`: public configuration and type definitions.
- `spectral_fd.presets`: measured platform presets.
- `spectral_fd.layouts`: pure slab-decomposition and shape logic.
- `spectral_fd.transforms`: local FFTs, z-axis operations, and slab exchanges.
- `spectral_fd.operators`: horizontal symbols and distributed vertical rows.
- `spectral_fd.tridiagonal`: PCR/Thomas factorization and solve kernels.
- `spectral_fd.spike`: SPIKE communication and reduced-interface solvers.
- `spectral_fd.spike_local`: local block factors and spike-vector assembly.
- `spectral_fd.spike_adaptive`: adaptive PDD closure and exact low-kh box.
- `spectral_fd.pipeline`: mapped stage assembly, operator bundles, and residuals.
- `spectral_fd.benchmark`: distributed timing and benchmark reporting.
- `spectral_fd.runtime`: JAX environment and distributed-launch policy.
- `spectral_fd.cli`: package-owned benchmark argument parsing and entry point.
- `spectral_fd.solver`: public callable solver facade.
- `spectral_fd._engine`: internal binding of factors to solve callables.
- `spectral_fd._compat`: internal public-config to benchmark-core adapter.
- Validation-data generation and top-level CLI orchestration currently remain
  in `poisson3d_distributed.py`.
- The benchmark CLI and library API share the same factor builders and solve
  callables, avoiding a second implementation.

This boundary is intentional. Subsequent refactors can move manufactured-
solution generation and validation policy into package modules without
changing external application code.
