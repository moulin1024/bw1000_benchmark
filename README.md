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

## Embedding in an existing JAX application

Applications that own JAX startup can pass an already initialized runtime.
This path does not set environment variables or call
`jax.distributed.initialize()` a second time:

```python
import jax

from spectral_fd import (
    Poisson3DConfig,
    Poisson3DSolver,
    runtime_from_initialized_jax,
)

# The application initializes JAX/distributed execution before this point.
runtime = runtime_from_initialized_jax(jax)
config = Poisson3DConfig(
    nx=64,
    ny=32,
    nz=128,
    lx=12.0,
    ly=6.0,
    lz=6.0,
    dtype="float32",
    method="transpose",
    data_layout="z-first",
    discretization="cell-centered-compatible",
)
solver = Poisson3DSolver(config, runtime=runtime)

# All nz cell-centered RHS planes participate in the solve. The facade
# removes the distributed constant RHS mode and returns a mean-zero pressure.
pressure = solver.solve(rhs)
```

`cell-centered-compatible` implements the compatible projection operator
`L = D G`: pressure is cell-centered, vertical gradients live on interior
faces, and homogeneous Neumann boundary fluxes close the first and last cell
rows. It excludes the self-conjugate horizontal Nyquist modes so that the
spectral symbols match a real-valued staggered gradient/divergence pair. This
mode supports the transpose, exact SPIKE, and adaptive SPIKE solvers. The
compatible SPIKE variants use the standard two-endpoint system with `2P`
interface rows for `P` z-slabs; they exchange the same two runtime endpoint
values per block and horizontal mode as the legacy implementation.

Cell-centered pressure does not imply a collocated vertical velocity. In a
staggered application, `w` remains on vertical faces, the Poisson right-hand
side is cell-centered, and the vertical pressure operator is the composed map
`D_z G_z: Cell -> ZFace -> Cell`. The solver stores only the resulting `nz`
cell-to-cell pressure system and does not impose an application-wide ghost or
velocity storage convention.

The facade remains composable under an enclosing `jax.jit`. For implicit
differentiation, supply the application's semantic matrix-vector product:

```python
pressure = solver.implicit_solve(
    lambda value: divergence(gradient(value)),
    rhs,
)
```

The default assumes that operator is symmetric. A nonsymmetric operator must
provide `transpose_solve`. Both the primal and cotangent spaces must obey the
same mean-zero compatibility constraint. When using `dtype="float64"` with an
application-owned runtime, enable `jax_enable_x64` before constructing the
runtime context.

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

Benchmark results can be emitted with a versioned JSON schema and a fixed-
column CSV row in addition to the human-readable report:

```bash
spectral-fd-poisson --method spike --report-json results/run.json \
  --report-csv results/run.csv
```

The numerical regression runner provides a nine-case covering matrix for CI
and the full 144-case Cartesian matrix. Device-count groups run in isolated
CPU/JAX subprocesses so 1/2/4-device configurations can coexist in one run:

```bash
spectral-fd-regression --quick --output results/regression-smoke.json
spectral-fd-regression --output results/regression-full.json
# Optional CI sharding:
spectral-fd-regression --shard-count 4 --shard-index 0
```

The repository workflow runs static checks, unit tests, and the nine-case
matrix for every push and pull request. The full matrix is deliberately
manual to avoid unexpected private-repository runner charges: open **Actions →
Numerical CI → Run workflow** and enable `full_matrix`. Its four shards upload
separate JSON artifacts.

Compare benchmark files or directories with a configurable regression gate:

```bash
spectral-fd-compare results/baseline results/candidate \
  --max-regression-percent 5 \
  --markdown results/comparison.md \
  --json results/comparison.json
```

Reports are matched by solver/grid/runtime configuration. A positive runtime
change means the candidate is slower; the command exits nonzero for a full-
solve regression above the threshold or a missing candidate configuration.
Use `--allow-missing` when intentionally comparing a partial matrix.

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
- `spectral_fd.factory`: JAX runtime initialization and solver assembly.
- `spectral_fd.validation`: random/MMS fields and distributed error metrics.
- `spectral_fd.benchmark`: distributed timing and benchmark reporting.
- `spectral_fd.reporting`: stable JSON/CSV benchmark serialization.
- `spectral_fd.compare`: matched benchmark comparison and regression gates.
- `spectral_fd.regression`: 1/2/4-device numerical regression matrices.
- `spectral_fd.runtime`: JAX environment and distributed-launch policy.
- `spectral_fd.driver`: package-owned benchmark orchestration.
- `spectral_fd.cli`: benchmark argument parsing and package entry point.
- `spectral_fd.solver`: public callable solver facade.
- `spectral_fd._engine`: internal binding of factors to solve callables.
- `poisson3d_distributed.py` is only a thin historical CLI forwarder.
- The benchmark CLI and library API share the same factor builders and solve
  callables, avoiding a second implementation.

The package CLI and public solver now construct the factory directly; the
legacy script contains no numerical or orchestration implementation.
