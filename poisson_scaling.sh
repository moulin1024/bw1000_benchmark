#!/usr/bin/env bash
#SBATCH --job-name=poisson3d-scale
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:8
# NOTE: this cluster ties memory to CPUs (at most DefMemPerCPU per requested
# CPU) and rejects --mem requests beyond that, so memory scales only through
# --cpus-per-task. 32 CPUs also keeps per-gpu mode (8 ROCm processes) clear
# of the 1-CPU-share OOM seen in job 686753.
#SBATCH --time=00:30:00
#SBATCH --output=poisson3d-scale-%j.out
#SBATCH --error=poisson3d-scale-%j.err

# Scaling driver for poisson3d_distributed.py (spectral xy + FD z pressure
# Poisson solve). Inherits the lessons from scaling.sh: memory scales only
# through --cpus-per-task on this cluster; per-gpu mode must NOT use per-task
# GRES binding (RCCL IPC breaks) and pools step memory with --mem=0.
#
# Overrides:
#   sbatch poisson_scaling.sh
#   sbatch --export=ALL,PRESET=dcu-rocm poisson_scaling.sh
#   sbatch --export=ALL,PRESET=none poisson_scaling.sh
#   sbatch --export=ALL,NX=2048,NY=2048,NZ=256 poisson_scaling.sh
#   sbatch --export=ALL,METHOD=spike,SPIKE_INTERFACE_COLLECTIVE=allgather poisson_scaling.sh
#   sbatch --export=ALL,METHOD=spike,SPIKE_INTERFACE_SOLVER=block-thomas poisson_scaling.sh
#   sbatch --export=ALL,METHOD=spike,SPIKE_INTERFACE_SOLVER=dense poisson_scaling.sh
#   sbatch --export=ALL,PIPELINE_EXECUTION=monolithic poisson_scaling.sh
#   sbatch --export=ALL,DATA_LAYOUT=xyz poisson_scaling.sh
#   sbatch --export=ALL,THOMAS_CHUNK=1 poisson_scaling.sh
#   sbatch --export=ALL,METHOD=spike-adaptive poisson_scaling.sh
#   sbatch --export=ALL,METHOD=spike-adaptive,TRIDIAG=thomas poisson_scaling.sh
#   sbatch --export=ALL,TASK_MODE=per-gpu poisson_scaling.sh
#   sbatch --export=ALL,RCCL_DEBUG=1,GPU_COUNTS=8,WARMUP=0,SAMPLES=1,ITERATIONS=1 poisson_scaling.sh

set -euo pipefail

PRESET="${PRESET:-dcu-rocm}"
PRESET_FILE="${SLURM_SUBMIT_DIR}/benchmark_presets.sh"
if [[ ! -f "${PRESET_FILE}" ]]; then
    echo "ERROR: benchmark preset file not found: ${PRESET_FILE}" >&2
    exit 2
fi
source "${PRESET_FILE}"
apply_poisson_benchmark_preset "${PRESET}"

BENCHMARK="${BENCHMARK:-${SLURM_SUBMIT_DIR}/poisson3d_distributed.py}"
NX="${NX:-1024}"
NY="${NY:-1024}"
NZ="${NZ:-128}"
DTYPE="${DTYPE:-float64}"
TRIDIAG="${TRIDIAG:-pcr}"        # pcr = log2(nz) vectorized steps; thomas = sequential scan A/B
THOMAS_CHUNK="${THOMAS_CHUNK:-16}" # BW candidate; 1 restores row-at-a-time scans
METHOD="${METHOD:-transpose}"    # transpose, spike, or spike-adaptive
SPIKE_INTERFACE_COLLECTIVE="${SPIKE_INTERFACE_COLLECTIVE:-alltoall}" # plain spike only
SPIKE_INTERFACE_SOLVER="${SPIKE_INTERFACE_SOLVER:-selected-rows}" # plain spike only
PIPELINE_EXECUTION="${PIPELINE_EXECUTION:-staged}" # staged avoids the observed ROCm monolithic penalty
DATA_LAYOUT="${DATA_LAYOUT:-z-first}" # contiguous x/y FFTs and leading-z Thomas scans
MMS="${MMS:-0}"                  # 1 = manufactured-solution validation (adds MMS error to summary)
MMS_KIND="${MMS_KIND:-broadband}" # broadband = full-spectrum random-phase MMS; modes = 5 eigenmodes
WARMUP="${WARMUP:-3}"
SAMPLES="${SAMPLES:-10}"
ITERATIONS="${ITERATIONS:-5}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"
STEP_CPUS="${STEP_CPUS:-${SLURM_CPUS_PER_TASK:-32}}"
TASK_MODE="${TASK_MODE:-single}" # single = one process drives all DCUs; per-gpu = one process per DCU
RCCL_DEBUG="${RCCL_DEBUG:-0}"
CONDA_ENV="${CONDA_ENV:-jax060}"
RESULT_DIR="${RESULT_DIR:-${SLURM_SUBMIT_DIR}/poisson3d_results_${SLURM_JOB_ID}}"

if [[ ! -f "${BENCHMARK}" ]]; then
    echo "ERROR: benchmark not found: ${BENCHMARK}" >&2
    exit 2
fi

module purge
module load compiler/dtk/26.04
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="/public/software/compiler/dtk-26.04/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}"
MPI_LIBDIR="$(mpicc --showme:libdirs | awk '{print $1}')"
export LD_PRELOAD="${MPI_LIBDIR}/libmpi.so${LD_PRELOAD:+:${LD_PRELOAD}}"

export JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${RESULT_DIR}"

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Benchmark    : ${BENCHMARK}"
echo "Preset       : ${PRESET}"
echo "Python       : $(command -v python)"
echo "Configuration: grid=${NX}x${NY}x${NZ}, dtype=${DTYPE}, tridiag=${TRIDIAG}, thomas_chunk=${THOMAS_CHUNK}, method=${METHOD}, spike_collective=${SPIKE_INTERFACE_COLLECTIVE}, spike_interface_solver=${SPIKE_INTERFACE_SOLVER}, pipeline=${PIPELINE_EXECUTION}, layout=${DATA_LAYOUT}, mms=${MMS}, task_mode=${TASK_MODE}"
echo "GPU counts   : ${GPU_COUNTS}"
echo "CPUs per step: ${STEP_CPUS}"
echo "Results      : ${RESULT_DIR}"
echo

scontrol show job "${SLURM_JOB_ID}" > "${RESULT_DIR}/slurm_job.txt"
(rocm-smi --showtopo || hy-smi --showtopo || echo "no smi topology tool found") \
    > "${RESULT_DIR}/topology.txt" 2>&1 || true

for ngpu in ${GPU_COUNTS}; do
    if (( ngpu < 1 || ngpu > 8 )); then
        echo "ERROR: GPU_COUNTS contains unsupported value: ${ngpu}" >&2
        exit 2
    fi

    if [[ "${RCCL_DEBUG}" == "1" ]]; then
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
        export NCCL_DEBUG_FILE="${RESULT_DIR}/rccl_p${ngpu}.%h.%p.log"
    fi

    bench_args=(
        --platform rocm
        --nx "${NX}"
        --ny "${NY}"
        --nz "${NZ}"
        --dtype "${DTYPE}"
        --tridiag "${TRIDIAG}"
        --thomas-chunk "${THOMAS_CHUNK}"
        --method "${METHOD}"
        --spike-interface-collective "${SPIKE_INTERFACE_COLLECTIVE}"
        --spike-interface-solver "${SPIKE_INTERFACE_SOLVER}"
        --pipeline-execution "${PIPELINE_EXECUTION}"
        --data-layout "${DATA_LAYOUT}"
        --warmup "${WARMUP}"
        --samples "${SAMPLES}"
        --iterations "${ITERATIONS}"
    )
    if [[ "${MMS}" == "1" ]]; then
        bench_args+=( --mms --mms-kind "${MMS_KIND}" )
    fi
    if [[ "${TASK_MODE}" == "per-gpu" ]]; then
        step_cpus_per_task=$(( STEP_CPUS / ngpu ))
        (( step_cpus_per_task >= 1 )) || step_cpus_per_task=1
        step_flags=(
            --ntasks="${ngpu}"
            --cpus-per-task="${step_cpus_per_task}"
            --cpu-bind=cores
            --mem=0
        )
        bench_args+=( --distributed )
    else
        step_flags=(
            --ntasks=1
            --cpus-per-task="${STEP_CPUS}"
            --tres-per-task="gres/dcu:${ngpu}"
        )
    fi

    log="${RESULT_DIR}/poisson_${NX}x${NY}x${NZ}_${DTYPE}_${TRIDIAG}_tc${THOMAS_CHUNK}_${METHOD}_${SPIKE_INTERFACE_COLLECTIVE}_${SPIKE_INTERFACE_SOLVER}_${PIPELINE_EXECUTION}_${DATA_LAYOUT}_p${ngpu}_${TASK_MODE}.log"
    echo "===== Starting ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="

    srun "${step_flags[@]}" --kill-on-bad-exit=1 \
        python "${BENCHMARK}" "${bench_args[@]}" \
        2>&1 | tee "${log}"

    echo "===== Finished ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="
    echo
done

echo "All runs completed successfully."
echo "Results: ${RESULT_DIR}"
