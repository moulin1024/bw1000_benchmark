#!/bin/bash -l
#
# MN5 single-node scaling run for poisson3d_distributed.py.
#
# Submit from the directory containing poisson3d_distributed.py:
#   sbatch submit_mn5.sh
#
# Runtime overrides:
#   sbatch --export=ALL,PRESET=mn5-cuda submit_mn5.sh
#   sbatch --export=ALL,PRESET=none submit_mn5.sh
#   sbatch --export=ALL,NX=2048,NY=2048,NZ=256 submit_mn5.sh
#   sbatch --export=ALL,GPU_COUNTS="1 2 4",WARMUP=1,SAMPLES=3,ITERATIONS=2 submit_mn5.sh
#   sbatch --export=ALL,SPIKE_INTERFACE_COLLECTIVE=alltoall submit_mn5.sh
#   sbatch --export=ALL,SPIKE_INTERFACE_SOLVER=block-thomas submit_mn5.sh
#   sbatch --export=ALL,SPIKE_INTERFACE_SOLVER=dense submit_mn5.sh
#   sbatch --export=ALL,PIPELINE_EXECUTION=staged submit_mn5.sh
#   sbatch --export=ALL,DATA_LAYOUT=z-first submit_mn5.sh
#   sbatch --export=ALL,THOMAS_CHUNK=16 submit_mn5.sh
#   sbatch --export=ALL,METHOD=spike-adaptive,TRIDIAG=thomas submit_mn5.sh
#
#SBATCH --job-name=poisson3d-scale
#SBATCH --account=ehpc537
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=00:30:00
#SBATCH --output=poisson3d-mn5.%j.out
#SBATCH --error=poisson3d-mn5.%j.err

set -euo pipefail

PRESET="${PRESET:-mn5-cuda}"
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
NZ="${NZ:-1024}"
DTYPE="${DTYPE:-float64}"
TRIDIAG="${TRIDIAG:-pcr}"
THOMAS_CHUNK="${THOMAS_CHUNK:-1}"
METHOD="${METHOD:-spike}"
SPIKE_INTERFACE_COLLECTIVE="${SPIKE_INTERFACE_COLLECTIVE:-allgather}"
SPIKE_INTERFACE_SOLVER="${SPIKE_INTERFACE_SOLVER:-selected-rows}"
PIPELINE_EXECUTION="${PIPELINE_EXECUTION:-monolithic}"
DATA_LAYOUT="${DATA_LAYOUT:-xyz}"
MMS="${MMS:-0}"
WARMUP="${WARMUP:-3}"
SAMPLES="${SAMPLES:-10}"
ITERATIONS="${ITERATIONS:-5}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-20}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/mpcd/mpcd549688/venvs/jax-cupy-cuda12/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-${SLURM_SUBMIT_DIR}/poisson3d_mn5_results_${SLURM_JOB_ID}}"

if [[ ! -f "${BENCHMARK}" ]]; then
    echo "ERROR: benchmark not found: ${BENCHMARK}" >&2
    exit 2
fi

module purge
module load anaconda/2023.07

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    echo "ERROR: virtual-environment activation script not found: ${VENV_ACTIVATE}" >&2
    exit 2
fi
source "${VENV_ACTIVATE}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${RESULT_DIR}"

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Benchmark    : ${BENCHMARK}"
echo "Preset       : ${PRESET}"
echo "Python       : $(command -v "${PYTHON_BIN}")"
echo "Virtual env  : ${VIRTUAL_ENV:-${VENV_ACTIVATE}}"
echo "Configuration: grid=${NX}x${NY}x${NZ}, dtype=${DTYPE}, tridiag=${TRIDIAG}, thomas_chunk=${THOMAS_CHUNK}, method=${METHOD}, spike_collective=${SPIKE_INTERFACE_COLLECTIVE}, spike_interface_solver=${SPIKE_INTERFACE_SOLVER}, pipeline=${PIPELINE_EXECUTION}, layout=${DATA_LAYOUT}, mms=${MMS}"
echo "GPU counts   : ${GPU_COUNTS}"
echo "Results      : ${RESULT_DIR}"
echo

scontrol show job "${SLURM_JOB_ID}" > "${RESULT_DIR}/slurm_job.txt"
nvidia-smi -L > "${RESULT_DIR}/nvidia_smi.txt" 2>&1 || true
nvidia-smi topo -m > "${RESULT_DIR}/topology.txt" 2>&1 || true

for ngpu in ${GPU_COUNTS}; do
    if (( ngpu != 1 && ngpu != 2 && ngpu != 4 )); then
        echo "ERROR: GPU_COUNTS contains unsupported value: ${ngpu}; use 1, 2, or 4" >&2
        exit 2
    fi

    step_cpus=$(( CPUS_PER_GPU * ngpu ))
    bench_args=(
        --platform cuda
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
        bench_args+=( --mms )
    fi

    log="${RESULT_DIR}/poisson_${NX}x${NY}x${NZ}_${DTYPE}_${TRIDIAG}_tc${THOMAS_CHUNK}_${METHOD}_${SPIKE_INTERFACE_COLLECTIVE}_${SPIKE_INTERFACE_SOLVER}_${PIPELINE_EXECUTION}_${DATA_LAYOUT}_p${ngpu}.log"
    echo "===== Starting ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="

    srun \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${step_cpus}" \
        --gres="gpu:${ngpu}" \
        --cpu-bind=cores \
        --kill-on-bad-exit=1 \
        "${PYTHON_BIN}" "${BENCHMARK}" "${bench_args[@]}" \
        2>&1 | tee "${log}"

    echo "===== Finished ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="
    echo
done

echo "All MN5 runs completed successfully."
echo "Results: ${RESULT_DIR}"
