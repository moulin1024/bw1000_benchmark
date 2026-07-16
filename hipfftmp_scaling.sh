#!/usr/bin/env bash
#SBATCH --job-name=hipfftmp-scale
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:8
#SBATCH --time=00:30:00
#SBATCH --output=hipfftmp-scale-%j.out
#SBATCH --error=hipfftmp-scale-%j.err

# hipFFT-Mp counterpart of scaling.sh: same problem (16384^2 complex128),
# same timing protocol, one MPI rank per DCU. Build first: ./build_hipfftmp.sh
#
# Overrides:
#   sbatch --export=ALL,SIZE=32768 hipfftmp_scaling.sh
#   sbatch --export=ALL,MPI_KIND=pmi2 hipfftmp_scaling.sh   # if pmix unsupported

set -euo pipefail

BINARY="${BINARY:-${SLURM_SUBMIT_DIR}/hipfftmp_benchmark}"
SIZE="${SIZE:-16384}"
WARMUP="${WARMUP:-3}"
SAMPLES="${SAMPLES:-10}"
ITERATIONS="${ITERATIONS:-5}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"
STEP_CPUS="${STEP_CPUS:-${SLURM_CPUS_PER_TASK:-32}}"
MPI_KIND="${MPI_KIND:-pmix}"     # check `srun --mpi=list`; fallback: pmi2 or mpirun
RESULT_DIR="${RESULT_DIR:-${SLURM_SUBMIT_DIR}/hipfftmp_results_${SLURM_JOB_ID}}"

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found or not executable: ${BINARY} (run ./build_hipfftmp.sh)" >&2
    exit 2
fi

module purge
module load compiler/dtk/26.04
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0

export LD_LIBRARY_PATH="/public/software/compiler/dtk-26.04/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RESULT_DIR}"

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Binary       : ${BINARY}"
echo "Configuration: size=${SIZE}, dtype=complex128 (Z2Z)"
echo "GPU counts   : ${GPU_COUNTS}"
echo "CPUs per step: ${STEP_CPUS}"
echo "MPI launcher : srun --mpi=${MPI_KIND}"
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

    step_cpus_per_task=$(( STEP_CPUS / ngpu ))
    (( step_cpus_per_task >= 1 )) || step_cpus_per_task=1

    log="${RESULT_DIR}/hipfftmp_${SIZE}_p${ngpu}.log"
    echo "===== Starting ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="

    # No per-task GRES binding, same lesson as the JAX per-gpu mode: every
    # rank must see all node DCUs (it claims device = node-local rank), or
    # cross-process IPC fails. --mem=0 pools the job's memory for the step.
    srun --mpi="${MPI_KIND}" \
        --ntasks="${ngpu}" \
        --cpus-per-task="${step_cpus_per_task}" \
        --cpu-bind=cores \
        --mem=0 \
        --kill-on-bad-exit=1 \
        "${BINARY}" \
            --size "${SIZE}" \
            --warmup "${WARMUP}" \
            --samples "${SAMPLES}" \
            --iterations "${ITERATIONS}" \
        2>&1 | tee "${log}"
    # If srun --mpi=${MPI_KIND} fails to bootstrap MPI, comment the srun block
    # and use OpenMPI's launcher inside the allocation instead:
    #   mpirun -np "${ngpu}" --bind-to core "${BINARY}" --size "${SIZE}" \
    #       --warmup "${WARMUP}" --samples "${SAMPLES}" --iterations "${ITERATIONS}" \
    #       2>&1 | tee "${log}"

    echo "===== Finished ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="
    echo
done

echo "All runs completed successfully."
echo "Results: ${RESULT_DIR}"
