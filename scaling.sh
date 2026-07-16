#!/usr/bin/env bash
#SBATCH --job-name=jax-fft-scale
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:8
# NOTE: this cluster ties memory to CPUs (at most DefMemPerCPU per requested
# CPU) and rejects --mem requests beyond that, so memory scales only through
# --cpus-per-task. per-gpu mode runs 8 ROCm processes and their 1-CPU shares
# get OOM-killed (job 686753): submit it with e.g. --cpus-per-task=32.
#SBATCH --time=00:30:00
#SBATCH --output=jax-fft-scale-%j.out
#SBATCH --error=jax-fft-scale-%j.err

set -euo pipefail

# Override these at submission time with, for example:
#   sbatch scaling.sh                                    # full single-process scaling
#   sbatch --export=ALL,SIZE=32768,DTYPE=complex64 scaling.sh
#   sbatch --export=ALL,RCCL_DEBUG=1,GPU_COUNTS=8,WARMUP=0,SAMPLES=1,ITERATIONS=1 scaling.sh
#   sbatch --cpus-per-task=32 --export=ALL,TASK_MODE=per-gpu scaling.sh   # full per-gpu scaling
BENCHMARK="${BENCHMARK:-${SLURM_SUBMIT_DIR}/jax_distributed_fft2d_benchmark_v4.py}"
SIZE="${SIZE:-16384}"
BATCH="${BATCH:-1}"              # independent N x N fields per transform
DECOMP="${DECOMP:-slab}"         # slab = split rows (all-to-all); batch = whole fields per GPU (no comm)
DTYPE="${DTYPE:-complex128}"
LAYOUT="${LAYOUT:-optimized}"
PIPELINE="${PIPELINE:-compare}"
WARMUP="${WARMUP:-3}"
SAMPLES="${SAMPLES:-10}"
ITERATIONS="${ITERATIONS:-5}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"
STEP_CPUS="${STEP_CPUS:-${SLURM_CPUS_PER_TASK:-8}}"  # follows sbatch --cpus-per-task
TRANSPOSE="${TRANSPOSE:-auto}"   # auto = pairs on 1 GPU, native on multi-GPU (jobs 686701/686709)
CHUNKS="${CHUNKS:-1}"            # chunks>1 lost 20-36% on jax 0.6.0/ROCm (job 686717)
TASK_MODE="${TASK_MODE:-single}" # single = one process drives all DCUs; per-gpu = one process per DCU
RCCL_DEBUG="${RCCL_DEBUG:-0}"    # 1 = write RCCL transport logs into RESULT_DIR
CONDA_ENV="${CONDA_ENV:-jax060}"
RESULT_DIR="${RESULT_DIR:-${SLURM_SUBMIT_DIR}/jax_fft_results_${SLURM_JOB_ID}}"

if [[ ! -f "${BENCHMARK}" ]]; then
    echo "ERROR: benchmark not found: ${BENCHMARK}" >&2
    exit 2
fi

module purge
module load compiler/dtk/26.04
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# The JAX ROCm plugin requires gcvm from DTK. hipFFT-Mp also references
# OpenMPI symbols (for example ompi_mpi_int), so preload the matching libmpi.
export LD_LIBRARY_PATH="/public/software/compiler/dtk-26.04/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}"
MPI_LIBDIR="$(mpicc --showme:libdirs | awk '{print $1}')"
export LD_PRELOAD="${MPI_LIBDIR}/libmpi.so${LD_PRELOAD:+:${LD_PRELOAD}}"

export JAX_PLATFORMS=rocm
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${RESULT_DIR}"

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Benchmark    : ${BENCHMARK}"
echo "Python       : $(command -v python)"
echo "Configuration: size=${SIZE}, batch=${BATCH}, decomp=${DECOMP}, dtype=${DTYPE}, layout=${LAYOUT}, pipeline=${PIPELINE}"
echo "Exchange     : transpose=${TRANSPOSE}, chunks=${CHUNKS}, task_mode=${TASK_MODE}, rccl_debug=${RCCL_DEBUG}"
echo "GPU counts   : ${GPU_COUNTS}"
echo "CPUs per step: ${STEP_CPUS} (kept fixed so every step receives full host memory)"
echo "Results      : ${RESULT_DIR}"
echo

scontrol show job "${SLURM_JOB_ID}" > "${RESULT_DIR}/slurm_job.txt"
module list 2> "${RESULT_DIR}/modules.txt"
conda list > "${RESULT_DIR}/conda_packages.txt"
# Link-type matrix (XGMI vs PCIe) and NUMA affinity, for reading next to the
# RCCL transport log. Tool name differs across ROCm/DTK versions.
(rocm-smi --showtopo || hy-smi --showtopo || echo "no smi topology tool found") \
    > "${RESULT_DIR}/topology.txt" 2>&1 || true

for ngpu in ${GPU_COUNTS}; do
    if (( ngpu < 1 || ngpu > 8 )); then
        echo "ERROR: GPU_COUNTS contains unsupported value: ${ngpu}" >&2
        exit 2
    fi

    # pairs only wins the isolated single-GPU transpose (-18% total); the
    # multi-GPU exchange is communication-bound and pairs adds a pass there.
    step_transpose="${TRANSPOSE}"
    if [[ "${step_transpose}" == "auto" ]]; then
        if (( ngpu == 1 )); then step_transpose=pairs; else step_transpose=native; fi
    fi

    if [[ "${RCCL_DEBUG}" == "1" ]]; then
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
        export NCCL_DEBUG_FILE="${RESULT_DIR}/rccl_p${ngpu}.%h.%p.log"
    fi

    bench_args=(
        --platform rocm
        --layout "${LAYOUT}"
        --pipeline "${PIPELINE}"
        --size "${SIZE}"
        --batch "${BATCH}"
        --decomp "${DECOMP}"
        --dtype "${DTYPE}"
        --warmup "${WARMUP}"
        --samples "${SAMPLES}"
        --iterations "${ITERATIONS}"
        --transpose "${step_transpose}"
        --chunks "${CHUNKS}"
    )
    if [[ "${TASK_MODE}" == "per-gpu" ]]; then
        # One process per DCU. Deliberately NO per-task GRES binding: per-task
        # cgroups hide peer devices and cross-process RCCL IPC then dies with
        # 'invalid device pointer' (job 686746). Tasks inherit the job's full
        # DCU visibility and the benchmark claims device SLURM_LOCALID.
        step_cpus_per_task=$(( STEP_CPUS / ngpu ))
        (( step_cpus_per_task >= 1 )) || step_cpus_per_task=1
        # --mem=0 gives the step the whole allocation's memory instead of
        # slicing it per task by CPU share (1-CPU shares OOM, job 686753).
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

    log="${RESULT_DIR}/fft_${SIZE}_b${BATCH}_${DECOMP}_${DTYPE}_p${ngpu}_${step_transpose}_c${CHUNKS}_${TASK_MODE}_${PIPELINE}.log"
    echo "===== Starting ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="

    srun "${step_flags[@]}" --kill-on-bad-exit=1 \
        python "${BENCHMARK}" "${bench_args[@]}" \
        2>&1 | tee "${log}"

    echo "===== Finished ${ngpu}-GPU run: $(date --iso-8601=seconds) ====="
    echo
done

echo "All runs completed successfully."
echo "Results: ${RESULT_DIR}"

