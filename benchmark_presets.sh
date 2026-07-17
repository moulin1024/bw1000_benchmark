#!/usr/bin/env bash
#
# Measured platform presets shared by the Slurm benchmark drivers.
# Environment variables set by the caller take precedence over preset values.

apply_poisson_benchmark_preset() {
    local preset="${1:?preset name is required}"

    case "${preset}" in
        mn5-cuda)
            : "${NX:=1024}"
            : "${NY:=1024}"
            : "${NZ:=1024}"
            : "${DTYPE:=float64}"
            : "${METHOD:=spike}"
            : "${TRIDIAG:=thomas}"
            : "${THOMAS_CHUNK:=1}"
            : "${SPIKE_INTERFACE_COLLECTIVE:=allgather}"
            : "${SPIKE_INTERFACE_SOLVER:=selected-rows}"
            : "${PIPELINE_EXECUTION:=staged}"
            : "${DATA_LAYOUT:=z-first}"
            : "${GPU_COUNTS:=4}"
            : "${TASK_MODE:=single}"
            : "${WARMUP:=5}"
            : "${SAMPLES:=20}"
            : "${ITERATIONS:=10}"
            ;;
        dcu-rocm)
            : "${NX:=1024}"
            : "${NY:=1024}"
            : "${NZ:=1024}"
            : "${DTYPE:=float64}"
            : "${METHOD:=spike}"
            : "${TRIDIAG:=thomas}"
            : "${THOMAS_CHUNK:=16}"
            : "${SPIKE_INTERFACE_COLLECTIVE:=allgather}"
            : "${SPIKE_INTERFACE_SOLVER:=selected-rows}"
            : "${PIPELINE_EXECUTION:=staged}"
            : "${DATA_LAYOUT:=z-first}"
            : "${GPU_COUNTS:=8}"
            : "${TASK_MODE:=single}"
            : "${WARMUP:=5}"
            : "${SAMPLES:=20}"
            : "${ITERATIONS:=10}"
            ;;
        none)
            ;;
        *)
            echo "ERROR: unknown PRESET=${preset}; use mn5-cuda, dcu-rocm, or none" >&2
            return 2
            ;;
    esac
}
