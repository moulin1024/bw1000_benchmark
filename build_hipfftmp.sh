#!/usr/bin/env bash
# Build hipfftmp_benchmark on the cluster (run on a login or compute node).
#
#   ./build_hipfftmp.sh
#
# If compilation fails on hipfftMp names, look at the probe output below for
# the actual header/library spellings in your DTK and adjust the include in
# hipfftmp_benchmark.cpp and/or FFT_LIBS here.
set -euo pipefail

module purge
module load compiler/dtk/26.04
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0

DTK_ROOT="${DTK_ROOT:-/public/software/compiler/dtk-26.04}"
SRC="${SRC:-hipfftmp_benchmark.cpp}"
OUT="${OUT:-hipfftmp_benchmark}"
# Adjust after checking the probe, e.g. "-lhipfft -lhipfftmp" or just "-lhipfft"
# if the Mp symbols live inside libhipfft itself.
FFT_LIBS="${FFT_LIBS:--lhipfft -lhipfftMp}"

echo "== probe: hipFFT headers and libraries in ${DTK_ROOT} =="
find "${DTK_ROOT}" \( -iname '*hipfft*' -o -iname '*rocfft*mp*' \) \
    \( -name '*.h' -o -name '*.hpp' -o -name '*.so*' \) 2>/dev/null | sort || true
echo "== end probe =="
echo

MPI_CFLAGS="$(mpicc --showme:compile)"
MPI_LDFLAGS="$(mpicc --showme:link)"

set -x
hipcc -O3 -std=c++17 "${SRC}" -o "${OUT}" \
    ${MPI_CFLAGS} ${MPI_LDFLAGS} ${FFT_LIBS}
set +x

echo "built ./${OUT}"
