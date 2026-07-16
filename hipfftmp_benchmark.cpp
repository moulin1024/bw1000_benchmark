// hipfftmp_benchmark.cpp
//
// Distributed 2D complex-to-complex FFT benchmark on hipFFT-Mp (the
// cuFFTMp-parity multi-process FFT library shipped with DTK), one MPI rank
// per DCU/GPU. Prints the same metrics as jax_distributed_fft2d_benchmark
// so the two are directly comparable:
//
//   input : natural X slabs, (N/P) x N complex128 per rank
//   output: library-chosen shuffled Y-slab distribution (the analogue of the
//           JAX benchmark's (ky, kx) spectral slabs)
//
// Timing runs forward+inverse pairs and reports per-transform time (both
// directions cost the same); the device data is re-uploaded between samples
// so the unnormalized transforms cannot overflow inside a timed region.
// Round-trip validation runs before any timing.
//
// Build: ./build_hipfftmp.sh  (its probe lists the actual DTK header/library
// names if the hipfftMp spellings below need adjusting).

#include <mpi.h>

#include <hip/hip_runtime.h>

#if __has_include(<hipfft/hipfft.h>)
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif

#if __has_include(<hipfft/hipfftXt.h>)
#include <hipfft/hipfftXt.h>
#elif __has_include(<hipfftXt.h>)
#include <hipfftXt.h>
#endif

// Multi-process API header: DTK versions differ in spelling.
#if __has_include(<hipfft/hipfftMp.h>)
#include <hipfft/hipfftMp.h>
#elif __has_include(<hipfftMp.h>)
#include <hipfftMp.h>
#elif __has_include(<hipfft/hipfftmp.h>)
#include <hipfft/hipfftmp.h>
#elif __has_include(<hipfftmp.h>)
#include <hipfftmp.h>
#else
#error "hipFFT-Mp header not found; run the probe in build_hipfftmp.sh and adjust this include"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                              \
            std::fprintf(stderr, "HIP error '%s' at %s:%d\n",                  \
                         hipGetErrorString(err_), __FILE__, __LINE__);         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#define FFT_CHECK(call)                                                        \
    do {                                                                       \
        hipfftResult res_ = (call);                                            \
        if (res_ != HIPFFT_SUCCESS) {                                          \
            std::fprintf(stderr, "hipFFT error %d at %s:%d\n", (int)res_,      \
                         __FILE__, __LINE__);                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

static double global_max(double value) {
    double result = value;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
}

static double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double pos = (values.size() - 1) * q;
    const size_t lo = (size_t)std::floor(pos);
    const size_t hi = (size_t)std::ceil(pos);
    if (lo == hi) return values[lo];
    return values[lo] * (hi - pos) + values[hi] * (pos - lo);
}

static std::string human_bytes(double n) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int u = 0;
    while (n >= 1024.0 && u < 4) { n /= 1024.0; ++u; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", n, units[u]);
    return buf;
}

static hipfftResult exec_desc(hipfftHandle plan, hipLibXtDesc* desc, int direction) {
    // Some hipFFT builds only ship the typed variant; if the generic call is
    // missing, swap in: return hipfftXtExecDescriptorZ2Z(plan, desc, desc, direction);
    return hipfftXtExecDescriptor(plan, desc, desc, direction);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    long n = 16384;
    int warmup = 3, samples = 10, iterations = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char* name) -> long {
            if (i + 1 >= argc) {
                if (rank == 0) std::fprintf(stderr, "missing value for %s\n", name);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            return std::atol(argv[++i]);
        };
        if (arg == "--size") n = next_value("--size");
        else if (arg == "--warmup") warmup = (int)next_value("--warmup");
        else if (arg == "--samples") samples = (int)next_value("--samples");
        else if (arg == "--iterations") iterations = (int)next_value("--iterations");
        else {
            if (rank == 0) std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    if (n <= 0 || n % nranks != 0) {
        if (rank == 0)
            std::fprintf(stderr, "size %ld must be positive and divisible by %d ranks\n", n, nranks);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (warmup < 0 || samples <= 0 || iterations <= 0) {
        if (rank == 0) std::fprintf(stderr, "bad warmup/samples/iterations\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // Claim the DCU matching the node-local rank. Every rank must see all
    // node devices (launch WITHOUT per-task GRES binding, like the JAX
    // per-gpu mode) or cross-process IPC may fail.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "rank %d sees no HIP devices\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    HIP_CHECK(hipSetDevice(local_rank % device_count));

    const long rows = n / nranks;
    const size_t elems = (size_t)rows * (size_t)n;
    const double global_bytes = (double)n * (double)n * sizeof(hipfftDoubleComplex);
    const double local_bytes = global_bytes / nranks;

    std::vector<hipfftDoubleComplex> input(elems), roundtrip(elems);
    srand48(1234 + rank);
    for (size_t i = 0; i < elems; ++i) {
        const double r = drand48();
        input[i].x = r;
        input[i].y = 0.5 * r;
    }

    // Plan: create, attach the MPI communicator, then plan the global 2D FFT.
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    hipfftHandle plan;
    FFT_CHECK(hipfftCreate(&plan));
    MPI_Comm fft_comm = MPI_COMM_WORLD;
    FFT_CHECK(hipfftMpAttachComm(plan, HIPFFT_COMM_MPI, &fft_comm));
    size_t workspace = 0;
    FFT_CHECK(hipfftMakePlan2d(plan, (int)n, (int)n, HIPFFT_Z2Z, &workspace));
    const double plan_s = global_max(MPI_Wtime() - t0);

    // Library-managed distributed descriptor memory.
    hipLibXtDesc* desc = nullptr;
    FFT_CHECK(hipfftXtMalloc(plan, &desc, HIPFFT_XT_FORMAT_INPLACE));

    // Validation before timing: forward + inverse, unnormalized => x * N^2.
    FFT_CHECK(hipfftXtMemcpy(plan, desc, input.data(), HIPFFT_COPY_HOST_TO_DEVICE));
    HIP_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    FFT_CHECK(exec_desc(plan, desc, HIPFFT_FORWARD));
    HIP_CHECK(hipDeviceSynchronize());
    const double first_s = global_max(MPI_Wtime() - t0);
    FFT_CHECK(exec_desc(plan, desc, HIPFFT_BACKWARD));
    HIP_CHECK(hipDeviceSynchronize());
    FFT_CHECK(hipfftXtMemcpy(plan, roundtrip.data(), desc, HIPFFT_COPY_DEVICE_TO_HOST));
    const double scale = (double)n * (double)n;
    double max_err = 0.0, max_mag = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        max_err = std::max(max_err, std::hypot(roundtrip[i].x / scale - input[i].x,
                                               roundtrip[i].y / scale - input[i].y));
        max_mag = std::max(max_mag, std::hypot(input[i].x, input[i].y));
    }
    const double rel_err = global_max(max_err) / std::max(global_max(max_mag), 1e-300);

    auto run_pairs = [&](int count) {
        for (int i = 0; i < count; ++i) {
            FFT_CHECK(exec_desc(plan, desc, HIPFFT_FORWARD));
            FFT_CHECK(exec_desc(plan, desc, HIPFFT_BACKWARD));
        }
        HIP_CHECK(hipDeviceSynchronize());
    };

    FFT_CHECK(hipfftXtMemcpy(plan, desc, input.data(), HIPFFT_COPY_HOST_TO_DEVICE));
    HIP_CHECK(hipDeviceSynchronize());
    run_pairs(warmup);

    std::vector<double> times(samples);
    for (int s = 0; s < samples; ++s) {
        // Untimed refresh: keeps unnormalized magnitude growth (x N^2 per
        // pair) bounded within one timed block.
        FFT_CHECK(hipfftXtMemcpy(plan, desc, input.data(), HIPFFT_COPY_HOST_TO_DEVICE));
        HIP_CHECK(hipDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        run_pairs(iterations);
        times[s] = global_max((MPI_Wtime() - t0) / (2.0 * iterations));
    }

    if (rank == 0) {
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted.size() % 2
                                  ? sorted[sorted.size() / 2]
                                  : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
        double mean = 0.0;
        for (double t : times) mean += t;
        mean /= times.size();
        const double flops = 5.0 * (double)n * (double)n * std::log2((double)n * (double)n);
        const double remote_payload = global_bytes * (nranks - 1) / nranks;

        std::printf("hipFFT-Mp distributed 2D FFT benchmark\n");
        std::printf("Ranks (1 per DCU)       : %d\n", nranks);
        std::printf("Global shape            : %ld x %ld\n", n, n);
        std::printf("Input row slab / rank   : %ld x %ld\n", rows, n);
        std::printf("Dtype                   : complex128 (Z2Z)\n");
        std::printf("Global / local data     : %s / %s\n",
                    human_bytes(global_bytes).c_str(), human_bytes(local_bytes).c_str());
        std::printf("Workspace / rank        : %s\n", human_bytes((double)workspace).c_str());
        std::printf("Warmup / timing         : %d / %d x %d (fwd+inv pairs)\n\n",
                    warmup, samples, iterations);
        std::printf("Plan + comm attach      : %.3f ms\n", plan_s * 1e3);
        std::printf("First forward exec      : %.3f ms\n\n", first_s * 1e3);
        std::printf("Per-transform time\n");
        std::printf("  median                 : %.3f ms\n", median * 1e3);
        std::printf("  best                   : %.3f ms\n", sorted.front() * 1e3);
        std::printf("  mean / p95             : %.3f / %.3f ms\n\n",
                    mean * 1e3, percentile(times, 0.95) * 1e3);
        std::printf("Summary\n");
        std::printf("  global transforms/s          : %.3f\n", 1.0 / median);
        std::printf("  estimated FFT rate           : %.3f GFLOP/s\n", flops / median / 1e9);
        std::printf("  round-trip relative max error: %.3e\n", rel_err);
        if (nranks > 1) {
            std::printf("  remote payload per transpose : %s\n", human_bytes(remote_payload).c_str());
        } else {
            std::printf("  remote payload per transpose : 0 B (single rank)\n");
        }
    }

    FFT_CHECK(hipfftXtFree(desc));
    FFT_CHECK(hipfftDestroy(plan));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
