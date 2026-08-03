// GPU matrix-product benchmark.
//
// It exists separately from bench.cpp for a practical reason: profiling with
// Nsight Compute needs an executable that launches **one** kernel on **one**
// shape. Handing the whole benchmark to ncu means waiting for it to profile
// dozens of unrelated launches to read one.
//
//   bench_matmul                                    # sweep everything, print the table
//   bench_matmul --kernel=register --size=2048      # one variant, one shape
//   ncu --set full -o profile bench_matmul --kernel=register --size=2048 --iters=10
//
// The figures it prints are the ones that go into docs/CUDA.md. None of them is
// written by hand in the documentation.

#include "engine/cuda.hpp"
#include "engine/random.hpp"
#include "engine/tensor.hpp"

#ifdef ENGINE_BENCH_CUBLAS
#include <cublas_v2.h>

#include "engine/detail/storage.hpp"
#include "engine/detail/tensor_impl.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using engine::Tensor;
namespace cuda = engine::cuda;

namespace {

struct Options {
    cuda::MatmulKernel kernel = cuda::MatmulKernel::Auto;
    bool sweep = true;    // with no arguments, sweep everything
    bool cublas = false;  // --kernel=cublas: the reference row, on its own
    size_t size = 2048;
    size_t iters = 0;  // 0 = by time, not by repetition count
};

bool parse_kernel(const std::string& value, cuda::MatmulKernel& out) {
    if (value == "auto") {
        out = cuda::MatmulKernel::Auto;
        return true;
    }
    if (value == "naive") {
        out = cuda::MatmulKernel::Naive;
        return true;
    }
    if (value == "tiled") {
        out = cuda::MatmulKernel::Tiled;
        return true;
    }
    if (value == "register") {
        out = cuda::MatmulKernel::RegisterTiled;
        return true;
    }
    if (value == "vectorized") {
        out = cuda::MatmulKernel::Vectorized;
        return true;
    }
    if (value == "tensorcore") {
        out = cuda::MatmulKernel::TensorCore;
        return true;
    }
    if (value == "tensorcore-fp16" || value == "fp16") {
        out = cuda::MatmulKernel::TensorCoreFp16;
        return true;
    }
    return false;
}

bool parse_args(int argc, char** argv, Options& opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--kernel=cublas") {
            opts.cublas = true;
            opts.sweep = false;
        } else if (arg.rfind("--kernel=", 0) == 0) {
            if (!parse_kernel(arg.substr(9), opts.kernel)) {
                printf("Unknown kernel: %s\n", arg.substr(9).c_str());
                return false;
            }
            opts.sweep = false;
        } else if (arg.rfind("--size=", 0) == 0) {
            opts.size = static_cast<size_t>(std::stoul(arg.substr(7)));
            opts.sweep = false;
        } else if (arg.rfind("--iters=", 0) == 0) {
            opts.iters = static_cast<size_t>(std::stoul(arg.substr(8)));
        } else {
            printf(
                "Usage: bench_matmul\n"
                "  [--kernel=auto|naive|tiled|register|vectorized|tensorcore]\n"
                "  [--size=N] [--iters=N]\n");
            return false;
        }
    }
    return true;
}

// How many timing windows each measurement takes, keeping the best.
//
// This is not caution, it is a bug fix. The first version of the sweep ran one
// window per kernel, back to back, and the numbers it produced were wrong in a
// way that mattered: `vectorized` measured 5259 GFLOP/s at 4096^3 inside the
// sweep and 7903 on its own. The card had simply heated up and dropped its
// clocks by the time the later rows ran, so every row was slower than the one
// above it partly *because* it was below it -- and the ordering of the table was
// deciding its own conclusion.
//
// Taking the best of several windows removes most of it, since thermal noise only
// ever adds time. What it cannot fix is the ordering bias, so the sweep also
// pauses between kernels to let clocks recover.
constexpr int kTimingWindows = 3;

// Returns the shortest observed time per product, in seconds.
//
// The operands are left resident on the device before starting: what is measured
// here is the kernel, not PCIe. The transfer cost is measured in bench.cpp, and
// separately on purpose.
double time_matmul_once(const Tensor& A, const Tensor& B, size_t iters, double min_seconds) {
    { Tensor warm = A.matmul(B); }  // uploads the operands and warms the path
    cuda::synchronize();

    const auto start = std::chrono::steady_clock::now();
    size_t done = 0;
    double elapsed = 0.0;
    while (iters > 0 ? (done < iters) : (elapsed < min_seconds)) {
        Tensor C = A.matmul(B);
        ++done;
        if (iters == 0) {
            cuda::synchronize();
            elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    }
    cuda::synchronize();
    elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return elapsed / static_cast<double>(done);
}

double time_matmul(const Tensor& A, const Tensor& B, size_t iters, double min_seconds) {
    double best = 0.0;
    for (int w = 0; w < kTimingWindows; ++w) {
        const double t = time_matmul_once(A, B, iters, min_seconds);
        if (best == 0.0 || t < best) best = t;
    }
    return best;
}

// Between kernels, so a row is not penalised for the rows above it.
void let_clocks_settle() {
    cuda::synchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
}

void report(const char* label, double seconds, size_t n, double peak) {
    const double gflops = 2.0 * (double)n * n * n / seconds / 1e9;
    const double share = (peak > 0.0) ? (gflops / peak * 100.0) : 0.0;
    printf("  %-14s %10.3f ms %12.1f %11.1f%%\n", label, seconds * 1000.0, gflops, share);
}

void run_one(cuda::MatmulKernel kernel, size_t n, size_t iters, double peak) {
    cuda::set_matmul_kernel(kernel);

    Tensor A = Tensor::randn({n, n});
    Tensor B = Tensor::randn({n, n});

    report(cuda::matmul_kernel_name(kernel), time_matmul(A, B, iters, 0.4), n, peak);
}

#ifdef ENGINE_BENCH_CUBLAS

// The reference row.
//
// cuBLAS is called here and **nowhere else in the project**. The engine's own
// kernels are the point of the exercise; this is the ruler they are held against.
// Leaving it out was worse than losing to it: a table whose only reference is the
// card's theoretical peak invites the reader to assume the gap is small.
//
// cuBLAS is column-major and the engine is row-major. Rather than transpose
// anything, the operands are swapped: in column-major terms, computing
// B_col x A_col with B_col = B_row^T and A_col = A_row^T yields C_col = C_row^T,
// which read back as row-major is exactly C_row. The alternative -- two explicit
// transposes -- would measure the transposes.
double time_cublas_once(const Tensor& A, const Tensor& B, size_t n, size_t iters,
                        double min_seconds) {
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 0.0;

    // Force both operands onto the device through the engine's own Storage, so
    // this row measures the same thing the others do: a kernel over resident
    // data, with no transfer inside the timed region.
    const float* dA = A.storage().device();
    const float* dB = B.storage().device();

    Tensor C({n, n}, 0.0f, false);
    float* dC = C.storage().device_write();

    const int m = static_cast<int>(n);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto gemm = [&] {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha, dB, m, dA, m, &beta, dC, m);
    };

    gemm();  // warm-up: the first call picks a kernel and allocates workspace
    cuda::synchronize();

    const auto start = std::chrono::steady_clock::now();
    size_t done = 0;
    double elapsed = 0.0;
    while (iters > 0 ? (done < iters) : (elapsed < min_seconds)) {
        gemm();
        ++done;
        if (iters == 0) {
            cuda::synchronize();
            elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    }
    cuda::synchronize();
    elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    cublasDestroy(handle);
    return elapsed / static_cast<double>(done);
}

double time_cublas(const Tensor& A, const Tensor& B, size_t n, size_t iters, double min_seconds) {
    double best = 0.0;
    for (int w = 0; w < kTimingWindows; ++w) {
        const double t = time_cublas_once(A, B, n, iters, min_seconds);
        if (t == 0.0) return 0.0;
        if (best == 0.0 || t < best) best = t;
    }
    return best;
}

// Not just timed: checked. A reference row that computed something else would
// make the whole comparison meaningless, and the row/column-major swap above is
// exactly the kind of thing that is easy to get subtly wrong.
bool cublas_agrees(size_t n) {
    Tensor A = Tensor::randn({n, n});
    Tensor B = Tensor::randn({n, n});

    cuda::set_matmul_kernel(cuda::MatmulKernel::Auto);
    const Tensor want = A.matmul(B);
    const std::vector<float> expected = want.to_vector();

    const double seconds = time_cublas_once(A, B, n, 1, 0.0);
    if (seconds == 0.0) return false;

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
    Tensor C({n, n}, 0.0f, false);
    const int m = static_cast<int>(n);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha, B.storage().device(), m,
                A.storage().device(), m, &beta, C.storage().device_write(), m);
    cuda::synchronize();
    cublasDestroy(handle);

    const std::vector<float> got = C.to_vector();

    // The scale is max(1, |expected|), the same one the parity test uses, and the
    // reason matters. Dividing by |expected| alone reports a huge relative error
    // for any element whose true value happens to sit near zero -- and in a sum
    // of n products of N(0,1) values, plenty do. The first version of this check
    // did exactly that and reported 7.2e-03, which looked like a broken
    // reference row and was really just catastrophic cancellation in the divisor.
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double scale = std::max(1.0, std::abs((double)expected[i]));
        worst = std::max(worst, std::abs((double)got[i] - expected[i]) / scale);
    }
    printf("cuBLAS agrees with the engine to %.1e relative error on %zux%zu\n\n", worst, n, n);
    return worst < 1e-4;
}

void run_cublas(size_t n, size_t iters, double peak) {
    Tensor A = Tensor::randn({n, n});
    Tensor B = Tensor::randn({n, n});
    const double seconds = time_cublas(A, B, n, iters, 0.4);
    if (seconds > 0.0) report("cuBLAS", seconds, n, peak);
}

#endif  // ENGINE_BENCH_CUBLAS

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) return 1;

    engine::manual_seed(1);

    if (!cuda::available()) {
        printf(
            "Built without CUDA, or no usable device.\n"
            "  cmake -B build-cuda -S . -DENGINE_CUDA=ON\n");
        return 0;
    }

    const cuda::DeviceInfo info = cuda::device_info();
    const double peak = cuda::peak_fp32_gflops();

    printf("Device: %s (cc %d.%d, %d SM, %zu MiB)\n", info.name.c_str(), info.compute_major,
           info.compute_minor, info.multiprocessors, info.total_memory >> 20);
    printf("Theoretical peak: %.0f GFLOP/s fp32, %.0f GB/s of bandwidth\n", peak,
           cuda::peak_bandwidth_gbs());

    // The roofline model's ridge point: below this arithmetic intensity memory
    // rules, above it the arithmetic does. An NxNxN matmul moves 3*N^2 values to
    // do 2*N^3 operations, that is N/6 FLOP/byte: any N above a few hundred is
    // squarely in the compute-bound region, which is why the work goes into the
    // kernel's arithmetic intensity and not into the transfers.
    const double ridge = peak / cuda::peak_bandwidth_gbs();
    printf("Roofline ridge point: %.1f FLOP/byte\n\n", ridge);

    // No threshold: everything is measured on the GPU here, even where it would
    // not be worth dispatching.
    cuda::set_thresholds(0, 0);

    if (!opts.sweep) {
        const double intensity = (double)opts.size / 6.0;
        printf("Shape %zux%zux%zu, arithmetic intensity %.0f FLOP/byte (%s)\n", opts.size,
               opts.size, opts.size, intensity,
               intensity > ridge ? "compute bound" : "memory bound");
        printf("  %-14s %10s %12s %11s\n", "kernel", "time", "GFLOP/s", "% of peak");
        if (opts.cublas) {
#ifdef ENGINE_BENCH_CUBLAS
            run_cublas(opts.size, opts.iters, peak);
#else
            printf("  built without cuBLAS\n");
#endif
        } else {
            run_one(opts.kernel, opts.size, opts.iters, peak);
        }
        printf("\n");
        return 0;
    }

#ifdef ENGINE_BENCH_CUBLAS
    // Checked once, on a small shape, before any of the timings are believed. A
    // reference row known to compute something else is worse than none at all,
    // so a disagreement removes the row rather than printing a warning above it
    // and carrying on.
    const bool cublas_ok = cublas_agrees(256);
    if (!cublas_ok) {
        printf("cuBLAS disagrees with the engine: the reference row is suppressed.\n\n");
    }
#endif

    // TensorCore only when the hardware has it. It is listed after the fp32
    // kernels rather than among them because it is not measuring the same thing:
    // see the tf32 note below the table.
    std::vector<cuda::MatmulKernel> variants = {
        cuda::MatmulKernel::Naive,
        cuda::MatmulKernel::Tiled,
        cuda::MatmulKernel::RegisterTiled,
        cuda::MatmulKernel::Vectorized,
    };
    if (cuda::tensor_cores_available()) {
        variants.push_back(cuda::MatmulKernel::TensorCore);
        variants.push_back(cuda::MatmulKernel::TensorCoreFp16);
    }

    printf(
        "NOTE: rows within one sweep are NOT comparable to each other. A consumer\n"
        "card throttles under sustained load, so a kernel measured after four\n"
        "others runs at lower clocks and the table's own ordering biases it --\n"
        "measured here at up to 1.6x between a kernel run inside the sweep and the\n"
        "same kernel run alone. For numbers that can be compared, run one process\n"
        "per kernel: tools/bench_matmul_isolated.sh (or .ps1).\n\n");

    for (size_t n : {size_t{512}, size_t{1024}, size_t{2048}, size_t{4096}}) {
        printf("=== %zux%zux%zu (%.0f FLOP/byte) ===\n", n, n, n, (double)n / 6.0);
        printf("  %-14s %10s %12s %11s\n", "kernel", "time", "GFLOP/s", "% of peak");
        for (cuda::MatmulKernel v : variants) {
            // The naive kernel on 4096^3 takes too long for what it adds.
            if (v == cuda::MatmulKernel::Naive && n > 2048) continue;
            let_clocks_settle();
            run_one(v, n, opts.iters, peak);
        }
#ifdef ENGINE_BENCH_CUBLAS
        if (cublas_ok) {
            let_clocks_settle();
            run_cublas(n, opts.iters, peak);
        }
#endif
        printf("\n");
    }

#ifdef ENGINE_BENCH_CUBLAS
    printf(
        "cuBLAS is the reference row, not a backend: the engine never calls it.\n"
        "The point of the project is to write the kernel. The point of the row is\n"
        "that a number with no ruler beside it is not a measurement.\n\n");
#else
    printf(
        "Built without cuBLAS, so the reference row is missing. The only ruler\n"
        "here is the card's theoretical peak, in the right-hand column.\n\n");
#endif

    cuda::set_matmul_kernel(cuda::MatmulKernel::Auto);
    return 0;
}
