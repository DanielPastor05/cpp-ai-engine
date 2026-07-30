// The engine's CUDA kernels.
//
// They cover what profiling flagged as hot: the matrix product (53% of the
// Transformer example's time), the element-wise operations, ReLU and softmax.
// The rest stays on the CPU, and that is fine: porting an operation that does
// not dominate the profile only adds transfers.
//
// Every entry point returns false when it decides not to take the work, and the
// caller carries on down the CPU path. Refusals come in two kinds:
//   - not worth it (size below the measured threshold), or
//   - it does not fit the launch geometry (huge batches or dimensions).
// The second kind should not happen with realistic shapes, but returning false
// beats launching an invalid grid and getting garbage.

#include "engine/cuda.hpp"
#include "engine/detail/cuda_ops.hpp"

#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace engine {
namespace cuda {
namespace ops {

namespace {

constexpr int kTile = 32;          // side of the matmul's shared-memory tile
constexpr int kBlock = 256;        // threads per block in the 1D kernels
constexpr int kReduceBlock = 256;  // a power of two: the reduction requires it

// Geometry of the register-tiled matmul. The five numbers are tied to each
// other and cannot be changed independently:
//   (kBM / kTM) * (kBN / kTN) == kRegBlock   -> one output square per thread
//   kBM * kBK == kBN * kBK == 1024           -> a tile fits in 4 scalar load
//                                               passes, or 1 vectorised
constexpr int kBM = 128;                              // output rows per block
constexpr int kBN = 128;                              // output columns per block
constexpr int kBK = 8;                                // step over K
constexpr int kTM = 8;                                // output rows per thread
constexpr int kTN = 8;                                // output columns per thread
constexpr int kRegBlock = (kBM / kTM) * (kBN / kTN);  // 256 threads

static_assert(kRegBlock == 256, "the load indices assume 256 threads per block");
static_assert(kBM * kBK == 1024 && kBN * kBK == 1024,
              "each tile must hold 1024 values for the split to work out");

// The gridDim.y and gridDim.z limit on every supported architecture.
constexpr size_t kMaxGridYZ = 65535;

constexpr size_t kMaxInt = static_cast<size_t>(std::numeric_limits<int>::max());

// Axes the permutation kernel accepts. Four is what the engine uses -- (B, H,
// S, d) in attention, (N, C, H, W) in convolution -- and eight leaves room
// while the plan still fits comfortably in the kernel argument space.
constexpr int kMaxPermuteDims = 8;

// cudaGetLastError() only sees **launch** failures. If the fault happens inside
// the kernel body -- an out-of-range access, say -- the error sticks to the
// context and surfaces at the next copy, which is usually three operations
// later and is in no way to blame.
//
// Synchronising here catches it at the guilty launch, but that is a barrier per
// kernel and the engine launches hundreds per step: it would mean paying in
// production for a diagnostic that only matters while chasing a fault. Hence
// the environment variable, off by default.
//
//   ENGINE_CUDA_SYNC=1 .\build-cuda\Release\test_engine.exe
bool sync_after_launch() {
    static const bool on = [] {
        const char* raw = std::getenv("ENGINE_CUDA_SYNC");
        return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    return on;
}

// A kernel that fails to launch does not abort the program: it is reported and
// false is returned, so the CPU computes the result. An engine that crashes
// because the GPU is busy is worse than one that runs slower.
//
// It is separate from launched_ok() because not every failed launch is undone
// the same way: whoever asked for the output with device_write() has to revert,
// and whoever used device_mut() must not. See accumulate_grad().
bool launch_ok(const char* what) {
    cudaError_t status = cudaGetLastError();
    // With ENGINE_CUDA_SYNC the kernel's execution error is picked up here, at
    // the launch that caused it, rather than at the next copy. The recovery
    // path is the same: report it and compute on the CPU.
    if (status == cudaSuccess && sync_after_launch()) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        detail::note_kernel_launched();
        return true;
    }

    detail::note_kernel_failed();

    // Only the first one, and with the full diagnosis. The previous version
    // printed a line per failed launch: in a test suite that is hundreds of
    // identical lines burying the real output without saying why. The rest are
    // counted and read back with cuda::kernels_failed().
    static bool reported = false;
    if (!reported) {
        reported = true;
        // The useful comparison is the toolkit that compiled this against the
        // driver. runtime_version() is no use here: it follows the driver, so it
        // never comes out ahead and the hint would never print.
        const int built = compiled_version();
        const int drv = driver_version();
        std::fprintf(stderr,
                     "\nengine: kernel %s could not be launched (%s).\n"
                     "  It is computed on the CPU, so the results are correct but slow.\n"
                     "  Built with CUDA %d.%d, installed driver CUDA %d.%d.\n",
                     what, cudaGetErrorString(status), built / 1000, (built % 1000) / 10,
                     drv / 1000, (drv % 1000) / 10);

        // The two hints are independent and can both apply: a binary with the wrong
        // architecture falls back to compiling the PTX, and that is when a driver
        // older than the toolkit finishes the failure off. Chaining them with
        // else-if would tell half the story.
        if (drv > 0 && built > drv) {
            std::fprintf(stderr,
                         "  The driver is older than the toolkit: update the NVIDIA driver, or\n"
                         "  build with a CUDA version the driver supports.\n");
        }
        if (status == cudaErrorUnsupportedPtxVersion || status == cudaErrorNoKernelImageForDevice) {
            const DeviceInfo info = device_info();
            std::fprintf(stderr,
                         "  The binary carries no native code for this card (cc %d.%d).\n"
                         "  Reconfigure with -DCMAKE_CUDA_ARCHITECTURES=%d%d\n",
                         info.compute_major, info.compute_minor, info.compute_major,
                         info.compute_minor);
        }
        std::fprintf(stderr,
                     "  Later failures are not repeated here;"
                     " they are counted in cuda::kernels_failed().\n\n");
    }
    return false;
}

// The same, for an output that was requested with device_write().
//
// That device_write() has to be undone. Otherwise the CPU path would ask for the
// host buffer, Storage would pull it uninitialised off the device, and matmul --
// which accumulates into an output it assumes is zeroed -- would return garbage
// instead of a correct, slower result.
bool launched_ok(const char* what, Storage& out) {
    if (launch_ok(what)) return true;
    out.revert_device_write();
    return false;
}

// ---------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------

template <int Op>
__device__ inline float apply(float x, float y) {
    if (Op == 0) return x + y;
    if (Op == 1) return x - y;
    if (Op == 2) return x * y;
    return x / y;
}

template <int Op>
__global__ void binary_contiguous(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ out, long long n) {
    // Grid-stride loop: this way the block count does not depend on n and
    // gridDim.x never overflows.
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = apply<Op>(a[i], b[i]);
    }
}

// Suffix broadcasting. As on the CPU, no modulo is computed: the grid's y axis
// walks the repetitions and the x axis the block being repeated.
// A 64-bit modulo per element would cost far more here than on the CPU.
template <int Op>
__global__ void binary_broadcast(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ out, long long inner) {
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= inner) return;
    const long long i = (long long)blockIdx.y * inner + j;
    out[i] = apply<Op>(a[i], b[j]);
}

__global__ void relu_forward(const float* __restrict__ x, float* __restrict__ out, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = fmaxf(0.0f, x[i]);
    }
}

__global__ void relu_grad(const float* __restrict__ x, const float* __restrict__ g,
                          float* __restrict__ out, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = (x[i] > 0.0f) ? g[i] : 0.0f;
    }
}

// The backward accumulator: out = g the first time, out += g afterwards.
//
// initialize is uniform across the grid -- the caller decides it by looking at
// whether the tensor already had a gradient -- so the branch splits no warp.
__global__ void grad_accumulate(const float* __restrict__ g, float* __restrict__ out, long long n,
                                bool initialize) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = initialize ? g[i] : out[i] + g[i];
    }
}

// out = x * mul + add. One shape for both scalar operations: multiplying is
// add = 0 and adding is mul = 1. With those values the product or the sum is
// redundant, and it does not matter: the kernel is memory bound, not bound by
// the two floating-point operations it does per element.
__global__ void scalar_affine(const float* __restrict__ x, float* __restrict__ out, float mul,
                              float add, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = x[i] * mul + add;
    }
}

// ---------------------------------------------------------
// Axis reordering
// ---------------------------------------------------------
//
// A flat output index is turned into coordinates and from there into an offset
// into the input, exactly as on the CPU path. The write is contiguous and the
// read jumps: that is the right way round of the two, because a scattered write
// serialises bandwidth far worse than a scattered read, which at least overlaps
// with the rest of the warp.
//
// The plan travels by value. Kernel arguments live in constant memory, so the
// eight pairs are read from there and need neither an allocation nor a separate
// copy to pass them.
//
// ponytail: integer division per element per axis; if the profile flags it,
// swap in the magic-reciprocal multiplication trick.
struct PermutePlan {
    int shape[kMaxPermuteDims];
    long long stride[kMaxPermuteDims];
};

__global__ void permute_gather(const float* __restrict__ x, float* __restrict__ out,
                               PermutePlan plan, int nd, long long n) {
    const long long grid_stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        long long rem = i;
        long long src = 0;
        // Back to front: the last axis is the fastest varying one.
        for (int d = nd - 1; d >= 0; --d) {
            const long long extent = plan.shape[d];
            src += (rem % extent) * plan.stride[d];
            rem /= extent;
        }
        out[i] = x[src];
    }
}

// Sum over one axis viewed as (outer, axis_len, inner): one thread per output
// element, walking the axis.
//
// The accumulation order matches the CPU -- the axis from low to high -- and
// there is no multiplication the compiler could fuse into an FMA either, so
// this kernel does agree bit for bit with the CPU path.
__global__ void sum_over_axis(const float* __restrict__ x, float* __restrict__ out,
                              long long axis_len, long long inner, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long o = i / inner;
        const long long j = i % inner;
        const float* base = x + o * axis_len * inner + j;
        float acc = 0.0f;
        for (long long a = 0; a < axis_len; ++a) acc += base[a * inner];
        out[i] = acc;
    }
}

// ---------------------------------------------------------
// im2col / col2im
// ---------------------------------------------------------
//
// Both walk the output rather than the input, and that alone is the difference
// between needing atomics and not.
//
// im2col writes each element of the columns exactly once, so it comes out
// directly. col2im is its adjoint: several overlapping windows contribute to the
// same pixel, and walking the windows would force atomic adds -- slow and, worse,
// with an accumulation order that changes from run to run. So the traversal is
// inverted: one thread per **input** pixel, which looks for the windows covering
// it. No atomics, and the same order as the CPU.

struct WindowDims {
    int channels, height, width;
    int kernel_h, kernel_w, stride, padding;
    int out_h, out_w;
};

__global__ void im2col_gather(const float* __restrict__ input, float* __restrict__ cols,
                              WindowDims d, long long n) {
    const long long K = (long long)d.channels * d.kernel_h * d.kernel_w;
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long row = i / K;
        const long long k = i % K;

        const long long b = row / spatial;
        const long long oh = (row % spatial) / d.out_w;
        const long long ow = row % d.out_w;

        const long long c = k / ((long long)d.kernel_h * d.kernel_w);
        const long long r = k % ((long long)d.kernel_h * d.kernel_w);
        const long long ki = r / d.kernel_w;
        const long long kj = r % d.kernel_w;

        const long long h = oh * d.stride + ki - d.padding;
        const long long w = ow * d.stride + kj - d.padding;

        // The zero is written explicitly. The CPU path relies on the tensor being
        // born zeroed and skips the padding positions; here the output was
        // requested with device_write(), which uploads nothing, so skipping a
        // position would leave it holding whatever was in that buffer before.
        float value = 0.0f;
        if (h >= 0 && h < d.height && w >= 0 && w < d.width) {
            value = input[((b * d.channels + c) * d.height + h) * d.width + w];
        }
        cols[i] = value;
    }
}

__global__ void col2im_scatter(const float* __restrict__ cols, float* __restrict__ input,
                               WindowDims d, long long n) {
    const long long K = (long long)d.channels * d.kernel_h * d.kernel_w;
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long w = i % d.width;
        const long long h = (i / d.width) % d.height;
        const long long c = (i / ((long long)d.width * d.height)) % d.channels;
        const long long b = i / ((long long)d.width * d.height * d.channels);

        // Windows covering this pixel: from h + padding = oh*stride + ki with
        // 0 <= ki < kH the range of oh follows, and inside it **every** oh is
        // valid, so no divisibility check is needed.
        const long long hp = h + d.padding;
        const long long wp = w + d.padding;
        long long oh_lo = (hp - d.kernel_h + d.stride) / d.stride;  // techo de (hp-kH+1)/stride
        long long ow_lo = (wp - d.kernel_w + d.stride) / d.stride;
        if (oh_lo < 0) oh_lo = 0;
        if (ow_lo < 0) ow_lo = 0;
        const long long oh_hi = min(hp / d.stride, (long long)d.out_h - 1);
        const long long ow_hi = min(wp / d.stride, (long long)d.out_w - 1);

        // oh and ow ascending: the same order the CPU loop accumulates in, so
        // that the sum is done with the same roundings.
        float acc = 0.0f;
        for (long long oh = oh_lo; oh <= oh_hi; ++oh) {
            const long long ki = hp - oh * d.stride;
            for (long long ow = ow_lo; ow <= ow_hi; ++ow) {
                const long long kj = wp - ow * d.stride;
                const long long row = (b * d.out_h + oh) * d.out_w + ow;
                const long long k = (c * d.kernel_h + ki) * d.kernel_w + kj;
                acc += cols[row * K + k];
            }
        }
        input[i] = acc;
    }
}

// ---------------------------------------------------------
// Max pooling
// ---------------------------------------------------------
//
// The forward saves, alongside the maximum, the flat index of the pixel that
// won it: that is all the backward pass needs, and it uses the CPU path's
// tie-break -- the first in traversal order wins, because the comparison is
// strict.
//
// The backward walks the **input**, like col2im and for the same reason: two
// overlapping windows may have chosen the same pixel, and walking the windows
// would force atomic adds. This way each pixel looks for the windows covering
// it, keeps the ones that chose it, and sums in a fixed order.

__global__ void maxpool_windows(const float* __restrict__ input, float* __restrict__ out,
                                float* __restrict__ argmax, WindowDims d, long long n) {
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long ow = i % d.out_w;
        const long long oh = (i / d.out_w) % d.out_h;
        const long long c = (i / spatial) % d.channels;
        const long long b = i / (spatial * d.channels);

        float best = -INFINITY;
        long long best_idx = 0;
        bool found = false;

        for (int ki = 0; ki < d.kernel_h; ++ki) {
            const long long h = oh * d.stride + ki - d.padding;
            if (h < 0 || h >= d.height) continue;
            for (int kj = 0; kj < d.kernel_w; ++kj) {
                const long long w = ow * d.stride + kj - d.padding;
                if (w < 0 || w >= d.width) continue;

                const long long idx = ((b * d.channels + c) * d.height + h) * d.width + w;
                const float v = input[idx];
                // Strict, not >=: on a tie the first one wins, exactly as on the CPU.
                if (!found || v > best) {
                    best = v;
                    best_idx = idx;
                    found = true;
                }
            }
        }
        out[i] = best;
        argmax[i] = (float)best_idx;
    }
}

__global__ void maxpool_windows_grad(const float* __restrict__ argmax, const float* __restrict__ g,
                                     float* __restrict__ dx, WindowDims d, long long n) {
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long w = i % d.width;
        const long long h = (i / d.width) % d.height;
        const long long c = (i / ((long long)d.width * d.height)) % d.channels;
        const long long b = i / ((long long)d.width * d.height * d.channels);

        const long long hp = h + d.padding;
        const long long wp = w + d.padding;
        long long oh_lo = (hp - d.kernel_h + d.stride) / d.stride;
        long long ow_lo = (wp - d.kernel_w + d.stride) / d.stride;
        if (oh_lo < 0) oh_lo = 0;
        if (ow_lo < 0) ow_lo = 0;
        const long long oh_hi = min(hp / d.stride, (long long)d.out_h - 1);
        const long long ow_hi = min(wp / d.stride, (long long)d.out_w - 1);

        float acc = 0.0f;
        for (long long oh = oh_lo; oh <= oh_hi; ++oh) {
            for (long long ow = ow_lo; ow <= ow_hi; ++ow) {
                const long long out_idx = ((b * d.channels + c) * d.out_h + oh) * d.out_w + ow;
                // Only adds if this window chose precisely this pixel.
                if ((long long)argmax[out_idx] == i) acc += g[out_idx];
            }
        }
        dx[i] = acc;
    }
}

// ---------------------------------------------------------
// Matrix product with shared-memory tiles
// ---------------------------------------------------------
//
// Each block computes a 32x32 tile of the output. The tile walks K: at each
// step the 1024 threads load one tile of A and one of B into shared memory and
// then each thread does its 32 products reading from there.
//
// The reason is global memory traffic: without tiles, each element of A is read
// N times and each of B M times. With tiles of side T they are read N/T and M/T
// times, that is 32 times fewer.
//
// The accumulation order is fixed (k ascending, as on the CPU), so the result is
// reproducible from run to run. It is not bit-identical to the CPU, and that is
// expected: the device compiler fuses multiply and add into a single FMA that
// rounds once instead of twice. That is why the parity test compares to a
// tolerance rather than for exact equality.
//
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N, long long a_stride,
                             long long b_stride) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;

    float acc = 0.0f;
    const int tiles = (K + kTile - 1) / kTile;

    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * kTile + threadIdx.x;
        const int b_row = t * kTile + threadIdx.y;

        // Edges are padded with zeros instead of shortening the loop: that way
        // every thread in the block reaches the same __syncthreads(), which is
        // mandatory for the barrier to be valid.
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[(long long)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[(long long)b_row * N + col] : 0.0f;

        __syncthreads();

// As[ty][k] is a broadcast within the warp and Bs[k][tx] walks consecutive
// banks: neither access produces conflicts, so shared memory needs no
// padding.
#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(long long)row * N + col] = acc;
    }
}

// ---------------------------------------------------------
// Matrix product without shared memory
// ---------------------------------------------------------
//
// It exists so the benchmark table has an honest lower bound. Each thread reads
// both operands straight from global memory, so each element of A is read N
// times and each of B M times.
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N, long long a_stride,
                             long long b_stride) {
    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[(long long)row * K + k] * B[(long long)k * N + col];
    }
    C[(long long)row * N + col] = acc;
}

// ---------------------------------------------------------
// Matrix product with register tiling
// ---------------------------------------------------------
//
// This is the real jump, and it is worth understanding **why** the tiled kernel
// above falls short. It is not occupancy, and it is not global memory traffic:
// the tiles already fixed that. It is arithmetic intensity *at the register
// level*.
//
// In matmul_tiled, each thread per step of K does:
//     1 FMA   against   2 shared-memory reads
// That 1:2 ratio is what rules, because the load units saturate long before the
// arithmetic ones. It makes no difference how many warps are in flight.
//
// The fix is for each thread to compute a **block** of results instead of a
// single one. With TM x TN = 8 x 8 outputs live in registers, per step of K a
// thread reads 8 values of A and 8 of B and does 64 products with them:
//     64 FMA   against   16 shared-memory reads
// The ratio goes from 1:2 to 4:1 -- eight times better. The 64 accumulators
// live in registers and are not touched until the end.
//
// Geometry: 128x128 output blocks, 256 threads, step BK=8 over K.
// Shared memory comes to 2 x 8 x 128 x 4 = 8 KB per block.
//
// `As` is stored **transposed** ([k][m] instead of [m][k]) so that each thread's
// eight reads of A are contiguous along m; without that they would go with stride
// K and each would hit a different bank.
//
// On occupancy: with 64 accumulators plus working registers, this burns on the
// order of 80-100 registers per thread and halves occupancy against the tiled
// kernel. **That is deliberate.** The instruction-level parallelism inside each
// thread more than makes up for having fewer warps: it is this kernel's classic
// trade-off, and it is exactly what shows up under the profiler.
//
// UseVector4 picks how the tiles are loaded from global memory. The vectorised
// version moves 16 bytes per instruction instead of 4, which cuts the number of
// load instructions on the global -> shared path. It requires K and N to be
// multiples of 4 so the addresses are aligned, which the host side checks before
// choosing this variant.
template <bool UseVector4>
__global__ void matmul_register_tiled(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int M, int K, int N,
                                      long long a_stride, long long b_stride) {
    // No alignment attribute, deliberately: nothing reads these two matrices in
    // 16-byte blocks. The vectorised load acts on **global** memory, which is
    // where the gain is; what is written and read here is scalar, so there is
    // no alignment requirement to meet. It is also the portable choice, because
    // __align__ is spelled differently depending on the host compiler.
    //
    __shared__ float As[kBK][kBM];  // transpuesta: [k][m]
    __shared__ float Bs[kBK][kBN];

    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;

    // Each thread takes a kTM x kTN square of the output tile.
    const int thread_row = threadIdx.x / (kBN / kTN);  // [0, 16)
    const int thread_col = threadIdx.x % (kBN / kTN);  // [0, 16)

    float acc[kTM][kTN];
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) acc[i][j] = 0.0f;
    }

    // Load indices, distinct from the compute ones: to bring the tiles in it
    // matters that consecutive threads read consecutive positions, not that each
    // reads what it will later use.
    const int a_load_row_v = threadIdx.x / (kBK / 4);  // [0, 128) with float4
    const int a_load_col_v = (threadIdx.x % (kBK / 4)) * 4;
    const int b_load_row_v = threadIdx.x / (kBN / 4);  // [0, 8) with float4
    const int b_load_col_v = (threadIdx.x % (kBN / 4)) * 4;

    const int a_load_row_s = threadIdx.x / kBK;  // [0, 32) escalar
    const int a_load_col_s = threadIdx.x % kBK;
    const int b_load_row_s = threadIdx.x / kBN;  // [0, 2) escalar
    const int b_load_col_s = threadIdx.x % kBN;

    for (int k_base = 0; k_base < K; k_base += kBK) {
        if (UseVector4) {
            // A single 16-byte instruction per thread covers the A tile:
            // 128 rows x 8 columns = 1024 values = 256 float4 = 256 threads.
            {
                const int g_row = block_row + a_load_row_v;
                const int g_col = k_base + a_load_col_v;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                // K is a multiple of 4 and so is g_col, so the float4 either fits
                // whole or not at all: there is no partial remainder to handle.
                if (g_row < M && g_col < K) {
                    v = *reinterpret_cast<const float4*>(A + (long long)g_row * K + g_col);
                }
                // The store is scalar, because it goes in transposed.
                As[a_load_col_v + 0][a_load_row_v] = v.x;
                As[a_load_col_v + 1][a_load_row_v] = v.y;
                As[a_load_col_v + 2][a_load_row_v] = v.z;
                As[a_load_col_v + 3][a_load_row_v] = v.w;
            }
            {
                const int g_row = k_base + b_load_row_v;
                const int g_col = block_col + b_load_col_v;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (g_row < K && g_col < N) {
                    v = *reinterpret_cast<const float4*>(B + (long long)g_row * N + g_col);
                }
                Bs[b_load_row_v][b_load_col_v + 0] = v.x;
                Bs[b_load_row_v][b_load_col_v + 1] = v.y;
                Bs[b_load_row_v][b_load_col_v + 2] = v.z;
                Bs[b_load_row_v][b_load_col_v + 3] = v.w;
            }
        } else {
// Without vectorising, four passes per tile are needed: 256 threads
// against 1024 values.
#pragma unroll
            for (int off = 0; off < kBM; off += 256 / kBK) {
                const int g_row = block_row + a_load_row_s + off;
                const int g_col = k_base + a_load_col_s;
                As[a_load_col_s][a_load_row_s + off] =
                    (g_row < M && g_col < K) ? A[(long long)g_row * K + g_col] : 0.0f;
            }
#pragma unroll
            for (int off = 0; off < kBK; off += 256 / kBN) {
                const int g_row = k_base + b_load_row_s + off;
                const int g_col = block_col + b_load_col_s;
                Bs[b_load_row_s + off][b_load_col_s] =
                    (g_row < K && g_col < N) ? B[(long long)g_row * N + g_col] : 0.0f;
            }
        }

        __syncthreads();

// The hot loop. The shared-memory reads are hoisted into registers before
// the products: without that the compiler would read from shared inside the
// double loop and the whole advantage would be lost.
#pragma unroll
        for (int dot = 0; dot < kBK; ++dot) {
            float reg_m[kTM];
            float reg_n[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) reg_m[i] = As[dot][thread_row * kTM + i];
#pragma unroll
            for (int j = 0; j < kTN; ++j) reg_n[j] = Bs[dot][thread_col * kTN + j];

#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) acc[i][j] += reg_m[i] * reg_n[j];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int row = block_row + thread_row * kTM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int col = block_col + thread_col * kTN + j;
            if (col < N) C[(long long)row * N + col] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------
// Matrix product on the tensor cores, in tf32
// ---------------------------------------------------------
//
// The four kernels above are all the same instruction set: FMA on the ordinary
// fp32 pipes. This one is a different machine. An Ampere SM has four tensor
// cores that each retire a 16x8x8 matrix multiply-accumulate as one instruction,
// and the peak they reach in tf32 is roughly double the fp32 pipes'.
//
// **It buys that with mantissa bits.** tf32 keeps fp32's 8-bit exponent and cuts
// the mantissa from 23 bits to 10; the accumulation stays fp32. Range is
// unchanged, so nothing overflows that would not have overflowed before, but the
// product is not the fp32 product. That is a decision, not a detail, and it is
// why `Auto` never picks this kernel: silently spending three digits of
// precision would invalidate the reference tests that agree with PyTorch to
// ~1e-7, which are the most valuable thing this engine has.
//
// **The geometry is the whole point, and the first version of this kernel proved
// it the hard way.** A block of 4 warps computing a 64x64 tile measured 4227
// GFLOP/s at 4096^3 -- *slower* than the fp32 register-tiled kernel's 4786,
// despite running on hardware with twice the peak. Reaching for the tensor cores
// bought nothing, because the bottleneck was never the multiply.
//
// The reason is data reuse. That version staged 64*32 + 32*64 = 4096 values to
// produce a 64x64 tile: each staged value fed 32 outputs. The fp32 kernel it lost
// to stages 2048 values to produce 128x128: 64 outputs each, twice the reuse. The
// tensor cores were sitting idle waiting on shared memory, and a faster multiplier
// does not help a kernel that is not multiply-bound.
//
// So the tile doubled in both directions: 8 warps, a 128x128 output tile, each
// warp owning 32x64 as a 2x4 grid of 16x16 accumulator fragments. Same reuse as
// the fp32 kernel, and now the extra arithmetic throughput has somewhere to go.
//
// The fragments are declared with wmma::precision::tf32, which means
// load_matrix_sync reads fp32 and the caller is responsible for rounding each
// element with __float_to_tf32. Skipping that step compiles and returns numbers
// that are close but wrong, which is the worst failure mode available.

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;  // tf32 fragments step 8 along K, not 16

constexpr int kTcBM = 128;  // output rows per block
constexpr int kTcBN = 128;  // output columns per block
constexpr int kTcBK = 32;   // K staged per iteration: four wmma steps
constexpr int kTcWarps = 8;
constexpr int kTcBlock = kTcWarps * 32;  // 256 threads

// Each warp owns a 32x64 slab of the output: 2 fragments down, 4 across.
constexpr int kTcWarpM = 2;
constexpr int kTcWarpN = 4;
static_assert(kTcBM / (kTcWarpM * kWmmaM) * (kTcBN / (kTcWarpN * kWmmaN)) == kTcWarps,
              "the warp grid must tile the block exactly");

// 128*32 + 32*128 = 8192 floats = 32 KB, inside the 48 KB a block gets without
// opting in. C does **not** get staged wholesale -- at 128x128 that would be
// another 64 KB -- so fragments go straight to global memory when they are fully
// inside the matrix, and only edge fragments detour through this same buffer.
constexpr int kTcSmem = kTcBM * kTcBK + kTcBK * kTcBN;
static_assert(kTcSmem >= kTcWarps * kWmmaM * kWmmaN, "edge staging must fit in the A/B buffer");

__global__ void matmul_tensor_core(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int M, int K, int N, long long a_stride,
                                   long long b_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    __shared__ float smem[kTcSmem];
    float* As = smem;                  // [kTcBM][kTcBK]
    float* Bs = smem + kTcBM * kTcBK;  // [kTcBK][kTcBN]

    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int block_row = blockIdx.y * kTcBM;
    const int block_col = blockIdx.x * kTcBN;

    // Warps tile the block 4 down by 2 across: each owns 32 rows and 64 columns.
    const int warp = threadIdx.x / 32;
    const int warp_row = (warp / 2) * (kTcWarpM * kWmmaM);
    const int warp_col = (warp % 2) * (kTcWarpN * kWmmaN);

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc[kTcWarpM][kTcWarpN];
#pragma unroll
    for (int i = 0; i < kTcWarpM; ++i) {
#pragma unroll
        for (int j = 0; j < kTcWarpN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    for (int k_base = 0; k_base < K; k_base += kTcBK) {
        // 256 threads stage 4096 values each for A and B: 16 apiece. Out-of-range
        // positions are written as zero rather than skipped -- the fragments read
        // the whole tile, so a hole would be whatever the previous iteration left.
#pragma unroll
        for (int t = 0; t < (kTcBM * kTcBK) / kTcBlock; ++t) {
            const int idx = t * kTcBlock + threadIdx.x;
            const int r = idx / kTcBK;
            const int c = idx % kTcBK;
            const int g_row = block_row + r;
            const int g_col = k_base + c;
            // Rounded to tf32 **here**, once per staged value, rather than after
            // every fragment load. See the note below the b-staging loop.
            As[idx] = (g_row < M && g_col < K)
                          ? wmma::__float_to_tf32(A[(long long)g_row * K + g_col])
                          : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < (kTcBK * kTcBN) / kTcBlock; ++t) {
            const int idx = t * kTcBlock + threadIdx.x;
            const int r = idx / kTcBN;
            const int c = idx % kTcBN;
            const int g_row = k_base + r;
            const int g_col = block_col + c;
            Bs[idx] = (g_row < K && g_col < N)
                          ? wmma::__float_to_tf32(B[(long long)g_row * N + g_col])
                          : 0.0f;
        }

        // Why the rounding moved up here, which was worth 8% and is the third
        // thing this kernel got wrong before it got it right.
        //
        // wmma::precision::tf32 fragments hold fp32 bits and it is the caller's
        // job to round them; the obvious place is right after load_matrix_sync.
        // That costs 24 conversions per lane per K-step -- 4 elements in each of
        // 2 A-fragments and 4 B-fragments -- against only 8 mma_sync instructions.
        // The conversion was outnumbering the arithmetic three to one.
        //
        // Staging does it once per value instead of once per load, and every warp
        // that reads a staged value gets it already rounded. Correctness is
        // unchanged: a value rounded to tf32 is exactly representable as fp32, so
        // storing it in shared memory and letting the hardware read it as tf32
        // rounds nothing twice. It is in fact slightly *more* accurate than
        // leaving it to the hardware, which truncates where __float_to_tf32
        // rounds to nearest.
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTcBK; kk += kWmmaK) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32,
                           wmma::row_major>
                a_frag[kTcWarpM];
            wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32,
                           wmma::row_major>
                b_frag[kTcWarpN];

            // No conversion loop here: shared memory already holds tf32-rounded
            // values, done once at staging time.
#pragma unroll
            for (int i = 0; i < kTcWarpM; ++i) {
                wmma::load_matrix_sync(a_frag[i], As + (warp_row + i * kWmmaM) * kTcBK + kk, kTcBK);
            }
#pragma unroll
            for (int j = 0; j < kTcWarpN; ++j) {
                wmma::load_matrix_sync(b_frag[j], Bs + kk * kTcBN + warp_col + j * kWmmaN, kTcBN);
            }

            // 8 fragments of output against 6 fragments loaded: the ratio the
            // 64x64 version got wrong.
#pragma unroll
            for (int i = 0; i < kTcWarpM; ++i) {
#pragma unroll
                for (int j = 0; j < kTcWarpN; ++j) {
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // store_matrix_sync cannot bounds-check. Staging the whole 128x128 tile would
    // cost another 64 KB of shared memory, so a fragment entirely inside the
    // matrix goes straight to global, and only the ones straddling an edge detour
    // through the (now dead) A/B buffer.
    //
    // The direct store carries one more condition than "fully in bounds", and it
    // is the sort that does not announce itself: **store_matrix_sync requires the
    // leading dimension to be a multiple of 4 floats.** Passing it an N of 67 is
    // undefined behaviour, not a slow path -- it writes plausible numbers in the
    // wrong places. The first version of this kernel omitted the check and the
    // exact-integer parity case caught it on 65x33x67 while every tolerance case
    // around it still passed, which is the entire argument for having that test.
    const bool direct_store_ok = (N % 4 == 0);
    __syncthreads();
    float* edge = smem + warp * (kWmmaM * kWmmaN);

#pragma unroll
    for (int i = 0; i < kTcWarpM; ++i) {
#pragma unroll
        for (int j = 0; j < kTcWarpN; ++j) {
            const int row = block_row + warp_row + i * kWmmaM;
            const int col = block_col + warp_col + j * kWmmaN;

            if (direct_store_ok && row + kWmmaM <= M && col + kWmmaN <= N) {
                wmma::store_matrix_sync(C + (long long)row * N + col, acc[i][j], N,
                                        wmma::mem_row_major);
                continue;
            }
            if (row >= M || col >= N) continue;

            wmma::store_matrix_sync(edge, acc[i][j], kWmmaN, wmma::mem_row_major);
            __syncwarp();
            for (int idx = (int)(threadIdx.x % 32); idx < kWmmaM * kWmmaN; idx += 32) {
                const int r = row + idx / kWmmaN;
                const int c = col + idx % kWmmaN;
                if (r < M && c < N) C[(long long)r * N + c] = edge[idx];
            }
            __syncwarp();
        }
    }
#else
    // Compiled for a card without tf32 tensor cores. The host side refuses to
    // dispatch here, so this is unreachable; silencing the parameters keeps the
    // build clean rather than leaving a warning per architecture.
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)K;
    (void)N;
    (void)a_stride;
    (void)b_stride;
#endif
}

// ---------------------------------------------------------
// Softmax over the last axis
// ---------------------------------------------------------
//
// One block per row. Two reductions over shared memory: first the maximum (to
// subtract it so the exponential does not overflow, as on the CPU) and then the
// sum.
//
// expf is used rather than __expf: the fast version saves a few cycles but loses
// precision, and these values are compared against PyTorch in the reference
// test. The exponential is not the bottleneck here.
__global__ void softmax_rows(const float* __restrict__ x, float* __restrict__ y, int cols) {
    __shared__ float shared[kReduceBlock];

    const long long row = blockIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) local_max = fmaxf(local_max, xr[j]);
    shared[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    const float max_v = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        const float e = expf(xr[j] - max_v);
        yr[j] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float denom = shared[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) yr[j] /= denom;
}

// dX_ij = y_ij * (dY_ij - sum_k dY_ik * y_ik)
__global__ void softmax_rows_grad(const float* __restrict__ y, const float* __restrict__ g,
                                  float* __restrict__ out, int cols) {
    __shared__ float shared[kReduceBlock];

    const long long row = blockIdx.x;
    const float* yr = y + row * cols;
    const float* gr = g + row * cols;
    float* orow = out + row * cols;
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) local += gr[j] * yr[j];
    shared[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float dot = shared[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) orow[j] = yr[j] * (gr[j] - dot);
}

// ---------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------

// Block count for a grid-stride kernel: enough to fill the device, without
// overdoing it.
int grid_for(long long n) {
    const long long want = (n + kBlock - 1) / kBlock;
    const long long cap = 65535 * 16;
    return (int)(want < cap ? want : cap);
}

bool elementwise_worth_it(size_t n) {
    return enabled() && n > 0 && n >= min_elementwise_elements();
}

// Translates the geometry into the kernel's integers, or says it does not fit.
// Both convolution entry points share these checks.
bool window_dims(const WindowShape& s, WindowDims& d) {
    const size_t all[] = {s.batch,    s.channels, s.height,  s.width, s.kernel_h,
                          s.kernel_w, s.stride,   s.padding, s.out_h, s.out_w};
    for (size_t v : all) {
        if (v > kMaxInt) return false;
    }
    if (s.batch == 0 || s.channels == 0 || s.height == 0 || s.width == 0) return false;
    if (s.kernel_h == 0 || s.kernel_w == 0 || s.stride == 0) return false;
    if (s.out_h == 0 || s.out_w == 0) return false;

    d.channels = (int)s.channels;
    d.height = (int)s.height;
    d.width = (int)s.width;
    d.kernel_h = (int)s.kernel_h;
    d.kernel_w = (int)s.kernel_w;
    d.stride = (int)s.stride;
    d.padding = (int)s.padding;
    d.out_h = (int)s.out_h;
    d.out_w = (int)s.out_w;
    return true;
}

template <int Op>
bool launch_binary(const Storage& a, const Storage& b, Storage& out, size_t inner, size_t repeat) {
    const size_t n = a.size();
    if (repeat <= 1) {
        binary_contiguous<Op><<<grid_for((long long)n), kBlock>>>(a.device(), b.device(),
                                                                  out.device_write(), (long long)n);
        return launched_ok("binary_contiguous", out);
    }
    if (repeat > kMaxGridYZ) return false;
    const dim3 grid((unsigned)((inner + kBlock - 1) / kBlock), (unsigned)repeat);
    binary_broadcast<Op>
        <<<grid, kBlock>>>(a.device(), b.device(), out.device_write(), (long long)inner);
    return launched_ok("binary_broadcast", out);
}

}  // namespace

bool binary(Binary op, const Storage& a, const Storage& b, Storage& out, size_t inner,
            size_t repeat) {
    if (!elementwise_worth_it(a.size())) return false;
    if (inner == 0 || repeat == 0) return false;
    if (inner * repeat != a.size() || out.size() != a.size()) return false;
    if (b.size() < inner) return false;
    // With broadcasting the grid is organised by repetitions, which limits how many
    // fit; past that point the CPU takes it.
    if (repeat > 1 && repeat > kMaxGridYZ) return false;

    switch (op) {
        case Binary::Add:
            return launch_binary<0>(a, b, out, inner, repeat);
        case Binary::Sub:
            return launch_binary<1>(a, b, out, inner, repeat);
        case Binary::Mul:
            return launch_binary<2>(a, b, out, inner, repeat);
        case Binary::Div:
            return launch_binary<3>(a, b, out, inner, repeat);
    }
    return false;
}

bool matmul(const Storage& a, const Storage& b, Storage& out, size_t batch, size_t rows,
            size_t inner_dim, size_t cols, bool a_batched, bool b_batched) {
    if (!enabled()) return false;
    if (batch == 0 || rows == 0 || inner_dim == 0 || cols == 0) return false;

    // The threshold is the batch's total work: a batch of small matrices can be
    // worth it even when none of them would be on its own.
    const double flops = 2.0 * (double)batch * rows * inner_dim * cols;
    if (flops < (double)min_matmul_flops()) return false;

    if (rows > kMaxInt || inner_dim > kMaxInt || cols > kMaxInt) return false;
    if (batch > kMaxGridYZ) return false;
    if (out.size() != batch * rows * cols) return false;

    // Stride 0 on the unbatched operand: the same matrix for the whole batch,
    // exactly as on the CPU path.
    const long long a_stride = a_batched ? (long long)rows * inner_dim : 0;
    const long long b_stride = b_batched ? (long long)inner_dim * cols : 0;

    const int M = (int)rows;
    const int K = (int)inner_dim;
    const int N = (int)cols;

    MatmulKernel choice = resolve_matmul_kernel(rows, inner_dim, cols);

    // If someone asked for the vectorised variant by hand on a shape that does not
    // meet the alignment, it degrades quietly to the register-tiled one instead of
    // computing wrongly. A misaligned float4 read does not raise an error: it
    // returns a different value, which is considerably worse.
    if (choice == MatmulKernel::Vectorized && (K % 4 != 0 || N % 4 != 0)) {
        choice = MatmulKernel::RegisterTiled;
    }

    // Asked for by hand on a build or a card without tf32 tensor cores. It does
    // NOT degrade to another kernel the way Vectorized does, and the difference
    // matters: the alignment fallback substitutes a kernel that computes the same
    // numbers, while substituting fp32 for a tf32 request would quietly hand back
    // three more digits of precision than the caller asked for. Silently better
    // is still silently different. Refusing sends the work to the CPU, which the
    // caller can see in the launch counters.
    if (choice == MatmulKernel::TensorCore && !tensor_cores_available()) return false;

    if (choice == MatmulKernel::TensorCore) {
        if ((rows + kTcBM - 1) / kTcBM > kMaxGridYZ) return false;
        const dim3 grid((unsigned)((cols + kTcBN - 1) / kTcBN),
                        (unsigned)((rows + kTcBM - 1) / kTcBM), (unsigned)batch);
        matmul_tensor_core<<<grid, kTcBlock>>>(a.device(), b.device(), out.device_write(), M, K, N,
                                               a_stride, b_stride);
        return launched_ok("matmul_tensor_core", out);
    }

    if (choice == MatmulKernel::RegisterTiled || choice == MatmulKernel::Vectorized) {
        if ((rows + kBM - 1) / kBM > kMaxGridYZ) return false;
        const dim3 grid((unsigned)((cols + kBN - 1) / kBN), (unsigned)((rows + kBM - 1) / kBM),
                        (unsigned)batch);
        if (choice == MatmulKernel::Vectorized) {
            matmul_register_tiled<true><<<grid, kRegBlock>>>(
                a.device(), b.device(), out.device_write(), M, K, N, a_stride, b_stride);
            return launched_ok("matmul_vectorized", out);
        }
        matmul_register_tiled<false><<<grid, kRegBlock>>>(
            a.device(), b.device(), out.device_write(), M, K, N, a_stride, b_stride);
        return launched_ok("matmul_register_tiled", out);
    }

    if ((rows + kTile - 1) / kTile > kMaxGridYZ) return false;
    const dim3 block(kTile, kTile);
    const dim3 grid((unsigned)((cols + kTile - 1) / kTile), (unsigned)((rows + kTile - 1) / kTile),
                    (unsigned)batch);

    if (choice == MatmulKernel::Naive) {
        matmul_naive<<<grid, block>>>(a.device(), b.device(), out.device_write(), M, K, N, a_stride,
                                      b_stride);
        return launched_ok("matmul_naive", out);
    }

    matmul_tiled<<<grid, block>>>(a.device(), b.device(), out.device_write(), M, K, N, a_stride,
                                  b_stride);
    return launched_ok("matmul_tiled", out);
}

bool scalar(const Storage& x, Storage& out, float mul, float add) {
    if (!elementwise_worth_it(x.size())) return false;
    if (out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    scalar_affine<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), mul, add, n);
    return launched_ok("scalar_affine", out);
}

bool permute(const Storage& x, Storage& out, const size_t* out_shape, const size_t* src_strides,
             size_t ndim) {
    if (!elementwise_worth_it(x.size())) return false;
    if (ndim == 0 || ndim > (size_t)kMaxPermuteDims) return false;
    if (out.size() != x.size()) return false;

    PermutePlan plan{};
    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) {
        if (out_shape[d] == 0 || out_shape[d] > kMaxInt) return false;
        plan.shape[d] = (int)out_shape[d];
        plan.stride[d] = (long long)src_strides[d];
        total *= out_shape[d];
    }
    // The output shape has to cover the whole input: if it does not add up, the
    // kernel would read past the buffer rather than give a wrong result.
    if (total != x.size()) return false;

    const long long n = (long long)x.size();
    permute_gather<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), plan, (int)ndim, n);
    return launched_ok("permute_gather", out);
}

bool sum_axis(const Storage& x, Storage& out, size_t outer, size_t axis_len, size_t inner) {
    if (!elementwise_worth_it(x.size())) return false;
    if (outer == 0 || axis_len == 0 || inner == 0) return false;
    if (outer * axis_len * inner != x.size()) return false;
    if (out.size() != outer * inner) return false;

    const long long n = (long long)(outer * inner);
    sum_over_axis<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), (long long)axis_len,
                                           (long long)inner, n);
    return launched_ok("sum_over_axis", out);
}

bool im2col(const Storage& input, Storage& cols, const WindowShape& s) {
    if (!elementwise_worth_it(cols.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    const size_t K = s.channels * s.kernel_h * s.kernel_w;
    if (input.size() != s.batch * s.channels * s.height * s.width) return false;
    if (cols.size() != s.batch * s.out_h * s.out_w * K) return false;

    const long long n = (long long)cols.size();
    im2col_gather<<<grid_for(n), kBlock>>>(input.device(), cols.device_write(), d, n);
    return launched_ok("im2col_gather", cols);
}

bool col2im(const Storage& cols, Storage& input, const WindowShape& s) {
    // Work is measured by the columns, the big side, even though the grid walks
    // the input: with a 3x3 kernel that is nine times more values.
    if (!elementwise_worth_it(cols.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    const size_t K = s.channels * s.kernel_h * s.kernel_w;
    if (input.size() != s.batch * s.channels * s.height * s.width) return false;
    if (cols.size() != s.batch * s.out_h * s.out_w * K) return false;

    const long long n = (long long)input.size();
    col2im_scatter<<<grid_for(n), kBlock>>>(cols.device(), input.device_write(), d, n);
    return launched_ok("col2im_scatter", input);
}

bool maxpool(const Storage& input, Storage& out, Storage& argmax, const WindowShape& s) {
    if (!elementwise_worth_it(input.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    if (input.size() != s.batch * s.channels * s.height * s.width) return false;
    if (out.size() != s.batch * s.channels * s.out_h * s.out_w) return false;
    if (argmax.size() != out.size()) return false;
    // The index travels as a float, which only represents integers exactly up to
    // 2^24. Above that size the rounding would pick a different pixel, so the
    // dispatch is refused and the CPU takes it.
    if (input.size() > (size_t{1} << 24)) return false;

    const long long n = (long long)out.size();
    maxpool_windows<<<grid_for(n), kBlock>>>(input.device(), out.device_write(),
                                             argmax.device_write(), d, n);
    // Two outputs, both requested with device_write(): if the launch fails both
    // have to be undone, or the CPU path would pull an uninitialised buffer down
    // to host.
    if (launch_ok("maxpool_windows")) return true;
    out.revert_device_write();
    argmax.revert_device_write();
    return false;
}

bool maxpool_backward(const Storage& argmax, const Storage& grad_out, Storage& dx,
                      const WindowShape& s) {
    if (!elementwise_worth_it(dx.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    if (dx.size() != s.batch * s.channels * s.height * s.width) return false;
    if (grad_out.size() != s.batch * s.channels * s.out_h * s.out_w) return false;
    if (argmax.size() != grad_out.size()) return false;
    if (dx.size() > (size_t{1} << 24)) return false;

    const long long n = (long long)dx.size();
    maxpool_windows_grad<<<grid_for(n), kBlock>>>(argmax.device(), grad_out.device(),
                                                  dx.device_write(), d, n);
    return launched_ok("maxpool_windows_grad", dx);
}

bool relu(const Storage& x, Storage& out) {
    if (!elementwise_worth_it(x.size())) return false;
    if (out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    relu_forward<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), n);
    return launched_ok("relu_forward", out);
}

bool relu_backward(const Storage& x, const Storage& grad_out, Storage& out) {
    if (!elementwise_worth_it(x.size())) return false;
    if (grad_out.size() != x.size() || out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    relu_grad<<<grid_for(n), kBlock>>>(x.device(), grad_out.device(), out.device_write(), n);
    return launched_ok("relu_grad", out);
}

bool accumulate_grad(Storage& grad, const Storage& g, bool initialize) {
    if (!elementwise_worth_it(g.size())) return false;
    if (grad.size() != g.size()) return false;
    // Only if it costs no transfer at all. The gradient ends up being read on the
    // host -- the optimiser runs on the CPU -- so uploading something to add it here
    // would pay the trip twice and come out worse than not accelerating at all. When
    // initialising, the destination is written in full and there is nothing to send.
    if (!g.resident_on_device()) return false;
    if (!initialize && !grad.resident_on_device()) return false;

    const long long n = (long long)g.size();
    float* out = initialize ? grad.device_write() : grad.device_mut();
    grad_accumulate<<<grid_for(n), kBlock>>>(g.device(), out, n, initialize);
    if (launch_ok("grad_accumulate")) return true;

    // A deliberate asymmetry, which is why this one does not use launched_ok().
    // device_write() marked the buffer valid without uploading it, so if the kernel
    // never ran that has to be undone. device_mut() did upload: the device holds the
    // good data and the host is marked stale, so the CPU path will pull it down
    // intact. Reverting there would promote a host copy that is not valid.
    if (initialize) grad.revert_device_write();
    return false;
}

bool softmax(const Storage& x, Storage& out, size_t rows, size_t cols) {
    if (!elementwise_worth_it(x.size())) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != x.size() || out.size() != x.size()) return false;
    if (rows > (size_t)std::numeric_limits<int>::max()) return false;

    softmax_rows<<<(unsigned)rows, kReduceBlock>>>(x.device(), out.device_write(), (int)cols);
    return launched_ok("softmax_rows", out);
}

bool softmax_backward(const Storage& y, const Storage& grad_out, Storage& out, size_t rows,
                      size_t cols) {
    if (!elementwise_worth_it(y.size())) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != y.size() || grad_out.size() != y.size() || out.size() != y.size()) {
        return false;
    }
    if (rows > (size_t)std::numeric_limits<int>::max()) return false;

    softmax_rows_grad<<<(unsigned)rows, kReduceBlock>>>(y.device(), grad_out.device(),
                                                        out.device_write(), (int)cols);
    return launched_ok("softmax_rows_grad", out);
}

}  // namespace ops
}  // namespace cuda
}  // namespace engine
