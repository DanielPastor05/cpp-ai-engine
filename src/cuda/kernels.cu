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
#include "kernels_common.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace engine {
namespace cuda {
namespace ops {

// Defined below in `detail` rather than in the anonymous namespace, because
// each owns a function-local static and two translation units must not get
// one each. kernels_common.cuh says why in full.
using shared::grid_for;
using shared::launch_ok;
using shared::launched_ok;
using shared::scratch_buffer;

namespace {

// The shared constants and launch helpers are in kernels_common.cuh.

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
}  // namespace

namespace shared {

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
            // The optional is the point here. This used to read a DeviceInfo whose
            // fields defaulted to zero when there was no usable device, so the very
            // failure it reports -- no native code for this card -- printed "cc 0.0"
            // and advised -DCMAKE_CUDA_ARCHITECTURES=00. Advice built on a number
            // nobody has is worse than no advice.
            if (const std::optional<DeviceInfo> info = device_info()) {
                std::fprintf(stderr,
                             "  The binary carries no native code for this card (cc %d.%d).\n"
                             "  Reconfigure with -DCMAKE_CUDA_ARCHITECTURES=%d%d\n",
                             info->compute_major, info->compute_minor, info->compute_major,
                             info->compute_minor);
            } else {
                std::fprintf(stderr,
                             "  The binary carries no native code for this device, and the"
                             " device could not be queried to say which one it needs.\n");
            }
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

}  // namespace shared

namespace {

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

// Swapping the last two axes, through shared memory.
//
// The general gather below reads x[i] at a stride the permutation decides, and
// for the one permutation this engine actually leans on -- Conv2d turning
// (N*oH*oW, C) into (N, C, oH*oW) after the matmul, twice per layer per
// direction -- that stride is the channel count. Consecutive threads then land
// 64 bytes apart, so every 32-byte sector fetched carries four useful bytes.
// Measured on an MNIST step: 33 GB/s on a card that does around 400.
//
// A tile staged in shared memory makes both halves contiguous: the read walks
// rows, the write walks the transposed rows, and neither is strided. The +1 on
// the row length is the usual bank-conflict padding -- without it, column
// threadIdx.x of the shared tile lands in one bank for all 32 threads.
constexpr int kTrTile = 32;
constexpr int kTrRows = 8;  // rows per pass, so a block is 32x8 rather than 32x32

__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int rows,
                                int cols, long long matrix) {
    __shared__ float tile[kTrTile][kTrTile + 1];

    in += (long long)blockIdx.z * matrix;
    out += (long long)blockIdx.z * matrix;

    int x = blockIdx.x * kTrTile + threadIdx.x;
    int y = blockIdx.y * kTrTile + threadIdx.y;
    for (int j = 0; j < kTrTile; j += kTrRows) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * cols + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * kTrTile + threadIdx.x;
    y = blockIdx.x * kTrTile + threadIdx.y;
    for (int j = 0; j < kTrTile; j += kTrRows) {
        if (x < rows && y + j < cols) {
            out[(long long)(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

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
}  // namespace

// Declared in kernels_common.cuh: split-K's second pass in
// kernels_matmul.cu launches this one, so it cannot have internal linkage.
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

namespace {

// The same sum, one block per output instead of one thread.
//
// The kernel above parallelises over outer*inner, which is the **output** size,
// not the work. For a convolution's bias gradient the output is the channel
// count: sixteen threads looping fifty thousand times each, on a card with
// thousands of cores. Measured on an MNIST step it was 42% of all GPU time,
// more than every matmul in the model put together.
//
// The reduction order changes -- a fixed tree instead of a serial walk -- so
// this does not match the CPU bit for bit. It is still deterministic, which is
// what the engine actually promises: the block size is a compile-time constant,
// so which thread reads which element and in what order the partials combine
// are the same on every run.
template <int kThreads>
__global__ void sum_over_axis_blocked(const float* __restrict__ x, float* __restrict__ out,
                                      long long axis_len, long long inner) {
    __shared__ float partial[kThreads];
    const long long i = blockIdx.x;
    const long long o = i / inner;
    const long long j = i % inner;
    const float* base = x + o * axis_len * inner + j;

    float acc = 0.0f;
    for (long long a = threadIdx.x; a < axis_len; a += kThreads) acc += base[a * inner];
    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[i] = partial[0];
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
// Whole-buffer reductions, in double
// ---------------------------------------------------------
//
// Two stages over a fixed block count. Stage one gives each block a slice and
// leaves one double behind; stage two sums those in a single block, in index
// order. The alternative -- one kernel and an atomicAdd on a double -- is
// shorter and makes the answer depend on which block finishes first, so two runs
// over the same data could disagree in the last bits. Everything else in this
// backend is reproducible run to run and this is not the place to stop.
//
// double all the way through, because Tensor::sum() accumulates in double
// deliberately and mean(), mse_loss and every backward's initial gradient hang
// off it.

constexpr int kReduceBlocks = 256;

// One buffer for the partials, allocated once and never freed. It is 2 KB, it
// outlives every call, and freeing it at exit would mean touching the CUDA
// context during shutdown -- which src/cuda/runtime.cu already documents as a
// thing not to do.
double* reduction_partials() {
    static double* buffer = [] {
        void* p = nullptr;
        if (cudaMalloc(&p, kReduceBlocks * sizeof(double)) != cudaSuccess) return (double*)nullptr;
        return (double*)p;
    }();
    return buffer;
}

// Scratch shared by the passes that need somewhere to put per-block partials:
// split-K's matmul and LayerNorm's cross-row dgamma/dbeta. Same reasoning as the
// reduction buffer above, except the size depends on the shape, so it grows to
// the largest asked for and stays there -- a training loop pays one allocation
// rather than one per step.
//
// One buffer for both is safe because the engine dispatches from a single
// thread and neither user holds it across a call.
}  // namespace

namespace shared {

float* scratch_buffer(size_t floats) {
    static float* buffer = nullptr;
    static size_t capacity = 0;
    if (floats <= capacity) return buffer;

    void* p = nullptr;
    if (cudaMalloc(&p, floats * sizeof(float)) != cudaSuccess) return nullptr;
    if (buffer != nullptr) cudaFree(buffer);
    buffer = (float*)p;
    capacity = floats;
    return buffer;
}

}  // namespace shared

namespace {

template <bool Square>
__global__ void reduce_stage1(const float* __restrict__ x, double* __restrict__ partials,
                              long long n) {
    __shared__ double shared[kReduceBlock];

    const long long stride = (long long)blockDim.x * gridDim.x;
    double acc = 0.0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double v = (double)x[i];
        acc += Square ? v * v : v;
    }

    const int tid = threadIdx.x;
    shared[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = shared[0];
}

// One block, walking the partials in index order: the same order every run.
__global__ void reduce_stage2(double* __restrict__ partials, int count) {
    __shared__ double shared[kReduceBlocks];
    const int tid = threadIdx.x;
    shared[tid] = (tid < count) ? partials[tid] : 0.0;
    __syncthreads();
    for (int s = kReduceBlocks / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[0] = shared[0];
}

__global__ void scale_buffer(float* __restrict__ x, float factor, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        x[i] *= factor;
    }
}

// ---------------------------------------------------------
// Optimiser steps
// ---------------------------------------------------------
//
// The dullest kernels here and the ones that save the most traffic. Each thread
// owns one weight, reads its gradient and its own slice of the optimiser state,
// and writes the weight back. Nothing is shared between indices, so there is no
// reduction and no barrier -- the arithmetic is identical to the CPU loop, line
// for line, which is why the parity test can hold them to a tight tolerance.

__global__ void sgd_update(float* __restrict__ w, const float* __restrict__ g,
                           float* __restrict__ velocity, float lr, float momentum,
                           float weight_decay, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float grad = g[i];
        if (weight_decay != 0.0f) grad += weight_decay * w[i];
        if (velocity != nullptr) {
            velocity[i] = momentum * velocity[i] + grad;
            grad = velocity[i];
        }
        w[i] -= lr * grad;
    }
}

__global__ void adam_update(float* __restrict__ w, const float* __restrict__ g,
                            float* __restrict__ m, float* __restrict__ v, float lr, float beta1,
                            float beta2, float eps, float weight_decay, float bias_c1,
                            float bias_c2, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float grad = g[i];
        if (weight_decay != 0.0f) grad += weight_decay * w[i];

        m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;

        // The bias corrections arrive precomputed: they depend on the step
        // count, not on the index, so working them out per element would be the
        // same two pow() calls repeated a million times.
        const float m_hat = m[i] / bias_c1;
        const float v_hat = v[i] / bias_c2;

        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
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
// ---------------------------------------------------------
// Layer normalisation
// ---------------------------------------------------------
//
// One block per row, two shared-memory trees: one for the mean, one for the
// variance around it. Two passes and not the one-pass sum-of-squares identity
// (var = E[x^2] - E[x]^2), which is algebraically the same and numerically much
// worse: it subtracts two large nearly-equal numbers, and for an activation with
// a large mean relative to its spread the result loses most of its digits. The
// CPU path takes two passes for the same reason and these have to agree.
__global__ void layernorm_rows(const float* __restrict__ x, const float* __restrict__ gamma,
                               const float* __restrict__ beta, float* __restrict__ out,
                               float* __restrict__ xhat, float* __restrict__ inv_std, int cols,
                               float eps) {
    __shared__ float shared[kReduceBlock];

    const long long row = blockIdx.x;
    const float* xr = x + row * cols;
    float* out_r = out + row * cols;
    float* xhat_r = xhat + row * cols;
    const int tid = threadIdx.x;
    const float inv_cols = 1.0f / (float)cols;

    float acc = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) acc += xr[j];
    shared[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float mean = shared[0] * inv_cols;
    __syncthreads();

    float var_acc = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        const float d = xr[j] - mean;
        var_acc += d * d;
    }
    shared[tid] = var_acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(shared[0] * inv_cols + eps);
    __syncthreads();

    if (tid == 0) inv_std[row] = inv;
    for (int j = tid; j < cols; j += blockDim.x) {
        const float h = (xr[j] - mean) * inv;
        xhat_r[j] = h;
        out_r[j] = gamma[j] * h + beta[j];
    }
}

// The backward, with the cross-row reduction that made this the last pair of
// operations left on the host.
//
// dx is row-local: the mean and the variance depend on the whole row, so each
// component drags two correction terms along, and both are row sums. That part
// is the forward's shape again.
//
// dgamma and dbeta are not. Every row contributes to the same `cols` values, so
// a grid of one block per row would need atomics -- and an atomicAdd makes the
// sum depend on which block finishes first, which is exactly the reproducibility
// this engine tests for. Instead a **fixed** number of blocks walk the rows in a
// grid-stride loop, each accumulating into a private slice of scratch that
// nothing else writes. layernorm_backward() then sums those slices in index
// order with sum_over_axis, so the accumulation order is a property of the shape
// and not of the scheduler.
__global__ void layernorm_backward_rows(
    const float* __restrict__ dy, const float* __restrict__ xhat, const float* __restrict__ gamma,
    const float* __restrict__ inv_std, float* __restrict__ dx, float* __restrict__ dgamma_partial,
    float* __restrict__ dbeta_partial, long long rows, int cols) {
    __shared__ float shared[kReduceBlock];

    const int tid = threadIdx.x;
    const float inv_cols = 1.0f / (float)cols;
    float* dg = dgamma_partial + (long long)blockIdx.x * cols;
    float* db = dbeta_partial + (long long)blockIdx.x * cols;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* dy_r = dy + row * cols;
        const float* h_r = xhat + row * cols;
        float* dx_r = dx + row * cols;

        float s_dxhat = 0.0f, s_dxhat_h = 0.0f;
        for (int j = tid; j < cols; j += blockDim.x) {
            const float dxhat = dy_r[j] * gamma[j];
            s_dxhat += dxhat;
            s_dxhat_h += dxhat * h_r[j];
        }

        // Both sums in one tree each, back to back rather than interleaved: two
        // reductions over kReduceBlock floats cost less than the register
        // pressure of carrying a pair through one.
        shared[tid] = s_dxhat;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        const float sum_dxhat = shared[0];
        __syncthreads();

        shared[tid] = s_dxhat_h;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        const float sum_dxhat_h = shared[0];
        __syncthreads();

        const float inv = inv_std[row];
        for (int j = tid; j < cols; j += blockDim.x) {
            const float dxhat = dy_r[j] * gamma[j];
            dx_r[j] = inv * (dxhat - inv_cols * sum_dxhat - h_r[j] * inv_cols * sum_dxhat_h);
            dg[j] += dy_r[j] * h_r[j];
            db[j] += dy_r[j];
        }
    }
}

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

// The size threshold answers one question: is this operation big enough to be
// worth moving the data across PCIe? When the data is **already on the device**
// that question does not apply -- there is nothing to move, and refusing is what
// costs the round trip, because the CPU path then has to pull the buffer down.
//
// This is the same rule the optimiser kernels use, and it is what the thresholds
// could not express. Tuned back when most of a step ran on the host, the default
// of 2^20 elements meant that MNIST -- whose largest tensor is 802,816 -- never
// dispatched a single elementwise operation, so the chain broke after every
// matmul. Measured: 15.8 s against 3.4 s for the same binary.
bool elementwise_worth_it(const Storage& input, size_t n) {
    if (!enabled() || n == 0) return false;
    return input.resident_on_device() || n >= min_elementwise_elements();
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
    if (!elementwise_worth_it(a, a.size())) return false;
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

bool scalar(const Storage& x, Storage& out, float mul, float add) {
    if (!elementwise_worth_it(x, x.size())) return false;
    if (out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    scalar_affine<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), mul, add, n);
    return launched_ok("scalar_affine", out);
}

bool permute(const Storage& x, Storage& out, const size_t* out_shape, const size_t* src_strides,
             size_t ndim) {
    if (!elementwise_worth_it(x, x.size())) return false;
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

    // Is this a swap of the last two axes over contiguous leading ones? Then it
    // is a batch of plain transposes and the tiled kernel applies. Writing the
    // output shape as (..., C, R) over a source of (..., R, C), the source
    // strides the caller computed have to read 1 for the axis that was C and R
    // for the one that was R, with every leading axis still packed.
    if (ndim >= 2) {
        const size_t rows = out_shape[ndim - 1], cols = out_shape[ndim - 2];
        bool swap = src_strides[ndim - 2] == 1 && src_strides[ndim - 1] == (long long)cols;
        size_t packed = rows * cols, batch = 1;
        for (size_t d = ndim - 2; d-- > 0 && swap;) {
            swap = src_strides[d] == (long long)packed;
            batch *= out_shape[d];
            packed *= out_shape[d];
        }
        // grid.y covers the source rows, so it is bounded like any other axis.
        if (swap && batch <= (size_t)kMaxGridYZ && (rows + kTrTile - 1) / kTrTile <= kMaxGridYZ) {
            const dim3 grid((unsigned)((cols + kTrTile - 1) / kTrTile),
                            (unsigned)((rows + kTrTile - 1) / kTrTile), (unsigned)batch);
            transpose_tiled<<<grid, dim3(kTrTile, kTrRows)>>>(
                x.device(), out.device_write(), (int)rows, (int)cols, (long long)(rows * cols));
            return launched_ok("transpose_tiled", out);
        }
    }

    const long long n = (long long)x.size();
    permute_gather<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), plan, (int)ndim, n);
    return launched_ok("permute_gather", out);
}

bool sum_axis(const Storage& x, Storage& out, size_t outer, size_t axis_len, size_t inner) {
    if (!elementwise_worth_it(x, x.size())) return false;
    if (outer == 0 || axis_len == 0 || inner == 0) return false;
    if (outer * axis_len * inner != x.size()) return false;
    if (out.size() != outer * inner) return false;

    // Which of the two mappings wins is decided by the *output* size, because
    // that is the flat kernel's whole parallelism. Few outputs over a long axis
    // -- a bias gradient -- and it runs on a handful of threads; many outputs
    // and it already fills the card while the blocked one would spend most of
    // its tree reduction on nothing.
    const long long n = (long long)(outer * inner);
    if (n <= 4096 && axis_len >= 64) {
        sum_over_axis_blocked<kBlock><<<(unsigned)n, kBlock>>>(
            x.device(), out.device_write(), (long long)axis_len, (long long)inner);
        return launched_ok("sum_over_axis_blocked", out);
    }
    sum_over_axis<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), (long long)axis_len,
                                           (long long)inner, n);
    return launched_ok("sum_over_axis", out);
}

bool im2col(const Storage& input, Storage& cols, const WindowShape& s) {
    if (!elementwise_worth_it(input, cols.size())) return false;

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
    if (!elementwise_worth_it(cols, cols.size())) return false;

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
    if (!elementwise_worth_it(input, input.size())) return false;

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
    if (!elementwise_worth_it(grad_out, dx.size())) return false;

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

namespace {

// Shared by both reductions. Like the optimisers, the admission rule is
// residency and not size: if the values are already on the device, reducing them
// here costs one 8-byte read back, and refusing costs pulling the whole buffer
// down.
template <bool Square>
bool reduce_impl(const Storage& x, double& out) {
    if (!enabled() || x.size() == 0) return false;
    if (!x.resident_on_device()) return false;

    double* partials = reduction_partials();
    if (partials == nullptr) return false;

    const long long n = (long long)x.size();
    const int blocks =
        (int)std::min<long long>(kReduceBlocks, (n + kReduceBlock - 1) / kReduceBlock);

    reduce_stage1<Square><<<blocks, kReduceBlock>>>(x.device(), partials, n);
    if (!launch_ok("reduce_stage1")) return false;
    reduce_stage2<<<1, kReduceBlocks>>>(partials, blocks);
    if (!launch_ok("reduce_stage2")) return false;

    double result = 0.0;
    if (cudaMemcpy(&result, partials, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    out = result;
    return true;
}

}  // namespace

bool reduce_sum(const Storage& x, double& out) {
    return reduce_impl<false>(x, out);
}

bool reduce_sum_squares(const Storage& x, double& out) {
    return reduce_impl<true>(x, out);
}

bool scale_in_place(Storage& x, float factor) {
    if (!enabled() || x.size() == 0) return false;
    if (!x.resident_on_device()) return false;

    const long long n = (long long)x.size();
    // device_mut() and not device_write(): the kernel reads each value before it
    // writes it.
    scale_buffer<<<grid_for(n), kBlock>>>(x.device_mut(), factor, n);
    return launch_ok("scale_buffer");
}

// Both optimiser entry points share the same admission rule, and it is not a
// size threshold. What decides is whether the backward left the gradient on the
// device: if it did, running here costs nothing and saves pulling the whole
// model down; if it did not, uploading a gradient to subtract it would pay the
// very transfer this is meant to remove.
//
// Note what is deliberately absent: elementwise_worth_it(). A ten-element bias
// still dispatches, because the alternative is not "a cheap CPU loop" but "a
// download, a CPU loop, and an upload before the next forward".
namespace {

bool optimiser_can_run(const Storage& param, const Storage& grad) {
    if (!enabled()) return false;
    if (param.size() == 0 || grad.size() != param.size()) return false;
    return grad.resident_on_device();
}

}  // namespace

bool sgd_step(Storage& param, const Storage& grad, Storage* velocity, float lr, float momentum,
              float weight_decay) {
    if (!optimiser_can_run(param, grad)) return false;
    if (velocity != nullptr && velocity->size() != param.size()) return false;

    const long long n = (long long)param.size();
    // device_mut() and not device_write(): every one of these reads the old
    // value before writing the new one, so the buffer has to be uploaded if it
    // is stale rather than merely reserved.
    float* w = param.device_mut();
    float* vel = (velocity != nullptr && momentum != 0.0f) ? velocity->device_mut() : nullptr;

    sgd_update<<<grid_for(n), kBlock>>>(w, grad.device(), vel, lr, momentum, weight_decay, n);
    // launch_ok and not launched_ok: nothing here was requested with
    // device_write(), so there is no reservation to revert. The host copy is
    // already marked stale and the device holds the good data, which is exactly
    // what the CPU fallback needs in order to pull it down intact.
    return launch_ok("sgd_update");
}

bool adam_step(Storage& param, const Storage& grad, Storage& m, Storage& v, float lr, float beta1,
               float beta2, float eps, float weight_decay, float bias_c1, float bias_c2) {
    if (!optimiser_can_run(param, grad)) return false;
    if (m.size() != param.size() || v.size() != param.size()) return false;

    const long long n = (long long)param.size();
    float* w = param.device_mut();
    float* mp = m.device_mut();
    float* vp = v.device_mut();

    adam_update<<<grid_for(n), kBlock>>>(w, grad.device(), mp, vp, lr, beta1, beta2, eps,
                                         weight_decay, bias_c1, bias_c2, n);
    return launch_ok("adam_update");
}

bool relu(const Storage& x, Storage& out) {
    if (!elementwise_worth_it(x, x.size())) return false;
    if (out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    relu_forward<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), n);
    return launched_ok("relu_forward", out);
}

bool relu_backward(const Storage& x, const Storage& grad_out, Storage& out) {
    if (!elementwise_worth_it(x, x.size())) return false;
    if (grad_out.size() != x.size() || out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    relu_grad<<<grid_for(n), kBlock>>>(x.device(), grad_out.device(), out.device_write(), n);
    return launched_ok("relu_grad", out);
}

bool accumulate_grad(Storage& grad, const Storage& g, bool initialize) {
    if (!elementwise_worth_it(g, g.size())) return false;
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

// LayerNorm is the one operation with a size floor that residency does not
// override, and the exception is measured rather than assumed.
//
// The residency rule -- dispatch if the data is already on the device, because
// refusing is what costs the round trip -- holds for one kernel replacing one
// pass over the host. LayerNorm's backward is four calls: a memset, the row
// kernel, and two reductions over the per-block partials. That fixed cost has to
// be earned.
//
// Measured, forward plus backward, CPU against device (scratchpad sweep, RTX
// 3060 Ti):
//
//     rows x cols      elements     speedup
//        64 x 32          2 048       0.18x
//       256 x 32          8 192       0.54x
//        64 x 128         8 192       0.58x
//        64 x 512        32 768       2.26x
//       256 x 128        32 768       2.05x
//      1024 x 512       524 288       3.96x
//
// The crossover sits at about 2^15 elements, so that is the floor. Note that
// 1024x32 and 64x512 are the same element count and give 1.31x against 2.26x:
// a 256-thread block on 32 columns wastes seven eighths of itself, so the count
// is a single-parameter approximation to a two-parameter shape. It is the right
// approximation for the one decision being made here.
//
// This is why examples/transformer_demo.cpp did not get faster when these
// kernels were written: its rows x d_model lands at 8 192, in the losing half of
// that table. The kernels are for the models this engine cannot run yet, not for
// the demo that fits in a terminal.
constexpr size_t kMinLayerNormElements = 1u << 15;

bool layernorm(const Storage& x, const Storage& gamma, const Storage& beta, Storage& out,
               Storage& xhat, Storage& inv_std, size_t rows, size_t cols, float eps) {
    if (!enabled() || x.size() < kMinLayerNormElements) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != x.size() || out.size() != x.size() || xhat.size() != x.size()) return false;
    if (gamma.size() != cols || beta.size() != cols || inv_std.size() != rows) return false;
    if (rows > kMaxInt) return false;

    layernorm_rows<<<(unsigned)rows, kReduceBlock>>>(x.device(), gamma.device(), beta.device(),
                                                     out.device_write(), xhat.device_write(),
                                                     inv_std.device_write(), (int)cols, eps);
    if (!launch_ok("layernorm_rows")) {
        out.revert_device_write();
        xhat.revert_device_write();
        inv_std.revert_device_write();
        return false;
    }
    detail::note_kernel_launched();
    return true;
}

bool layernorm_backward(const Storage& grad_out, const Storage& xhat, const Storage& gamma,
                        const Storage& inv_std, Storage& dx, Storage& dgamma, Storage& dbeta,
                        size_t rows, size_t cols) {
    if (!enabled() || grad_out.size() < kMinLayerNormElements) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != grad_out.size() || xhat.size() != grad_out.size()) return false;
    if (dx.size() != grad_out.size() || gamma.size() != cols) return false;
    if (dgamma.size() != cols || dbeta.size() != cols || inv_std.size() != rows) return false;

    // Fixed by the shape and nothing else, so two runs reduce in the same order.
    // Capped at 64 because the scratch is blocks x cols and the whole point is
    // that it stays small enough to sum in one pass afterwards.
    const size_t blocks = std::min<size_t>(rows, 64);
    float* scratch = scratch_buffer(2 * blocks * cols);
    if (scratch == nullptr) return false;
    float* dg_partial = scratch;
    float* db_partial = scratch + blocks * cols;
    if (cudaMemset(scratch, 0, 2 * blocks * cols * sizeof(float)) != cudaSuccess) return false;

    layernorm_backward_rows<<<(unsigned)blocks, kReduceBlock>>>(
        grad_out.device(), xhat.device(), gamma.device(), inv_std.device(), dx.device_write(),
        dg_partial, db_partial, (long long)rows, (int)cols);
    if (!launch_ok("layernorm_backward_rows")) {
        dx.revert_device_write();
        return false;
    }
    detail::note_kernel_launched();

    // Stage two: sum the per-block partials in index order. This is exactly
    // sum_over_axis with outer=1 over `blocks` slices of `cols`, which is the
    // reduction kernel that already exists.
    const long long n = (long long)cols;
    sum_over_axis<<<grid_for(n), kBlock>>>(dg_partial, dgamma.device_write(), (long long)blocks, n,
                                           n);
    if (!launch_ok("sum_over_axis")) {
        dgamma.revert_device_write();
        return false;
    }
    detail::note_kernel_launched();

    sum_over_axis<<<grid_for(n), kBlock>>>(db_partial, dbeta.device_write(), (long long)blocks, n,
                                           n);
    return launched_ok("sum_over_axis", dbeta);
}

bool softmax(const Storage& x, Storage& out, size_t rows, size_t cols) {
    if (!elementwise_worth_it(x, x.size())) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != x.size() || out.size() != x.size()) return false;
    if (rows > (size_t)std::numeric_limits<int>::max()) return false;

    softmax_rows<<<(unsigned)rows, kReduceBlock>>>(x.device(), out.device_write(), (int)cols);
    return launched_ok("softmax_rows", out);
}

bool softmax_backward(const Storage& y, const Storage& grad_out, Storage& out, size_t rows,
                      size_t cols) {
    if (!elementwise_worth_it(y, y.size())) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != y.size() || grad_out.size() != y.size() || out.size() != y.size()) {
        return false;
    }
    if (rows > (size_t)std::numeric_limits<int>::max()) return false;

    softmax_rows_grad<<<(unsigned)rows, kReduceBlock>>>(y.device(), grad_out.device(),
                                                        out.device_write(), (int)cols);
    return launched_ok("softmax_rows_grad", out);
}

// ---------------------------------------------------------
// Occupancy, from the runtime rather than from a profiler
// ---------------------------------------------------------
//
// This lives here and not in runtime.cu for one reason: the kernels are in an
// anonymous namespace in this file, so this is the only translation unit that
// can take their addresses. Everything it needs is a plain runtime call --
// cudaFuncGetAttributes for registers and shared memory,
// cudaOccupancyMaxActiveBlocksPerMultiprocessor for how many blocks fit -- and
// neither wants the elevated permissions that kept `ncu` from being usable on
// the machine this was developed on.
//
// `limiter` is a deduction, not a reported field: the runtime says how many
// blocks fit but not what stopped it at that number, so each candidate ceiling
// is computed and whichever binds first is named. It is the part a reader
// actually wants -- "0.5 occupancy" says nothing you can act on, "0.5, limited
// by registers" says which knob exists.

namespace {

// One row per kernel worth asking about. Not all 26: the point is the shapes of
// the problem -- a heavily register-tiled GEMM, a shared-memory reduction, a
// plain element-wise pass -- not a full inventory.
const char* what_binds(const cudaDeviceProp& p, const cudaFuncAttributes& a, int block_threads,
                       int blocks_per_sm) {
    if (blocks_per_sm <= 0) return "does not fit";

    // Each ceiling, computed the way the hardware does, then compared. Whichever
    // sits at the answer is the one that bound it.
    const int by_blocks = p.maxBlocksPerMultiProcessor;
    const int by_warps = (p.maxThreadsPerMultiProcessor / block_threads);
    const int by_regs = a.numRegs > 0
                            ? (int)(p.regsPerMultiprocessor / ((size_t)a.numRegs * block_threads))
                            : by_blocks;
    const int by_smem =
        a.sharedSizeBytes > 0 ? (int)(p.sharedMemPerMultiprocessor / a.sharedSizeBytes) : by_blocks;

    if (blocks_per_sm == by_regs && by_regs <= by_smem && by_regs <= by_warps) return "registers";
    if (blocks_per_sm == by_smem && by_smem <= by_warps) return "shared memory";
    if (blocks_per_sm == by_warps) return "threads per SM";
    return "blocks per SM";
}

}  // namespace

std::vector<KernelOccupancy> occupancy_report() {
    std::vector<KernelOccupancy> out;
    if (!available()) return out;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return out;

    std::vector<shared::Probe> probes = {
        {"transpose_tiled", (const void*)transpose_tiled, kTrTile * kTrRows},
        {"permute_gather", (const void*)permute_gather, kBlock},
        {"sum_over_axis", (const void*)sum_over_axis, kBlock},
        {"sum_over_axis_blocked", (const void*)sum_over_axis_blocked<kBlock>, kBlock},
        {"im2col_gather", (const void*)im2col_gather, kBlock},
        {"col2im_scatter", (const void*)col2im_scatter, kBlock},
        {"maxpool_windows", (const void*)maxpool_windows, kBlock},
        {"reduce_stage1", (const void*)reduce_stage1<false>, kReduceBlock},
        {"grad_accumulate", (const void*)grad_accumulate, kBlock},
        {"adam_update", (const void*)adam_update, kBlock},
        {"softmax_rows", (const void*)softmax_rows, kBlock},
    };
    // The matmul kernels live in another translation unit's anonymous
    // namespace, so their addresses are not nameable here.
    shared::collect_matmul_probes(probes);

    const double warps_per_sm = prop.maxThreadsPerMultiProcessor / 32.0;

    for (const shared::Probe& probe : probes) {
        cudaFuncAttributes attr{};
        if (cudaFuncGetAttributes(&attr, probe.fn) != cudaSuccess) continue;

        int blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, probe.fn, probe.block_threads,
                                                          0) != cudaSuccess) {
            continue;
        }

        KernelOccupancy row;
        row.name = probe.name;
        row.block_threads = probe.block_threads;
        row.registers = attr.numRegs;
        row.shared_bytes = attr.sharedSizeBytes;
        row.blocks_per_sm = blocks;
        row.occupancy = (blocks * probe.block_threads / 32.0) / warps_per_sm;
        row.limiter = what_binds(prop, attr, probe.block_threads, blocks);
        out.push_back(row);
    }
    // The error state is not the caller's problem, but a failed query would
    // otherwise look like a kernel that does not exist.
    cudaGetLastError();
    return out;
}

}  // namespace ops

// The public entry point. The body is in ops:: because that is the only scope
// that can take the address of a kernel in this file's anonymous namespace.
std::vector<KernelOccupancy> kernel_occupancy() {
    return ops::occupancy_report();
}

}  // namespace cuda
}  // namespace engine
