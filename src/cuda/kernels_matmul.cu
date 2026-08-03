// The matrix product, all six variants of it.
//
// Lifted out of kernels.cu, which was 2 361 lines. This family is the one that
// comes out cleanly: nothing else in the engine launches these kernels, and the
// only things they need from the rest are the tile constants and the launch
// helpers, both of which are in kernels_common.cuh.
//
// The variants, in the order they were written, and none of them is dead code --
// the dispatch at the bottom picks between them by shape, and the parity suite
// exercises every path:
//
//   matmul_naive           one thread per output, no shared memory
//   matmul_tiled           32x32 shared-memory tile, the general case
//   matmul_tiled_split_k   for tall and thin, with the chunk count fixed on the
//                          host so partial sums combine in the same order
//   matmul_register_tiled  128x128 blocks, 8x8 per thread; <true> vectorises
//                          the loads through float4
//   matmul_tensor_core     wmma in tf32, which measured slower than the
//                          register-tiled one on consumer Ampere and is never
//                          selected automatically
//   matmul_tensor_core_fp16  the same in fp16, which is the one that pays

#include "engine/cuda.hpp"
#include "engine/detail/cuda_ops.hpp"
#include "kernels_common.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace engine {
namespace cuda {
namespace ops {

using shared::grid_for;
using shared::launch_ok;
using shared::launched_ok;
using shared::scratch_buffer;

namespace {

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
// A note on `beta`, which three of these kernels now take.
//
// It is BLAS's: C = A*B + beta*C, with beta == 0 meaning "overwrite" and
// therefore the behaviour every caller had before. It exists because the
// backward pass was spending 13.5% of GPU time in grad_accumulate -- twenty
// launches per training step -- writing each gradient to a fresh tensor and then
// adding it to the parameter's gradient in a second kernel. With beta == 1 the
// product lands in the gradient directly and both the launch and the temporary
// go away. cuBLAS and PyTorch both take this parameter for the same reason.
//
// **beta != 0 changes which accessor the output needs, and getting that wrong
// loses the gradient silently.** device_write() reserves the buffer without
// uploading, which is correct exactly when the kernel overwrites every element;
// with beta the kernel *reads* what is there, so the host side has to be
// uploaded first and device_mut() is the accessor. The dispatch below picks
// between them, and tests/test_cuda_parity.cpp checks accumulation onto a
// **non-zero** gradient, because onto a zero one both spellings agree and the
// bug would not show.
//
// Only naive, tiled and register-tiled take it. Split-K sums per-block partials
// with sum_over_axis, which overwrites, and the tensor-core kernels store
// through store_matrix_sync, which has no accumulate form -- both would need
// their own work and neither is on the path this was measured for. The dispatch
// refuses beta != 0 for them and the caller falls back, which is the same
// contract as every other refusal in this file.

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N, long long a_stride,
                             long long b_stride, float beta) {
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
        float* dst = C + (long long)row * N + col;
        *dst = (beta == 0.0f) ? acc : acc + beta * *dst;
    }
}

// The same tiled product, but with the K axis cut into a fixed number of chunks
// and one grid slice per chunk.
//
// Every variant above draws its parallelism from the **output**: one block per
// output tile. A convolution's weight gradient is cols^T x dout, and for MNIST's
// first layer that is (9 x 50176) x (50176 x 16) -- nine by sixteen outputs, so
// a single block walking a K of fifty thousand while the rest of the card sits
// idle. Measured, that one product was 4.5 ms.
//
// Each slice leaves its own partial in `partials`, and sum_over_axis adds them
// up afterwards. The chunk boundaries come from a count fixed on the host, so
// the split does not shift between runs and neither does the result.
__global__ void matmul_tiled_split_k(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ partials, int M, int K, int N, int chunk) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    const int k_begin = (int)blockIdx.z * chunk;
    const int k_end = min(K, k_begin + chunk);

    float acc = 0.0f;
    for (int t = k_begin; t < k_end; t += kTile) {
        const int a_col = t + threadIdx.x;
        const int b_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < k_end) ? A[(long long)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < k_end && col < N) ? B[(long long)b_row * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        partials[(long long)blockIdx.z * M * N + (long long)row * N + col] = acc;
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
                             long long b_stride, float beta) {
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
    float* dst = C + (long long)row * N + col;
    *dst = (beta == 0.0f) ? acc : acc + beta * *dst;
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
                                      long long a_stride, long long b_stride, float beta) {
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

    const int a_load_row_s = threadIdx.x / kBK;  // [0, 32), a scalar
    const int a_load_col_s = threadIdx.x % kBK;
    const int b_load_row_s = threadIdx.x / kBN;  // [0, 2), a scalar
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
            if (col < N) {
                float* dst = C + (long long)row * N + col;
                *dst = (beta == 0.0f) ? acc[i][j] : acc[i][j] + beta * *dst;
            }
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
// The same tile in fp16, which is the one the tensor cores were built for
// ---------------------------------------------------------
//
// docs/CUDA.md records that the tf32 kernel above lands at 31.5% of peak against
// the fp32 kernel's 46.5%, and that the reason is the card rather than the code:
// on consumer Ampere, dense tf32 tensor throughput is *the same* 16.2 TFLOP/s as
// fp32. The famous 2x needs fp16. This is that kernel, so the claim stops being
// an argument and becomes a row in a table.
//
// Three things change and nothing else does. The fragments step **16** along K
// rather than 8, so a 32-deep staging pass is two wmma steps instead of four.
// Shared memory holds __half, which halves the tile's 32 KB to 16. And the
// accumulator stays fp32 -- that is what "mixed precision" means, and dropping it
// to fp16 would lose far more than the multiply gains.
//
// The conversion sits at staging time for the same reason it does above, and the
// same 8% is at stake: one __float2half per staged value rather than one per
// fragment load, where the conversions would outnumber the mma instructions
// three to one.
//
// **The precision is genuinely worse and in a different way from tf32.** Both
// keep 10 mantissa bits, so the rounding error per product is comparable. What
// fp16 also loses is *range*: 5 exponent bits against fp32's 8, so values above
// 65504 become infinity and values below about 6e-5 flush to zero, where tf32
// keeps fp32's full exponent and cannot overflow anything fp32 could hold. For a
// benchmark of unit-normal matrices that is invisible; for training it is the
// entire reason loss scaling exists, and this engine has none. Hence: opt-in,
// never selected by Auto, and documented as a measurement rather than wired into
// the optimiser.
constexpr int kHWmmaK = 16;                              // fp16 fragments step 16 along K
constexpr int kHTcSmem = kTcBM * kTcBK + kTcBK * kTcBN;  // in halves: 16 KB

__global__ void matmul_tensor_core_fp16(const float* __restrict__ A, const float* __restrict__ B,
                                        float* __restrict__ C, int M, int K, int N,
                                        long long a_stride, long long b_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    __shared__ __half smem[kHTcSmem];
    __half* As = smem;
    __half* Bs = smem + kTcBM * kTcBK;

    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int block_row = blockIdx.y * kTcBM;
    const int block_col = blockIdx.x * kTcBN;

    const int warp = threadIdx.x / 32;
    const int warp_row = (warp / 2) * (kTcWarpM * kWmmaM);
    const int warp_col = (warp % 2) * (kTcWarpN * kWmmaN);

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kHWmmaK, float> acc[kTcWarpM][kTcWarpN];
#pragma unroll
    for (int i = 0; i < kTcWarpM; ++i) {
#pragma unroll
        for (int j = 0; j < kTcWarpN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    for (int k_base = 0; k_base < K; k_base += kTcBK) {
#pragma unroll
        for (int t = 0; t < (kTcBM * kTcBK) / kTcBlock; ++t) {
            const int idx = t * kTcBlock + threadIdx.x;
            const int r = idx / kTcBK;
            const int c = idx % kTcBK;
            const int g_row = block_row + r;
            const int g_col = k_base + c;
            As[idx] = (g_row < M && g_col < K) ? __float2half(A[(long long)g_row * K + g_col])
                                               : __float2half(0.0f);
        }
#pragma unroll
        for (int t = 0; t < (kTcBK * kTcBN) / kTcBlock; ++t) {
            const int idx = t * kTcBlock + threadIdx.x;
            const int r = idx / kTcBN;
            const int c = idx % kTcBN;
            const int g_row = k_base + r;
            const int g_col = block_col + c;
            Bs[idx] = (g_row < K && g_col < N) ? __float2half(B[(long long)g_row * N + g_col])
                                               : __float2half(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTcBK; kk += kHWmmaK) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kHWmmaK, __half, wmma::row_major>
                a_frag[kTcWarpM];
            wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kHWmmaK, __half, wmma::row_major>
                b_frag[kTcWarpN];

#pragma unroll
            for (int i = 0; i < kTcWarpM; ++i) {
                wmma::load_matrix_sync(a_frag[i], As + (warp_row + i * kWmmaM) * kTcBK + kk, kTcBK);
            }
#pragma unroll
            for (int j = 0; j < kTcWarpN; ++j) {
                wmma::load_matrix_sync(b_frag[j], Bs + kk * kTcBN + warp_col + j * kWmmaN, kTcBN);
            }
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

    // Same store as the tf32 kernel, including the multiple-of-4 requirement on
    // the leading dimension that the exact-integer parity case caught there. The
    // edge buffer reinterprets the (now dead) half staging area: 8 warps x 256
    // floats is 8 KB, inside the 16 KB it holds.
    const bool direct_store_ok = (N % 4 == 0);
    __syncthreads();
    float* edge = reinterpret_cast<float*>(smem) + warp * (kWmmaM * kWmmaN);

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
}  // namespace

bool matmul(const Storage& a, const Storage& b, Storage& out, size_t batch, size_t rows,
            size_t inner_dim, size_t cols, bool a_batched, bool b_batched, float beta) {
    if (!enabled()) return false;
    if (batch == 0 || rows == 0 || inner_dim == 0 || cols == 0) return false;

    // The threshold is the batch's total work: a batch of small matrices can be
    // worth it even when none of them would be on its own.
    //
    // Unless both operands are already on the device, in which case there is no
    // transfer for the threshold to weigh -- same rule as elementwise_worth_it().
    // Both, not either: with one side on the host this would trade a download for
    // an upload rather than avoiding one. In a training loop both is the normal
    // case, because the weights stay resident between steps once the optimiser
    // updates them there.
    const double flops = 2.0 * (double)batch * rows * inner_dim * cols;
    if (flops < (double)min_matmul_flops() && !(a.resident_on_device() && b.resident_on_device())) {
        return false;
    }

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

    // The output accessor depends on beta, and this is the line the parity test
    // for accumulation exists to protect. device_write() reserves without
    // uploading, which is right only when the kernel overwrites every element.
    // With beta != 0 the kernel reads C, so the host side has to be there first.
    const auto output = [&out, beta]() {
        return beta == 0.0f ? out.device_write() : out.device_mut();
    };

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
    if ((choice == MatmulKernel::TensorCore || choice == MatmulKernel::TensorCoreFp16) &&
        (!tensor_cores_available() || beta != 0.0f)) {
        return false;
    }

    if (choice == MatmulKernel::TensorCoreFp16) {
        if ((rows + kTcBM - 1) / kTcBM > kMaxGridYZ) return false;
        const dim3 grid((unsigned)((cols + kTcBN - 1) / kTcBN),
                        (unsigned)((rows + kTcBM - 1) / kTcBM), (unsigned)batch);
        matmul_tensor_core_fp16<<<grid, kTcBlock>>>(a.device(), b.device(), out.device_write(), M,
                                                    K, N, a_stride, b_stride);
        return launched_ok("matmul_tensor_core_fp16", out);
    }

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
            matmul_register_tiled<true><<<grid, kRegBlock>>>(a.device(), b.device(), output(), M, K,
                                                             N, a_stride, b_stride, beta);
            return launched_ok("matmul_vectorized", out);
        }
        matmul_register_tiled<false><<<grid, kRegBlock>>>(a.device(), b.device(), output(), M, K, N,
                                                          a_stride, b_stride, beta);
        return launched_ok("matmul_register_tiled", out);
    }

    if ((rows + kTile - 1) / kTile > kMaxGridYZ) return false;
    const dim3 block(kTile, kTile);
    const dim3 grid((unsigned)((cols + kTile - 1) / kTile), (unsigned)((rows + kTile - 1) / kTile),
                    (unsigned)batch);

    if (choice == MatmulKernel::Naive) {
        matmul_naive<<<grid, block>>>(a.device(), b.device(), output(), M, K, N, a_stride, b_stride,
                                      beta);
        return launched_ok("matmul_naive", out);
    }

    // Split-K, when the output is too small to fill the card and K is long
    // enough to be worth cutting. Only unbatched: a batch already supplies the
    // blocks this exists to manufacture.
    const int tiles_m = (M + kTile - 1) / kTile;
    const int tiles_n = (N + kTile - 1) / kTile;
    if (beta == 0.0f && batch == 1 && choice == MatmulKernel::Tiled && tiles_m * tiles_n <= 8 &&
        K >= 4096) {
        // Aim for about 256 blocks: enough to fill the card, few enough that the
        // partials stay scratch rather than an allocation worth worrying about.
        int split = std::min(256 / (tiles_m * tiles_n), kMaxSplitK);
        split = (int)std::min<long long>(split, (K + kTile - 1) / kTile);
        float* partials = split > 1 ? scratch_buffer((size_t)split * M * N) : nullptr;
        if (partials != nullptr) {
            // Chunks are a whole number of tiles so every slice starts on a tile
            // boundary. Rounding up can leave the last slices with nothing to do,
            // which costs a launch and contributes an exact zero.
            const int chunk = ((K + split - 1) / split + kTile - 1) / kTile * kTile;
            const dim3 split_grid((unsigned)tiles_n, (unsigned)tiles_m, (unsigned)split);
            matmul_tiled_split_k<<<split_grid, block>>>(a.device(), b.device(), partials, M, K, N,
                                                        chunk);
            if (!launch_ok("matmul_tiled_split_k")) return false;

            const long long mn = (long long)M * N;
            sum_over_axis<<<grid_for(mn), kBlock>>>(partials, out.device_write(), (long long)split,
                                                    mn, mn);
            return launched_ok("matmul_split_k", out);
        }
    }

    matmul_tiled<<<grid, block>>>(a.device(), b.device(), output(), M, K, N, a_stride, b_stride,
                                  beta);
    return launched_ok("matmul_tiled", out);
}

namespace shared {

// The kernels above are in this file's anonymous namespace, so occupancy_report
// in kernels.cu cannot name them. It asks instead.
void collect_matmul_probes(std::vector<Probe>& out) {
    out.push_back({"matmul_naive", (const void*)matmul_naive, kTile * kTile});
    out.push_back({"matmul_tiled", (const void*)matmul_tiled, kTile * kTile});
    out.push_back({"matmul_tiled_split_k", (const void*)matmul_tiled_split_k, kTile * kTile});
    out.push_back({"matmul_register_tiled", (const void*)matmul_register_tiled<false>, kRegBlock});
    out.push_back({"matmul_vectorized", (const void*)matmul_register_tiled<true>, kRegBlock});
    out.push_back({"matmul_tensor_core", (const void*)matmul_tensor_core, kTcBlock});
}

}  // namespace shared

}  // namespace ops
}  // namespace cuda
}  // namespace engine
