// What the kernel translation units share, and why each thing is here.
//
// src/cuda/kernels.cu reached 2 361 lines, which is more than any other file in
// this repository and more than one sitting can hold. The matrix product came
// out first because it is the most self-contained family -- six kernel variants,
// one dispatch function, and no other family calls into it.
//
// The split has one trap and it is worth stating, because it is the reason this
// header holds declarations rather than definitions. Four of these helpers own
// function-local statics:
//
//   sync_after_launch()  caches the environment variable
//   launch_ok()          reports the *first* failed launch and only the first
//   scratch_buffer()     caches a device allocation and grows it
//
// Defining those inline in a header, inside the anonymous namespace they used to
// live in, gives every translation unit its own copy. Nothing would crash. The
// error report would just start firing once per file instead of once per run --
// exactly the behaviour launch_ok's comment says it exists to prevent -- and the
// scratch allocation would silently double. So they live in `shared`, with
// external linkage, defined once in kernels.cu.
//
// The constants below are `constexpr` and safe to duplicate: they are values,
// not state.

#ifndef ENGINE_CUDA_KERNELS_COMMON_CUH
#define ENGINE_CUDA_KERNELS_COMMON_CUH

#include "engine/detail/cuda_ops.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace engine {
namespace cuda {
namespace ops {

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

// The largest number of chunks split-K will cut the K dimension into. Fixed on
// the host so the partial sums are combined in the same order every run.
constexpr int kMaxSplitK = 64;

// `shared` and not `detail`: engine::cuda::detail already exists, and a nested
// ops::detail shadows it -- every unqualified detail::note_kernel_launched()
// inside ops:: would resolve to the wrong namespace. It compiled straight into
// five "has no member" errors, which is the good outcome.
namespace shared {

// A kernel that fails to launch does not abort the program: it is reported and
// false is returned, so the CPU computes the result. See the definitions in
// kernels.cu for what each does with the output buffer -- they differ, and the
// difference is whether device_write() has to be undone.
bool launch_ok(const char* what);
bool launched_ok(const char* what, Storage& out);

// Block count for a grid-stride kernel: enough to fill the device without
// overdoing it. Stateless, so unlike the three above it is safe to duplicate
// per translation unit.
inline int grid_for(long long n) {
    const long long want = (n + kBlock - 1) / kBlock;
    const long long cap = 65535 * 16;
    return (int)(want < cap ? want : cap);
}

// Scratch shared by the passes that need somewhere to put per-block partials:
// split-K's matmul and LayerNorm's cross-row dgamma/dbeta. It grows to the
// largest size asked for and stays there, so a training loop pays one
// allocation rather than one per step. Safe to share because the engine
// dispatches from a single thread and neither user holds it across a call.
float* scratch_buffer(size_t floats);

// One occupancy question per kernel. Each family fills in its own, because a
// kernel in another translation unit's anonymous namespace has no address this
// one can take -- which is the whole reason this struct is here rather than
// next to occupancy_report().
struct Probe {
    const char* name;
    const void* fn;
    int block_threads;
};

void collect_matmul_probes(std::vector<Probe>& out);

}  // namespace shared

// Split-K's second pass sums the per-chunk partials, and the kernel that does it
// is the reductions family's. It is declared here, outside any anonymous
// namespace, so both translation units launch the same one -- a `__global__` at
// namespace scope has external linkage, and a launch from another unit resolves
// through the host-side stub without needing relocatable device code.
__global__ void sum_over_axis(const float* __restrict__ x, float* __restrict__ out,
                              long long axis_len, long long inner, long long n);

}  // namespace ops
}  // namespace cuda
}  // namespace engine

#endif  // ENGINE_CUDA_KERNELS_COMMON_CUH
