#ifndef ENGINE_CUDA_HPP
#define ENGINE_CUDA_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace engine {
namespace cuda {

// ---------------------------------------------------------
// The CUDA backend.
//
// This header always exists. If the engine was built without CUDA, available()
// returns false, everything runs on the CPU, and a program including it still
// compiles: nothing has to be wrapped in an #ifdef to use the library.
// ---------------------------------------------------------

// True if the engine was built with -DENGINE_CUDA and there is a usable device.
// The first call interrogates the runtime; later ones return the memoised
// answer.
bool available();

// Runtime switch. It starts out following available(), and can be turned off
// with ENGINE_CUDA=0 in the environment or with set_enabled(false). The parity
// tests use it to compute the same expression both ways and compare.
bool enabled();
void set_enabled(bool on);

struct DeviceInfo {
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessors = 0;
    size_t total_memory = 0;
};

// The card, if there is one.
//
// This used to return a DeviceInfo whose name defaulted to "no device" and whose
// numbers defaulted to zero, and that is a sentinel dressed as a value. It was
// already producing wrong output: the diagnostic in kernels.cu that prints "the
// binary carries no native code for this card (cc %d.%d)" reads compute_major
// and compute_minor straight out of it, so on a machine with no usable device it
// advised reconfiguring with -DCMAKE_CUDA_ARCHITECTURES=00.
//
// An optional will not compile if the caller forgets.
std::optional<DeviceInfo> device_info();

// Waits for launched kernels to finish. Only needed for measurement: the copies
// are synchronising, so correctness never depends on this.
void synchronize();

// The CUDA versions in play, in the toolkit's own format (12040 = 12.4).
//
// There are three of them, and the distinction matters: if the toolkit that
// compiled the binary is ahead of the installed driver, the binary carries no
// usable native code for the card, the driver falls back to compiling the PTX
// at run time, and that is where it fails with "unsupported toolchain". It is
// the most common reason a freshly built backend does not start.
//
// Detecting it means comparing compiled_version() against driver_version().
// runtime_version() is **no** use for this: cudaRuntimeGetVersion() follows the
// driver, so it reports 13.2 on a machine with driver 13.2 even when the linked
// runtime came from toolkit 13.3. Measured, not assumed.
int compiled_version();  // CUDART_VERSION: the toolkit that compiled this
int runtime_version();   // what the runtime reports at execution time
int driver_version();    // the installed driver

// ---------------------------------------------------------
// Launch counters.
//
// The engine falls back to the CPU path when a kernel cannot be launched, which
// is correct in production and a trap in the tests: parity would compare CPU
// against CPU and pass with exactly zero error without ever touching the
// device. A green test that exercised nothing is worse than a red one.
// ---------------------------------------------------------
size_t kernels_launched();
size_t kernels_failed();
void reset_kernel_counters();

// ---------------------------------------------------------
// The device's theoretical ceiling.
//
// Without this, saying "the kernel does 4 TFLOP/s" means nothing: the figure
// that matters is what fraction of peak it reaches, because that is the one
// that says whether there is work left to do. It is derived from
// cudaDeviceProp, so nobody has to look up each card's specifications by hand.
// ---------------------------------------------------------

// SMs x cores per SM x 2 (an FMA counts as two operations) x clock.
double peak_fp32_gflops();
// Memory clock x 2 (double data rate) x bus width.
double peak_bandwidth_gbs();

// ---------------------------------------------------------
// What each kernel costs the scheduler.
//
// Registers per thread and static shared memory decide how many blocks fit on
// an SM at once, and that decides how much latency the hardware can hide. It is
// the first thing anybody asks about a CUDA kernel and this engine could not
// answer it: `ncu` reports it, and `ncu` needs administrator rights for its
// performance counters, which stopped the question being asked here at all.
//
// It does not need them. cudaFuncGetAttributes and
// cudaOccupancyMaxActiveBlocksPerMultiprocessor are plain runtime calls, and
// between them they give the whole static picture. What they cannot give is the
// *achieved* occupancy -- whether the blocks that fit are actually resident,
// which depends on the launch's grid and on what else is running. This is the
// ceiling, and a kernel far below its ceiling is a different problem from one
// whose ceiling is low.
//
// Empty without CUDA, or without a device.
// ---------------------------------------------------------
struct KernelOccupancy {
    std::string name;
    int block_threads = 0;     // the block size this kernel is launched with
    int registers = 0;         // per thread
    size_t shared_bytes = 0;   // static shared memory per block
    int blocks_per_sm = 0;     // how many fit at once
    double occupancy = 0.0;    // resident warps as a fraction of the SM's maximum
    const char* limiter = "";  // what runs out first: registers, shared memory, blocks
};
std::vector<KernelOccupancy> kernel_occupancy();

// ---------------------------------------------------------
// Matrix product variants.
//
// The engine carries four kernels for the same operation, from the most naive
// to the most tuned. This is not indecision: **the progression is the result**.
// Keeping them all alive makes it possible to measure them against each other
// on the same machine and, above all, to parity-check each one separately — a
// kernel built on 128x128 tiles fails precisely on the shapes with remainders,
// and without being able to select it there would be no way to pin that down in
// a test.
// ---------------------------------------------------------
enum class MatmulKernel {
    Auto,           // picks by shape and alignment
    Naive,          // no shared memory; the honest lower bound
    Tiled,          // 32x32 tiles in shared memory, one result per thread
    RegisterTiled,  // 8x8 results per thread, held in registers
    Vectorized,     // the same, with float4 loads
    TensorCore,     // tf32 on the tensor cores; see the precision note below
    TensorCoreFp16  // fp16 in, fp32 accumulate: the one the tensor cores were built for
};

// Whether this build can run MatmulKernel::TensorCore: compiled for sm_80 or
// later, and running on a card of that generation. Asking for it anywhere else
// falls back to the CPU rather than launching something the hardware cannot do.
bool tensor_cores_available();

// ---------------------------------------------------------
// A precision note, because TensorCore is not a free speed-up.
//
// The tensor-core path multiplies in **tf32**: the same 8-bit exponent as fp32
// with the mantissa cut from 23 bits to 10, accumulating in fp32. Nothing about
// the engine's memory layout changes — tensors stay fp32 and the conversion
// happens in registers — but the product is not the fp32 product. Measured
// against the fp32 kernels it lands around 1e-3 relative error rather than the
// 1e-7 the other four hold.
//
// That is why `Auto` never selects it and why it is not the default. An engine
// that silently traded three digits of precision for speed would invalidate its
// own PyTorch reference tests, and the interesting part of this project is that
// those tests agree to ~1e-7. Opting in is a decision the caller makes with the
// error budget in front of them.
//
// `TensorCoreFp16` is the same bargain with a second cost on top. It keeps the
// same 10 mantissa bits as tf32, so the rounding per product is comparable, and
// it loses *range*: 5 exponent bits against fp32's 8, so anything above 65504
// becomes infinity and anything below about 6e-5 flushes to zero. tf32 keeps
// fp32's exponent and cannot overflow something fp32 could hold; fp16 can. That
// is what loss scaling exists to manage in a training loop, and this engine has
// none -- so this variant is a measurement of what the hardware can do, opt-in
// and never automatic, not a mode to train in.
// ---------------------------------------------------------

MatmulKernel matmul_kernel();
void set_matmul_kernel(MatmulKernel kernel);
const char* matmul_kernel_name(MatmulKernel kernel);

// Which variant `Auto` would resolve to for a given shape. Used by the
// benchmark and the documentation; not needed to compute anything.
MatmulKernel resolve_matmul_kernel(size_t rows, size_t inner_dim, size_t cols);

// ---------------------------------------------------------
// Dispatch thresholds.
//
// Launching a kernel costs a few microseconds, so below a certain size the GPU
// loses to a single CPU core. These two numbers decide when it is worth it;
// they can be set with ENGINE_CUDA_MIN_FLOPS and ENGINE_CUDA_MIN_ELEMENTS to
// sweep them without recompiling.
// ---------------------------------------------------------

// Minimum operation count (2*M*K*N) to send a matrix product to the GPU.
size_t min_matmul_flops();
// Minimum element count to send an element-wise operation.
size_t min_elementwise_elements();
void set_thresholds(size_t matmul_flops, size_t elementwise_elements);

// ---------------------------------------------------------
// Host <-> device transfer accounting.
//
// Measured separately from kernel time, deliberately. In a real engine the PCIe
// link is the bottleneck long before the arithmetic is, and a CPU/GPU table
// that hides that cost inside the total says nothing useful.
// ---------------------------------------------------------
struct TransferStats {
    size_t to_device_bytes = 0;
    size_t to_host_bytes = 0;
    size_t to_device_count = 0;
    size_t to_host_count = 0;
    double to_device_seconds = 0.0;
    double to_host_seconds = 0.0;
};
TransferStats transfer_stats();
void reset_transfer_stats();

namespace detail {

// The memory primitives Storage uses. They are not part of the public API: they
// are declared here so that engine/detail/storage.hpp does not have to include
// the CUDA headers, which would drag nvcc into every translation unit.
// Launch accounting, kept by the kernel dispatcher.
void note_kernel_launched();
void note_kernel_failed();

float* device_alloc(size_t elements);
void device_free(float* ptr);
void copy_to_device(float* dst, const float* src, size_t elements);
void copy_to_host(float* dst, const float* src, size_t elements);
// Duplicates a buffer without leaving the device. Deliberately not counted in
// TransferStats: those figures measure PCIe, and putting a copy that never
// crosses it in there would stop the number meaning what it claims to.
void copy_device_to_device(float* dst, const float* src, size_t elements);

}  // namespace detail

}  // namespace cuda
}  // namespace engine

#endif  // ENGINE_CUDA_HPP
