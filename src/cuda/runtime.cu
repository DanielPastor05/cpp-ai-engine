// Context, memory and transfer accounting for the CUDA backend.
//
// The kernels live in src/cuda/kernels.cu. Only what surrounds the computation
// is here: discovering the device, allocating and freeing memory, and measuring
// what crossing PCIe costs.

#include "engine/cuda.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace engine {
namespace cuda {

namespace {

// A failing CUDA call must not go unnoticed: unchecked, an allocation error
// shows up much later as silently incorrect results.
//
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(status));
    }
}

size_t env_size(const char* name, size_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw) return fallback;
    return static_cast<size_t>(value);
}

// CUDA cores per multiprocessor. No runtime property provides it: it depends on
// the architecture and has to be looked up in a table, exactly as the toolkit
// samples' helper_cuda.h does. Without this number the theoretical peak cannot
// be computed, and without the peak a measured GFLOP/s does not say whether the
// kernel is doing well or badly.
int cores_per_sm(int major, int minor) {
    switch (major) {
        case 3:
            return 192;  // Kepler
        case 5:
            return 128;  // Maxwell
        case 6:
            return (minor == 0) ? 64 : 128;  // Pascal
        case 7:
            return 64;  // Volta and Turing
        case 8:
            return (minor == 0) ? 64 : 128;  // Ampere: GA100 vs the rest
        case 9:
            return 128;  // Hopper
        default:
            return 128;  // the most likely from here on
    }
}

struct Context {
    bool usable = false;
    bool on = false;
    DeviceInfo info;
    double peak_gflops = 0.0;
    double peak_bandwidth = 0.0;
    // Whether the card supports the driver's memory pool. Checked once and constant
    // for the life of the process: device_alloc() and device_free() must always
    // pick the same route, because freeing a cudaMallocAsync pointer with cudaFree
    // (or the other way round) is undefined behaviour.
    bool memory_pools = false;

    Context() {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return;

        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) return;
        if (cudaSetDevice(0) != cudaSuccess) return;

        info.name = props.name;
        info.compute_major = props.major;
        info.compute_minor = props.minor;
        info.multiprocessors = props.multiProcessorCount;
        info.total_memory = props.totalGlobalMem;

        // The clocks are queried by attribute rather than from a cudaDeviceProp field:
        // CUDA 13 removed props.clockRate and props.memoryClockRate, and
        // cudaDeviceGetAttribute is the route that works the same on 11, 12 and 13.
        int clock_khz = 0;         // core clock, in kHz
        int memory_clock_khz = 0;  // memory clock, in kHz
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
        cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0);

        // The factor of 2 is because an FMA counts as two floating-point operations,
        // which is the convention the cards' published figures use.
        //
        peak_gflops = static_cast<double>(props.multiProcessorCount) *
                      cores_per_sm(props.major, props.minor) * 2.0 *
                      (static_cast<double>(clock_khz) * 1e3) / 1e9;

        // The other 2 is graphics memory's double data rate.
        peak_bandwidth = (static_cast<double>(memory_clock_khz) * 1e3) * 2.0 *
                         (static_cast<double>(props.memoryBusWidth) / 8.0) / 1e9;

        int pools = 0;
        cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, 0);
        memory_pools = (pools != 0);

        // The pool's release threshold defaults to **zero**, which means "give
        // every free block back to the operating system at the next
        // synchronisation". An engine that synchronises to read a result -- which
        // is every engine -- therefore re-acquires each output buffer from the
        // driver on every operation, and that cost grows with the buffer.
        //
        // Measured on a 16.7M-value addition: 3.5 ms of wall clock around a
        // kernel that nsys times at 0.48 ms. The kernel is at 93% of the card's
        // bandwidth; the other 3 ms was the allocator handing 64 MiB back and
        // asking for it again, twenty times a second.
        //
        // UINT64_MAX means never release, and it is not a guess: PyTorch's
        // c10/cuda/CUDAMallocAsyncAllocator.cpp sets the same attribute to the
        // same value, citing the same NVIDIA note on retaining memory in the
        // pool. Finding that afterwards is the only external corroboration any
        // decision in this engine has, so it is worth the two lines.
        //
        // The pool still bounds itself -- it reuses rather than grows -- and
        // freeing the context at exit returns everything. An earlier note here
        // recorded that raising this to 512 MiB
        // was worth only 6%; that measurement did not synchronise between
        // iterations, so the block was never released and there was nothing for
        // the threshold to change.
        if (memory_pools) {
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
                uint64_t never_release = UINT64_MAX;
                cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &never_release);
            }
        }

        usable = true;

        // ENGINE_CUDA=0 turns the backend off without recompiling, to compare against
        // the CPU path on the same machine and the same binary.
        const char* flag = std::getenv("ENGINE_CUDA");
        on = !(flag != nullptr && (flag[0] == '0' || flag[0] == 'n' || flag[0] == 'N'));
    }
};

Context& context() {
    static Context ctx;
    return ctx;
}

// Only the dispatching thread touches the thresholds and counters: the engine's
// thread splitting lives on the CPU path, and that never reaches here.
size_t g_min_matmul_flops = env_size("ENGINE_CUDA_MIN_FLOPS", size_t{1} << 22);
size_t g_min_elements = env_size("ENGINE_CUDA_MIN_ELEMENTS", size_t{1} << 20);

MatmulKernel g_matmul_kernel = MatmulKernel::Auto;

TransferStats g_stats;
size_t g_launched = 0;
size_t g_failed = 0;

double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

bool available() {
    return context().usable;
}

bool enabled() {
    const Context& ctx = context();
    return ctx.usable && ctx.on;
}

void set_enabled(bool on) {
    Context& ctx = context();
    ctx.on = on && ctx.usable;
}

DeviceInfo device_info() {
    return context().info;
}

void synchronize() {
    if (!context().usable) return;
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

// A compile-time constant: it is the only one of the three that really says
// which toolkit produced this binary.
int compiled_version() {
    return CUDART_VERSION;
}

int runtime_version() {
    int version = 0;
    return (cudaRuntimeGetVersion(&version) == cudaSuccess) ? version : 0;
}

int driver_version() {
    int version = 0;
    return (cudaDriverGetVersion(&version) == cudaSuccess) ? version : 0;
}

size_t kernels_launched() {
    return g_launched;
}
size_t kernels_failed() {
    return g_failed;
}
void reset_kernel_counters() {
    g_launched = 0;
    g_failed = 0;
}

double peak_fp32_gflops() {
    return context().peak_gflops;
}
double peak_bandwidth_gbs() {
    return context().peak_bandwidth;
}

MatmulKernel matmul_kernel() {
    return g_matmul_kernel;
}
void set_matmul_kernel(MatmulKernel kernel) {
    g_matmul_kernel = kernel;
}

const char* matmul_kernel_name(MatmulKernel kernel) {
    switch (kernel) {
        case MatmulKernel::Auto:
            return "auto";
        case MatmulKernel::Naive:
            return "naive";
        case MatmulKernel::Tiled:
            return "tiled";
        case MatmulKernel::RegisterTiled:
            return "register";
        case MatmulKernel::Vectorized:
            return "vectorized";
        case MatmulKernel::TensorCore:
            return "tensorcore";
        case MatmulKernel::TensorCoreFp16:
            return "tensorcore-fp16";
    }
    return "unknown";
}

// tf32 WMMA needs Ampere or later, both at compile time -- the kernel body is
// guarded on __CUDA_ARCH__ >= 800 -- and at run time. CUDART_VERSION covers the
// toolkit; the compute capability covers the card. A build for sm_75 running on
// an Ampere card would still have no tensor-core code in the binary, which is why
// the compiled architecture is not something this can infer and the kernel guard
// has to agree with it: both are >= 800 or the dispatch is refused.
bool tensor_cores_available() {
    const Context& ctx = context();
    if (!ctx.usable) return false;
#if defined(__CUDA_ARCH_LIST__)
    // Available since CUDA 11.5: the architectures this binary actually carries.
    // Without at least one >= 800 there is no tf32 kernel to launch, whatever
    // card is plugged in.
    constexpr int arch_list[] = {__CUDA_ARCH_LIST__};
    bool built_for_ampere = false;
    for (int a : arch_list) {
        if (a >= 800) built_for_ampere = true;
    }
    if (!built_for_ampere) return false;
#endif
    return ctx.info.compute_major >= 8;
}

// The resolution of `Auto`, in one place so the benchmark can ask what will run
// without duplicating the criterion.
//
// Two conditions, and both are about correctness before speed:
//   - float4 loads require rows to start at an address that is a multiple of
//     16 bytes, that is K and N multiples of 4. It is the same kind of
//     alignment-driven kernel selection cuBLAS does internally.
//   - With small matrices a 128x128 block wastes almost the whole grid padding
//     with zeros, so the tiled kernel wins there.
MatmulKernel resolve_matmul_kernel(size_t rows, size_t inner_dim, size_t cols) {
    if (g_matmul_kernel != MatmulKernel::Auto) return g_matmul_kernel;
    if (rows < 128 || cols < 128) return MatmulKernel::Tiled;
    if (inner_dim % 4 == 0 && cols % 4 == 0) return MatmulKernel::Vectorized;
    return MatmulKernel::RegisterTiled;
}

size_t min_matmul_flops() {
    return g_min_matmul_flops;
}
size_t min_elementwise_elements() {
    return g_min_elements;
}

void set_thresholds(size_t matmul_flops, size_t elementwise_elements) {
    g_min_matmul_flops = matmul_flops;
    g_min_elements = elementwise_elements;
}

TransferStats transfer_stats() {
    return g_stats;
}
void reset_transfer_stats() {
    g_stats = TransferStats{};
}

namespace detail {

// Incremented by launched_ok(), in kernels.cu.
void note_kernel_launched() {
    ++g_launched;
}
void note_kernel_failed() {
    ++g_failed;
}

// cudaMalloc and cudaFree synchronise the device and go down to the system,
// which costs milliseconds on large buffers. Since every engine operation
// creates an output tensor, that is paid per operation and ends up dominating
// the time: on a 16.7M-value addition I measured 3.64 ms of allocation and free
// against 0.51 for the kernel, so 86% of the time was not computing.
//
// The pool the driver already keeps behind cudaMallocAsync returns the block to
// a free list instead of to the system, and the next allocation of the same size
// reuses it. No custom allocator is needed for this.
//
// It only reuses the block if the pool is allowed to keep it, which by default
// it is not -- see the release threshold set in Context(). That one attribute
// took the same 16.7M addition from 3.5 ms to 0.50, against a kernel nsys times
// at 0.48: the engine's share of the operation went from 3 ms to 23 us.
//
// This paragraph used to say the threshold was worth 6% and not worth setting.
// That measurement never synchronised between iterations, so the block was
// never released and there was nothing for the threshold to change. The
// benchmark it came from was measuring the wrong thing, which is the whole
// reason bench/bench_kernels.cpp now measures at two sizes.
//
// The whole engine launches on the default stream, so the stream order
// cudaFreeAsync respects is the same one the memory was used in.
float* device_alloc(size_t elements) {
    if (elements == 0) return nullptr;
    void* ptr = nullptr;
    const size_t bytes = elements * sizeof(float);
    if (context().memory_pools) {
        check(cudaMallocAsync(&ptr, bytes, 0), "cudaMallocAsync");
    } else {
        check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    }
    return static_cast<float*>(ptr);
}

void device_free(float* ptr) {
    if (ptr == nullptr) return;
    // Storage's destructor calls in here, so it cannot throw: during process
    // shutdown the CUDA context may already have been destroyed, and that is not a
    // failure worth terminating the program over.
    const cudaError_t status = context().memory_pools ? cudaFreeAsync(ptr, 0) : cudaFree(ptr);

    // Not throwing is not the same as not noticing. Without this, a double free or
    // a pointer that did not come from this allocator are completely invisible: the
    // error sticks to the thread and shows up at the next CUDA call that does check,
    // pointing at somewhere that is not to blame. It is reported once, as with the
    // kernels, and cleared so as not to contaminate.
    if (status != cudaSuccess && status != cudaErrorCudartUnloading) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            std::fprintf(stderr,
                         "\nengine: could not free device memory (%s).\n"
                         "  Later ones are not repeated here.\n\n",
                         cudaGetErrorString(status));
        }
        cudaGetLastError();
    }
}

void copy_to_device(float* dst, const float* src, size_t elements) {
    if (elements == 0) return;
    const auto start = std::chrono::steady_clock::now();
    check(cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    g_stats.to_device_seconds += seconds_since(start);
    g_stats.to_device_bytes += elements * sizeof(float);
    ++g_stats.to_device_count;
}

// Not accounted for: it does not cross PCIe. It runs at the card's memory speed
// -- hundreds of GB/s against a PCIe 3.0 x16's ~12 -- which is exactly why it
// exists: reshape copies the whole buffer by definition, and doing it here
// avoids pulling it down and pushing it back up.
void copy_device_to_device(float* dst, const float* src, size_t elements) {
    if (elements == 0) return;
    check(cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyDeviceToDevice),
          "cudaMemcpy D2D");
}

void copy_to_host(float* dst, const float* src, size_t elements) {
    if (elements == 0) return;
    const auto start = std::chrono::steady_clock::now();
    // cudaMemcpy is synchronising, so this time includes waiting for kernels still
    // pending on the source buffer. That is exactly what should be measured: the
    // real cost of reading a result from the host.
    check(cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    g_stats.to_host_seconds += seconds_since(start);
    g_stats.to_host_bytes += elements * sizeof(float);
    ++g_stats.to_host_count;
}

}  // namespace detail

}  // namespace cuda
}  // namespace engine
