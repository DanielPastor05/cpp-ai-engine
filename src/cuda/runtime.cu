// Contexto, memoria y contabilidad de transferencias del backend CUDA.
//
// Los kernels están en src/cuda/kernels.cu. Aquí sólo vive lo que rodea al
// cálculo: descubrir el dispositivo, reservar y liberar memoria, y medir lo
// que cuesta cruzar el PCIe.

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

// Una llamada de CUDA que falla no debe pasar desapercibida: sin comprobarlas,
// un error de reserva se manifiesta mucho más tarde como resultados
// silenciosamente incorrectos.
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + what + ": " +
                                 cudaGetErrorString(status));
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

// Núcleos CUDA por multiprocesador. No hay ninguna propiedad del runtime que
// lo dé: depende de la arquitectura y hay que mirarlo en una tabla, igual que
// hace el helper_cuda.h de los ejemplos del toolkit. Sin este número no se
// puede calcular el pico teórico, y sin el pico teórico un GFLOP/s medido no
// dice si el kernel va bien o mal.
int cores_per_sm(int major, int minor) {
    switch (major) {
        case 3: return 192;                          // Kepler
        case 5: return 128;                          // Maxwell
        case 6: return (minor == 0) ? 64 : 128;      // Pascal
        case 7: return 64;                           // Volta y Turing
        case 8: return (minor == 0) ? 64 : 128;      // Ampere: GA100 vs el resto
        case 9: return 128;                          // Hopper
        default: return 128;                         // lo más probable de aquí en adelante
    }
}

struct Context {
    bool usable = false;
    bool on = false;
    DeviceInfo info;
    double peak_gflops = 0.0;
    double peak_bandwidth = 0.0;

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

        // clockRate viene en kHz. El factor 2 es porque una FMA cuenta como
        // dos operaciones en punto flotante, que es el convenio con el que se
        // publican las cifras de las tarjetas.
        peak_gflops = static_cast<double>(props.multiProcessorCount) *
                      cores_per_sm(props.major, props.minor) * 2.0 *
                      (static_cast<double>(props.clockRate) * 1e3) / 1e9;

        // El otro 2 es la doble tasa de transferencia de la memoria gráfica.
        peak_bandwidth = (static_cast<double>(props.memoryClockRate) * 1e3) * 2.0 *
                         (static_cast<double>(props.memoryBusWidth) / 8.0) / 1e9;

        usable = true;

        // ENGINE_CUDA=0 apaga el backend sin recompilar, para comparar contra
        // el camino de CPU en la misma máquina y el mismo binario.
        const char* flag = std::getenv("ENGINE_CUDA");
        on = !(flag != nullptr && (flag[0] == '0' || flag[0] == 'n' || flag[0] == 'N'));
    }
};

Context& context() {
    static Context ctx;
    return ctx;
}

// Los umbrales y los contadores los toca sólo el hilo que despacha: el reparto
// entre hilos del motor vive en el camino de CPU, y ese no llega hasta aquí.
size_t g_min_matmul_flops = env_size("ENGINE_CUDA_MIN_FLOPS", size_t{1} << 22);
size_t g_min_elements = env_size("ENGINE_CUDA_MIN_ELEMENTS", size_t{1} << 20);

MatmulKernel g_matmul_kernel = MatmulKernel::Auto;

TransferStats g_stats;
size_t g_launched = 0;
size_t g_failed = 0;

double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

bool available() { return context().usable; }

bool enabled() {
    const Context& ctx = context();
    return ctx.usable && ctx.on;
}

void set_enabled(bool on) {
    Context& ctx = context();
    ctx.on = on && ctx.usable;
}

DeviceInfo device_info() { return context().info; }

void synchronize() {
    if (!context().usable) return;
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

int runtime_version() {
    int version = 0;
    return (cudaRuntimeGetVersion(&version) == cudaSuccess) ? version : 0;
}

int driver_version() {
    int version = 0;
    return (cudaDriverGetVersion(&version) == cudaSuccess) ? version : 0;
}

size_t kernels_launched() { return g_launched; }
size_t kernels_failed() { return g_failed; }
void reset_kernel_counters() { g_launched = 0; g_failed = 0; }

double peak_fp32_gflops() { return context().peak_gflops; }
double peak_bandwidth_gbs() { return context().peak_bandwidth; }

MatmulKernel matmul_kernel() { return g_matmul_kernel; }
void set_matmul_kernel(MatmulKernel kernel) { g_matmul_kernel = kernel; }

const char* matmul_kernel_name(MatmulKernel kernel) {
    switch (kernel) {
        case MatmulKernel::Auto: return "auto";
        case MatmulKernel::Naive: return "naive";
        case MatmulKernel::Tiled: return "tiled";
        case MatmulKernel::RegisterTiled: return "register";
        case MatmulKernel::Vectorized: return "vectorized";
    }
    return "desconocido";
}

// La resolución de `Auto`, en un solo sitio para que el banco de pruebas pueda
// preguntar qué se va a ejecutar sin duplicar el criterio.
//
// Dos condiciones, y las dos son de corrección antes que de velocidad:
//   - Las cargas float4 exigen que las filas empiecen en una dirección
//     múltiplo de 16 bytes, o sea K y N múltiplos de 4. Es la misma clase de
//     selección de kernel por alineación que hace cuBLAS por dentro.
//   - Con matrices pequeñas, un bloque de 128x128 desperdicia casi toda la
//     malla rellenando con ceros, así que ahí gana el kernel de teselas.
MatmulKernel resolve_matmul_kernel(size_t rows, size_t inner_dim, size_t cols) {
    if (g_matmul_kernel != MatmulKernel::Auto) return g_matmul_kernel;
    if (rows < 128 || cols < 128) return MatmulKernel::Tiled;
    if (inner_dim % 4 == 0 && cols % 4 == 0) return MatmulKernel::Vectorized;
    return MatmulKernel::RegisterTiled;
}

size_t min_matmul_flops() { return g_min_matmul_flops; }
size_t min_elementwise_elements() { return g_min_elements; }

void set_thresholds(size_t matmul_flops, size_t elementwise_elements) {
    g_min_matmul_flops = matmul_flops;
    g_min_elements = elementwise_elements;
}

TransferStats transfer_stats() { return g_stats; }
void reset_transfer_stats() { g_stats = TransferStats{}; }

namespace detail {

// Las incrementa launched_ok(), en kernels.cu.
void note_kernel_launched() { ++g_launched; }
void note_kernel_failed() { ++g_failed; }

float* device_alloc(size_t elements) {
    if (elements == 0) return nullptr;
    void* ptr = nullptr;
    check(cudaMalloc(&ptr, elements * sizeof(float)), "cudaMalloc");
    return static_cast<float*>(ptr);
}

void device_free(float* ptr) {
    if (ptr == nullptr) return;
    // El destructor de Storage llama aquí, así que no puede lanzar: durante el
    // apagado del proceso el contexto de CUDA puede haberse destruido ya, y
    // eso no es un fallo que merezca terminar el programa.
    cudaFree(ptr);
}

void copy_to_device(float* dst, const float* src, size_t elements) {
    if (elements == 0) return;
    const auto start = std::chrono::steady_clock::now();
    check(cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
    g_stats.to_device_seconds += seconds_since(start);
    g_stats.to_device_bytes += elements * sizeof(float);
    ++g_stats.to_device_count;
}

void copy_to_host(float* dst, const float* src, size_t elements) {
    if (elements == 0) return;
    const auto start = std::chrono::steady_clock::now();
    // cudaMemcpy es sincronizante, así que este tiempo incluye la espera a que
    // terminen los kernels pendientes sobre el búfer de origen. Es justo lo
    // que se quiere medir: el coste real de leer un resultado desde host.
    check(cudaMemcpy(dst, src, elements * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");
    g_stats.to_host_seconds += seconds_since(start);
    g_stats.to_host_bytes += elements * sizeof(float);
    ++g_stats.to_host_count;
}

} // namespace detail

} // namespace cuda
} // namespace engine
