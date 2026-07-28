// Backend CUDA cuando el motor se compila sin CUDA.
//
// No es un fichero de relleno: es lo que permite que engine/cuda.hpp forme
// parte de la API pública sin obligar a nadie a instalar el toolkit, y que el
// despacho en src/tensor.cpp sea una condición normal en lugar de un #ifdef
// por operación. Todas las operaciones devuelven false —«no lo hice»— y el
// llamante sigue por CPU.

#ifndef ENGINE_CUDA

#include "engine/cuda.hpp"
#include "engine/detail/cuda_ops.hpp"

#include <stdexcept>

namespace engine {
namespace cuda {

bool available() { return false; }
bool enabled() { return false; }
void set_enabled(bool) {}

DeviceInfo device_info() { return DeviceInfo{}; }
void synchronize() {}

int compiled_version() { return 0; }
int runtime_version() { return 0; }
int driver_version() { return 0; }
size_t kernels_launched() { return 0; }
size_t kernels_failed() { return 0; }
void reset_kernel_counters() {}

double peak_fp32_gflops() { return 0.0; }
double peak_bandwidth_gbs() { return 0.0; }

MatmulKernel matmul_kernel() { return MatmulKernel::Auto; }
void set_matmul_kernel(MatmulKernel) {}
MatmulKernel resolve_matmul_kernel(size_t, size_t, size_t) { return MatmulKernel::Auto; }

// Este sí devuelve algo útil sin CUDA: el banco de pruebas y las pruebas
// imprimen el nombre de la variante aunque no haya dispositivo.
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

size_t min_matmul_flops() { return 0; }
size_t min_elementwise_elements() { return 0; }
void set_thresholds(size_t, size_t) {}

TransferStats transfer_stats() { return TransferStats{}; }
void reset_transfer_stats() {}

namespace detail {

// Storage sólo llama a estas cuando alguien pide el lado del dispositivo, y
// sin ENGINE_CUDA ese código ni se compila. Llegar aquí sería un error de
// programación, no una condición de ejecución.
void note_kernel_launched() {}
void note_kernel_failed() {}

float* device_alloc(size_t) {
    throw std::logic_error("El motor se compilo sin CUDA: no hay memoria de dispositivo.");
}
void device_free(float*) {}
void copy_to_device(float*, const float*, size_t) {
    throw std::logic_error("El motor se compilo sin CUDA: no hay memoria de dispositivo.");
}
void copy_to_host(float*, const float*, size_t) {
    throw std::logic_error("El motor se compilo sin CUDA: no hay memoria de dispositivo.");
}

} // namespace detail

namespace ops {

bool binary(Binary, const Storage&, const Storage&, Storage&, size_t, size_t) { return false; }
bool matmul(const Storage&, const Storage&, Storage&, size_t, size_t, size_t, size_t, bool, bool) {
    return false;
}
bool relu(const Storage&, Storage&) { return false; }
bool relu_backward(const Storage&, const Storage&, Storage&) { return false; }
bool accumulate_grad(Storage&, const Storage&, bool) { return false; }
bool softmax(const Storage&, Storage&, size_t, size_t) { return false; }
bool softmax_backward(const Storage&, const Storage&, Storage&, size_t, size_t) { return false; }

} // namespace ops

} // namespace cuda
} // namespace engine

#endif // !ENGINE_CUDA
