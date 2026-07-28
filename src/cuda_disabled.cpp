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

size_t min_matmul_flops() { return 0; }
size_t min_elementwise_elements() { return 0; }
void set_thresholds(size_t, size_t) {}

TransferStats transfer_stats() { return TransferStats{}; }
void reset_transfer_stats() {}

namespace detail {

// Storage sólo llama a estas cuando alguien pide el lado del dispositivo, y
// sin ENGINE_CUDA ese código ni se compila. Llegar aquí sería un error de
// programación, no una condición de ejecución.
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
bool softmax(const Storage&, Storage&, size_t, size_t) { return false; }
bool softmax_backward(const Storage&, const Storage&, Storage&, size_t, size_t) { return false; }

} // namespace ops

} // namespace cuda
} // namespace engine

#endif // !ENGINE_CUDA
