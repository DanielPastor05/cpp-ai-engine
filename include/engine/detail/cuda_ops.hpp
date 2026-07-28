#ifndef ENGINE_DETAIL_CUDA_OPS_HPP
#define ENGINE_DETAIL_CUDA_OPS_HPP

#include <cstddef>

#include "engine/detail/storage.hpp"

namespace engine {
namespace cuda {
namespace ops {

// ---------------------------------------------------------
// Puntos de entrada a los kernels.
//
// Todas devuelven **true si el trabajo se hizo en el dispositivo**. Devolver
// false significa «aquí no compensa» o «este binario no lleva CUDA», y el
// llamante sigue por el camino de CPU sin enterarse de nada más. Ese contrato
// es lo que mantiene src/tensor.cpp legible: una condición por operación, no
// dos implementaciones enredadas.
//
// Sin ENGINE_CUDA existen igualmente, implementadas en src/cuda_disabled.cpp
// devolviendo false, así que el despacho no necesita #ifdef.
// ---------------------------------------------------------

enum class Binary { Add, Sub, Mul, Div };

// out = a `op` b, con difusión por sufijo: b tiene `inner` elementos que se
// repiten `repeat` veces para cubrir a. Sin difusión, inner == a.size() y
// repeat == 1.
bool binary(Binary op, const Storage& a, const Storage& b, Storage& out,
            size_t inner, size_t repeat);

// out = a x b por lotes. Un operando no lotificado (a_batched/b_batched a
// false) se reutiliza para todas las matrices del lote, igual que en CPU.
bool matmul(const Storage& a, const Storage& b, Storage& out,
            size_t batch, size_t rows, size_t inner_dim, size_t cols,
            bool a_batched, bool b_batched);

bool relu(const Storage& x, Storage& out);
bool relu_backward(const Storage& x, const Storage& grad_out, Storage& out);

// Softmax sobre el último eje: `rows` filas de `cols` valores contiguos.
bool softmax(const Storage& x, Storage& out, size_t rows, size_t cols);
// y es la salida guardada del forward, no la entrada.
bool softmax_backward(const Storage& y, const Storage& grad_out, Storage& out,
                      size_t rows, size_t cols);

} // namespace ops
} // namespace cuda
} // namespace engine

#endif // ENGINE_DETAIL_CUDA_OPS_HPP
