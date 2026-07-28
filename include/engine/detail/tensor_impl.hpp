#ifndef ENGINE_DETAIL_TENSOR_IMPL_HPP
#define ENGINE_DETAIL_TENSOR_IMPL_HPP

// Representación interna de un nodo del grafo. Solo la necesitan la propia
// librería y quien registre operaciones nuevas; se mantiene fuera de
// engine/tensor.hpp para no arrastrar <functional> a cada unidad de
// traducción que solo quiere usar tensores.

#include <functional>
#include <memory>
#include <vector>

#include "engine/detail/storage.hpp"

namespace engine {

class Tensor;

// Estructura interna para almacenar el estado y los nodos del grafo Autograd.
//
// Nota sobre la propiedad de la memoria: un nodo referencia a sus padres con
// shared_ptr (aristas hijo -> padre) y nunca a sus hijos, de modo que el grafo
// es acíclico también en el conteo de referencias. Por eso backward_fn recibe
// el gradiente de salida como argumento en lugar de capturar su propio tensor:
// capturarlo crearía un ciclo y el grafo jamás se liberaría.
struct TensorImpl {
    // El búfer y de qué lado vive. Antes era un std::vector<float> a secas;
    // sacarlo a un tipo propio es lo que permite que un tensor resida en la GPU
    // entre operaciones en vez de ir y volver por PCIe en cada una.
    Storage storage;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    bool requires_grad = false;
    std::shared_ptr<TensorImpl> grad = nullptr;

    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void(const Tensor&)> backward_fn = nullptr;

    TensorImpl() = default;
    TensorImpl(const std::vector<size_t>& s, float fill_val = 0.0f, bool req_grad = false);
    TensorImpl(const std::vector<size_t>& s, const std::vector<float>& d, bool req_grad = false);

    void compute_strides();
    size_t get_flat_index(const std::vector<size_t>& indices) const;
};

} // namespace engine

#endif // ENGINE_DETAIL_TENSOR_IMPL_HPP
