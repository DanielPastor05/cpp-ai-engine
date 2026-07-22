#include "engine/autograd.hpp"
#include "engine/tensor.hpp"

namespace engine {
namespace autograd {

// Función auxiliar recursiva para ordenamiento topológico mediante DFS
static void build_topological_sort(const std::shared_ptr<TensorImpl>& node,
                                    std::vector<std::shared_ptr<TensorImpl>>& topo_order,
                                    std::unordered_set<TensorImpl*>& visited) {
    if (!node || visited.find(node.get()) != visited.end()) {
        return;
    }
    visited.insert(node.get());

    for (const auto& parent : node->parents) {
        if (parent) {
            build_topological_sort(parent, topo_order, visited);
        }
    }
    topo_order.push_back(node);
}

// Retropropagación de gradientes (Backpropagation)
void backward(Tensor& root_tensor) {
    auto root_impl = root_tensor.get_impl();
    if (!root_impl) return;

    // Si la raíz no tiene gradiente asignado aún, asignamos 1.0f (dLoss/dLoss = 1)
    if (!root_impl->grad) {
        root_impl->grad = std::make_shared<TensorImpl>(root_impl->shape, 1.0f, false);
    }

    // 1. Obtener el ordenamiento topológico del grafo
    std::vector<std::shared_ptr<TensorImpl>> topo_order;
    std::unordered_set<TensorImpl*> visited;
    build_topological_sort(root_impl, topo_order, visited);

    // 2. Ejecutar las funciones de gradiente en orden topológico inverso
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

} // namespace autograd
} // namespace engine
