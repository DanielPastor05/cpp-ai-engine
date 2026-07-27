#include "engine/autograd.hpp"
#include "engine/tensor.hpp"

#include <utility>

#include <stdexcept>

namespace engine {
namespace autograd {

namespace {

// Estado por hilo: cada hilo construye (o no) su propio grafo.
thread_local bool g_grad_enabled = true;

// Ordenamiento topológico por DFS post-orden.
//
// Se implementa de forma iterativa a propósito: una versión recursiva agota la
// pila del proceso en grafos profundos (redes con muchas capas o bucles de
// entrenamiento largos desenrollados).
std::vector<std::shared_ptr<TensorImpl>> topological_sort(const std::shared_ptr<TensorImpl>& root) {
    std::vector<std::shared_ptr<TensorImpl>> topo_order;
    std::unordered_set<TensorImpl*> visited;
    // (nodo, índice del siguiente padre por visitar)
    std::vector<std::pair<std::shared_ptr<TensorImpl>, size_t>> stack;

    if (!root) return topo_order;

    stack.emplace_back(root, 0);
    visited.insert(root.get());

    while (!stack.empty()) {
        auto& frame = stack.back();
        const auto& parents = frame.first->parents;

        if (frame.second < parents.size()) {
            const auto& parent = parents[frame.second++];
            if (parent && visited.insert(parent.get()).second) {
                stack.emplace_back(parent, 0);
            }
        } else {
            topo_order.push_back(std::move(frame.first));
            stack.pop_back();
        }
    }
    return topo_order;
}

} // namespace

bool grad_enabled() {
    return g_grad_enabled;
}

void set_grad_enabled(bool enabled) {
    g_grad_enabled = enabled;
}

NoGradGuard::NoGradGuard() : previous_(g_grad_enabled) {
    g_grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    g_grad_enabled = previous_;
}

// Retropropagación de gradientes (Backpropagation)
void backward(Tensor& root_tensor) {
    auto root_impl = root_tensor.get_impl();
    if (!root_impl) return;

    // Si la raíz no tiene gradiente asignado aún, asignamos 1.0f (dLoss/dLoss = 1).
    // Solo tiene sentido para un escalar; para cualquier otra forma hay que
    // indicar explícitamente el gradiente inicial con backward(grad_output).
    if (!root_impl->grad) {
        if (root_impl->data.size() != 1) {
            throw std::runtime_error(
                "backward() implícito solo esta permitido sobre un escalar; el tensor raíz tiene forma " +
                root_tensor.shape_str() + ". Usa backward(grad_output).");
        }
        root_impl->grad = std::make_shared<TensorImpl>(root_impl->shape, 1.0f, false);
    }

    // El cálculo de los gradientes no debe, a su vez, construir grafo:
    // sin esto cada backward dejaría atrás un grafo de segundo orden inútil.
    NoGradGuard no_grad;

    // 1. Obtener el ordenamiento topológico del grafo
    std::vector<std::shared_ptr<TensorImpl>> topo_order = topological_sort(root_impl);

    // 2. Descartar el gradiente de los nodos intermedios de una llamada previa.
    //    Un nodo con backward_fn es el resultado de una operación, no una hoja:
    //    su gradiente es un valor temporal de este recorrido. Si se conservase
    //    entre llamadas, un segundo backward sobre el mismo grafo propagaría la
    //    suma de ambos recorridos y multiplicaría los gradientes de las hojas.
    //    Solo las hojas (sin backward_fn) acumulan, igual que en PyTorch.
    for (const auto& node : topo_order) {
        if (node != root_impl && node->backward_fn) {
            node->grad = nullptr;
        }
    }

    // 3. Ejecutar las funciones de gradiente en orden topológico inverso.
    //
    //    El gradiente de un nodo intermedio se libera en cuanto se ha
    //    consumido. En orden inverso, al llegar a un nodo ya han pasado todos
    //    sus hijos, así que su gradiente está completo y, una vez propagado a
    //    los padres, nadie más lo necesita. Conservarlo hasta que muera el
    //    grafo multiplicaba por más de veinte la memoria de un backward.
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        const auto& node = *it;
        // Un nodo sin gradiente acumulado no propaga nada (rama muerta del grafo).
        if (node->backward_fn && node->grad) {
            node->backward_fn(Tensor::from_impl(node->grad));
            if (node != root_impl) node->grad = nullptr;
        }
    }
}

} // namespace autograd
} // namespace engine
