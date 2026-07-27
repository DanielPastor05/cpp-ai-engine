#ifndef ENGINE_AUTOGRAD_HPP
#define ENGINE_AUTOGRAD_HPP

#include <vector>
#include <unordered_set>
#include <memory>
#include <functional>

namespace engine {

class Tensor;

namespace autograd {

// Estado global de construccion del grafo. Cuando esta desactivado, las
// operaciones no registran nodos ni funciones de gradiente.
bool grad_enabled();
void set_grad_enabled(bool enabled);

// Guarda RAII para desactivar temporalmente el registro del grafo.
// Se usa durante la propagacion hacia atras (los gradientes no necesitan
// gradientes) y en los optimizadores.
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

// Ejecuta el ordenamiento topologico (DFS) y la propagacion de gradientes
// hacia atras (Backpropagation) partiendo de root_tensor.
void backward(Tensor& root_tensor);

} // namespace autograd
} // namespace engine

#endif // ENGINE_AUTOGRAD_HPP
