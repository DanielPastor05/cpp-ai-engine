#ifndef ENGINE_AUTOGRAD_HPP
#define ENGINE_AUTOGRAD_HPP

#include <vector>
#include <unordered_set>
#include <memory>
#include <functional>

namespace engine {

class Tensor;

namespace autograd {

// Ejecuta el algoritmo de ordenamiento topológico (DFS) y la propagación de gradientes hacia atrás (Backpropagation)
void backward(Tensor& root_tensor);

} // namespace autograd
} // namespace engine

#endif // ENGINE_AUTOGRAD_HPP
