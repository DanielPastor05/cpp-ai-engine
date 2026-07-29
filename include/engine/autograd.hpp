#ifndef ENGINE_AUTOGRAD_HPP
#define ENGINE_AUTOGRAD_HPP

#include <vector>
#include <unordered_set>
#include <memory>
#include <functional>

namespace engine {

class Tensor;

namespace autograd {

// Global graph-construction state. While it is off, operations register neither
// nodes nor gradient functions.
bool grad_enabled();
void set_grad_enabled(bool enabled);

// RAII guard that temporarily disables graph recording. Used during the
// backward pass (gradients do not need gradients) and inside the optimisers.
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

// Runs the topological sort (DFS) and propagates gradients backwards from
// root_tensor.
void backward(Tensor& root_tensor);

}  // namespace autograd
}  // namespace engine

#endif  // ENGINE_AUTOGRAD_HPP
