#include "engine/autograd.hpp"
#include "engine/tensor.hpp"

#include <utility>

#include <stdexcept>

namespace engine {
namespace autograd {

namespace {

// Per-thread state: each thread builds (or does not build) its own graph.
thread_local bool g_grad_enabled = true;

// Topological ordering by post-order DFS.
//
// Implemented iteratively on purpose: a recursive version exhausts the process
// stack on deep graphs (networks with many layers, or long training loops
// unrolled).
std::vector<std::shared_ptr<TensorImpl>> topological_sort(const std::shared_ptr<TensorImpl>& root) {
    std::vector<std::shared_ptr<TensorImpl>> topo_order;
    std::unordered_set<TensorImpl*> visited;
    // (node, index of the next parent to visit)
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

}  // namespace

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

// Gradient backpropagation
void backward(Tensor& root_tensor) {
    auto root_impl = root_tensor.get_impl();
    if (!root_impl) return;

    // If the root has no gradient assigned yet, assign 1.0f (dLoss/dLoss = 1).
    // That only makes sense for a scalar; for any other shape the initial gradient
    // has to be given explicitly with backward(grad_output).
    if (!root_impl->grad) {
        if (root_impl->storage.size() != 1) {
            throw std::runtime_error(
                "An implicit backward() is only allowed on a scalar; the root tensor has "
                "forma " +
                root_tensor.shape_str() + ". Usa backward(grad_output).");
        }
        root_impl->grad = std::make_shared<TensorImpl>(root_impl->shape, 1.0f, false);
    }

    // Computing the gradients must not itself build a graph: without this every
    // backward would leave a useless second-order graph behind.
    NoGradGuard no_grad;

    // 1. Obtain the graph's topological ordering
    std::vector<std::shared_ptr<TensorImpl>> topo_order = topological_sort(root_impl);

    // 2. Discard intermediate nodes' gradients from a previous call.
    //    A node with a backward_fn is the result of an operation, not a leaf: its
    //    gradient is a temporary of this traversal. Kept between calls, a second
    //    backward over the same graph would propagate the sum of both traversals
    //    and multiply the leaves' gradients.
    //    Only leaves (no backward_fn) accumulate, exactly as in PyTorch.
    for (const auto& node : topo_order) {
        if (node != root_impl && node->backward_fn) {
            node->grad = nullptr;
        }
    }

    // 3. Run the gradient functions in reverse topological order.
    //
    //    An intermediate node's gradient is freed as soon as it has been consumed.
    //    In reverse order, by the time a node is reached all its children have
    //    passed, so its gradient is complete and, once propagated to the parents,
    //    nobody else needs it. Keeping it until the graph died multiplied a
    //    backward's memory by more than twenty.
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        const auto& node = *it;
        // A node with no accumulated gradient propagates nothing (dead branch).
        if (node->backward_fn && node->grad) {
            node->backward_fn(Tensor::from_impl(node->grad));
            if (node != root_impl) node->grad = nullptr;
        }
    }
}

}  // namespace autograd
}  // namespace engine
