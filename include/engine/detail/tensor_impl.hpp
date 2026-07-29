#ifndef ENGINE_DETAIL_TENSOR_IMPL_HPP
#define ENGINE_DETAIL_TENSOR_IMPL_HPP

// The internal representation of a graph node. Only the library itself and
// anyone registering new operations need it; it is kept out of
// engine/tensor.hpp so that <functional> is not dragged into every translation
// unit that just wants to use tensors.

#include <functional>
#include <memory>
#include <vector>

#include "engine/detail/storage.hpp"

namespace engine {

class Tensor;

// Internal structure holding the state and the autograd graph edges.
//
// A note on ownership: a node references its parents with shared_ptr (child ->
// parent edges) and never its children, so the graph is acyclic in the
// reference count as well. That is why backward_fn takes the output gradient as
// an argument instead of capturing its own tensor: capturing it would create a
// cycle and the graph would never be freed.
struct TensorImpl {
    // The buffer and which side it lives on. This used to be a plain
    // std::vector<float>; lifting it into a type of its own is what lets a
    // tensor stay resident on the GPU between operations instead of crossing
    // PCIe on each one.
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

}  // namespace engine

#endif  // ENGINE_DETAIL_TENSOR_IMPL_HPP
