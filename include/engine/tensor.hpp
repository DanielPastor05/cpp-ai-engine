#ifndef ENGINE_TENSOR_HPP
#define ENGINE_TENSOR_HPP

// Every translation unit in the engine includes this header, so it pulls in
// nothing beyond what declaring Tensor needs. The rest lives where it is used:
//   - TensorImpl (and <functional>) in engine/detail/tensor_impl.hpp
//   - global_rng() (and <random>) in engine/random.hpp
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/detail/tensor_impl.hpp"

namespace engine {

struct TensorImpl;

// Seeds the global generator used by rand()/randn() and by layer
// initialisation, so a training run can be reproduced. The generator itself is
// declared in <engine/random.hpp>.
//
// Warning: it is a single shared object with no protection against concurrent
// access. The engine does not thread through it; if that ever changes, it needs
// either one generator per thread or a lock.
void manual_seed(uint64_t seed);

class Tensor {
private:
    std::shared_ptr<TensorImpl> impl_;

    // Private constructor wrapping an existing TensorImpl
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

public:
    // Public constructors
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape, float fill_value = 0.0f, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad = false);

    // Static helper for wrapping a shared implementation
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    // Factory methods
    static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor rand(const std::vector<size_t>& shape, float min_val = -1.0f, float max_val = 1.0f, bool requires_grad = false);
    static Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float stddev = 1.0f, bool requires_grad = false);

    // Autograd and gradients
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    Tensor grad() const;
    bool has_grad() const;
    void zero_grad();
    void add_grad(const Tensor& g);
    void backward();
    void backward(const Tensor& grad_output);

    // Access to the shared internal implementation
    std::shared_ptr<TensorImpl> get_impl() const { return impl_; }

    // Indexing and properties.
    //
    // The vector overload is generic but allocates on every access; to walk a
    // matrix, prefer the (row, col) overload.
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& at(size_t flat_index);
    const float& at(size_t flat_index) const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    const std::vector<float>& data() const;
    std::vector<float>& data();
    size_t size() const;
    size_t ndim() const;
    std::string shape_str() const;

    // Arithmetic, with autograd support.
    //
    // All four operators broadcast the right-hand operand by suffix: (N,) or
    // (1, N) over (M, N) for a bias, (S, D) over (B, S, D) for a positional
    // encoding, (S, S) over (B, H, S, S) for a mask, and {1} as a scalar over
    // any shape.
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // Concatenates along an axis; every other dimension must match.
    static Tensor concat(const std::vector<Tensor>& parts, size_t axis);
    // Stacks, creating a new axis at the given position.
    static Tensor stack(const std::vector<Tensor>& parts, size_t axis = 0);

    // matmul is batched: (B..., M, K) x (B..., K, N) -> (B..., M, N).
    // transpose swaps the last two axes; permute reorders all of them.
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor permute(const std::vector<size_t>& order) const;
    Tensor relu() const;
    Tensor softmax() const;   // over the last axis
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    // Reductions. With no argument they reduce to a {1} scalar; with an axis
    // they drop it (or leave it at 1 with keepdim), as in NumPy or PyTorch.
    Tensor sum() const;
    Tensor mean() const;
    Tensor sum(size_t axis, bool keepdim = false) const;
    Tensor mean(size_t axis, bool keepdim = false) const;
    Tensor max(size_t axis, bool keepdim = false) const;

    // Contiguous sub-tensor along an axis: [start, start + count).
    Tensor slice(size_t axis, size_t start, size_t count) const;

    // Gathers the given elements of the first axis into a mini-batch: rows from
    // an (M, N), whole images from an (N, C, H, W). This is the operation that
    // makes mini-batch training possible. Indices may repeat: their gradients
    // accumulate into the source element.
    Tensor select_rows(const std::vector<size_t>& indices) const;

    // A copy detached from the graph (shares shape and values, not history)
    Tensor detach() const;

    // Formatting and printing
    void print(const std::string& name = "") const;
};

// Scalar-on-the-left overloads, so that 2.0f * t is valid
Tensor operator+(float scalar, const Tensor& t);
Tensor operator-(float scalar, const Tensor& t);
Tensor operator*(float scalar, const Tensor& t);

} // namespace engine

#endif // ENGINE_TENSOR_HPP
