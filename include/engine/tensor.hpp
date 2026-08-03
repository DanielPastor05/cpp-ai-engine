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
    explicit Tensor(const std::vector<size_t>& shape, float fill_value = 0.0f,
                    bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data,
           bool requires_grad = false);

    // Static helper for wrapping a shared implementation
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    // Factory methods
    static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor rand(const std::vector<size_t>& shape, float min_val = -1.0f,
                       float max_val = 1.0f, bool requires_grad = false);
    static Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float stddev = 1.0f,
                        bool requires_grad = false);

    // Autograd and gradients
    [[nodiscard]] bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    [[nodiscard]] Tensor grad() const;
    [[nodiscard]] bool has_grad() const;
    void zero_grad();
    void add_grad(const Tensor& g);
    void backward();
    void backward(const Tensor& grad_output);

    // The element buffer and its device mirror.
    //
    // Fifty-three of the seventy-eight uses of get_impl() below reached straight
    // through it for this member, and every one of them copied a shared_ptr --
    // an atomic increment and decrement -- to travel two hops. That happens on
    // the dispatch path of every operation. This is the same thing without the
    // ownership handle escaping and without the refcount.
    //
    // Storage lives under detail/ and is not part of the stable API. It is here
    // because the CUDA entry points in engine/detail/cuda_ops.hpp take one, and
    // a caller writing a kernel dispatch needs to hand it over.
    [[nodiscard]] Storage& storage() noexcept { return impl_->storage; }
    [[nodiscard]] const Storage& storage() const noexcept { return impl_->storage; }

    // The whole node: buffer, shape, gradient, and the edges of the autograd
    // graph. **Not part of the stable API** -- see the stability policy in the
    // README. It exists because building a graph node from outside src/ needs to
    // write `parents` and `backward_fn`, and the twenty-five call sites that do
    // are all inside this engine.
    //
    // Returns a reference rather than a copy: callers overwhelmingly want to
    // reach through it, not to keep it alive, and the copy was a refcount they
    // were not asking for.
    [[nodiscard]] const std::shared_ptr<TensorImpl>& get_impl() const noexcept { return impl_; }

    // Indexing and properties.
    //
    // The vector overload is generic but allocates on every access; to walk a
    // matrix, prefer the (row, col) overload.
    [[nodiscard]] size_t get_flat_index(const std::vector<size_t>& indices) const;
    float& operator()(const std::vector<size_t>& indices);
    [[nodiscard]] const float& operator()(const std::vector<size_t>& indices) const;
    float& operator()(size_t row, size_t col);
    [[nodiscard]] const float& operator()(size_t row, size_t col) const;
    float& at(size_t flat_index);
    [[nodiscard]] const float& at(size_t flat_index) const;

    [[nodiscard]] const std::vector<size_t>& shape() const;
    [[nodiscard]] const std::vector<size_t>& strides() const;

    // The element buffer, as a pointer, with `size()` for the count.
    //
    // These used to return `const std::vector<float>&` and `std::vector<float>&`,
    // and the writable one was a hole: a caller could `t.data().resize(0)` and
    // leave the tensor claiming a shape whose elements no longer exist -- no
    // error anywhere, a segfault somewhere else entirely. `src/serialize.cpp`
    // came close, replacing a tensor's whole buffer by assigning over the
    // reference. Nothing checked that the new one was the right length. A
    // library that publishes an API stability policy should not also hand out a
    // handle to its own invariants.
    //
    // A pointer carries exactly the authority a caller needs -- read the
    // elements, write the elements -- and none of the authority to change how
    // many there are. It is also what the engine wanted all along: about twenty
    // call sites inside src/ immediately did `.data().data()` to get back to a
    // pointer, and only a handful used the container.
    //
    // Neither is a plain field access. `data()` is the door to the host side of
    // `Storage`: calling it downloads from the device if the host copy has gone
    // stale, and the non-const one additionally marks the device copy stale.
    // Hoist it out of loops -- that is worth ~10% on the element-wise operators,
    // measured, and `docs/PERFORMANCE.md` has the profile that found it.
    [[nodiscard]] const float* data() const;
    [[nodiscard]] float* data();

    // The elements as a container, by value.
    //
    // Wanting a vector of the values is a real need -- comparing two results,
    // holding a snapshot across an operation that overwrites the original -- and
    // it is the need the old `data()` was being used for. The difference is the
    // copy: this hands over a vector the caller owns, so resizing it or keeping
    // it is their business and the tensor's invariants are not involved.
    [[nodiscard]] std::vector<float> to_vector() const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t ndim() const;
    [[nodiscard]] std::string shape_str() const;

    // Every method above and below that returns a value carries [[nodiscard]].
    // These are pure functions over an immutable handle -- `a.relu();` on its own
    // line computes a tensor and throws it away, which is never what anybody
    // meant. The compiler can say so, so it should.
    //
    // The mutators are deliberately without it: zero_grad(), add_grad(),
    // backward(), set_requires_grad() and print() are called for their effect.
    // So are the non-const accessors, which hand out a reference to write
    // through.
    //
    // Arithmetic, with autograd support.
    //
    // All four operators broadcast the right-hand operand by suffix: (N,) or
    // (1, N) over (M, N) for a bias, (S, D) over (B, S, D) for a positional
    // encoding, (S, S) over (B, H, S, S) for a mask, and {1} as a scalar over
    // any shape.
    [[nodiscard]] Tensor operator+(const Tensor& other) const;
    [[nodiscard]] Tensor operator-(const Tensor& other) const;
    [[nodiscard]] Tensor operator*(const Tensor& other) const;
    [[nodiscard]] Tensor operator/(const Tensor& other) const;

    [[nodiscard]] Tensor operator+(float scalar) const;
    [[nodiscard]] Tensor operator-(float scalar) const;
    [[nodiscard]] Tensor operator*(float scalar) const;
    [[nodiscard]] Tensor operator/(float scalar) const;

    // Concatenates along an axis; every other dimension must match.
    [[nodiscard]] static Tensor concat(const std::vector<Tensor>& parts, size_t axis);
    // Stacks, creating a new axis at the given position.
    [[nodiscard]] static Tensor stack(const std::vector<Tensor>& parts, size_t axis = 0);

    // matmul is batched: (B..., M, K) x (B..., K, N) -> (B..., M, N).
    // transpose swaps the last two axes; permute reorders all of them.
    [[nodiscard]] Tensor matmul(const Tensor& other) const;
    [[nodiscard]] Tensor transpose() const;
    [[nodiscard]] Tensor permute(const std::vector<size_t>& order) const;
    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor softmax() const;  // over the last axis
    [[nodiscard]] Tensor reshape(const std::vector<size_t>& new_shape) const;
    // Reductions. With no argument they reduce to a {1} scalar; with an axis
    // they drop it (or leave it at 1 with keepdim), as in NumPy or PyTorch.
    [[nodiscard]] Tensor sum() const;
    [[nodiscard]] Tensor mean() const;
    [[nodiscard]] Tensor sum(size_t axis, bool keepdim = false) const;
    [[nodiscard]] Tensor mean(size_t axis, bool keepdim = false) const;
    [[nodiscard]] Tensor max(size_t axis, bool keepdim = false) const;

    // Contiguous sub-tensor along an axis: [start, start + count).
    [[nodiscard]] Tensor slice(size_t axis, size_t start, size_t count) const;

    // Gathers the given elements of the first axis into a mini-batch: rows from
    // an (M, N), whole images from an (N, C, H, W). This is the operation that
    // makes mini-batch training possible. Indices may repeat: their gradients
    // accumulate into the source element.
    [[nodiscard]] Tensor select_rows(const std::vector<size_t>& indices) const;

    // A copy detached from the graph (shares shape and values, not history)
    [[nodiscard]] Tensor detach() const;

    // Formatting and printing. The only const member here without
    // [[nodiscard]]: it returns nothing, so there is nothing to discard.
    void print(const std::string& name = "") const;
};

// Scalar-on-the-left overloads, so that 2.0f * t is valid
[[nodiscard]] Tensor operator+(float scalar, const Tensor& t);
[[nodiscard]] Tensor operator-(float scalar, const Tensor& t);
[[nodiscard]] Tensor operator*(float scalar, const Tensor& t);

}  // namespace engine

#endif  // ENGINE_TENSOR_HPP
