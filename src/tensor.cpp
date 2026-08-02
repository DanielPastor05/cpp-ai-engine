#include "engine/tensor.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"
#include "engine/detail/cuda_ops.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "engine/random.hpp"

namespace engine {

// ---------------------------------------------------------
// Global random generator
// ---------------------------------------------------------

std::mt19937& global_rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

void manual_seed(uint64_t seed) {
    global_rng().seed(static_cast<std::mt19937::result_type>(seed));
}

namespace {

// The graph is only built if the tensor asks for it and autograd mode is on
// (during backward and inside the optimisers it is off).
// Split thresholds, calibrated against the measured cost of dispatching a
// parallel region: about 8 us on four threads on this machine. Below roughly
// 100 us of work, splitting costs more than computing.
//
// The first attempt used thresholds ten times lower and the examples got
// SLOWER: the Transformer one went from 15.9 s to 22.6 s, because it chains
// many small products and each paid for synchronisation without gaining.

// A product of 1M multiply-adds is about 130 us at 15 GFLOP/s.
constexpr size_t kMatmulParallelFloor = 1u << 20;
// Chunks of ~65k operations: enough for the dynamic split to balance, without
// the cost of asking for the next chunk weighing in.
constexpr size_t kMatmulChunkWork = 1u << 16;

using parallel::kElementsPerThread;

// Returns 0 -- "run inline" -- when the product is too small to split.
inline size_t matmul_rows_per_thread(size_t rows, size_t K, size_t N) {
    const size_t per_row = std::max<size_t>(1, K * N);
    if (rows * per_row < kMatmulParallelFloor) return 0;
    return std::max<size_t>(1, kMatmulChunkWork / per_row);
}

inline bool track(bool requires_grad) {
    return requires_grad && autograd::grad_enabled();
}

size_t product(const std::vector<size_t>& dims, size_t from = 0, size_t to = 0) {
    const size_t end = (to == 0) ? dims.size() : to;
    size_t total = 1;
    for (size_t i = from; i < end; ++i) total *= dims[i];
    return total;
}

// Suffix broadcasting: `other` broadcasts over `base` if, after dropping its
// leading ones, its shape is a suffix of `base`'s. It covers a dense layer's
// bias (1, N) over (M, N), the positional encoding (S, D) over (B, S, D) and
// the mask (S, S) over (B, H, S, S).
//
// Because the tensor is contiguous in C order, broadcasting over the leading
// axes reduces to repeating the final block: base[i] + other[i % inner].
struct BroadcastPlan {
    bool valid = false;
    size_t inner = 1;   // size of the block that repeats
    size_t repeat = 1;  // how many times it repeats
};

BroadcastPlan plan_broadcast(const std::vector<size_t>& base, const std::vector<size_t>& other) {
    BroadcastPlan plan;

    size_t first = 0;
    while (first < other.size() && other[first] == 1) ++first;
    const size_t core = other.size() - first;

    if (core > base.size()) return plan;

    const size_t offset = base.size() - core;
    for (size_t i = 0; i < core; ++i) {
        if (base[offset + i] != other[first + i]) return plan;
    }

    plan.valid = true;
    plan.inner = product(base, offset);
    // The repetition count is derived from the total, not from a partial product:
    // with offset == 0 (same rank after dropping leading ones) a product over the
    // empty range would be mistaken for the full product.
    plan.repeat = (plan.inner == 0) ? 0 : product(base) / plan.inner;
    return plan;
}

// Materialises the broadcast operand at the base's shape. Only used in the
// derivatives, where having both shapes equal simplifies the formulas.
Tensor expand_operand(const Tensor& other, const std::vector<size_t>& base_shape, size_t total,
                      size_t inner) {
    // The accessors are hoisted out of the loop: data() is not a pointer, it is
    // the door to the host side, and calling it per element checks the device
    // mirror's validity once per value copied.
    Tensor full(base_shape, 0.0f, false);
    const float* ENGINE_RESTRICT src = other.data().data();
    float* ENGINE_RESTRICT dst = full.data().data();
    for (size_t i = 0; i < total; ++i) dst[i] = src[i % inner];
    return full;
}

// Sums the leading axes of `full` down to the shape `target`. It is the adjoint
// of broadcasting an operand over a batch, used both by the broadcast addition
// and by matmul with a shared operand.
Tensor fold_leading(const Tensor& full, const std::vector<size_t>& target) {
    const size_t inner = product(target);
    const size_t repeat = full.size() / inner;

    Tensor folded(target, 0.0f, false);

    // This is sum_axis with a single outer block: out[j] = sum over r of
    // full[r * inner + j]. The kernel already existed, and this was the single
    // largest remaining download in a training step -- measured at 5.8 MiB per
    // step on MNIST, almost all of it the bias gradient of the first
    // convolution, where `full` is (50176, 16) and `target` is (16).
    //
    // The accumulation order matches the CPU loop's, r ascending, so the two
    // agree bit for bit.
    if (!cuda::ops::sum_axis(full.get_impl()->storage, folded.get_impl()->storage, 1, repeat,
                             inner)) {
        const float* ENGINE_RESTRICT src = full.data().data();
        float* ENGINE_RESTRICT dst = folded.data().data();
        for (size_t r = 0; r < repeat; ++r) {
            for (size_t j = 0; j < inner; ++j) {
                dst[j] += src[r * inner + j];
            }
        }
    }
    return folded;
}

// A new tensor with the same values, a different shape and no history.
//
// The difference from Tensor(shape, data(), ...) is that this data() **is a
// trip down to host**: the buffer is copied either way, but crossing PCIe twice
// if the tensor lived on the GPU. clone() copies keeping the side it is on.
// reshape() and detach() use it, and those are exactly the two applied to
// outputs a kernel has just computed.
Tensor clone_with_shape(const Storage& src, const std::vector<size_t>& shape, bool req_grad) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = src.clone();
    impl->shape = shape;
    impl->compute_strides();
    impl->requires_grad = req_grad;
    return Tensor::from_impl(impl);
}

// The same, but sharing the buffer instead of copying it: a view.
//
// Only reshape uses this, and only because reshape is the one operation that
// changes nothing about the data -- same values, same order, same count, a
// different label on the axes. Everything else in the engine that produces a
// new shape (transpose, permute, slice off axis 0) genuinely moves elements
// around, and a view of those would need non-contiguous strides threaded
// through every kernel and every CPU loop.
//
// The aliasing this introduces is real and is the point: writing through the
// reshaped tensor is visible in the original, exactly as in PyTorch. It is safe
// here because nothing in the engine writes through a reshape result -- checked
// across the twelve in-place write sites in src/, all of which write to tensors
// they created on the line above. tests/test_tensor.cpp pins the semantics so a
// future one cannot start doing it by accident.
Tensor view_with_shape(const Storage& src, const std::vector<size_t>& shape, bool req_grad) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = src.share();
    impl->shape = shape;
    impl->compute_strides();
    impl->requires_grad = req_grad;
    return Tensor::from_impl(impl);
}

// Offers the operation to the CUDA backend. Returns true if the GPU took it;
// false means "no device" or "not worth it at this size", and the caller
// carries on down the CPU path. Without CUDA the call is a function returning
// false and the linker removes it.
//
// Translating the broadcast: the right operand has `inner` elements repeated
// `repeat` times, exactly the plan the CPU loop uses.
bool offer_to_device(cuda::ops::Binary op, const Tensor& a, const Tensor& b, Tensor& out,
                     bool broadcast, const BroadcastPlan& plan) {
    const size_t inner = broadcast ? plan.inner : a.size();
    const size_t repeat = broadcast ? plan.repeat : 1;
    return cuda::ops::binary(op, a.get_impl()->storage, b.get_impl()->storage,
                             out.get_impl()->storage, inner, repeat);
}

}  // namespace

// ---------------------------------------------------------
// TensorImpl implementation
// ---------------------------------------------------------

TensorImpl::TensorImpl(const std::vector<size_t>& s, float fill_val, bool req_grad)
    : shape(s), requires_grad(req_grad) {
    compute_strides();
    size_t total_elements = 1;
    for (size_t dim : shape) total_elements *= dim;
    storage.assign(total_elements, fill_val);
}

TensorImpl::TensorImpl(const std::vector<size_t>& s, const std::vector<float>& d, bool req_grad)
    : storage(d), shape(s), requires_grad(req_grad) {
    compute_strides();
    size_t total_elements = 1;
    for (size_t dim : shape) total_elements *= dim;
    if (storage.size() != total_elements) {
        throw std::invalid_argument(
            "The number of elements in data does not match the given shape.");
    }
}

void TensorImpl::compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

size_t TensorImpl::get_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Index count incompatible with ndim.");
    }
    size_t flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of range.");
        }
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

// ---------------------------------------------------------
// The Tensor wrapper class
// ---------------------------------------------------------

Tensor::Tensor() : impl_(std::make_shared<TensorImpl>(std::vector<size_t>{0}, 0.0f, false)) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument("Cannot construct a Tensor over a null implementation.");
    }
}

Tensor::Tensor(const std::vector<size_t>& shape, float fill_value, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, fill_value, requires_grad)) {}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, data, requires_grad)) {}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(std::move(impl));
}

// Factory methods
Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    return Tensor(shape, 0.0f, requires_grad);
}

Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    return Tensor(shape, 1.0f, requires_grad);
}

Tensor Tensor::rand(const std::vector<size_t>& shape, float min_val, float max_val,
                    bool requires_grad) {
    Tensor t(shape, 0.0f, requires_grad);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dis(global_rng());
    }
    return t;
}

Tensor Tensor::randn(const std::vector<size_t>& shape, float mean, float stddev,
                     bool requires_grad) {
    Tensor t(shape, 0.0f, requires_grad);
    std::normal_distribution<float> dis(mean, stddev);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dis(global_rng());
    }
    return t;
}

// Autograd and gradients
bool Tensor::requires_grad() const {
    return impl_->requires_grad;
}

void Tensor::set_requires_grad(bool req_grad) {
    impl_->requires_grad = req_grad;
}

Tensor Tensor::grad() const {
    if (!impl_->grad) {
        throw std::runtime_error("The tensor has no computed gradient.");
    }
    return Tensor(impl_->grad);
}

bool Tensor::has_grad() const {
    return impl_->grad != nullptr;
}

void Tensor::zero_grad() {
    // assign() rather than host_mut() + fill(): host_mut() synchronises, which
    // pulls a gradient off the device only to overwrite it with zeros, once per
    // parameter per step. assign() marks host valid and device stale without
    // copying anything, and keeps the GPU allocation because the size does not
    // change.
    if (impl_->grad) {
        impl_->grad->storage.assign(impl_->grad->storage.size(), 0.0f);
    }
}

void Tensor::add_grad(const Tensor& g) {
    if (!requires_grad()) return;
    if (g.shape() != shape()) {
        throw std::invalid_argument("Gradient shape " + g.shape_str() +
                                    " incompatible with the tensor's " + shape_str() + ".");
    }
    // This is where the backward's whole residency was being decided by accident.
    // Building the gradient from g.data() pulled it off the device, and autograd
    // hands it straight to the next backward_fn, which had to upload it again: not
    // one node stayed on the GPU, whatever the size and whatever the thresholds.
    // The forward has had this invariant right since Phase 6; this applies the same
    // one to the gradient.
    //
    // The branch that matters is the first write, not the +=: autograd discards
    // intermediate nodes' gradients before the traversal and frees them as soon as
    // they are consumed, so a node with a single consumer -- the normal case --
    // always comes through here and never through the accumulation.
    const bool initialize = !impl_->grad;
    // Zeroed so the CPU path is a single loop: over zeros, adding is assigning.
    // Copying g's Storage would not do as a shortcut, because its copy constructor
    // takes only the host side and synchronises to do so: that would be exactly the
    // download being removed here.
    if (initialize) impl_->grad = std::make_shared<TensorImpl>(shape(), 0.0f, false);

    Storage& acc = impl_->grad->storage;
    if (!cuda::ops::accumulate_grad(acc, g.get_impl()->storage, initialize)) {
        // g.data() stays inside the if deliberately: outside, the download would come
        // back in through the back door on the path the device does accelerate.
        std::vector<float>& dst = acc.host_mut();
        const float* ENGINE_RESTRICT src = g.data().data();
        float* ENGINE_RESTRICT out = dst.data();
        parallel::parallel_for(dst.size(), kElementsPerThread, [&](size_t from, size_t to) {
            for (size_t i = from; i < to; ++i) out[i] += src[i];
        });
    }
}

void Tensor::backward() {
    autograd::backward(*this);
}

void Tensor::backward(const Tensor& grad_output) {
    if (grad_output.shape() != shape()) {
        throw std::invalid_argument("Initial gradient shape " + grad_output.shape_str() +
                                    " incompatible with the root's " + shape_str() + ".");
    }
    impl_->grad = std::make_shared<TensorImpl>(shape(), grad_output.data(), false);
    autograd::backward(*this);
}

// Accessors and properties
size_t Tensor::get_flat_index(const std::vector<size_t>& indices) const {
    return impl_->get_flat_index(indices);
}

float& Tensor::operator()(const std::vector<size_t>& indices) {
    return impl_->storage[get_flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    return impl_->storage[get_flat_index(indices)];
}

float& Tensor::operator()(size_t row, size_t col) {
    return const_cast<float&>(static_cast<const Tensor&>(*this)(row, col));
}

const float& Tensor::operator()(size_t row, size_t col) const {
    if (ndim() != 2) {
        throw std::invalid_argument("(row, column) indexing requires a 2D tensor; this one is " +
                                    shape_str() + ".");
    }
    if (row >= shape()[0] || col >= shape()[1]) {
        throw std::out_of_range("Index (" + std::to_string(row) + ", " + std::to_string(col) +
                                ") out of range for a tensor " + shape_str() + ".");
    }
    return impl_->storage[row * shape()[1] + col];
}

float& Tensor::at(size_t flat_index) {
    return impl_->storage.at(flat_index);
}

const float& Tensor::at(size_t flat_index) const {
    return impl_->storage.at(flat_index);
}

const std::vector<size_t>& Tensor::shape() const {
    return impl_->shape;
}
const std::vector<size_t>& Tensor::strides() const {
    return impl_->strides;
}
// data() is the door to the host side. The mutable overload also marks the
// device mirror stale, which is what keeps the two copies coherent without any
// operation having to remember to do it.
const std::vector<float>& Tensor::data() const {
    return impl_->storage.host();
}
std::vector<float>& Tensor::data() {
    return impl_->storage.host_mut();
}
size_t Tensor::size() const {
    return impl_->storage.size();
}
size_t Tensor::ndim() const {
    return impl_->shape.size();
}

std::string Tensor::shape_str() const {
    std::string s = "(";
    for (size_t i = 0; i < shape().size(); ++i) {
        s += std::to_string(shape()[i]);
        if (i + 1 < shape().size()) s += ", ";
    }
    return s + ")";
}

Tensor Tensor::detach() const {
    // This is where the output the backward pass will need gets saved -- softmax's
    // Jacobian depends on it -- so it went through host once per softmax in the
    // model even when the tensor was going to stay up there.
    return clone_with_shape(impl_->storage, shape(), false);
}

// ---------------------------------------------------------
// Arithmetic operators, with autograd registration
//
// Every lambda takes the output gradient (grad_out) as a parameter and captures
// only the input tensors, never the result: capturing the result would form a
// shared_ptr cycle and the graph would never be freed.
// ---------------------------------------------------------

// Tensor addition, broadcasting the right operand by suffix
Tensor Tensor::operator+(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Incompatible shapes for tensor addition: " + shape_str() +
                                        " and " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Add, *this, other, res, broadcast, plan)) {
        // The accessors are hoisted out of the loop: calling them per element stops
        // the compiler from vectorising.
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] + rhs[i];
            });
        } else {
            // By blocks rather than with a modulo per element: the broadcast operand
            // repeats verbatim, so the inner loop vectorises.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] + rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_, other.impl_};
        Tensor self_copy = *this;
        Tensor other_copy = other;

        // `plan` is deliberately not captured: the broadcast backward used to walk
        // it by hand and now calls fold_leading, which recomputes the decomposition
        // from the shapes. Clang catches the leftover capture, MSVC does not.
        res.impl_->backward_fn = [self_copy, other_copy,
                                  broadcast](const Tensor& grad_out) mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out);
            if (!other_copy.requires_grad()) return;
            if (!broadcast) {
                other_copy.add_grad(grad_out);
            } else {
                // The broadcast operand was repeated `repeat` times, so its gradient
                // is the sum of all those copies -- which is exactly what
                // fold_leading does, and what the other three operators already
                // called. This one had its own inline loop with a
                // grad_out.data() in it, and that one accessor was **the largest
                // download left in a training step**: 5.8 MiB per step on MNIST,
                // almost all of it the first convolution's bias gradient, where
                // grad_out is (50176, 16) folded down to (16).
                //
                // Four operators, four broadcast backwards, and the one that
                // duplicated the shared helper instead of calling it was the one
                // that stayed on the host.
                other_copy.add_grad(fold_leading(grad_out, other_copy.shape()));
            }
        };
    }
    return res;
}

// Tensor subtraction, broadcasting the right operand
Tensor Tensor::operator-(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Incompatible shapes for tensor subtraction: " +
                                        shape_str() + " and " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Sub, *this, other, res, broadcast, plan)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] - rhs[i];
            });
        } else {
            // By blocks rather than with a modulo per element: the broadcast operand
            // repeats verbatim, so the inner loop vectorises.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] - rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_, other.impl_};
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy,
                                  broadcast](const Tensor& grad_out) mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out);
            if (!other_copy.requires_grad()) return;
            // The broadcast operand collects the sum of all its copies.
            // Subtraction does not need the plan: its derivative depends on the
            // operand's shape, not its values.
            Tensor d = grad_out * -1.0f;
            other_copy.add_grad(broadcast ? fold_leading(d, other_copy.shape()) : d);
        };
    }
    return res;
}

// Element-wise (Hadamard) multiplication, broadcasting the right operand
Tensor Tensor::operator*(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Incompatible shapes for tensor multiplication: " +
                                        shape_str() + " and " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Mul, *this, other, res, broadcast, plan)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] * rhs[i];
            });
        } else {
            // By blocks rather than with a modulo per element: the broadcast operand
            // repeats verbatim, so the inner loop vectorises.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] * rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_, other.impl_};
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy, broadcast,
                                  plan](const Tensor& grad_out) mutable {
            const Tensor rhs = broadcast ? expand_operand(other_copy, self_copy.shape(),
                                                          self_copy.size(), plan.inner)
                                         : other_copy;
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out * rhs);
            if (other_copy.requires_grad()) {
                Tensor d = grad_out * self_copy;
                other_copy.add_grad(broadcast ? fold_leading(d, other_copy.shape()) : d);
            }
        };
    }
    return res;
}

// Element-wise division, broadcasting the right operand
Tensor Tensor::operator/(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Incompatible shapes for tensor division: " + shape_str() +
                                        " and " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Div, *this, other, res, broadcast, plan)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] / rhs[i];
            });
        } else {
            // By blocks rather than with a modulo per element: the broadcast operand
            // repeats verbatim, so the inner loop vectorises.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] / rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_, other.impl_};
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy, broadcast,
                                  plan](const Tensor& grad_out) mutable {
            const Tensor rhs = broadcast ? expand_operand(other_copy, self_copy.shape(),
                                                          self_copy.size(), plan.inner)
                                         : other_copy;
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out / rhs);
            if (other_copy.requires_grad()) {
                // d/db (a/b) = -a/b^2
                Tensor d = (grad_out * self_copy * -1.0f) / (rhs * rhs);
                other_copy.add_grad(broadcast ? fold_leading(d, other_copy.shape()) : d);
            }
        };
    }
    return res;
}

// Scalar addition
Tensor Tensor::operator+(float scalar) const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    // out = x * 1 + k. The accessors stay inside the else: calling them outside
    // would pull the tensor down to host on the very path that does not need it.
    if (!cuda::ops::scalar(impl_->storage, res.impl_->storage, 1.0f, scalar)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] + scalar;
    }
    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out);
        };
    }
    return res;
}

// Scalar subtraction
Tensor Tensor::operator-(float scalar) const {
    return (*this) + (-scalar);
}

// Scalar multiplication
Tensor Tensor::operator*(float scalar) const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    // out = x * k + 0. This is the attention scaling, between two matmuls that do
    // have kernels: without it the chain went down to host right in the middle.
    if (!cuda::ops::scalar(impl_->storage, res.impl_->storage, scalar, 0.0f)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] * scalar;
    }
    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, scalar](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out * scalar);
        };
    }
    return res;
}

// Scalar division
Tensor Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::invalid_argument("Division by zero.");
    }
    return (*this) * (1.0f / scalar);
}

// Transpose: swaps the last two axes.
// For a 2D tensor it is the ordinary transpose; for (B, ..., M, N) it
// transposes each matrix in the batch, which is what attention needs.
Tensor Tensor::transpose() const {
    if (ndim() < 2) {
        throw std::invalid_argument("Transpose requires at least 2 dimensions; this tensor is " +
                                    shape_str() + ".");
    }
    const size_t nd = ndim();
    const size_t rows = shape()[nd - 2];
    const size_t cols = shape()[nd - 1];
    const size_t batch = size() / (rows * cols);

    std::vector<size_t> out_shape = shape();
    std::swap(out_shape[nd - 2], out_shape[nd - 1]);

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);

    // Transposing is permuting with the last two axes swapped, so it shares a
    // kernel with permute(): the input strides read in the output's order are the
    // whole difference between the two operations.
    std::vector<size_t> src_strides = strides();
    std::swap(src_strides[nd - 2], src_strides[nd - 1]);

    if (!cuda::ops::permute(impl_->storage, res.impl_->storage, out_shape.data(),
                            src_strides.data(), nd)) {
        const float* ENGINE_RESTRICT src_base = data().data();
        float* ENGINE_RESTRICT dst_base = res.data().data();
        // Split by source row: each writes one whole column of the destination and
        // none touches another's. The threshold is counted in rows because each row's
        // work is its `cols` elements.
        const size_t rows_per_thread =
            std::max<size_t>(1, kElementsPerThread / std::max<size_t>(1, cols));
        parallel::parallel_for(batch * rows, rows_per_thread, [&](size_t from, size_t to) {
            for (size_t r = from; r < to; ++r) {
                const size_t b = r / rows;
                const size_t i = r % rows;
                const float* src = src_base + b * rows * cols + i * cols;
                float* dst = dst_base + b * rows * cols + i;
                for (size_t j = 0; j < cols; ++j) dst[j * rows] = src[j];
            }
        });
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out.transpose());
        };
    }
    return res;
}

// General axis permutation: order[i] says which input axis moves into position
// i. It is what reorders (B, S, H, d) into (B, H, S, d) so that each attention
// head operates on its own sequence.
Tensor Tensor::permute(const std::vector<size_t>& order) const {
    const size_t nd = ndim();
    if (order.size() != nd) {
        throw std::invalid_argument("permute needs an order of " + std::to_string(nd) +
                                    " axes for a tensor " + shape_str() + ".");
    }
    std::vector<bool> seen(nd, false);
    for (size_t axis : order) {
        if (axis >= nd || seen[axis]) {
            throw std::invalid_argument("permute needs a valid permutation of the axes.");
        }
        seen[axis] = true;
    }

    std::vector<size_t> out_shape(nd);
    for (size_t i = 0; i < nd; ++i) out_shape[i] = shape()[order[i]];

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);

    // The output tensor's strides expressed over the input's memory
    std::vector<size_t> src_strides(nd);
    for (size_t i = 0; i < nd; ++i) src_strides[i] = strides()[order[i]];

    if (!cuda::ops::permute(impl_->storage, res.impl_->storage, out_shape.data(),
                            src_strides.data(), nd)) {
        const float* ENGINE_RESTRICT src_data = data().data();
        float* ENGINE_RESTRICT dst_data = res.data().data();
        // The carry counter is cheap per element but chains each one to the previous,
        // and that dependency is what forced a serial traversal. Seeding it is enough:
        // each chunk computes the coordinates of its first element -- one division per
        // axis, paid once per chunk -- and carries on with the usual carry from there.
        //
        //
        // It went unnoticed while permute was an attention concern, with tensors of a
        // few thousand elements. Conv2d permutes each layer's entire output.
        parallel::parallel_for(size(), kElementsPerThread, [&](size_t from, size_t to) {
            std::vector<size_t> idx(nd, 0);
            size_t rem = from;
            for (size_t i = nd; i-- > 0;) {
                idx[i] = rem % out_shape[i];
                rem /= out_shape[i];
            }

            for (size_t flat = from; flat < to; ++flat) {
                size_t src = 0;
                for (size_t i = 0; i < nd; ++i) src += idx[i] * src_strides[i];
                dst_data[flat] = src_data[src];

                for (size_t i = nd; i-- > 0;) {
                    if (++idx[i] < out_shape[i]) break;
                    idx[i] = 0;
                }
            }
        });
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        // The derivative is the inverse permutation
        std::vector<size_t> inverse(nd);
        for (size_t i = 0; i < nd; ++i) inverse[order[i]] = i;

        res.impl_->backward_fn = [self_copy, inverse](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out.permute(inverse));
        };
    }
    return res;
}

// Matrix multiplication, batched over the leading axes.
// (M, K) x (K, N) -> (M, N) y (B..., M, K) x (B..., K, N) -> (B..., M, N).
Tensor Tensor::matmul(const Tensor& B) const {
    if (ndim() < 2 || B.ndim() < 2) {
        throw std::invalid_argument("MatMul requires at least 2 dimensions in both operands.");
    }
    const size_t nd_a = ndim();
    const size_t nd_b = B.ndim();

    const std::vector<size_t> batch_a(shape().begin(), shape().end() - 2);
    const std::vector<size_t> batch_b(B.shape().begin(), B.shape().end() - 2);

    // A 2D operand is shared with every matrix in the other's batch:
    // (B, M, K) x (K, N) applies the same matrix to each batch element.
    const bool a_batched = !batch_a.empty();
    const bool b_batched = !batch_b.empty();
    if (a_batched && b_batched && batch_a != batch_b) {
        throw std::invalid_argument("Batched MatMul needs identical batch axes: " + shape_str() +
                                    " and " + B.shape_str() + ".");
    }
    const std::vector<size_t>& batch_dims = a_batched ? batch_a : batch_b;

    const size_t M = shape()[nd_a - 2];
    const size_t K = shape()[nd_a - 1];
    const size_t K2 = B.shape()[nd_b - 2];
    const size_t N = B.shape()[nd_b - 1];

    if (K != K2) {
        throw std::invalid_argument("Incompatible dimensions for MatMul: " + shape_str() + " and " +
                                    B.shape_str() + ".");
    }

    const size_t batch = product(batch_dims);
    std::vector<size_t> out_shape = batch_dims;
    out_shape.push_back(M);
    out_shape.push_back(N);

    bool req_g = track(requires_grad() || B.requires_grad());
    Tensor C(out_shape, 0.0f, req_g);

    // An unbatched operand uses stride 0: it is reused on every iteration
    const size_t a_stride = a_batched ? M * K : 0;
    const size_t b_stride = b_batched ? K * N : 0;

    // It is offered to the device first. If the device takes it, C stays resident
    // on the GPU and never comes down to host: only reading a value from the
    // program triggers the copy back.
    if (!cuda::ops::matmul(impl_->storage, B.impl_->storage, C.impl_->storage, batch, M, K, N,
                           a_batched, b_batched)) {
        const std::vector<float>& a_data = data();
        const std::vector<float>& b_data = B.data();
        std::vector<float>& c_data = C.data();

        // Loop ordered i -> k -> j: walking j is contiguous in memory, so B's row is
        // read and C's is written sequentially.
        //
        // The row pointers are hoisted out of the inner loop and marked with
        // ENGINE_RESTRICT: without that promise of no overlap the compiler cannot
        // vectorise the accumulator.
        //
        // The a_ik == 0 check does pay, even though it prevents vectorising that
        // branch. On dense data it costs 40%, but the matrices arriving here are often
        // ReLU outputs with half the values at exact zero -- the second dense layer of
        // every Transformer block, for instance -- and there it skips half the work.
        // Measured both ways on the Transformer example: without the branch 18.7 s,
        // with it 15.9 s.
        // The split goes by output row: each row is computed by a single thread from
        // start to finish, so the accumulation order never changes and the result is
        // identical bit for bit whatever the thread count.
        const size_t rows = batch * M;
        parallel::parallel_for(
            rows, matmul_rows_per_thread(rows, K, N), [&](size_t from, size_t to) {
                for (size_t row = from; row < to; ++row) {
                    const size_t b = row / M;
                    const size_t i = row % M;

                    const float* ENGINE_RESTRICT a_row = a_data.data() + b * a_stride + i * K;
                    const float* ENGINE_RESTRICT bm = b_data.data() + b * b_stride;
                    float* ENGINE_RESTRICT c_row = c_data.data() + b * M * N + i * N;

                    for (size_t k = 0; k < K; ++k) {
                        const float a_ik = a_row[k];
                        if (a_ik == 0.0f) continue;
                        const float* ENGINE_RESTRICT b_row = bm + k * N;
                        for (size_t j = 0; j < N; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            });
    }

    if (req_g) {
        C.impl_->parents = {impl_, B.impl_};
        Tensor self_copy = *this;
        Tensor B_copy = B;

        C.impl_->backward_fn = [self_copy, B_copy](const Tensor& grad_out) mutable {
            // The same formulas as in 2D, applied to each matrix in the batch:
            // dL/dA = dL/dC x B^T   y   dL/dB = A^T x dL/dC.
            // If an operand was shared with the whole batch, its gradient is the
            // sum of each element's contribution.
            if (self_copy.requires_grad()) {
                Tensor dA = grad_out.matmul(B_copy.transpose());
                self_copy.add_grad(
                    dA.shape() == self_copy.shape() ? dA : fold_leading(dA, self_copy.shape()));
            }
            if (B_copy.requires_grad()) {
                Tensor dB = self_copy.transpose().matmul(grad_out);
                B_copy.add_grad(dB.shape() == B_copy.shape() ? dB
                                                             : fold_leading(dB, B_copy.shape()));
            }
        };
    }
    return C;
}

// ReLU activation
Tensor Tensor::relu() const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    if (!cuda::ops::relu(impl_->storage, res.impl_->storage)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = std::max(0.0f, lhs[i]);
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;

        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            Tensor dX(self_copy.shape(), 0.0f, false);
            if (!cuda::ops::relu_backward(self_copy.get_impl()->storage,
                                          grad_out.get_impl()->storage, dX.get_impl()->storage)) {
                const size_t n = self_copy.size();
                const float* ENGINE_RESTRICT x = self_copy.data().data();
                const float* ENGINE_RESTRICT g = grad_out.data().data();
                float* ENGINE_RESTRICT out = dX.data().data();
                for (size_t i = 0; i < n; ++i) out[i] = (x[i] > 0.0f) ? g[i] : 0.0f;
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Softmax over the last axis (numerically stable: the maximum is subtracted).
Tensor Tensor::softmax() const {
    if (ndim() == 0 || size() == 0) {
        throw std::invalid_argument("Softmax needs a non-empty tensor.");
    }
    // It always normalises over the last axis: for (N, C) those are the rows, and
    // for (B, H, S, S) the attention scores of each query.
    const size_t cols = shape().back();
    const size_t rows = size() / cols;

    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!cuda::ops::softmax(impl_->storage, res.impl_->storage, rows, cols)) {
        const float* ENGINE_RESTRICT src = data().data();
        float* ENGINE_RESTRICT dst = res.data().data();
        for (size_t i = 0; i < rows; ++i) {
            const float* row = src + i * cols;
            float max_v = *std::max_element(row, row + cols);
            float denom = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                float e = std::exp(row[j] - max_v);
                dst[i * cols + j] = e;
                denom += e;
            }
            for (size_t j = 0; j < cols; ++j) dst[i * cols + j] /= denom;
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        // Softmax's Jacobian depends on its output, so a copy detached from the graph
        // is saved (the same thing PyTorch does with save_for_backward).
        Tensor saved = res.detach();

        res.impl_->backward_fn = [self_copy, saved, rows, cols](const Tensor& grad_out) mutable {
            // dX_ij = y_ij * (dY_ij - sum_k dY_ik * y_ik)
            Tensor dX(self_copy.shape(), 0.0f, false);
            if (!cuda::ops::softmax_backward(saved.get_impl()->storage,
                                             grad_out.get_impl()->storage, dX.get_impl()->storage,
                                             rows, cols)) {
                const float* ENGINE_RESTRICT y = saved.data().data();
                const float* ENGINE_RESTRICT g = grad_out.data().data();
                float* ENGINE_RESTRICT out = dX.data().data();
                for (size_t i = 0; i < rows; ++i) {
                    float dot = 0.0f;
                    for (size_t j = 0; j < cols; ++j) dot += g[i * cols + j] * y[i * cols + j];
                    for (size_t j = 0; j < cols; ++j) {
                        out[i * cols + j] = y[i * cols + j] * (g[i * cols + j] - dot);
                    }
                }
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Sum reduction to a {1} scalar
Tensor Tensor::sum() const {
    // The accumulator is double even though the data is float, as in
    // clip_grad_norm. Summing a million values in float loses the low bits as soon
    // as the total grows against each addend, and mean() hangs off this reduction,
    // which means mse_loss and the initial gradient of every backward.
    // Offered to the device first, and the accumulator stays double on both
    // paths. A float reduction on the GPU would be shorter and would undo the
    // reason this one is double: mean(), mse_loss and the initial gradient of
    // every backward hang off this sum.
    double total = 0.0;
    if (!cuda::ops::reduce_sum(impl_->storage, total)) {
        total = 0.0;
        for (float v : data()) total += v;
    }
    bool req_g = track(requires_grad());
    Tensor res({1}, std::vector<float>{static_cast<float>(total)}, req_g);

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;

        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            Tensor dX(self_copy.shape(), grad_out.data()[0], false);
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Mean reduction to a {1} scalar
Tensor Tensor::mean() const {
    if (size() == 0) {
        throw std::invalid_argument("Cannot take the mean of an empty tensor.");
    }
    Tensor s = sum();
    return s * (1.0f / static_cast<float>(size()));
}

// Reshape
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_total = 1;
    for (size_t dim : new_shape) new_total *= dim;
    if (new_total != size()) {
        throw std::invalid_argument("Incompatible element count for Reshape.");
    }
    bool req_g = track(requires_grad());

    // A view: no bytes move, and neither does the residency. Reshaping a tensor
    // that is on the GPU gives another handle on the same device buffer.
    //
    // This used to copy the whole buffer, and it was the most expensive line in
    // the engine that computed nothing. Conv2d reshapes three times per forward,
    // Linear twice on a 3D input, attention twice per projection -- each of them
    // megabytes, each of them with a mirror in the backward. It was also what
    // made the composed Conv2d slower than the hand-written one it replaced.
    Tensor res = view_with_shape(impl_->storage, new_shape, req_g);

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out.reshape(self_copy.shape()));
        };
    }
    return res;
}

// ---------------------------------------------------------
// Reductions along an axis
//
// Relative to one axis, a contiguous tensor looks like an
// (outer, axis_len, inner) block: outer is the axes before it, inner the ones
// after. With that, reducing is walking axis_len for each (outer, inner).
// ---------------------------------------------------------

namespace {

struct AxisView {
    size_t outer;
    size_t axis_len;
    size_t inner;
};

AxisView axis_view(const std::vector<size_t>& shape, size_t axis) {
    AxisView v{1, shape[axis], 1};
    for (size_t i = 0; i < axis; ++i) v.outer *= shape[i];
    for (size_t i = axis + 1; i < shape.size(); ++i) v.inner *= shape[i];
    return v;
}

std::vector<size_t> reduced_shape(const std::vector<size_t>& shape, size_t axis, bool keepdim) {
    std::vector<size_t> out = shape;
    if (keepdim) {
        out[axis] = 1;
    } else {
        out.erase(out.begin() + static_cast<long>(axis));
        if (out.empty()) out.push_back(1);  // reducing a 1D tensor leaves a scalar {1}
    }
    return out;
}

void check_axis(size_t axis, size_t ndim, const std::string& what, const std::string& shape_txt) {
    if (axis >= ndim) {
        throw std::out_of_range(what + ": axis " + std::to_string(axis) +
                                " does not exist in a tensor " + shape_txt + ".");
    }
}

}  // namespace

Tensor Tensor::sum(size_t axis, bool keepdim) const {
    check_axis(axis, ndim(), "sum", shape_str());
    const AxisView v = axis_view(shape(), axis);

    bool req_g = track(requires_grad());
    Tensor res(reduced_shape(shape(), axis, keepdim), 0.0f, req_g);

    if (!cuda::ops::sum_axis(impl_->storage, res.impl_->storage, v.outer, v.axis_len, v.inner)) {
        const float* ENGINE_RESTRICT src_base = data().data();
        float* ENGINE_RESTRICT dst_base = res.data().data();
        for (size_t o = 0; o < v.outer; ++o) {
            for (size_t a = 0; a < v.axis_len; ++a) {
                const float* src = src_base + (o * v.axis_len + a) * v.inner;
                float* dst = dst_base + o * v.inner;
                for (size_t i = 0; i < v.inner; ++i) dst[i] += src[i];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, v](const Tensor& grad_out) mutable {
            // Each element contributed once, so it receives the whole gradient
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t o = 0; o < v.outer; ++o) {
                const float* g = grad_out.data().data() + o * v.inner;
                for (size_t a = 0; a < v.axis_len; ++a) {
                    float* d = dX.data().data() + (o * v.axis_len + a) * v.inner;
                    for (size_t i = 0; i < v.inner; ++i) d[i] = g[i];
                }
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

Tensor Tensor::mean(size_t axis, bool keepdim) const {
    check_axis(axis, ndim(), "mean", shape_str());
    const float n = static_cast<float>(shape()[axis]);
    return sum(axis, keepdim) * (1.0f / n);
}

Tensor Tensor::max(size_t axis, bool keepdim) const {
    check_axis(axis, ndim(), "max", shape_str());
    if (size() == 0) throw std::invalid_argument("max over an empty tensor.");
    const AxisView v = axis_view(shape(), axis);

    bool req_g = track(requires_grad());
    Tensor res(reduced_shape(shape(), axis, keepdim), 0.0f, req_g);
    // The winning position of each reduction, for routing the gradient
    std::vector<size_t> argmax(res.size(), 0);

    for (size_t o = 0; o < v.outer; ++o) {
        for (size_t i = 0; i < v.inner; ++i) {
            float best = data()[o * v.axis_len * v.inner + i];
            size_t best_a = 0;
            for (size_t a = 1; a < v.axis_len; ++a) {
                const float value = data()[(o * v.axis_len + a) * v.inner + i];
                if (value > best) {
                    best = value;
                    best_a = a;
                }
            }
            res.data()[o * v.inner + i] = best;
            argmax[o * v.inner + i] = (o * v.axis_len + best_a) * v.inner + i;
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, argmax](const Tensor& grad_out) mutable {
            // Only the maximum influenced the output
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t k = 0; k < argmax.size(); ++k) {
                dX.data()[argmax[k]] += grad_out.data()[k];
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// ---------------------------------------------------------
// Slicing, concatenation and stacking
// ---------------------------------------------------------

Tensor Tensor::slice(size_t axis, size_t start, size_t count) const {
    check_axis(axis, ndim(), "slice", shape_str());
    if (count == 0) throw std::invalid_argument("slice needs at least one element.");
    if (start + count > shape()[axis]) {
        throw std::out_of_range("slice [" + std::to_string(start) + ", " +
                                std::to_string(start + count) + ") runs off axis " +
                                std::to_string(axis) + " of a tensor " + shape_str() + ".");
    }

    const AxisView v = axis_view(shape(), axis);
    std::vector<size_t> out_shape = shape();
    out_shape[axis] = count;

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);

    for (size_t o = 0; o < v.outer; ++o) {
        const float* src = data().data() + (o * v.axis_len + start) * v.inner;
        float* dst = res.data().data() + o * count * v.inner;
        std::copy_n(src, count * v.inner, dst);
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, v, start, count](const Tensor& grad_out) mutable {
            // The gradient goes back to its slot; the rest of the tensor gets zero
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t o = 0; o < v.outer; ++o) {
                const float* g = grad_out.data().data() + o * count * v.inner;
                float* d = dX.data().data() + (o * v.axis_len + start) * v.inner;
                std::copy_n(g, count * v.inner, d);
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

Tensor Tensor::concat(const std::vector<Tensor>& parts, size_t axis) {
    if (parts.empty()) throw std::invalid_argument("concat needs at least one tensor.");
    const std::vector<size_t>& first = parts[0].shape();
    check_axis(axis, first.size(), "concat", parts[0].shape_str());

    size_t total_axis = 0;
    bool req_g = false;
    for (const Tensor& t : parts) {
        if (t.ndim() != first.size()) {
            throw std::invalid_argument("concat needs the same number of axes in every part.");
        }
        for (size_t d = 0; d < first.size(); ++d) {
            if (d != axis && t.shape()[d] != first[d]) {
                throw std::invalid_argument("concat: the parts may only differ along axis " +
                                            std::to_string(axis) + "; " + parts[0].shape_str() +
                                            " against " + t.shape_str() + ".");
            }
        }
        total_axis += t.shape()[axis];
        req_g = req_g || t.requires_grad();
    }
    req_g = track(req_g);

    std::vector<size_t> out_shape = first;
    out_shape[axis] = total_axis;
    Tensor res(out_shape, 0.0f, req_g);

    const AxisView out_view = axis_view(out_shape, axis);
    size_t offset = 0;
    for (const Tensor& t : parts) {
        const size_t len = t.shape()[axis];
        const AxisView v = axis_view(t.shape(), axis);
        for (size_t o = 0; o < v.outer; ++o) {
            std::copy_n(t.data().data() + o * len * v.inner, len * v.inner,
                        res.data().data() + (o * out_view.axis_len + offset) * v.inner);
        }
        offset += len;
    }

    if (req_g) {
        std::vector<std::shared_ptr<TensorImpl>> parents;
        parents.reserve(parts.size());
        std::vector<Tensor> copies = parts;
        for (const Tensor& t : parts) parents.push_back(t.get_impl());
        res.impl_->parents = parents;

        res.impl_->backward_fn = [copies, axis, out_view](const Tensor& grad_out) mutable {
            // Each part receives the band of the gradient it contributed
            size_t off = 0;
            for (Tensor& t : copies) {
                const size_t len = t.shape()[axis];
                const AxisView v = axis_view(t.shape(), axis);
                if (t.requires_grad()) {
                    Tensor d(t.shape(), 0.0f, false);
                    for (size_t o = 0; o < v.outer; ++o) {
                        std::copy_n(
                            grad_out.data().data() + (o * out_view.axis_len + off) * v.inner,
                            len * v.inner, d.data().data() + o * len * v.inner);
                    }
                    t.add_grad(d);
                }
                off += len;
            }
        };
    }
    return res;
}

Tensor Tensor::stack(const std::vector<Tensor>& parts, size_t axis) {
    if (parts.empty()) throw std::invalid_argument("stack needs at least one tensor.");
    if (axis > parts[0].ndim()) {
        throw std::out_of_range("stack: axis " + std::to_string(axis) +
                                " does not fit in a tensor " + parts[0].shape_str() + ".");
    }
    // Stacking is concatenating after inserting a size-1 axis into each part
    std::vector<Tensor> expanded;
    expanded.reserve(parts.size());
    for (const Tensor& t : parts) {
        if (t.shape() != parts[0].shape()) {
            throw std::invalid_argument("stack needs every part to have the same shape: " +
                                        parts[0].shape_str() + " against " + t.shape_str() + ".");
        }
        std::vector<size_t> s = t.shape();
        s.insert(s.begin() + static_cast<long>(axis), 1);
        expanded.push_back(t.reshape(s));
    }
    return concat(expanded, axis);
}

// Row selection (gathering a mini-batch)
Tensor Tensor::select_rows(const std::vector<size_t>& indices) const {
    if (ndim() < 2) {
        throw std::invalid_argument("select_rows requires at least 2 dimensions; this tensor is " +
                                    shape_str() + ".");
    }
    if (indices.empty()) {
        throw std::invalid_argument("select_rows received an empty index list.");
    }

    const size_t rows = shape()[0];
    // Everything after the first axis travels together: for (N, C, H, W) each
    // "row" is a whole image of C*H*W contiguous values.
    const size_t row_size = (rows == 0) ? 0 : size() / rows;

    for (size_t idx : indices) {
        if (idx >= rows) {
            throw std::out_of_range("select_rows: row " + std::to_string(idx) +
                                    " does not exist in a tensor " + shape_str() + ".");
        }
    }

    std::vector<size_t> out_shape = shape();
    out_shape[0] = indices.size();

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);
    for (size_t i = 0; i < indices.size(); ++i) {
        std::copy_n(data().begin() + indices[i] * row_size, row_size,
                    res.data().begin() + i * row_size);
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        const std::vector<size_t>& idx_copy = indices;

        res.impl_->backward_fn = [self_copy, idx_copy, row_size](const Tensor& grad_out) mutable {
            // Inverse scatter: each row returns its gradient to the source row. The += is
            // necessary because an index may repeat.
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t i = 0; i < idx_copy.size(); ++i) {
                for (size_t j = 0; j < row_size; ++j) {
                    dX.data()[idx_copy[i] * row_size + j] += grad_out.data()[i * row_size + j];
                }
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// ---------------------------------------------------------
// Scalar-on-the-left operators
// ---------------------------------------------------------

Tensor operator+(float scalar, const Tensor& t) {
    return t + scalar;
}
Tensor operator*(float scalar, const Tensor& t) {
    return t * scalar;
}
Tensor operator-(float scalar, const Tensor& t) {
    return (t * -1.0f) + scalar;
}

// Console printing
void Tensor::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " = ";
    }
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape().size(); ++i) {
        std::cout << shape()[i] << (i + 1 < shape().size() ? ", " : "");
    }
    std::cout << "], requires_grad=" << (requires_grad() ? "true" : "false") << ")\n";

    if (ndim() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape()[0]; ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data()[i];
            if (i + 1 < shape()[0]) std::cout << ", ";
        }
        std::cout << "]\n";
    } else if (ndim() == 2) {
        std::cout << "[\n";
        for (size_t i = 0; i < shape()[0]; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < shape()[1]; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << (*this)({i, j});
                if (j + 1 < shape()[1]) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "]\n";
    } else if (size() > 0) {
        std::cout << "[ ";
        for (size_t i = 0; i < size(); ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data()[i] << " ";
            if ((i + 1) % shape().back() == 0 && i + 1 < size()) std::cout << "\n  ";
        }
        std::cout << "]\n";
    } else {
        std::cout << "[]\n";
    }

    if (has_grad()) {
        std::cout << "  grad = \n";
        grad().print("  gradientes");
    }
    std::cout << "\n";
}

}  // namespace engine
