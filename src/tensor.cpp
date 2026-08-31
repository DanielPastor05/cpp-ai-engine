#include "engine/tensor.hpp"
#include "tensor_detail.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"
#include "engine/detail/cuda_ops.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "engine/random.hpp"

namespace engine {

using namespace tensor_detail;

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
    // Two postconditions every index computation in this engine assumes and none
    // of them states: there is one stride per axis, and the last one is 1 because
    // the buffer is contiguous in C order. Every kernel, every parallel_for and
    // every im2col gather is written against both.
    assert(strides.size() == shape.size() && "one stride per axis");
    assert(strides.back() == 1 && "the tensor is contiguous, so the last stride is 1");
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
    if (!cuda::ops::accumulate_grad(acc, g.storage(), initialize)) {
        // g.data() stays inside the if deliberately: outside, the download would come
        // back in through the back door on the path the device does accelerate.
        std::vector<float>& dst = acc.host_mut();
        const float* ENGINE_RESTRICT src = g.data();
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
    impl_->grad = std::make_shared<TensorImpl>(shape(), grad_output.to_vector(), false);
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
const float* Tensor::data() const {
    return impl_->storage.host().data();
}
float* Tensor::data() {
    return impl_->storage.host_mut().data();
}
std::vector<float> Tensor::to_vector() const {
    return impl_->storage.host();
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
        const float* ENGINE_RESTRICT src_base = data();
        float* ENGINE_RESTRICT dst_base = res.data();
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
                const float* g = grad_out.data() + o * v.inner;
                for (size_t a = 0; a < v.axis_len; ++a) {
                    float* d = dX.data() + (o * v.axis_len + a) * v.inner;
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

    // The device path, admitted on residency rather than size, and for the same
    // reason copy_into's is: a tensor that lives on the card should be narrowed
    // on the card. The caller this exists for is a key/value cache -- a cached
    // forward attends over its whole capacity, so a server hands the model a
    // cache cut to the width its batch has reached, every step. Cutting it
    // through the host moves the whole cache across PCIe to read a prefix.
    //
    // The gradient path stays on the host. Slicing inside a live graph is a
    // training operation, and training does not slice a cache per step.
    bool on_device = false;
#ifdef ENGINE_CUDA
    on_device = !req_g && cuda::ops::slice(res.storage(), storage(), v.outer, v.axis_len, count,
                                           start, v.inner);
#endif
    if (!on_device) {
        for (size_t o = 0; o < v.outer; ++o) {
            const float* src = data() + (o * v.axis_len + start) * v.inner;
            float* dst = res.data() + o * count * v.inner;
            std::copy_n(src, count * v.inner, dst);
        }
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, v, start, count](const Tensor& grad_out) mutable {
            // The gradient goes back to its slot; the rest of the tensor gets zero
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t o = 0; o < v.outer; ++o) {
                const float* g = grad_out.data() + o * count * v.inner;
                float* d = dX.data() + (o * v.axis_len + start) * v.inner;
                std::copy_n(g, count * v.inner, d);
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

void Tensor::copy_into(const Tensor& src, size_t axis, size_t start) {
    check_axis(axis, ndim(), "copy_into", shape_str());
    if (src.ndim() != ndim()) {
        throw std::invalid_argument("copy_into needs the same number of axes: " + shape_str() +
                                    " against " + src.shape_str() + ".");
    }
    for (size_t d = 0; d < ndim(); ++d) {
        if (d != axis && src.shape()[d] != shape()[d]) {
            throw std::invalid_argument("copy_into: the two may only differ along axis " +
                                        std::to_string(axis) + "; " + shape_str() + " against " +
                                        src.shape_str() + ".");
        }
    }
    const size_t count = src.shape()[axis];
    if (count == 0) throw std::invalid_argument("copy_into needs at least one element.");
    if (start + count > shape()[axis]) {
        throw std::out_of_range("copy_into [" + std::to_string(start) + ", " +
                                std::to_string(start + count) + ") runs off axis " +
                                std::to_string(axis) + " of a tensor " + shape_str() + ".");
    }

    // Refused rather than handled. Every other operation here records how its
    // output was computed; this one changes a value that some earlier output may
    // already have been computed from, and no backward can describe that. A
    // cache is an inference structure and this is an inference operation.
    if (requires_grad() || src.requires_grad()) {
        throw std::invalid_argument(
            "copy_into writes in place and cannot run on a tensor that requires grad.");
    }

    const AxisView v = axis_view(shape(), axis);

#ifdef ENGINE_CUDA
    // No size threshold. The alternative to a small device copy is not a cheap
    // host loop -- it is pulling the destination down, writing, and pushing it
    // back, which for a cache is the entire cache over PCIe to append one
    // position. Residency decides, not size.
    if (cuda::ops::copy_into(storage(), src.storage(), v.outer, v.axis_len, count, start,
                             v.inner)) {
        return;
    }
#endif

    float* dst = data();
    const float* values = src.data();
    for (size_t o = 0; o < v.outer; ++o) {
        std::copy_n(values + o * count * v.inner, count * v.inner,
                    dst + (o * v.axis_len + start) * v.inner);
    }
}

void Tensor::copy_into_rows(const Tensor& src, size_t axis, const std::vector<size_t>& offsets) {
    check_axis(axis, ndim(), "copy_into_rows", shape_str());
    if (axis == 0) {
        throw std::invalid_argument(
            "copy_into_rows writes along an axis after the one the offsets index, so the axis "
            "cannot be 0.");
    }
    if (src.ndim() != ndim()) {
        throw std::invalid_argument("copy_into_rows needs the same number of axes: " + shape_str() +
                                    " against " + src.shape_str() + ".");
    }
    for (size_t d = 0; d < ndim(); ++d) {
        if (d != axis && src.shape()[d] != shape()[d]) {
            throw std::invalid_argument("copy_into_rows: the two may only differ along axis " +
                                        std::to_string(axis) + "; " + shape_str() + " against " +
                                        src.shape_str() + ".");
        }
    }
    const size_t rows = shape()[0];
    // Checked for its own sake and for the division below, which the static
    // analyser is right to refuse to take on trust: a tensor with a zero-length
    // first axis has no rows to give offsets to.
    if (rows == 0) {
        throw std::invalid_argument("copy_into_rows needs at least one row, and " + shape_str() +
                                    " has none.");
    }
    if (offsets.size() != rows) {
        throw std::invalid_argument("copy_into_rows was given " + std::to_string(offsets.size()) +
                                    " offsets for " + std::to_string(rows) + " rows.");
    }
    const size_t count = src.shape()[axis];
    if (count == 0) throw std::invalid_argument("copy_into_rows needs at least one element.");
    for (size_t r = 0; r < rows; ++r) {
        if (offsets[r] + count > shape()[axis]) {
            throw std::out_of_range("copy_into_rows: row " + std::to_string(r) + " writes [" +
                                    std::to_string(offsets[r]) + ", " +
                                    std::to_string(offsets[r] + count) + ") off axis " +
                                    std::to_string(axis) + " of a tensor " + shape_str() + ".");
        }
    }
    if (requires_grad() || src.requires_grad()) {
        throw std::invalid_argument(
            "copy_into_rows writes in place and cannot run on a tensor that requires grad.");
    }

    const AxisView v = axis_view(shape(), axis);
    // Everything between the row axis and the written axis. For a key/value
    // cache of (batch, heads, positions, head_dim) written along positions, this
    // is the heads: every head of a row shares that row's offset.
    const size_t per_row = v.outer / rows;

#ifdef ENGINE_CUDA
    if (cuda::ops::copy_into_rows(storage(), src.storage(), rows, per_row, v.axis_len, count,
                                  offsets.data(), v.inner)) {
        return;
    }
#endif

    float* dst = data();
    const float* values = src.data();
    for (size_t r = 0; r < rows; ++r) {
        for (size_t o = 0; o < per_row; ++o) {
            const size_t block = r * per_row + o;
            std::copy_n(values + block * count * v.inner, count * v.inner,
                        dst + (block * v.axis_len + offsets[r]) * v.inner);
        }
    }
}

void Tensor::scatter_rows(const Tensor& src, size_t axis, const std::vector<size_t>& into,
                          const std::vector<size_t>& offsets) {
    check_axis(axis, ndim(), "scatter_rows", shape_str());
    if (axis == 0) {
        throw std::invalid_argument(
            "scatter_rows writes along an axis after the one the indices select, so the axis "
            "cannot be 0.");
    }
    if (src.ndim() != ndim()) {
        throw std::invalid_argument("scatter_rows needs the same number of axes: " + shape_str() +
                                    " against " + src.shape_str() + ".");
    }
    for (size_t d = 1; d < ndim(); ++d) {
        if (d != axis && src.shape()[d] != shape()[d]) {
            throw std::invalid_argument(
                "scatter_rows: the two may only differ along axis 0 and axis " +
                std::to_string(axis) + "; " + shape_str() + " against " + src.shape_str() + ".");
        }
    }
    const size_t rows = src.shape()[0];
    if (rows == 0) {
        throw std::invalid_argument("scatter_rows needs at least one row to write.");
    }
    if (into.size() != rows || offsets.size() != rows) {
        throw std::invalid_argument("scatter_rows was given " + std::to_string(into.size()) +
                                    " indices and " + std::to_string(offsets.size()) +
                                    " offsets for " + std::to_string(rows) + " rows of source.");
    }
    const size_t count = src.shape()[axis];
    if (count == 0) throw std::invalid_argument("scatter_rows needs at least one element.");
    for (size_t r = 0; r < rows; ++r) {
        if (into[r] >= shape()[0]) {
            throw std::out_of_range("scatter_rows: row " + std::to_string(r) + " names slot " +
                                    std::to_string(into[r]) + " of a tensor " + shape_str() + ".");
        }
        if (offsets[r] + count > shape()[axis]) {
            throw std::out_of_range("scatter_rows: row " + std::to_string(r) + " writes [" +
                                    std::to_string(offsets[r]) + ", " +
                                    std::to_string(offsets[r] + count) + ") off axis " +
                                    std::to_string(axis) + " of a tensor " + shape_str() + ".");
        }
    }
    // Two source rows naming the same destination would race on the device and
    // would be order-dependent on the host. Refused in both, so the two paths
    // agree about what is a program and what is a bug.
    for (size_t a = 0; a < rows; ++a) {
        for (size_t b = a + 1; b < rows; ++b) {
            if (into[a] == into[b]) {
                throw std::invalid_argument("scatter_rows: rows " + std::to_string(a) + " and " +
                                            std::to_string(b) + " both write slot " +
                                            std::to_string(into[a]) + ".");
            }
        }
    }
    if (requires_grad() || src.requires_grad()) {
        throw std::invalid_argument(
            "scatter_rows writes in place and cannot run on a tensor that requires grad.");
    }

    const AxisView v = axis_view(shape(), axis);
    const size_t per_row = v.outer / shape()[0];

#ifdef ENGINE_CUDA
    if (cuda::ops::scatter_rows(storage(), src.storage(), rows, per_row, v.axis_len, count,
                                into.data(), offsets.data(), v.inner)) {
        return;
    }
#endif

    float* dst = data();
    const float* values = src.data();
    for (size_t r = 0; r < rows; ++r) {
        for (size_t o = 0; o < per_row; ++o) {
            const size_t from = r * per_row + o;
            const size_t to = into[r] * per_row + o;
            std::copy_n(values + from * count * v.inner, count * v.inner,
                        dst + (to * v.axis_len + offsets[r]) * v.inner);
        }
    }
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
            std::copy_n(t.data() + o * len * v.inner, len * v.inner,
                        res.data() + (o * out_view.axis_len + offset) * v.inner);
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
                        std::copy_n(grad_out.data() + (o * out_view.axis_len + off) * v.inner,
                                    len * v.inner, d.data() + o * len * v.inner);
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

#ifdef ENGINE_CUDA
    // Admitted on residency, like the other two. The caller this was added for
    // is a key/value cache that lives on the card and is indexed by slot: the
    // batch being stepped is some subset of those slots in some order, so
    // reading it is a gather, and doing it through the host would move the whole
    // cache across PCIe twice a step to read a part of it.
    //
    // The gradient path stays on the host. A gather inside a live graph is a
    // training operation and training does not have this problem.
    const bool on_device =
        !req_g &&
        cuda::ops::gather_rows(res.storage(), storage(), indices.data(), indices.size(), row_size);
    if (!on_device)
#endif
    {
        for (size_t i = 0; i < indices.size(); ++i) {
            std::copy_n(data() + indices[i] * row_size, row_size, res.data() + i * row_size);
        }
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

}  // namespace engine
