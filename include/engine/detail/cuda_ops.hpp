#ifndef ENGINE_DETAIL_CUDA_OPS_HPP
#define ENGINE_DETAIL_CUDA_OPS_HPP

#include <cstddef>

#include "engine/detail/storage.hpp"

namespace engine {
namespace cuda {
namespace ops {

// ---------------------------------------------------------
// Kernel entry points.
//
// Every one of them returns **true if the work was done on the device**.
// Returning false means "not worth it here" or "this binary has no CUDA", and
// the caller carries on down the CPU path without needing to know anything
// else. That contract is what keeps src/tensor.cpp readable: one condition per
// operation, not two implementations tangled together.
//
// Without ENGINE_CUDA they still exist, implemented in src/cuda_disabled.cpp
// returning false, so dispatch needs no #ifdef.
// ---------------------------------------------------------

enum class Binary { Add, Sub, Mul, Div };

// out = a `op` b, broadcasting by suffix: b has `inner` elements repeated
// `repeat` times to cover a. Without broadcasting, inner == a.size() and
// repeat == 1.
bool binary(Binary op, const Storage& a, const Storage& b, Storage& out,
            size_t inner, size_t repeat);

// out = a x b, batched. An unbatched operand (a_batched/b_batched false) is
// reused for every matrix in the batch, exactly as on the CPU path.
bool matmul(const Storage& a, const Storage& b, Storage& out,
            size_t batch, size_t rows, size_t inner_dim, size_t cols,
            bool a_batched, bool b_batched);

bool relu(const Storage& x, Storage& out);
bool relu_backward(const Storage& x, const Storage& grad_out, Storage& out);

// The backward accumulator: grad = g the first time, grad += g afterwards.
//
// It cannot be binary(Add, grad, g, grad, ...) with the output aliased onto the
// input: that path asks for the output with device_write(), which marks the
// buffer valid **without uploading it**, so the input's device() then sees it
// is already valid and does not upload either. The accumulated value would be
// lost. On top of that the evaluation order of the launch arguments is
// unspecified, so it would depend on the compiler. Hence a kernel of its own
// rather than reusing binary().
bool accumulate_grad(Storage& grad, const Storage& g, bool initialize);

// out = x * mul + add, in a single pass. Both of the engine's scalar operations
// land here: `t * k` is (k, 0) and `t + k` is (1, k). It does not look like
// much, but without a kernel the attention scaling — a `* 1/sqrt(d_k)` between
// two matmuls — pulled the whole tensor down to host and pushed it back up.
bool scalar(const Storage& x, Storage& out, float mul, float add);

// Axis reordering: out[flat] = x[sum_d coord_d * src_strides[d]], where the
// coordinates come from out_shape. Covers permute() and transpose(), the latter
// being the special case of swapping the last two axes.
//
// Both vectors describe the output over the input's memory, exactly as on the
// CPU path: the caller has already computed them.
bool permute(const Storage& x, Storage& out,
             const size_t* out_shape, const size_t* src_strides, size_t ndim);

// Sum over one axis, viewing the tensor as (outer, axis_len, inner). It is the
// same decomposition AxisView uses in src/tensor.cpp.
bool sum_axis(const Storage& x, Storage& out, size_t outer, size_t axis_len, size_t inner);

// The geometry of a sliding window over a batch of volumes. It repeats what
// nn::Window2d already says plus the tensor dimensions, deliberately: this way
// the header does not depend on engine/conv.hpp and the backend still knows
// nothing about layers.
struct WindowShape {
    size_t batch, channels, height, width;
    size_t kernel_h, kernel_w, stride, padding;
    size_t out_h, out_w;
};

// im2col flattens each window into a row: (N,C,H,W) -> (N*oH*oW, C*kH*kW).
// col2im is its adjoint and sums where windows overlap.
//
// Without these two, giving the convolution's matrix product a kernel buys
// nothing: the columns are kH*kW times the input, so uploading them costs more
// than multiplying them. Measured on MNIST: 24.6 s with the product on the GPU
// and the columns built on the host, against 19.0 s doing it all on the CPU.
bool im2col(const Storage& input, Storage& cols, const WindowShape& s);
bool col2im(const Storage& cols, Storage& input, const WindowShape& s);

// Max pooling. `argmax` holds the flat index of each window's winning pixel,
// which is all the backward pass needs.
//
// Without these two the chain breaks **between the two convolutions**: the
// first one's output goes down to host to be pooled and comes back up for the
// second. With large evaluation batches that is tens of MB per pass.
//
// The backward pass walks the input rather than the output, just like col2im:
// that way each pixel sums the windows that chose it with no atomics and a
// fixed accumulation order, which is what keeps the result reproducible.
//
// ponytail: the index travels as a float, exact up to 2^24; above that the
// dispatch is refused and the CPU takes it. A Storage of integers would be the
// right answer if that ceiling ever matters.
bool maxpool(const Storage& input, Storage& out, Storage& argmax, const WindowShape& s);
bool maxpool_backward(const Storage& argmax, const Storage& grad_out, Storage& dx,
                      const WindowShape& s);

// Softmax over the last axis: `rows` rows of `cols` contiguous values.
bool softmax(const Storage& x, Storage& out, size_t rows, size_t cols);
// y is the saved forward output, not the input.
bool softmax_backward(const Storage& y, const Storage& grad_out, Storage& out,
                      size_t rows, size_t cols);

} // namespace ops
} // namespace cuda
} // namespace engine

#endif // ENGINE_DETAIL_CUDA_OPS_HPP
