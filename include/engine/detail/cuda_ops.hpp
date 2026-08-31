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
bool binary(Binary op, const Storage& a, const Storage& b, Storage& out, size_t inner,
            size_t repeat);

// out = a x b + beta * out, batched. An unbatched operand (a_batched/b_batched
// false) is reused for every matrix in the batch, exactly as on the CPU path.
//
// `beta` is BLAS's, and beta == 0 is what every caller meant before it existed:
// overwrite. beta == 1 accumulates, which is what lets the backward pass write a
// gradient straight into the parameter's gradient instead of building a
// temporary and adding it in a second kernel -- grad_accumulate was twenty
// launches and 13.5% of GPU time per training step.
//
// Only the naive, tiled and register-tiled variants implement it. Split-K
// finishes with sum_over_axis, which overwrites, and the tensor-core kernels
// store through store_matrix_sync, which has no accumulate form; both refuse
// beta != 0 and the caller takes the CPU path, same as any other refusal here.
bool matmul(const Storage& a, const Storage& b, Storage& out, size_t batch, size_t rows,
            size_t inner_dim, size_t cols, bool a_batched, bool b_batched, float beta = 0.0f);

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
bool permute(const Storage& x, Storage& out, const size_t* out_shape, const size_t* src_strides,
             size_t ndim);

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

// Whole-buffer reductions to one scalar, on the device.
//
// **double and not float**, and that is not a detail. Tensor::sum() accumulates
// in double on purpose -- summing a million floats loses the low bits as soon as
// the total grows against each addend, and mean(), mse_loss and the initial
// gradient of every backward all hang off it. A float reduction here would
// quietly undo a fix that has a commit of its own.
//
// Two stages with a fixed block count, not an atomicAdd. Atomics would be
// shorter and would make the result depend on the order blocks happen to finish
// in, so two runs on the same data could disagree in the last bits. The engine
// promises that they do not.
//
// These exist because one host-side reduction in the middle of a step poisons
// everything after it: clip_grad_norm reading p.grad().data() pulled every
// gradient down, which left the optimiser kernel nothing device-resident to work
// on and it never fired at all.
bool reduce_sum(const Storage& x, double& out);
bool reduce_sum_squares(const Storage& x, double& out);

// x *= factor, in place. Distinct from scalar() because that one writes to a
// separate output and aliasing it onto its own input is the device_write() trap:
// the buffer would be marked valid without being uploaded, and the read would
// see whatever was there before.
bool scale_in_place(Storage& x, float factor);

// Writes `src` into a window of `dst`, in place, on the device. This is
// Tensor::copy_into's device path and the operation a key/value cache appends
// with: the whole point is that a cache which lives on the card is appended to
// on the card, instead of coming down and going back up to add one position.
//
// The window is described the way every axis operation here describes one --
// (outer, axis_len, inner) -- with `count` rows of the axis written starting at
// `start`. dst is expected to be resident; src is uploaded if it is not, which
// is one small transfer rather than the whole destination.
// Reads a window out of `src` on the device: Tensor::slice's device path, and
// the mirror of copy_into below.
//
// The asymmetry this closes: copy_into has had a device path since the key/value
// cache landed, on the argument that a cache living on the card should be
// appended to on the card. Narrowing one is the same argument -- a cached
// forward costs what its capacity is, so a server wants to hand the model a
// cache cut to the width its batch has reached, and cutting it through the host
// moves the whole thing across PCIe twice to read a prefix of it.
//
// Measured before it was written: gathering sixteen slots of a 1024-position
// pool is 1.86 ms for a whole model's keys and values, and slicing the result to
// 464 positions took that to 149 ms. Two orders of magnitude, all of it PCIe.
//
// The window is described as every axis operation here describes one --
// (outer, axis_len, inner) -- reading `count` rows of the axis from `start`.
// `src` is expected to be resident; if it is not, the host loop is right.
bool slice(Storage& out, const Storage& src, size_t outer, size_t src_axis_len, size_t count,
           size_t start, size_t inner);

bool copy_into(Storage& dst, const Storage& src, size_t outer, size_t dst_axis_len, size_t count,
               size_t start, size_t inner);

// copy_into with a start per row of the first axis. `rows` is that axis's
// length, `per_row` how many (outer) blocks belong to each of its indices, and
// `offsets` has `rows` entries.
//
// The offsets travel as a kernel argument rather than a device buffer: there are
// at most a few hundred of them, they change every step, and a buffer would mean
// an allocation and an upload per step to carry a kilobyte. Above the cap the
// call declines and the caller falls back, which for a resident cache means a
// round trip -- so the cap is set well above any batch size worth serving.
bool copy_into_rows(Storage& dst, const Storage& src, size_t rows, size_t per_row,
                    size_t dst_axis_len, size_t count, const size_t* offsets, size_t inner);

// copy_into_rows with an index list: row i of `src` goes into row `into[i]` of
// `dst`, at `offsets[i]` along the axis. The write counterpart of gather_rows,
// and the call a slot-indexed cache ends a step with.
//
// Neither existing operation does it. select_rows takes indices and no offset;
// copy_into_rows takes an offset per row and no indices. Putting a compact batch
// back into the slots it was gathered from therefore took one call per row per
// block -- 192 launches at a batch of sixteen, measured at 2.2 ms of launch
// overhead against a cached step of three to four milliseconds.
//
// Two rows naming the same destination are refused rather than ordered: they
// would race, and which won would depend on which block finished last.
//
// The cap is 256 rows rather than 512, because two lists travel where one did
// and the limit is the 4 KB CUDA allows for kernel arguments.
bool scatter_rows(Storage& dst, const Storage& src, size_t rows, size_t per_row,
                  size_t dst_axis_len, size_t count, const size_t* into, const size_t* offsets,
                  size_t inner);

// select_rows on the device: gathers whole rows of the first axis into a new
// tensor, in the order given.
//
// The read counterpart of copy_into_rows, and it exists for the same caller. A
// key/value cache lives on the card and is indexed by slot; the batch being
// stepped is some subset of those slots in some order, so getting from one to
// the other is a gather. Doing it through the host would move the whole cache
// across PCIe twice per step to read a part of it.
//
// The indices travel as a kernel argument for the same reasons as the offsets
// do; above the cap the call declines and the caller falls back to host.
bool gather_rows(Storage& out, const Storage& src, const size_t* indices, size_t count,
                 size_t row_size);

// One optimiser step, in place on the parameter.
//
// These two are the largest transfer saving in the engine and the least
// interesting kernels in it: element-wise, no reductions, no shared memory. The
// size is the point. Reading `p.data()` and `grad.data()` for every parameter
// pulls the whole model down and pushes it back up once per step -- on MNIST,
// 206,922 parameters, which is **606 MiB over PCIe per training run** from the
// optimiser alone, on a model whose forward pass had just finished computing
// entirely on the device.
//
// `velocity`, `m` and `v` are the optimiser's own state and have to be device
// resident too, or the transfers only move rather than disappear.
//
// The dispatch condition is not a size threshold. It is whether the **gradient
// is already on the device**: if the backward ran there, this costs nothing and
// saves a round trip, and if it ran on the host, uploading a gradient just to
// subtract it would pay exactly the cost this exists to avoid. Below any
// sensible size threshold the launch is still cheaper than the download it
// replaces, because the download also forces a synchronisation.
bool sgd_step(Storage& param, const Storage& grad, Storage* velocity, float lr, float momentum,
              float weight_decay);
bool adam_step(Storage& param, const Storage& grad, Storage& m, Storage& v, float lr, float beta1,
               float beta2, float eps, float weight_decay, float bias_c1, float bias_c2);

// Layer normalisation over the last axis, and its derivative.
//
// The forward is row-local and unremarkable: one block per row, a shared-memory
// tree for the mean and another for the variance. `xhat` and `inv_std` come back
// out because the backward needs both, exactly as the CPU path saves them.
//
// The backward is the one with a design problem, and it is why this pair was
// deferred twice. `dx` is row-local like the forward, but **dgamma and dbeta
// accumulate across rows** -- every row contributes to the same `cols` values.
// Doing that with atomicAdd would be four lines and would make the result depend
// on the order blocks happen to finish in, which this engine promises it does
// not: `tests/test_transformer.cpp` compares against PyTorch fixtures, and the
// parity suite compares against the CPU.
//
// So a fixed number of blocks each own a private partial in `dgamma`/`dbeta`
// scratch and accumulate the rows they are given, and a second pass sums the
// partials in index order. The second pass is sum_over_axis with outer=1: the
// reduction kernel that already exists, doing what it was written for.
//
// Splitting the pair is not an option worth having. A forward on the device
// whose backward reads xhat on the host pulls everything down anyway, so the
// chain breaks in the same place and the only thing gained is a second copy of
// the formula.
bool layernorm(const Storage& x, const Storage& gamma, const Storage& beta, Storage& out,
               Storage& xhat, Storage& inv_std, size_t rows, size_t cols, float eps);
bool layernorm_backward(const Storage& grad_out, const Storage& xhat, const Storage& gamma,
                        const Storage& inv_std, Storage& dx, Storage& dgamma, Storage& dbeta,
                        size_t rows, size_t cols);

// Softmax over the last axis: `rows` rows of `cols` contiguous values.
bool softmax(const Storage& x, Storage& out, size_t rows, size_t cols);
// y is the saved forward output, not the input.
bool softmax_backward(const Storage& y, const Storage& grad_out, Storage& out, size_t rows,
                      size_t cols);

}  // namespace ops
}  // namespace cuda
}  // namespace engine

#endif  // ENGINE_DETAIL_CUDA_OPS_HPP
