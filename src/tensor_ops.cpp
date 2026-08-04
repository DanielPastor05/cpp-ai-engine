// The arithmetic operators, and the autograd nodes they register.
//
// Lifted out of src/tensor.cpp, which was 1 602 lines. Every function here
// produces a tensor and, if the graph is live, a backward_fn that knows how to
// push a gradient through it. Nothing else in the engine calls into them: they
// are reached through Tensor's public operators, which is what made this the
// clean seam.
//
// The helpers they share with the rest -- broadcasting plans, the device offer,
// the parallel-split thresholds -- are in src/tensor_detail.hpp, and that header
// says why they can be inline when the CUDA ones could not.

#include "tensor_detail.hpp"

#include "engine/detail/tensor_impl.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace engine {

using namespace tensor_detail;  // NOLINT: this file is the other half of tensor.cpp

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
        const float* ENGINE_RESTRICT lhs = data();
        const float* ENGINE_RESTRICT rhs = other.data();
        float* ENGINE_RESTRICT out = res.data();
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
        const float* ENGINE_RESTRICT lhs = data();
        const float* ENGINE_RESTRICT rhs = other.data();
        float* ENGINE_RESTRICT out = res.data();
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
        const float* ENGINE_RESTRICT lhs = data();
        const float* ENGINE_RESTRICT rhs = other.data();
        float* ENGINE_RESTRICT out = res.data();
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
        const float* ENGINE_RESTRICT lhs = data();
        const float* ENGINE_RESTRICT rhs = other.data();
        float* ENGINE_RESTRICT out = res.data();
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
        const float* ENGINE_RESTRICT lhs = data();
        float* ENGINE_RESTRICT out = res.data();
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
        const float* ENGINE_RESTRICT lhs = data();
        float* ENGINE_RESTRICT out = res.data();
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
        transpose_blocked(data(), res.data(), batch, rows, cols);
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

    // Whether this permutation is only a swap of the last two axes. That is the
    // shape Conv2d asks for -- (N, oH*oW, C) back into (N, C, oH*oW) -- and it is
    // the one case where the generic gather below is doing far more work than the
    // problem requires. Measured at (64, 784, 16): 2.494 ms through the general
    // path, 46% of the whole first convolution of the MNIST model.
    bool last_two_swapped = nd >= 2 && order[nd - 2] == nd - 1 && order[nd - 1] == nd - 2;
    for (size_t i = 0; i + 2 < nd && last_two_swapped; ++i) {
        if (order[i] != i) last_two_swapped = false;
    }

    if (!cuda::ops::permute(impl_->storage, res.impl_->storage, out_shape.data(),
                            src_strides.data(), nd)) {
        if (last_two_swapped) {
            const size_t rows = shape()[nd - 2];
            const size_t cols = shape()[nd - 1];
            transpose_blocked(data(), res.data(), size() / std::max<size_t>(1, rows * cols), rows,
                              cols);
            if (req_g) {
                res.impl_->parents = {impl_};
                Tensor self_copy = *this;
                std::vector<size_t> inverse(nd);
                for (size_t i = 0; i < nd; ++i) inverse[order[i]] = i;
                res.impl_->backward_fn = [self_copy, inverse](const Tensor& grad_out) mutable {
                    self_copy.add_grad(grad_out.permute(inverse));
                };
            }
            return res;
        }

        const float* ENGINE_RESTRICT src_data = data();
        float* ENGINE_RESTRICT dst_data = res.data();
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
        const float* ENGINE_RESTRICT a_data = data();
        const float* ENGINE_RESTRICT b_data = B.data();
        float* ENGINE_RESTRICT c_data = C.data();

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

                    const float* ENGINE_RESTRICT a_row = a_data + b * a_stride + i * K;
                    const float* ENGINE_RESTRICT bm = b_data + b * b_stride;
                    float* ENGINE_RESTRICT c_row = c_data + b * M * N + i * N;

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
        const float* ENGINE_RESTRICT lhs = data();
        float* ENGINE_RESTRICT out = res.data();
        for (size_t i = 0; i < n; ++i) out[i] = std::max(0.0f, lhs[i]);
    }

    if (req_g) {
        res.impl_->parents = {impl_};
        Tensor self_copy = *this;

        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            Tensor dX(self_copy.shape(), 0.0f, false);
            if (!cuda::ops::relu_backward(self_copy.storage(), grad_out.storage(), dX.storage())) {
                const size_t n = self_copy.size();
                const float* ENGINE_RESTRICT x = self_copy.data();
                const float* ENGINE_RESTRICT g = grad_out.data();
                float* ENGINE_RESTRICT out = dX.data();
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
        const float* ENGINE_RESTRICT src = data();
        float* ENGINE_RESTRICT dst = res.data();
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
            if (!cuda::ops::softmax_backward(saved.storage(), grad_out.storage(), dX.storage(),
                                             rows, cols)) {
                const float* ENGINE_RESTRICT y = saved.data();
                const float* ENGINE_RESTRICT g = grad_out.data();
                float* ENGINE_RESTRICT out = dX.data();
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
        const float* ENGINE_RESTRICT values = data();
        for (size_t i = 0; i < size(); ++i) total += values[i];
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
