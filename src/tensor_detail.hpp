// The helpers src/tensor.cpp and src/tensor_ops.cpp both need.
//
// src/tensor.cpp reached 1 602 lines, which two separate plans pointed at and
// neither split. The arithmetic operators and their autograd registration are
// the half that comes out cleanly: 720 lines that nothing else calls into.
//
// Everything here is `inline` at namespace scope, and unlike the CUDA kernels'
// shared header that is safe rather than dangerous. Not one of these functions
// owns a function-local static -- they are pure, taking arguments and returning
// values -- so a copy per translation unit is a copy of stateless code that the
// linker folds. src/cuda/kernels_common.cuh had the opposite problem and says so
// at length: four helpers there cache things, and duplicating them would have
// given each unit its own cache.

#ifndef ENGINE_SRC_TENSOR_DETAIL_HPP
#define ENGINE_SRC_TENSOR_DETAIL_HPP

#include "engine/autograd.hpp"
#include "engine/detail/cuda_ops.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace engine {
namespace tensor_detail {

using parallel::kElementsPerThread;

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

// Batched 2D transpose, blocked. dst[b][j][i] = src[b][i][j].
//
// This exists because the post-convolution axis swap was measured at **2.57
// GB/s** on a machine whose memory does about 35, and it was 46% of the first
// convolution of the MNIST model -- more than im2col and the matrix product
// together. Both of the loops that used to do it were limited by the same thing
// and neither addressed it.
//
// A transpose cannot make both sides contiguous: one of read and write has to
// stride. The old transpose() strided its writes, permute() paid per-element
// index arithmetic on top, and docs/PERFORMANCE.md records permute coming out 5%
// ahead -- a real measurement of two variants of the same mistake.
//
// Blocking sidesteps the choice. A 32x32 tile is 4 KB, so once it is read the
// whole tile is in L1 and the strided side is served from cache rather than from
// memory. Inside a tile the write is the contiguous one, deliberately: a
// scattered write costs a read-for-ownership of the line as well.
//
// Deterministic by construction -- every destination element is written exactly
// once by exactly one thread, and no value is accumulated -- so the bit-for-bit
// promise the rest of the engine makes is not at risk here.
constexpr size_t kTransposeTile = 32;

inline void transpose_blocked(const float* ENGINE_RESTRICT src, float* ENGINE_RESTRICT dst,
                              size_t batch, size_t rows, size_t cols) {
    const size_t row_tiles = (rows + kTransposeTile - 1) / kTransposeTile;
    const size_t work_per_tile = std::max<size_t>(1, kTransposeTile * cols);
    const size_t tiles_per_thread = std::max<size_t>(1, kElementsPerThread / work_per_tile);

    parallel::parallel_for(batch * row_tiles, tiles_per_thread, [&](size_t from, size_t to) {
        for (size_t t = from; t < to; ++t) {
            const size_t b = t / row_tiles;
            const size_t i0 = (t % row_tiles) * kTransposeTile;
            const size_t i_end = std::min(i0 + kTransposeTile, rows);

            const float* ENGINE_RESTRICT s = src + b * rows * cols;
            float* ENGINE_RESTRICT d = dst + b * rows * cols;

            for (size_t j0 = 0; j0 < cols; j0 += kTransposeTile) {
                const size_t j_end = std::min(j0 + kTransposeTile, cols);
                for (size_t j = j0; j < j_end; ++j) {
                    float* ENGINE_RESTRICT drow = d + j * rows;
                    for (size_t i = i0; i < i_end; ++i) {
                        drow[i] = s[i * cols + j];
                    }
                }
            }
        }
    });
}

inline bool track(bool requires_grad) {
    return requires_grad && autograd::grad_enabled();
}

inline size_t product(const std::vector<size_t>& dims, size_t from = 0, size_t to = 0) {
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

inline BroadcastPlan plan_broadcast(const std::vector<size_t>& base,
                                    const std::vector<size_t>& other) {
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
    // The broadcast loop walks `repeat` blocks of `inner` elements and expects
    // that to be every element of the base, exactly. A plan where it is not sends
    // the loop off the end of the buffer or silently skips a tail, and neither
    // shows up as anything but wrong numbers.
    assert(plan.inner * plan.repeat == product(base) &&
           "a broadcast plan has to cover the base exactly");
    return plan;
}

// Materialises the broadcast operand at the base's shape. Only used in the
// derivatives, where having both shapes equal simplifies the formulas.
inline Tensor expand_operand(const Tensor& other, const std::vector<size_t>& base_shape,
                             size_t total, size_t inner) {
    // The accessors are hoisted out of the loop. data() returns a pointer, but
    // getting one is not free: it is the door to the host side of Storage, and
    // calling it per element checks the device mirror's validity once per value
    // copied.
    Tensor full(base_shape, 0.0f, false);
    const float* ENGINE_RESTRICT src = other.data();
    float* ENGINE_RESTRICT dst = full.data();
    for (size_t i = 0; i < total; ++i) dst[i] = src[i % inner];
    return full;
}

// Sums the leading axes of `full` down to the shape `target`. It is the adjoint
// of broadcasting an operand over a batch, used both by the broadcast addition
// and by matmul with a shared operand.
inline Tensor fold_leading(const Tensor& full, const std::vector<size_t>& target) {
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
    if (!cuda::ops::sum_axis(full.storage(), folded.storage(), 1, repeat, inner)) {
        const float* ENGINE_RESTRICT src = full.data();
        float* ENGINE_RESTRICT dst = folded.data();
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
inline Tensor clone_with_shape(const Storage& src, const std::vector<size_t>& shape,
                               bool req_grad) {
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
inline Tensor view_with_shape(const Storage& src, const std::vector<size_t>& shape, bool req_grad) {
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
inline bool offer_to_device(cuda::ops::Binary op, const Tensor& a, const Tensor& b, Tensor& out,
                            bool broadcast, const BroadcastPlan& plan) {
    const size_t inner = broadcast ? plan.inner : a.size();
    const size_t repeat = broadcast ? plan.repeat : 1;
    return cuda::ops::binary(op, a.storage(), b.storage(), out.storage(), inner, repeat);
}
}  // namespace tensor_detail
}  // namespace engine

#endif  // ENGINE_SRC_TENSOR_DETAIL_HPP
