#include "test_support.hpp"

#include <cstdlib>
#include <string>

#include "engine/parallel.hpp"

using namespace testing;

namespace {

void test_tensor_basics() {
    section("Tensor: shape, strides and indexing");

    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    check(A.shape() == std::vector<size_t>({2, 3}), "the shape is (2, 3)");
    check(A.strides() == std::vector<size_t>({3, 1}), "the row-major strides are (3, 1)");
    check(A.size() == 6, "the tensor has 6 elements");
    check_close(A({1, 2}), 6.0f, "A[1, 2] == 6");

    check_throws([&] { (void)A({2, 0}); }, "indexing out of range throws");
    check_throws([&] { (void)Tensor({2, 3}, {1.0f, 2.0f}); }, "data of the wrong size throws");
    check_throws([&] { (void)A.reshape({4, 2}); }, "an incompatible reshape throws");

    Tensor R = A.reshape({3, 2});
    check(R.shape() == std::vector<size_t>({3, 2}), "reshape to (3, 2) preserves the data");
    check_close(R({2, 1}), 6.0f, "reshape keeps the memory order");

    // reshape is a **view**: same buffer, different label on the axes. This is
    // PyTorch's semantics and it is a deliberate choice, not a side effect, so
    // it gets pinned here rather than left to be discovered.
    //
    // It is also what makes the operation free. When this copied the buffer it
    // was the most expensive line in the engine that computed nothing -- three
    // copies per Conv2d forward, two per attention projection, each of them
    // megabytes, each with a mirror in the backward.
    //
    // Nothing in src/ writes through a reshape result today. If something
    // starts to, this check is what turns that into a decision instead of a
    // surprise.
    R.data()[0] = 99.0f;
    check_close(A({0, 0}), 99.0f, "a write through a reshaped tensor is visible in the original");
    A.data()[5] = -7.0f;
    check_close(R({2, 1}), -7.0f, "and the other way round: they are one buffer");
}

void test_matmul() {
    section("Tensor: matrix multiplication");

    Tensor M1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor M2({3, 2}, {7, 8, 9, 1, 2, 3});
    Tensor R = M1.matmul(M2);

    check(R.shape() == std::vector<size_t>({2, 2}), "(2,3) x (3,2) gives (2,2)");
    check_close(R({0, 0}), 31.0f, "R[0,0] == 31");
    check_close(R({0, 1}), 19.0f, "R[0,1] == 19");
    check_close(R({1, 0}), 85.0f, "R[1,0] == 85");
    check_close(R({1, 1}), 55.0f, "R[1,1] == 55");

    check_throws([&] { (void)M1.matmul(M1); }, "incompatible inner dimensions throw");

    Tensor T = M1.transpose();
    check(T.shape() == std::vector<size_t>({3, 2}), "transpose flips the shape");
    check_close(T({2, 1}), 6.0f, "transpose swaps the indices");
}

void test_broadcast_add() {
    section("Tensor: row-vector broadcasting in addition");

    Tensor X({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({1, 3}, {10, 20, 30}, true);
    Tensor R = X + b;

    check_close(R({0, 0}), 11.0f, "the first row receives the bias");
    check_close(R({1, 2}), 36.0f, "the second row receives the same bias");

    Tensor loss = R.sum();
    loss.backward();
    check_close(b.grad().data()[0], 2.0f, "the bias gradient sums by column (2 rows)");

    Tensor bad({1, 4}, 0.0f);
    check_throws([&] { (void)(X + bad); }, "a broadcast with a different width throws");

    // Regression: with more axes than the base but leading ones, the repetition count
    // was computed with a product over an empty range that was mistaken for the full
    // product, and the backward read past the buffer.
    Tensor same_rank({3, 4}, 1.0f, false);
    Tensor leading_one({1, 3, 4}, 2.0f, true);
    Tensor bc = same_rank + leading_one;
    check(bc.shape() == std::vector<size_t>({3, 4}),
          "broadcasting (1,3,4) over (3,4) preserves the shape");
    check_close(bc.data()[0], 3.0f, "broadcasting with a leading axis of size 1 adds correctly");
    bc.sum().backward();
    check_close(leading_one.grad().data()[0], 1.0f,
                "its gradient is 1, not the sum of repetitions that do not exist");

    // A scalar also broadcasts over any shape
    Tensor scalar({1}, std::vector<float>{5.0f}, true);
    Tensor plus_scalar = X + scalar;
    check_close(plus_scalar.data()[0], 6.0f, "a one-element tensor broadcasts as a scalar");
    plus_scalar.sum().backward();
    check_close(scalar.grad().data()[0], 6.0f,
                "the scalar accumulates the gradient of all 6 elements");
}

void test_nd_tensor_ops() {
    section("Tensor: N-dimensional operations");

    // transpose swaps the last two axes at any rank
    Tensor T3({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor Tt = T3.transpose();
    check(Tt.shape() == std::vector<size_t>({2, 3, 2}), "3D transpose swaps the last two axes");
    check_close(Tt.data()[0], 1.0f, "3D transpose: first element of the first batch");
    check_close(Tt.data()[1], 4.0f, "3D transpose: transposes each matrix in the batch");
    check_close(Tt.data()[6], 7.0f, "3D transpose: the second batch is independent");

    // permute
    Tensor P({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < P.size(); ++i) P.data()[i] = static_cast<float>(i);
    Tensor Pp = P.permute({2, 0, 1});
    check(Pp.shape() == std::vector<size_t>({4, 2, 3}), "permute reorders the shape");
    // Element (i, j, k) of P must sit at (k, i, j) of Pp
    check_close(Pp.data()[(2 * 2 + 1) * 3 + 0], P.data()[(1 * 3 + 0) * 4 + 2],
                "permute relocates the elements correctly");
    check(P.permute({0, 1, 2}).shape() == P.shape(), "the identity permutation changes nothing");
    check_throws([&] { (void)P.permute({0, 1}); }, "permute with too few axes throws");
    check_throws([&] { (void)P.permute({0, 1, 1}); }, "permute with a repeated axis throws");
    check_throws([&] { (void)P.permute({0, 1, 5}); }, "permute with a nonexistent axis throws");

    // Swapping the last two axes takes a blocked path in permute() and in
    // transpose(); everything else takes the general per-element gather. Two code
    // paths computing the same function need a test that they agree, and it has to
    // run on shapes the 32x32 tile does not divide -- a blocked loop that is right
    // on multiples of its tile and wrong on the remainder is the classic way to
    // get this wrong, and every other shape in this file is a single partial tile.
    //
    // The oracle is the definition written out, out[b][j][i] == in[b][i][j], and
    // not a composition of two other permutes. The first draft of this test used
    // one, got the composition wrong, and reported four failures against correct
    // code -- an oracle that needs its own proof is not an oracle.
    // A sweep rather than four hand-picked shapes. 31, 33, 63, 64, 65 and 1
    // straddle the tile in every direction: one under, exact, one over, and the
    // degenerate single row or column where the tile is larger than the whole
    // dimension.
    size_t mismatches = 0;
    std::string first_bad;
    std::vector<std::vector<size_t>> shapes;
    for (size_t batch : {1u, 2u}) {
        for (size_t rows : {1u, 31u, 32u, 33u, 64u, 65u}) {
            for (size_t cols : {1u, 31u, 32u, 33u, 64u, 65u}) {
                shapes.push_back({batch, rows, cols});
            }
        }
    }
    for (const std::vector<size_t>& dims : shapes) {
        Tensor A(dims, 0.0f);
        for (size_t i = 0; i < A.size(); ++i) A.data()[i] = static_cast<float>(i % 251);

        const Tensor swapped = A.permute({0, 2, 1});
        const Tensor via_transpose = A.transpose();
        const size_t rows = dims[1], cols = dims[2];

        bool same = swapped.shape() == std::vector<size_t>{dims[0], cols, rows} &&
                    via_transpose.shape() == swapped.shape();
        for (size_t b = 0; same && b < dims[0]; ++b) {
            for (size_t i = 0; same && i < rows; ++i) {
                for (size_t j = 0; same && j < cols; ++j) {
                    const float want = A.data()[(b * rows + i) * cols + j];
                    same = swapped.data()[(b * cols + j) * rows + i] == want &&
                           via_transpose.data()[(b * cols + j) * rows + i] == want;
                }
            }
        }
        if (!same && first_bad.empty()) first_bad = A.shape_str();
        if (!same) ++mismatches;
    }
    check(shapes.size() > 60, "the sweep straddles the 32-wide tile in both directions");
    check(mismatches == 0, mismatches == 0
                               ? "the blocked axis swap is exact on every shape in it"
                               : "the blocked axis swap is wrong, first at " + first_bad);

    // batched matmul: each matrix in the batch is multiplied separately
    Tensor A({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor B({2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 1, 1});
    Tensor C = A.matmul(B);
    check(C.shape() == std::vector<size_t>({2, 2, 2}), "batched matmul gives (B, M, N)");
    check_close(C.data()[0], 4.0f, "batch 0: 1*1 + 2*0 + 3*1 == 4");
    check_close(C.data()[1], 5.0f, "batch 0: 1*0 + 2*1 + 3*1 == 5");
    check_close(C.data()[4], 23.0f, "batch 1: 7*2 + 8*0 + 9*1 == 23");

    // A 2D operand is shared with the whole batch
    Tensor shared({3, 2}, {1, 0, 0, 1, 1, 1});
    Tensor Cs = A.matmul(shared);
    check(Cs.shape() == std::vector<size_t>({2, 2, 2}),
          "matmul with a shared 2D matrix gives (B, M, N)");
    check_close(Cs.data()[0], 4.0f, "the shared matrix applies to the first batch");
    check_close(Cs.data()[4], 16.0f, "and the same matrix to the second (7 + 9)");

    check_throws([&] { (void)Tensor({2, 2, 3}, 1.0f).matmul(Tensor({3, 3, 2}, 1.0f)); },
                 "matmul with mismatched batches throws");
    check_throws([&] { (void)Tensor({4}, 1.0f).matmul(Tensor({4}, 1.0f)); },
                 "matmul of 1D vectors throws");

    // softmax over the last axis of a 3D tensor
    Tensor S3 = Tensor::randn({2, 3, 4}).softmax();
    check(S3.shape() == std::vector<size_t>({2, 3, 4}), "3D softmax preserves the shape");
    bool all_rows_sum_one = true;
    for (size_t r = 0; r < 6; ++r) {
        float total = 0.0f;
        for (size_t j = 0; j < 4; ++j) total += S3.data()[r * 4 + j];
        if (std::fabs(total - 1.0f) > 1e-4f) all_rows_sum_one = false;
    }
    check(all_rows_sum_one, "each vector along the last axis sums to 1 after softmax");

    // Suffix broadcasting
    Tensor base({2, 3, 4}, 1.0f);
    Tensor row({3, 4}, 2.0f, true);
    Tensor sum_bc = base + row;
    check(sum_bc.shape() == std::vector<size_t>({2, 3, 4}),
          "broadcasting (3,4) over (2,3,4) works");
    check_close(sum_bc.data()[0], 3.0f, "broadcasting adds the repeated block");
    sum_bc.sum().backward();
    check_close(row.grad().data()[0], 2.0f,
                "the broadcast operand's gradient sums both repetitions");

    check_throws([&] { (void)(base + Tensor({5, 4}, 1.0f)); },
                 "a broadcast with an incompatible suffix throws");

    // Numerical gradients of the new operations
    Tensor G({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.3f * static_cast<float>(i) - 3.1f;

    check_gradient("gradient of permute()", G, [](Tensor& t) {
        Tensor w = Tensor({4, 2, 3}, 0.0f);
        for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.1f * static_cast<float>(i) - 1.0f;
        return (t.permute({2, 0, 1}) * w).sum();
    });
    check_gradient("gradient of 3D transpose()", G, [](Tensor& t) { return t.transpose().sum(); });
    // The weighting tensor is built OUTSIDE the closure: generated inside, each
    // numerical evaluation would use different weights and the check would not be
    // comparing the same function with itself.
    Tensor w_soft = Tensor::randn({2, 3, 4});
    check_gradient("gradient of 3D softmax()", G,
                   [&](Tensor& t) { return (t.softmax() * w_soft).sum(); });
    check_gradient("gradient of batched matmul", G, [](Tensor& t) {
        Tensor B2({2, 4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f, 2, 1, -1, 0.5f, 3, -2, 1, 1});
        return t.matmul(B2).sum();
    });
    check_gradient("gradient of matmul with a shared matrix", G, [](Tensor& t) {
        Tensor shared2({4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f});
        return t.matmul(shared2).sum();
    });
    {
        // The shared matrix receives the sum of the batch's contributions
        Tensor shared3({4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f});
        Tensor batched = Tensor::randn({3, 2, 4});
        check_gradient("gradient of the shared matrix in matmul", shared3,
                       [&](Tensor& t) { return batched.matmul(t).sum(); });
    }

    // Linear over inputs with more than 2 axes
    engine::manual_seed(19);
    nn::Linear proj(4, 5);
    Tensor seq_input = Tensor::randn({2, 3, 4});
    check(proj(seq_input).shape() == std::vector<size_t>({2, 3, 5}),
          "Linear applies the projection to the last axis of a 3D tensor");
    Tensor w_proj = Tensor::randn({2, 3, 5});
    check_gradient("gradient of Linear over a 3D input", seq_input,
                   [&](Tensor& t) { return (proj(t) * w_proj).sum(); });
}

void test_reductions() {
    section("Tensor: reductions along an axis");

    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});

    Tensor s0 = A.sum(0);
    check(s0.shape() == std::vector<size_t>({3}), "sum(0) drops the first axis");
    check_close(s0.data()[0], 5.0f, "sum(0) sums by column: 1 + 4");
    check_close(s0.data()[2], 9.0f, "sum(0) third column: 3 + 6");

    Tensor s1 = A.sum(1);
    check(s1.shape() == std::vector<size_t>({2}), "sum(1) drops the second axis");
    check_close(s1.data()[0], 6.0f, "sum(1) sums by row: 1+2+3");

    Tensor sk = A.sum(1, true);
    check(sk.shape() == std::vector<size_t>({2, 1}), "keepdim leaves the reduced axis at 1");

    Tensor m = A.mean(0);
    check_close(m.data()[0], 2.5f, "mean(0) averages by column");

    Tensor mx = A.max(0);
    check(mx.shape() == std::vector<size_t>({3}), "max(0) drops the first axis");
    check_close(mx.data()[1], 5.0f, "max(0) takes the largest in each column");

    // Reducing a 1D tensor leaves a scalar
    Tensor v({4}, {1, 7, 3, 2});
    check(v.sum(0).shape() == std::vector<size_t>({1}), "reducing a 1D tensor gives a {1} scalar");
    check_close(v.max(0).data()[0], 7.0f, "max of a vector");

    check_throws([&] { (void)A.sum(5); }, "reducing a nonexistent axis throws");
    check_throws([&] { (void)A.max(2); }, "max over a nonexistent axis throws");

    // Gradientes
    Tensor G({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.37f * static_cast<float>(i) - 4.1f;
    Tensor w_sum = Tensor::randn({2, 4});
    check_gradient("gradient of sum(axis)", G, [&](Tensor& t) { return (t.sum(1) * w_sum).sum(); });
    check_gradient("gradient of mean(axis)", G,
                   [&](Tensor& t) { return (t.mean(1) * w_sum).sum(); });
    // Well-separated values: the maximum is not differentiable at a tie
    Tensor Gm({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < Gm.size(); ++i) Gm.data()[i] = static_cast<float>((i * 37) % 24) * 0.5f;
    check_gradient("gradient of max(axis)", Gm,
                   [&](Tensor& t) { return (t.max(1) * w_sum).sum(); });

    // The maximum's gradient goes only to the winner
    Tensor mg({1, 3}, {1.0f, 9.0f, 2.0f}, true);
    mg.max(1).sum().backward();
    check_close(mg.grad().data()[1], 1.0f, "the maximum receives the whole gradient");
    check_close(mg.grad().data()[0], 0.0f, "the rest receive nothing");
}

void test_slice_concat_stack() {
    section("Tensor: slice, concat and stack");

    Tensor A({3, 4}, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A.data()[i] = static_cast<float>(i);

    Tensor r = A.slice(0, 1, 2);
    check(r.shape() == std::vector<size_t>({2, 4}), "slice along the first axis");
    check_close(r(0, 0), 4.0f, "slice starts at the requested row");

    Tensor c = A.slice(1, 1, 2);
    check(c.shape() == std::vector<size_t>({3, 2}), "slice along the second axis");
    check_close(c(0, 0), 1.0f, "a column slice takes the right column");
    check_close(c(1, 1), 6.0f, "a column slice respects the rows");

    check_throws([&] { (void)A.slice(0, 2, 5); }, "a slice that runs off the end throws");
    check_throws([&] { (void)A.slice(0, 0, 0); }, "an empty slice throws");
    check_throws([&] { (void)A.slice(7, 0, 1); }, "a slice along a nonexistent axis throws");

    // concat
    Tensor P({2, 2}, {1, 2, 3, 4});
    Tensor Q({1, 2}, {5, 6});
    Tensor cc = Tensor::concat({P, Q}, 0);
    check(cc.shape() == std::vector<size_t>({3, 2}), "concat along the first axis adds the rows");
    check_close(cc(2, 0), 5.0f, "concat places the second part after");

    Tensor R({2, 3}, {7, 8, 9, 10, 11, 12});
    Tensor cc2 = Tensor::concat({P, R}, 1);
    check(cc2.shape() == std::vector<size_t>({2, 5}),
          "concat along the second axis adds the columns");
    check_close(cc2(0, 2), 7.0f, "column-wise concat interleaves correctly");
    check_close(cc2(1, 0), 3.0f, "column-wise concat preserves the rows");

    check_throws([&] { (void)Tensor::concat({}, 0); }, "concat with no parts throws");
    check_throws([&] { (void)Tensor::concat({P, Tensor({3, 3}, 1.0f)}, 0); },
                 "concat with incompatible dimensions throws");

    // stack
    Tensor st = Tensor::stack({P, P}, 0);
    check(st.shape() == std::vector<size_t>({2, 2, 2}), "stack creates a new axis");
    Tensor st1 = Tensor::stack({P, P}, 1);
    check(st1.shape() == std::vector<size_t>({2, 2, 2}), "stack accepts a middle axis");
    check_throws([&] { (void)Tensor::stack({P, Q}, 0); }, "stack with mismatched shapes throws");

    // Gradientes
    Tensor G({3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.31f * static_cast<float>(i) - 2.0f;
    Tensor w_sl = Tensor::randn({2, 4});
    check_gradient("gradient of slice()", G,
                   [&](Tensor& t) { return (t.slice(0, 1, 2) * w_sl).sum(); });

    Tensor other({2, 4}, 1.5f);
    Tensor w_cc = Tensor::randn({5, 4});
    check_gradient("gradient of concat() (first part)", G,
                   [&](Tensor& t) { return (Tensor::concat({t, other}, 0) * w_cc).sum(); });
    check_gradient("gradient of concat() (second part)", other,
                   [&](Tensor& t) { return (Tensor::concat({G, t}, 0) * w_cc).sum(); });

    // Concatenating a tensor with itself accumulates into both slices
    Tensor twice({1, 2}, {1.0f, 2.0f}, true);
    Tensor::concat({twice, twice}, 0).sum().backward();
    check_close(twice.grad().data()[0], 2.0f, "concatenating a tensor with itself accumulates");
}

void test_broadcast_all_operators() {
    section("Tensor: broadcasting in all four operators");

    Tensor X({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor row({3}, {1.0f, 2.0f, 4.0f}, true);

    check_close((X - row).data()[0], 0.0f, "subtraction broadcasts: 1 - 1");
    check_close((X - row).data()[4], 3.0f, "subtraction broadcasts on the second row: 5 - 2");
    check_close((X * row).data()[2], 12.0f, "multiplication broadcasts: 3 * 4");
    check_close((X / row).data()[1], 1.0f, "division broadcasts: 2 / 2");

    // A one-element tensor acts as a scalar
    Tensor k({1}, std::vector<float>{10.0f}, true);
    check_close((X * k).data()[3], 40.0f, "a {1} tensor broadcasts as a scalar");

    check_throws([&] { (void)(X - Tensor({4}, 1.0f)); },
                 "a subtraction with an incompatible suffix throws");
    check_throws([&] { (void)(X * Tensor({5, 3}, 1.0f)); }, "an incompatible product throws");

    // Gradients of the broadcast operand
    Tensor base({2, 3}, {1, 2, 3, 4, 5, 6}, false);
    Tensor b1({3}, {1.0f, 2.0f, 4.0f}, true);
    (base * b1).sum().backward();
    check_close(b1.grad().data()[0], 5.0f,
                "broadcast multiplication: the gradient sums the column (1+4)");

    Tensor b2({3}, {1.0f, 2.0f, 4.0f}, true);
    (base - b2).sum().backward();
    check_close(b2.grad().data()[0], -2.0f, "broadcast subtraction: the gradient is -1 per row");

    Tensor G({2, 3}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.5f * static_cast<float>(i) + 1.0f;
    Tensor d({3}, {2.0f, 3.0f, 4.0f});
    Tensor w = Tensor::randn({2, 3});
    check_gradient("gradient of broadcast subtraction", G,
                   [&](Tensor& t) { return ((t - d) * w).sum(); });
    check_gradient("gradient of broadcast multiplication", G,
                   [&](Tensor& t) { return ((t * d) * w).sum(); });
    check_gradient("gradient of broadcast division", G,
                   [&](Tensor& t) { return ((t / d) * w).sum(); });
    check_gradient("gradient of the broadcast divisor", d,
                   [&](Tensor& t) { return ((G / t) * w).sum(); });
}

void test_parallelism() {
    section("parallel: work splitting");

    namespace par = engine::parallel;
    const size_t original = par::num_threads();

    check(par::num_threads() >= 1, "the pool starts with at least one thread");
    check(!par::inside_parallel_region(), "the main thread is not inside a region");

    // Coverage: each index is visited exactly once
    for (size_t threads : {size_t(1), size_t(2), size_t(4)}) {
        par::set_num_threads(threads);
        const size_t n = 100000;
        std::vector<int> visits(n, 0);
        par::parallel_for(n, 1000, [&](size_t from, size_t to) {
            for (size_t i = from; i < to; ++i) visits[i]++;
        });
        bool exactly_once = true;
        for (int v : visits)
            if (v != 1) exactly_once = false;
        check(exactly_once,
              "with " + std::to_string(threads) + " threads each index is visited exactly once");
    }

    // Determinism: splitting by rows does not change the accumulation order, so the
    // result must be identical BIT FOR BIT, not merely close.
    engine::manual_seed(7);
    Tensor A = Tensor::randn({200, 150});
    Tensor B = Tensor::randn({150, 120});
    Tensor E1 = Tensor::randn({400, 400});
    Tensor E2 = Tensor::randn({400, 400});

    par::set_num_threads(1);
    Tensor mm_serial = A.matmul(B);
    Tensor add_serial = E1 + E2;

    par::set_num_threads(4);
    Tensor mm_par = A.matmul(B);
    Tensor add_par = E1 + E2;

    bool identical = true;
    for (size_t i = 0; i < mm_serial.size(); ++i) {
        if (mm_serial.data()[i] != mm_par.data()[i]) identical = false;
    }
    check(identical, "matmul is identical bit for bit with 1 and with 4 threads");

    identical = true;
    for (size_t i = 0; i < add_serial.size(); ++i) {
        if (add_serial.data()[i] != add_par.data()[i]) identical = false;
    }
    check(identical, "addition is identical bit for bit with 1 and with 4 threads");

    // Nested regions run inline: multiplying threads inside a thread only adds
    // contention
    par::set_num_threads(4);
    bool nested_inline = true;
    par::parallel_for(10000, 100, [&](size_t, size_t) {
        if (!par::inside_parallel_region()) nested_inline = false;
        size_t calls = 0;
        par::parallel_for(10000, 100, [&](size_t, size_t) { ++calls; });
        if (calls != 1) nested_inline = false;  // un solo trozo = ejecutado en linea
    });
    check(nested_inline, "a nested region runs inline");

    // An exception in a worker reaches the thread that split the work
    check_throws(
        [&] {
            (void)par::parallel_for(100000, 1000, [](size_t from, size_t) {
                if (from > 0) throw std::runtime_error("failure in a worker");
            });
        },
        "an exception in a worker propagates to the splitter");

    // Un rango vacio no hace nada
    size_t calls = 0;
    par::parallel_for(0, 10, [&](size_t, size_t) { ++calls; });
    check(calls == 0, "an empty range does not run the body");

    par::set_num_threads(1);
    check(par::num_threads() == 1, "set_num_threads(1) leaves only the calling thread");
    calls = 0;
    par::parallel_for(1000000, 1, [&](size_t, size_t) { ++calls; });
    check(calls == 1, "with one thread the body runs exactly once, inline");

    par::set_num_threads(original);
}

// The performance guard, and it deliberately measures nothing.
//
// The host buffer pool is worth 1.41x on MNIST, and nothing in this repository
// stopped that from being lost again. A wall-clock threshold in CI is the wrong
// answer -- tools/check_perf.py's header explains at length why a shared runner
// cannot hold one, and it is right. But the pool's guarantee is not a time. It is
// that a loop which has seen a shape once never allocates for it again, and that
// is exact on any machine at any speed.
//
// If an operation starts allocating per call again, `fresh` grows with the
// iteration count and this fails on all four compilers in milliseconds.
void test_buffer_pool_recycles() {
    section("Tensor: the buffer pool stops allocating once it has seen a shape");

    // Turning the pool off is a supported configuration, so the guard skips
    // itself rather than failing -- the same way the CUDA parity cases skip when
    // there is no card. Verified by running the suite with it set to 0 before
    // this branch existed: both checks below failed, which is what makes them a
    // guard rather than decoration.
    const char* pool_mb = std::getenv("ENGINE_BUFFER_POOL_MB");
    if (pool_mb != nullptr && std::string(pool_mb) == "0") {
        check(true, "the host buffer pool is disabled; the recycling guard skips itself");
        return;
    }

    const std::vector<size_t> shape = {64, 512};

    // Warm-up: the first pass is where the shapes are legitimately new, and how
    // many buffers that takes is an implementation detail nobody should pin.
    for (int i = 0; i < 4; ++i) {
        Tensor a(shape, 1.5f, false);
        Tensor b(shape, 2.5f, false);
        Tensor c = (a + b).relu();
        (void)c.data()[0];
    }

    const engine::BufferPoolStats before = engine::buffer_pool_stats();
    constexpr int kRounds = 40;
    for (int i = 0; i < kRounds; ++i) {
        Tensor a(shape, 1.5f, false);
        Tensor b(shape, 2.5f, false);
        Tensor c = (a + b).relu();
        (void)c.data()[0];
    }
    const engine::BufferPoolStats after = engine::buffer_pool_stats();

    const size_t fresh = after.fresh - before.fresh;
    const size_t recycled = after.recycled - before.recycled;

    check(recycled >= static_cast<size_t>(kRounds),
          "forty rounds of the same shapes are served from the free list");
    // Not zero: `a` and `b` are constructed and destroyed inside the loop body,
    // so the free list can be momentarily empty when the next round asks. What
    // must not happen is fresh growing *with* the round count, which is what an
    // allocation per operation looks like.
    check(fresh < static_cast<size_t>(kRounds),
          "and the allocator is asked for fewer buffers than there are rounds");
}

}  // namespace

void run_tensor_tests() {
    test_tensor_basics();
    test_matmul();
    test_broadcast_add();
    test_nd_tensor_ops();
    test_reductions();
    test_slice_concat_stack();
    test_broadcast_all_operators();
    test_parallelism();
    test_buffer_pool_recycles();
}
