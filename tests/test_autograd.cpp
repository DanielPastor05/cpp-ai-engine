#include "test_support.hpp"

using namespace testing;

namespace {

void test_autograd_scalar() {
    section("Autograd: derivadas analiticas conocidas");

    // L = a*b + relu(a), with a=2, b=3 -> dL/da = b+1 = 4, dL/db = a = 2
    Tensor a({1}, std::vector<float>{2.0f}, true);
    Tensor b({1}, std::vector<float>{3.0f}, true);
    Tensor L = (a * b) + a.relu();
    L.backward();

    check_close(L.data()[0], 8.0f, "L = a*b + relu(a) = 8");
    check_close(a.grad().data()[0], 4.0f, "dL/da == b + 1 == 4");
    check_close(b.grad().data()[0], 2.0f, "dL/db == a == 2");

    // A reused node must accumulate gradient through both branches: L = x + x
    Tensor x({1}, std::vector<float>{5.0f}, true);
    Tensor y = (x + x).sum();
    y.backward();
    check_close(x.grad().data()[0], 2.0f,
                "a shared node accumulates the gradient of both branches");

    // Calling backward twice accumulates (as in PyTorch)
    Tensor z({1}, std::vector<float>{3.0f}, true);
    Tensor w = (z * 2.0f).sum();
    w.backward();
    Tensor w2 = (z * 2.0f).sum();
    w2.backward();
    check_close(z.grad().data()[0], 4.0f, "backward accumulates gradients across calls");

    z.zero_grad();
    check_close(z.grad().data()[0], 0.0f, "zero_grad clears the gradient");
}

void test_autograd_numeric() {
    section("Autograd: verificacion numerica de gradients");

    // No value may land exactly on 0: ReLU is not differentiable there and the
    // centred difference would give 0.5 against the analytic 0.
    Tensor A({3, 4}, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A.data()[i] = 0.5f * static_cast<float>(i) - 2.25f;

    check_gradient("gradient of sum()", A, [](Tensor& t) { return t.sum(); });
    check_gradient("gradient of mean()", A, [](Tensor& t) { return t.mean(); });
    check_gradient("gradient of relu()", A, [](Tensor& t) { return t.relu().sum(); });
    check_gradient("gradient of transpose()", A, [](Tensor& t) { return t.transpose().sum(); });
    check_gradient("gradient of reshape()", A, [](Tensor& t) { return t.reshape({2, 6}).sum(); });
    check_gradient("gradient of x*x (Hadamard)", A, [](Tensor& t) { return (t * t).sum(); });
    check_gradient("gradient of softmax()", A, [](Tensor& t) {
        Tensor w({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3});
        return (t.softmax() * w).sum();
    });
    check_gradient("gradient of matmul()", A, [](Tensor& t) {
        Tensor B({4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f});
        return t.matmul(B).sum();
    });
    check_gradient("gradient of division", A, [](Tensor& t) {
        Tensor d({3, 4}, 3.0f);
        return (t / d).sum();
    });
    check_gradient("gradient of a deep composition", A, [](Tensor& t) {
        Tensor B({4, 3}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f, 1, 2, -3, 0.75f});
        Tensor h = t.matmul(B).relu();
        return (h * h).mean();
    });

    Tensor logits({4, 3},
                  {0.4f, -1.2f, 2.0f, 1.1f, 0.3f, -0.7f, -2.0f, 0.9f, 0.2f, 0.6f, 0.6f, -1.5f});
    std::vector<size_t> targets = {2, 0, 1, 1};
    check_gradient("gradient of cross_entropy_loss()", logits,
                   [&](Tensor& t) { return nn::cross_entropy_loss(t, targets); });
}

void test_repeated_backward() {
    section("Autograd: repeated backward over the same graph");

    // Regression: intermediate nodes kept the previous call's gradient, so the second
    // traversal propagated the sum of both and multiplied the leaves' gradients (it
    // gave 18 instead of 6).
    Tensor x({1}, std::vector<float>{1.0f}, true);
    Tensor h = x * 2.0f;
    Tensor L = (h * 3.0f).sum();

    L.backward();
    check_close(x.grad().data()[0], 6.0f, "primer backward: dL/dx == 6");

    x.zero_grad();
    L.backward();
    check_close(x.grad().data()[0], 6.0f, "a second backward over the same graph: dL/dx == 6");

    // Without clearing the leaf, the result must be exactly double: only leaves
    // accumulate.
    L.backward();
    check_close(x.grad().data()[0], 12.0f, "without zero_grad the leaves accumulate (6 + 6)");
}

void test_row_indexing_and_batches() {
    section("Tensor: indexacion (fila, col) y mini-lotes");

    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    check_close(A(1, 2), 6.0f, "A(1, 2) accesses without a heap allocation");
    check_close(A(0, 0), 1.0f, "A(0, 0) is the first element");
    A(0, 1) = 99.0f;
    check_close(A({0, 1}), 99.0f, "A(fila, col) permite escritura");
    check_throws([&] { (void)A(2, 0); }, "A(fila, col) out of range throws");
    check_throws([&] { (void)Tensor({4}, 1.0f)(0, 0); }, "A(fila, col) on a 1D tensor throws");

    // Scalar on the left
    Tensor t({2}, {1.0f, 2.0f});
    check_close((2.0f * t).data()[1], 4.0f, "2.0f * t multiplies from the left");
    check_close((1.0f + t).data()[0], 2.0f, "1.0f + t adds from the left");
    check_close((10.0f - t).data()[1], 8.0f, "10.0f - t subtracts from the left");

    // select_rows
    Tensor X({4, 2}, {0, 0, 1, 1, 2, 2, 3, 3}, true);
    Tensor batch = X.select_rows({3, 1});
    check(batch.shape() == std::vector<size_t>({2, 2}), "select_rows da (n_indices, cols)");
    check_close(batch(0, 0), 3.0f, "select_rows respects the requested order");
    check_close(batch(1, 0), 1.0f, "select_rows takes the right row");

    // With repeated indices the gradient must accumulate into the source row
    X.zero_grad();
    Tensor loss = X.select_rows({0, 0, 2}).sum();
    loss.backward();
    check_close(X.grad()(0, 0), 2.0f, "a repeated index accumulates gradient");
    check_close(X.grad()(1, 0), 0.0f, "an unselected row receives no gradient");
    check_close(X.grad()(2, 0), 1.0f, "a row selected once receives gradient 1");

    check_throws([&] { (void)X.select_rows({}); }, "select_rows with no indices throws");
    check_throws([&] { (void)X.select_rows({9}); }, "select_rows with an invalid index throws");

    Tensor G({3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.5f * static_cast<float>(i) - 2.25f;
    check_gradient("gradient of select_rows()", G,
                   [](Tensor& t) { return t.select_rows({2, 0, 2}).sum(); });
}

void test_no_grad_and_errors() {
    section("Autograd: NoGradGuard y errores");

    Tensor a({2, 2}, 1.0f, true);
    {
        engine::autograd::NoGradGuard no_grad;
        Tensor b = (a * 2.0f).sum();
        check(!b.requires_grad(), "NoGradGuard prevents recording the graph");
        check(b.get_impl()->parents.empty(), "the result records no parents under NoGradGuard");
    }
    check(engine::autograd::grad_enabled(),
          "autograd mode is restored when the guard goes out of scope");

    Tensor c({2, 2}, 1.0f, true);
    Tensor d = c * 2.0f;
    check_throws([&] { (void)d.backward(); }, "an implicit backward on a non-scalar throws");

    d.backward(Tensor({2, 2}, 1.0f));
    check_close(c.grad().data()[0], 2.0f, "backward(grad_output) explicito si funciona");

    Tensor e({2, 2}, 1.0f, true);
    check_throws([&] { (void)e.add_grad(Tensor({3, 3}, 1.0f)); },
                 "add_grad with an incompatible shape throws");
    check_throws([&] { (void)Tensor({2, 2}, 1.0f).grad(); },
                 "accessing a nonexistent gradient throws");
}

void test_graph_is_released() {
    section("Autograd: the graph is freed (no shared_ptr cycles)");

    // If backward_fn captured its own output tensor there would be a shared_ptr cycle
    // and the node would never be destroyed when it went out of scope.
    std::weak_ptr<engine::TensorImpl> weak_node;
    {
        Tensor a({4, 4}, 2.0f, true);
        Tensor intermediate = (a * a).relu();
        weak_node = intermediate.get_impl();
        Tensor loss = intermediate.sum();
        loss.backward();
        check(!weak_node.expired(), "the intermediate node is still alive inside the scope");
    }
    check(weak_node.expired(), "the intermediate node is freed when it goes out of scope");
}

}  // namespace

void run_autograd_tests() {
    test_autograd_scalar();
    test_autograd_numeric();
    test_repeated_backward();
    test_row_indexing_and_batches();
    test_no_grad_and_errors();
    test_graph_is_released();
}
