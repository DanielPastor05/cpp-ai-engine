#include "test_support.hpp"

#include "engine/parallel.hpp"

#include <tuple>

using namespace testing;

namespace {

void test_im2col() {
    section("conv: im2col y col2im");

    // A 3x3 window over a 4x4 image -> 2x2 positions, rows of 9 values
    Tensor img({1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, false);
    nn::Window2d w3(3, 3, 1, 0);

    check(w3.out_h(4) == 2 && w3.out_w(4) == 2, "(4 - 3)/1 + 1 == 2 positions per axis");
    check(nn::Window2d(3, 3, 1, 1).out_h(4) == 4, "padding 1 with kernel 3 preserves the size");
    check(nn::Window2d(2, 2, 2, 0).out_h(4) == 2, "a stride of 2 with kernel 2 halves the size");

    Tensor cols = nn::im2col(img, w3);
    check(cols.shape() == std::vector<size_t>({4, 9}), "im2col da (N*oH*oW, C*kH*kW)");
    // Primera ventana: esquina superior izquierda 3x3
    check_close(cols(0, 0), 1.0f, "the first window starts at pixel (0,0)");
    check_close(cols(0, 8), 11.0f, "the first window ends at pixel (2,2)");
    // Last window: shifted one pixel along both axes
    check_close(cols(3, 0), 6.0f, "the last window starts at pixel (1,1)");
    check_close(cols(3, 8), 16.0f, "the last window ends at pixel (3,3)");

    // With padding, the kernel's corners fall outside and are zero
    Tensor padded_cols = nn::im2col(img, nn::Window2d(3, 3, 1, 1));
    check(padded_cols.shape() == std::vector<size_t>({16, 9}), "with padding there are 16 windows");
    check_close(padded_cols(0, 0), 0.0f, "the padded area contributes zeros");
    check_close(padded_cols(0, 4), 1.0f, "the centre of the first padded window is pixel (0,0)");

    // col2im returns the original shape and accumulates the overlaps
    Tensor ones({4, 9}, 1.0f, false);
    Tensor scattered = nn::col2im(ones, {1, 1, 4, 4}, w3);
    check(scattered.shape() == std::vector<size_t>({1, 1, 4, 4}),
          "col2im restores the input shape");
    check_close(scattered.data()[0], 1.0f, "a corner belongs to a single window");
    check_close(scattered.data()[5], 4.0f, "pixel (1,1) belongs to all four windows");

    // The adjoint test: <im2col(x), y> == <x, col2im(y)>.
    // It is the exact statement that col2im is im2col's transpose, and therefore the
    // correct derivative of the convolution.
    engine::manual_seed(31);
    for (const nn::Window2d& win : {nn::Window2d(3, 3, 1, 0), nn::Window2d(3, 3, 1, 1),
                                    nn::Window2d(2, 2, 2, 0), nn::Window2d(2, 3, 1, 1)}) {
        Tensor x = Tensor::randn({2, 3, 5, 6});
        Tensor cols_x = nn::im2col(x, win);
        Tensor y = Tensor::randn(cols_x.shape());
        Tensor back = nn::col2im(y, x.shape(), win);

        float lhs = 0.0f;
        for (size_t i = 0; i < cols_x.size(); ++i) lhs += cols_x.data()[i] * y.data()[i];
        float rhs = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) rhs += x.data()[i] * back.data()[i];

        check_close(lhs, rhs, "col2im is the adjoint of im2col for " + win_name(win), 1e-2f);
    }

    check_throws([&] { nn::im2col(Tensor({4, 4}, 1.0f), w3); }, "im2col on a 2D tensor throws");
    check_throws([&] { nn::im2col(img, nn::Window2d(5, 5, 1, 0)); },
                 "a kernel larger than the image throws");
    check_throws([&] { nn::col2im(ones, {1, 1, 8, 8}, w3); },
                 "col2im with inconsistently sized columns throws");
}

void test_conv_layers() {
    section("conv: Conv2d, MaxPool2d y Flatten");

    engine::manual_seed(11);

    // Output shape
    nn::Conv2d conv(3, 5, nn::Window2d(3, 3, 1, 1));
    Tensor input = Tensor::randn({2, 3, 8, 8});
    Tensor out = conv(input);
    check(out.shape() == std::vector<size_t>({2, 5, 8, 8}),
          "Conv2d with padding 1 preserves the size");
    check(conv.weight().shape() == std::vector<size_t>({5, 3, 3, 3}),
          "the weights are (outC, inC, kH, kW)");
    check(conv.num_parameters() == 5 * 3 * 3 * 3 + 5, "Conv2d counts weights and biases");

    nn::Conv2d strided(3, 2, nn::Window2d(3, 3, 2, 0));
    check(strided(input).shape() == std::vector<size_t>({2, 2, 3, 3}),
          "Conv2d with stride 2 and no padding reduces 8 -> 3");

    check_throws([&] { conv(Tensor({2, 8, 8}, 1.0f)); }, "Conv2d with a 3D input throws");
    check_throws([&] { conv(Tensor::randn({2, 4, 8, 8})); },
                 "Conv2d with the wrong channel count throws");
    check_throws([&] { nn::Conv2d(0, 4, nn::Window2d(3)); },
                 "Conv2d with no input channels throws");

    // A 1x1 convolution with a known weight: the output is an exact rescaling
    nn::Conv2d unit(1, 1, nn::Window2d(1, 1, 1, 0));
    unit.weight() = Tensor({1, 1, 1, 1}, std::vector<float>{2.0f}, true);
    unit.bias() = Tensor({1}, std::vector<float>{0.5f}, true);
    Tensor pixels({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, false);
    Tensor scaled = unit(pixels);
    check_close(scaled.data()[0], 2.5f, "conv 1x1: 1*2 + 0.5 == 2.5");
    check_close(scaled.data()[3], 8.5f, "conv 1x1: 4*2 + 0.5 == 8.5");

    // A 3x3 kernel of ones with no padding: each output is the 3x3 block's sum
    nn::Conv2d summer(1, 1, nn::Window2d(3, 3, 1, 0), false);
    summer.weight() = Tensor({1, 1, 3, 3}, std::vector<float>(9, 1.0f), true);
    Tensor grid({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, false);
    check_close(summer(grid).data()[0], 45.0f, "a kernel of ones sums the whole block");

    // MaxPool2d
    nn::MaxPool2d pool(2, 2);
    Tensor pool_in({1, 1, 4, 4}, {1, 2, 9, 4, 5, 6, 7, 8, 13, 10, 11, 12, 3, 14, 15, 16}, false);
    Tensor pooled = pool(pool_in);
    check(pooled.shape() == std::vector<size_t>({1, 1, 2, 2}),
          "MaxPool 2x2 with stride 2 halves the size");
    check_close(pooled.data()[0], 6.0f, "maximum of the upper-left block");
    check_close(pooled.data()[1], 9.0f, "maximum of the upper-right block");
    check_close(pooled.data()[2], 14.0f, "maximum of the lower-left block");
    check_close(pooled.data()[3], 16.0f, "maximum of the lower-right block");
    check_throws([&] { pool(Tensor({4, 4}, 1.0f)); }, "MaxPool2d with a 2D input throws");
    // With padding >= kernel there would be windows with no real value to maximise
    check_throws([&] { nn::MaxPool2d(nn::Window2d(2, 2, 1, 2)); },
                 "MaxPool2d with padding greater than or equal to the kernel throws");

    // The gradient goes only to the winning position
    Tensor pool_grad({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    nn::MaxPool2d pool2(2, 2);
    Tensor small_out = pool2(pool_grad);
    small_out.sum().backward();
    check_close(pool_grad.grad().data()[3], 1.0f, "the maximum receives the whole gradient");
    check_close(pool_grad.grad().data()[0], 0.0f, "the losers receive no gradient");

    // Flatten
    nn::Flatten flat;
    check(flat(Tensor({2, 3, 4, 5}, 1.0f)).shape() == std::vector<size_t>({2, 60}),
          "Flatten keeps the batch and flattens the rest");
    check_throws([&] { flat(Tensor({6}, 1.0f)); }, "Flatten with a 1D input throws");
}

void test_conv_gradients() {
    section("conv: verificacion numerica de gradients");

    engine::manual_seed(5);

    // Gradient with respect to the input, with and without padding and with stride 2
    {
        nn::Conv2d conv(2, 3, nn::Window2d(3, 3, 1, 1));
        Tensor x = Tensor::randn({2, 2, 5, 5});
        Tensor w = Tensor::randn({2, 3, 5, 5});
        check_gradient("gradient of Conv2d with respect to the input", x,
                       [&](Tensor& t) { return (conv(t) * w).sum(); });
    }
    {
        nn::Conv2d conv(1, 2, nn::Window2d(2, 2, 2, 0));
        Tensor x = Tensor::randn({2, 1, 6, 6});
        check_gradient("gradient of Conv2d with stride 2 with respect to the input", x,
                       [&](Tensor& t) { return conv(t).sum(); });
    }

    // Gradient with respect to the weights and the bias. The perturbed tensor is the
    // layer's own parameter (they share an implementation), so the layer sees the
    // change and re-evaluating the forward is enough.
    {
        nn::Conv2d conv(2, 3, nn::Window2d(3, 3, 1, 1));
        Tensor x = Tensor::randn({2, 2, 4, 4});
        // Non-uniform weighting: with a bare sum() the output gradient is all ones, and
        // that can hide an indexing error.
        Tensor w = Tensor::randn({2, 3, 4, 4});

        check_gradient("gradient of Conv2d with respect to the weights", conv.weight(),
                       [&](Tensor&) { return (conv(x) * w).sum(); });
        check_gradient("gradient of Conv2d respecto to sesgo", conv.bias(),
                       [&](Tensor&) { return (conv(x) * w).sum(); });
    }

    // MaxPool: well-separated values so that no perturbation changes a window's
    // winner (the maximum is not differentiable at a tie).
    {
        Tensor x({1, 2, 4, 4}, 0.0f);
        for (size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<float>((i * 37) % 64) * 0.25f;
        }
        nn::MaxPool2d pool(2, 2);
        check_gradient("gradient of MaxPool2d", x, [&](Tensor& t) { return pool(t).sum(); });
    }

    // Flatten and a full conv -> relu -> pool -> flatten -> linear stack
    {
        Tensor x = Tensor::randn({2, 3, 4, 4});
        nn::Flatten flat;
        check_gradient("gradient of Flatten", x, [&](Tensor& t) { return flat(t).sum(); });
    }
    {
        nn::Sequential net{nn::make<nn::Conv2d>(1, 2, nn::Window2d(3, 3, 1, 1)),
                           nn::make<nn::ReLU>(), nn::make<nn::MaxPool2d>(2, 2),
                           nn::make<nn::Flatten>(), nn::make<nn::Linear>(8, 2)};
        Tensor x = Tensor::randn({2, 1, 4, 4});
        std::vector<size_t> targets = {0, 1};
        check_gradient("gradient of a whole CNN with cross entropy", x,
                       [&](Tensor& t) { return nn::cross_entropy_loss(net(t), targets); });
    }
}

void test_cnn_training() {
    section("conv: training a CNN");

    engine::manual_seed(3);

    // Two classes separable by a local feature: a bright pixel at the top left or the
    // bottom right, at varying positions.
    const size_t N = 40;
    Tensor X({N, 1, 6, 6}, 0.0f, false);
    std::vector<size_t> y(N, 0);
    std::uniform_int_distribution<size_t> jitter(0, 1);

    for (size_t n = 0; n < N; ++n) {
        const size_t label = n % 2;
        const size_t r = (label == 0 ? 0 : 4) + jitter(engine::global_rng());
        const size_t c = (label == 0 ? 0 : 4) + jitter(engine::global_rng());
        X.data()[n * 36 + r * 6 + c] = 1.0f;
        y[n] = label;
    }

    nn::Sequential cnn{nn::make<nn::Conv2d>(1, 4, nn::Window2d(3, 3, 1, 1)), nn::make<nn::ReLU>(),
                       nn::make<nn::MaxPool2d>(2, 2), nn::make<nn::Flatten>(),
                       nn::make<nn::Linear>(4 * 3 * 3, 2)};
    optim::Adam opt(cnn.parameters(), 0.05f);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 60; ++epoch) {
        opt.zero_grad();
        Tensor loss = nn::cross_entropy_loss(cnn(X), y);
        loss.backward();
        opt.step();
        if (epoch == 0) first_loss = loss.data()[0];
        last_loss = loss.data()[0];
    }

    check(last_loss < first_loss, "the CNN's loss decreases");
    check_close(nn::accuracy(cnn(X), y), 1.0f, "the CNN learns the task at 100%");

    // Mini-batches must work with 4D volumes too
    Tensor batch = X.select_rows({0, 1, 2});
    check(batch.shape() == std::vector<size_t>({3, 1, 6, 6}),
          "select_rows takes whole images from a 4D tensor");
    check_close(batch.data()[0], X.data()[0], "the 4D mini-batch copies the right data");
}

void test_conv_determinism() {
    section("conv: determinism across threads");

    namespace par = engine::parallel;
    const size_t original = par::num_threads();

    engine::manual_seed(13);
    Tensor x = Tensor::randn({8, 4, 16, 16}, 0.0f, 1.0f, true);
    Tensor upstream = Tensor::randn({8, 6, 16, 16});

    auto run = [&](size_t threads) {
        par::set_num_threads(threads);
        engine::manual_seed(13);
        nn::Conv2d conv(4, 6, nn::Window2d(3, 3, 1, 1));
        Tensor input = x.detach();
        input.set_requires_grad(true);
        Tensor out = conv(input);
        out.backward(upstream);
        return std::make_tuple(out, input.grad(), conv.weight().grad(), conv.bias().grad());
    };

    auto serial = run(1);
    auto parallel = run(4);

    // Conv2d's backward is split into two passes over disjoint axes precisely so that
    // the accumulation order does not depend on the threads.
    // Equality here is BIT FOR BIT, not approximate.
    const char* names[] = {"the output", "the gradient of the input", "the gradient of the kernel",
                           "the gradient of the bias"};
    const Tensor* s_all[] = {&std::get<0>(serial), &std::get<1>(serial), &std::get<2>(serial),
                             &std::get<3>(serial)};
    const Tensor* p_all[] = {&std::get<0>(parallel), &std::get<1>(parallel), &std::get<2>(parallel),
                             &std::get<3>(parallel)};

    for (size_t i = 0; i < 4; ++i) {
        bool identical = s_all[i]->size() == p_all[i]->size();
        for (size_t j = 0; identical && j < s_all[i]->size(); ++j) {
            if (s_all[i]->data()[j] != p_all[i]->data()[j]) identical = false;
        }
        check(identical, std::string("Conv2d: ") + names[i] +
                             " es identical bit for bit with 1 and with 4 threads");
    }

    // im2col y col2im tambien
    par::set_num_threads(1);
    Tensor cols_serial = nn::im2col(x, nn::Window2d(3, 3, 1, 1));
    Tensor back_serial = nn::col2im(cols_serial, x.shape(), nn::Window2d(3, 3, 1, 1));
    par::set_num_threads(4);
    Tensor cols_par = nn::im2col(x, nn::Window2d(3, 3, 1, 1));
    Tensor back_par = nn::col2im(cols_par, x.shape(), nn::Window2d(3, 3, 1, 1));

    bool cols_same = true;
    for (size_t i = 0; i < cols_serial.size(); ++i) {
        if (cols_serial.data()[i] != cols_par.data()[i]) cols_same = false;
    }
    check(cols_same, "im2col es identical bit for bit with 1 and with 4 threads");

    bool back_same = true;
    for (size_t i = 0; i < back_serial.size(); ++i) {
        if (back_serial.data()[i] != back_par.data()[i]) back_same = false;
    }
    check(back_same, "col2im es identical bit for bit with 1 and with 4 threads");

    par::set_num_threads(original);
}

}  // namespace

void run_conv_tests() {
    test_im2col();
    test_conv_layers();
    test_conv_gradients();
    test_cnn_training();
    test_conv_determinism();
}
