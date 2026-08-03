#include "test_support.hpp"

#include "engine/serialize.hpp"
#include "engine/data.hpp"

#include <cstdio>
#include <fstream>
#include <set>

using namespace testing;

namespace {

void test_nn_layers() {
    section("nn: layers");

    engine::manual_seed(7);

    nn::Linear layer(4, 3);
    check(layer.weight().shape() == std::vector<size_t>({4, 3}),
          "Linear creates (in, out) weights");
    check(layer.bias().shape() == std::vector<size_t>({1, 3}), "Linear creates a (1, out) bias");
    check(layer.num_parameters() == 15, "Linear(4,3) has 4*3 + 3 = 15 parameters");

    Tensor input({5, 4}, 1.0f);
    Tensor out = layer(input);
    check(out.shape() == std::vector<size_t>({5, 3}), "Linear preserves the batch size");

    check_throws([&] { (void)layer(Tensor({5, 7}, 1.0f)); },
                 "an input with the wrong in_features throws");
    check_throws([&] { (void)layer(Tensor({5}, 1.0f)); }, "a 1D input throws");

    nn::Linear no_bias(4, 3, false);
    check(no_bias.parameters().size() == 1, "Linear without bias exposes a single parameter");

    nn::Sequential model{nn::make<nn::Linear>(4, 8), nn::make<nn::ReLU>(),
                         nn::make<nn::Linear>(8, 2)};
    check(model.parameters().size() == 4, "Sequential aggregates its layers' parameters");
    check(model.num_parameters() == 4 * 8 + 8 + 8 * 2 + 2,
          "Sequential sums the parameters correctly");
    check(model(input).shape() == std::vector<size_t>({5, 2}), "Sequential chains the layers");

    check_throws(
        [&] {
            nn::Sequential s;
            s.add(nullptr);
        },
        "Sequential rejects null layers");
}

void test_softmax_and_losses() {
    section("nn: softmax and loss functions");

    Tensor logits({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    Tensor probs = logits.softmax();

    float row0 = probs({0, 0}) + probs({0, 1}) + probs({0, 2});
    float row1 = probs({1, 0}) + probs({1, 1}) + probs({1, 2});
    check_close(row0, 1.0f, "each softmax row sums to 1");
    check_close(row1, 1.0f, "the uniform row also sums to 1");
    check_close(probs({1, 0}), 1.0f / 3.0f, "equal logits give equal probabilities");
    check(probs({0, 2}) > probs({0, 1}), "softmax preserves the order of the logits");

    // Numerical stability: without subtracting the maximum, exp(1000) would overflow
    Tensor huge({1, 3}, {1000.0f, 1000.0f, 1000.0f});
    Tensor huge_probs = huge.softmax();
    check_close(huge_probs({0, 0}), 1.0f / 3.0f, "softmax is stable with enormous logits");

    // Cross entropy of a perfect prediction -> ~0
    Tensor confident({1, 3}, {50.0f, 0.0f, 0.0f});
    check_close(nn::cross_entropy_loss(confident, {0}).data()[0], 0.0f,
                "cross entropy of a perfect prediction is ~0");

    // A uniform distribution over C classes -> log(C)
    Tensor uniform({1, 4}, 0.0f);
    check_close(nn::cross_entropy_loss(uniform, {0}).data()[0], std::log(4.0f),
                "uniform cross entropy equals log(C)");

    check_throws([&] { (void)nn::cross_entropy_loss(logits, {0}); },
                 "a wrong number of labels throws");
    check_throws([&] { (void)nn::cross_entropy_loss(logits, {0, 9}); },
                 "a label out of range throws");

    // MSE
    Tensor pred({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor target({1, 3}, {1.0f, 4.0f, 3.0f});
    check_close(nn::mse_loss(pred, target).data()[0], 4.0f / 3.0f, "mse_loss computes the mean");

    // Metrics
    check(nn::argmax_rows(logits) == std::vector<size_t>({2, 0}),
          "argmax_rows takes the maximum per row");
    check_close(nn::accuracy(logits, {2, 0}), 1.0f, "accuracy with everything correct is 1");
    check_close(nn::accuracy(logits, {0, 0}), 0.5f, "accuracy with half correct is 0.5");
}

void test_optimizers() {
    section("optim: SGD y Adam");

    // Minimise f(w) = (w - 3)^2, whose minimum is at w = 3
    auto minimize = [](optim::Optimizer& opt, Tensor& w, int steps) {
        for (int i = 0; i < steps; ++i) {
            opt.zero_grad();
            Tensor diff = w - Tensor({1}, std::vector<float>{3.0f});
            Tensor loss = (diff * diff).sum();
            loss.backward();
            opt.step();
        }
    };

    Tensor w_sgd({1}, std::vector<float>{0.0f}, true);
    optim::SGD sgd({w_sgd}, 0.1f);
    minimize(sgd, w_sgd, 200);
    check_close(w_sgd.data()[0], 3.0f, "SGD converges to the minimum", 1e-3f);

    Tensor w_mom({1}, std::vector<float>{0.0f}, true);
    optim::SGD sgd_mom({w_mom}, 0.05f, 0.9f);
    minimize(sgd_mom, w_mom, 200);
    check_close(w_mom.data()[0], 3.0f, "SGD with momentum converges to the minimum", 1e-3f);

    Tensor w_adam({1}, std::vector<float>{0.0f}, true);
    optim::Adam adam({w_adam}, 0.1f);
    minimize(adam, w_adam, 300);
    check_close(w_adam.data()[0], 3.0f, "Adam converges to the minimum", 1e-3f);

    // The bias correction makes Adam's first step worth ~lr
    Tensor w_first({1}, std::vector<float>{0.0f}, true);
    optim::Adam adam_first({w_first}, 0.1f);
    adam_first.zero_grad();
    Tensor l = (w_first * 2.0f).sum();
    l.backward();
    adam_first.step();
    check_close(w_first.data()[0], -0.1f, "Adam's first step is ~lr thanks to the bias correction",
                1e-3f);
    check(adam_first.steps() == 1, "Adam counts the steps applied");

    // A parameter with no gradient must not move
    Tensor untouched({2}, {1.0f, 2.0f}, true);
    optim::SGD idle({untouched}, 0.5f);
    idle.step();
    check_close(untouched.data()[0], 1.0f, "a parameter with no gradient is not modified");

    check_throws([&] { (void)(optim::SGD({w_sgd}, -1.0f)); }, "a negative learning rate throws");
    check_throws([&] { (void)optim::Adam({w_sgd}, 0.1f, 1.5f); }, "an invalid beta1 throws");
}

void test_end_to_end_training() {
    section("end-to-end training");

    engine::manual_seed(123);

    // XOR: the smallest case that is not linearly separable
    Tensor X({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1}, false);
    std::vector<size_t> y = {0, 1, 1, 0};

    nn::Sequential model{nn::make<nn::Linear>(2, 16), nn::make<nn::ReLU>(),
                         nn::make<nn::Linear>(16, 2)};
    optim::Adam opt(model.parameters(), 0.1f);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 400; ++epoch) {
        opt.zero_grad();
        Tensor logits = model(X);
        Tensor loss = nn::cross_entropy_loss(logits, y);
        loss.backward();
        opt.step();

        if (epoch == 0) first_loss = loss.data()[0];
        last_loss = loss.data()[0];
    }

    check(last_loss < first_loss, "the loss decreases during training");
    check(last_loss < 0.05f, "the MLP fits XOR (loss < 0.05)");
    check_close(nn::accuracy(model(X), y), 1.0f, "the MLP classifies XOR at 100% accuracy");

    // The same problem cannot be solved with a linear model
    engine::manual_seed(123);
    nn::Linear linear(2, 2);
    optim::Adam linear_opt(linear.parameters(), 0.1f);
    for (int epoch = 0; epoch < 400; ++epoch) {
        linear_opt.zero_grad();
        Tensor loss = nn::cross_entropy_loss(linear(X), y);
        loss.backward();
        linear_opt.step();
    }
    check(nn::accuracy(linear(X), y) < 1.0f, "a linear model does not solve XOR (control)");
}

void test_activations() {
    section("nn: activaciones nuevas");

    Tensor x({5}, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});

    nn::Sigmoid sig;
    Tensor s = sig(x);
    check_close(s.data()[2], 0.5f, "sigmoid(0) == 0.5");
    check(s.data()[0] > 0.0f && s.data()[0] < 0.5f, "sigmoid of a negative falls in (0, 0.5)");
    check(s.data()[4] > 0.5f && s.data()[4] < 1.0f, "sigmoid of a positive falls in (0.5, 1)");
    // Stability at the extremes: without the branched computation, exp() would overflow
    Tensor extreme({2}, {-100.0f, 100.0f});
    check_close(sig(extreme).data()[0], 0.0f, "sigmoid es estable en -100");
    check_close(sig(extreme).data()[1], 1.0f, "sigmoid es estable en +100");

    nn::Tanh th;
    check_close(th(x).data()[2], 0.0f, "tanh(0) == 0");
    check_close(th(x).data()[4], std::tanh(2.0f), "tanh(2) agrees with the library");

    nn::GELU gelu;
    Tensor g = gelu(x);
    check_close(g.data()[2], 0.0f, "gelu(0) == 0");
    check(g.data()[0] < 0.0f, "gelu lets some negative signal through");
    check(g.data()[0] > -0.2f, "but heavily damped");
    check(g.data()[4] > 1.9f, "gelu is nearly the identity for large positives");

    // Gradientes
    Tensor G({4}, {-1.5f, -0.3f, 0.7f, 2.1f});
    Tensor w = Tensor::randn({4});
    check_gradient("gradient of Sigmoid", G, [&](Tensor& t) { return (sig(t) * w).sum(); });
    check_gradient("gradient of Tanh", G, [&](Tensor& t) { return (th(t) * w).sum(); });
    check_gradient("gradient of GELU", G, [&](Tensor& t) { return (gelu(t) * w).sum(); });
}

void test_train_eval_and_dropout() {
    section("nn: modo train/eval y Dropout");

    engine::manual_seed(77);

    nn::Dropout drop(0.5f);
    check(drop.is_training(), "modules start in training mode");

    Tensor x({1000}, 1.0f, false);
    Tensor trained = drop(x);
    size_t zeros = 0;
    for (size_t i = 0; i < trained.size(); ++i) {
        if (trained.data()[i] == 0.0f) ++zeros;
    }
    check(zeros > 400 && zeros < 600, "in training it zeroes about half");

    // The mean is preserved thanks to the 1/(1-p) scaling
    float mean = 0.0f;
    for (size_t i = 0; i < trained.size(); ++i) mean += trained.data()[i];
    mean /= 1000.0f;
    check(std::fabs(mean - 1.0f) < 0.1f, "the scaling preserves the mean");

    drop.eval();
    check(!drop.is_training(), "eval() turns training mode off");
    Tensor evaluated = drop(x);
    bool identical = true;
    for (size_t i = 0; i < evaluated.size(); ++i) {
        if (evaluated.data()[i] != 1.0f) identical = false;
    }
    check(identical, "at evaluation Dropout is the identity");

    check_throws([&] { (void)nn::Dropout(1.0f); }, "a probability of 1 throws");
    check_throws([&] { (void)(nn::Dropout(-0.1f)); }, "a negative probability throws");

    // The switch propagates through the container
    nn::Sequential model{nn::make<nn::Linear>(4, 4), nn::make<nn::Dropout>(0.5f),
                         nn::make<nn::Linear>(4, 2)};
    model.eval();
    check(!model.at(1).is_training(), "Sequential propagates eval() to its layers");
    model.train();
    check(model.at(1).is_training(), "Sequential propagates train() to its layers");

    // The gradient passes only through the positions that survived
    engine::manual_seed(5);
    nn::Dropout d2(0.5f);
    Tensor gx({20}, 1.0f, true);
    Tensor out = d2(gx);
    out.sum().backward();
    bool coherent = true;
    for (size_t i = 0; i < gx.size(); ++i) {
        const bool alive = out.data()[i] != 0.0f;
        if (alive != (gx.grad().data()[i] != 0.0f)) coherent = false;
    }
    check(coherent, "the gradient passes exactly where the activation did");
}

void test_named_parameters() {
    section("nn: named parameters");

    nn::Sequential model{nn::make<nn::Linear>(3, 4), nn::make<nn::ReLU>(),
                         nn::make<nn::Linear>(4, 2)};

    auto named = model.named_parameters();
    check(named.size() == model.parameters().size(), "there is one name per parameter");

    std::set<std::string> unique;
    for (const auto& entry : named) unique.insert(entry.first);
    check(unique.size() == named.size(), "the names do not repeat");

    // The layer's index goes into the name, so two identical layers
    // no colisionan
    check(named[0].first != named[2].first, "two identical Linear layers get different names");

    // They share their implementation with the real weights
    named[0].second.data()[0] = 42.0f;
    check_close(model.parameters()[0].data()[0], 42.0f,
                "the named tensors share data with the layer");
}

void test_clip_and_schedulers() {
    section("optim: recorte de gradient y planificadores");

    // Recorte
    Tensor a({2}, {3.0f, 4.0f}, true);  // norma 5
    (a * 1.0f).sum().backward();        // gradiente (1, 1) -> norma sqrt(2)
    float norm = optim::clip_grad_norm({a}, 10.0f);
    check_close(norm, std::sqrt(2.0f), "clip returns the prior norm");
    check_close(a.grad().data()[0], 1.0f, "below the limit it does not clip");

    Tensor b({2}, {1.0f, 1.0f}, true);
    b.add_grad(Tensor({2}, {3.0f, 4.0f}));  // norma 5
    float n2 = optim::clip_grad_norm({b}, 1.0f);
    check_close(n2, 5.0f, "clip measures the global norm");
    float after = std::sqrt(b.grad().data()[0] * b.grad().data()[0] +
                            b.grad().data()[1] * b.grad().data()[1]);
    check_close(after, 1.0f, "after clipping the norm is the maximum", 1e-3f);
    check_close(b.grad().data()[0] / b.grad().data()[1], 3.0f / 4.0f,
                "clipping preserves the direction", 1e-3f);

    // The norm is global, not per parameter
    Tensor p1({1}, std::vector<float>{0.0f}, true);
    Tensor p2({1}, std::vector<float>{0.0f}, true);
    p1.add_grad(Tensor({1}, std::vector<float>{3.0f}));
    p2.add_grad(Tensor({1}, std::vector<float>{4.0f}));
    check_close(optim::clip_grad_norm({p1, p2}, 100.0f), 5.0f,
                "the joint norm of two parameters is 5");

    check_throws([&] { (void)optim::clip_grad_norm({a}, 0.0f); }, "a non-positive max_norm throws");

    // Planificadores
    Tensor w({1}, std::vector<float>{0.0f}, true);
    optim::SGD opt({w}, 1.0f);

    optim::StepLR step(opt, 2, 0.5f);
    check_close(opt.learning_rate(), 1.0f, "the learning rate starts at the base");
    step.step();
    check_close(opt.learning_rate(), 1.0f, "StepLR holds the lr within a step");
    step.step();
    check_close(opt.learning_rate(), 0.5f, "StepLR multiplies it by gamma at a step change");
    step.step();
    step.step();
    check_close(opt.learning_rate(), 0.25f, "StepLR accumulates the steps");

    optim::SGD opt2({w}, 1.0f);
    optim::CosineAnnealingLR cos(opt2, 10, 0.0f);
    for (int i = 0; i < 5; ++i) cos.step();
    check_close(opt2.learning_rate(), 0.5f, "halfway down the cosine the lr is halved", 1e-3f);
    for (int i = 0; i < 5; ++i) cos.step();
    check_close(opt2.learning_rate(), 0.0f, "at the end of the cosine it reaches the minimum",
                1e-3f);

    optim::SGD opt3({w}, 1.0f);
    optim::WarmupCosineLR warm(opt3, 3, 10, 0.0f);
    warm.step();
    check(opt3.learning_rate() < 0.5f, "during warm-up the lr is low");
    warm.step();
    warm.step();
    check_close(opt3.learning_rate(), 1.0f, "when the warm-up ends it reaches the base", 1e-3f);
    for (int i = 0; i < 7; ++i) warm.step();
    check(opt3.learning_rate() < 0.01f, "then decays as a cosine");

    check_throws([&] { (void)optim::StepLR(opt, 0); }, "a zero step throws");
    check_throws([&] { (void)optim::WarmupCosineLR(opt, 10, 5); },
                 "a warm-up longer than the total throws");
}

void test_serialization() {
    section("serialize: saving and loading weights");

    const std::string path = "test_weights.bin";
    engine::manual_seed(99);

    nn::Sequential model{nn::make<nn::Linear>(4, 6), nn::make<nn::ReLU>(),
                         nn::make<nn::Linear>(6, 3)};
    Tensor x = Tensor::randn({5, 4});
    Tensor before = model(x);

    engine::save_parameters(model, path);

    // Another model with the same architecture but different weights
    engine::manual_seed(1234);
    nn::Sequential loaded{nn::make<nn::Linear>(4, 6), nn::make<nn::ReLU>(),
                          nn::make<nn::Linear>(6, 3)};
    Tensor different = loaded(x);
    bool differs = false;
    for (size_t i = 0; i < before.size(); ++i) {
        if (std::fabs(before.data()[i] - different.data()[i]) > 1e-5f) differs = true;
    }
    check(differs, "before loading, the two models give different outputs");

    const size_t n = engine::load_parameters(loaded, path);
    check(n == model.parameters().size(), "every parameter is loaded");

    Tensor after = loaded(x);
    float max_diff = 0.0f;
    for (size_t i = 0; i < before.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(before.data()[i] - after.data()[i]));
    }
    check_close(max_diff, 0.0f, "after loading, the model reproduces the exact output");

    // Inspect without loading
    auto summary = engine::inspect_parameters(path);
    check(summary.size() == model.parameters().size(), "inspect lists every tensor");
    check(summary[0].second == std::vector<size_t>({4, 6}), "inspect returns the shapes");

    // A different architecture is rejected rather than loaded wrongly
    nn::Sequential wrong{nn::make<nn::Linear>(4, 8)};
    check_throws([&] { (void)engine::load_parameters(wrong, path); },
                 "loading into a different architecture throws");

    // With strict=false only the ones matching by name are loaded. Since the default
    // name includes the layer's dimensions, a different architecture simply matches
    // nothing.
    check(engine::load_parameters(wrong, path, false) == 0,
          "without strict, a different architecture loads nothing");

    // --- hostile checkpoints ---
    //
    // A weight file is the only input to this engine that does not come from its
    // own API, and it is a thing people download. Three of its header fields are
    // 32-bit sizes used to allocate, and the element count was a product of
    // 64-bit dimensions accumulated into a size_t with no overflow check.
    //
    // Each case below is a file that is structurally valid up to the field being
    // attacked, so the parser has to reject it on the value rather than on the
    // magic number.
    {
        const char magic[8] = {'C', 'P', 'P', 'A', 'I', 'E', 'N', 'G'};
        auto craft = [&](const std::string& name, const std::vector<uint32_t>& u32s,
                         const std::vector<uint64_t>& u64s) {
            const std::string p = name;
            std::ofstream f(p, std::ios::binary);
            f.write(magic, sizeof(magic));
            const uint32_t version = 1;
            f.write(reinterpret_cast<const char*>(&version), sizeof(version));
            for (uint32_t v : u32s) f.write(reinterpret_cast<const char*>(&v), sizeof(v));
            for (uint64_t v : u64s) f.write(reinterpret_cast<const char*>(&v), sizeof(v));
            return p;
        };

        // Four billion tensors in a twenty-byte file.
        const std::string huge_count = craft("hostile_count.bin", {0xFFFFFFFFu}, {});
        check_throws([&] { (void)engine::inspect_parameters(huge_count); },
                     "a tensor count larger than the file is rejected");

        // One tensor, a name four gigabytes long.
        const std::string huge_name = craft("hostile_name.bin", {1u, 0xFFFFFFFFu}, {});
        check_throws([&] { (void)engine::inspect_parameters(huge_name); },
                     "a name length larger than the file is rejected");

        // One tensor, one short name, four billion dimensions.
        const std::string huge_ndim = craft("hostile_ndim.bin", {1u, 0u, 0xFFFFFFFFu}, {});
        check_throws([&] { (void)engine::inspect_parameters(huge_ndim); },
                     "a dimension count larger than the file is rejected");

        // The one that used to pass. Two dimensions of 2^32: their product
        // overflows size_t to exactly zero, so the tensor came out empty while
        // its shape claimed 2^64 elements. Nothing crashed and nothing was
        // reported -- the file simply loaded as something it was not.
        const std::string overflow =
            craft("hostile_overflow.bin", {1u, 0u, 2u}, {uint64_t(1) << 32, uint64_t(1) << 32});
        check_throws([&] { (void)engine::inspect_parameters(overflow); },
                     "a shape whose element count overflows is rejected, not silently wrapped");

        for (const char* f : {"hostile_count.bin", "hostile_name.bin", "hostile_ndim.bin",
                              "hostile_overflow.bin"}) {
            std::remove(f);
        }
    }

    // The shape check fires when the name DOES match but the shape does not: that is
    // the case that would leave a silently broken model.
    {
        Tensor good({2, 2}, 1.0f, true);
        std::vector<std::pair<std::string, Tensor>> saved = {{"weight", good}};
        engine::save_parameters(saved, "shape_test.bin");

        Tensor mismatched({3, 3}, 0.0f, true);
        std::vector<std::pair<std::string, Tensor>> target = {{"weight", mismatched}};
        check_throws([&] { (void)engine::load_parameters(target, "shape_test.bin"); },
                     "the same name with another shape is rejected");
        std::remove("shape_test.bin");
    }

    // Two parameters with the same name would be indistinguishable on load
    {
        Tensor t1({1}, std::vector<float>{1.0f}, true);
        Tensor t2({1}, std::vector<float>{2.0f}, true);
        std::vector<std::pair<std::string, Tensor>> dup = {{"a", t1}, {"a", t2}};
        check_throws([&] { (void)engine::save_parameters(dup, "dup.bin"); },
                     "duplicate names are rejected on save");
        std::remove("dup.bin");
    }

    check_throws([&] { (void)engine::load_parameters(model, "no_existe.bin"); },
                 "loading a nonexistent file throws");

    // A file that is not ours
    {
        std::ofstream bad("not_weights.bin", std::ios::binary);
        bad << "these are not weights at all";
    }
    check_throws([&] { (void)engine::load_parameters(model, "not_weights.bin"); },
                 "a file with a wrong signature is rejected");

    // A whole transformer round-trips too
    engine::manual_seed(3);
    nn::TransformerBlock block(8, 2, 16);
    Tensor seq = Tensor::randn({2, 4, 8});
    Tensor block_before = block(seq);
    engine::save_parameters(block, path);

    engine::manual_seed(444);
    nn::TransformerBlock block2(8, 2, 16);
    engine::load_parameters(block2, path);
    Tensor block_after = block2(seq);
    float block_diff = 0.0f;
    for (size_t i = 0; i < block_before.size(); ++i) {
        block_diff =
            std::max(block_diff, std::fabs(block_before.data()[i] - block_after.data()[i]));
    }
    check_close(block_diff, 0.0f, "a whole TransformerBlock is saved and restored");

    std::remove(path.c_str());
    std::remove("not_weights.bin");
}

void test_idx_reader() {
    section("data: IDX format reader");

    // The subset shipped with the repository
    engine::data::MnistPaths paths;
    bool found = true;
    try {
        paths = engine::data::find_mnist(ENGINE_DATA_DIR "/mnist");
    } catch (const std::exception&) {
        found = false;
    }
    check(found, "find_mnist locates the repository's set");
    if (!found) return;

    engine::data::Dataset train = engine::data::load_mnist_train(paths, 64);
    check(train.images.shape() == std::vector<size_t>({64, 1, 28, 28}),
          "the images come out as (N, 1, 28, 28), ready for Conv2d");
    check(train.labels.size() == 64, "as many labels as images are read");

    // Normalisation to [0, 1]: with 0-255 the first layer's gradients would be two
    // orders of magnitude larger
    float min_v = 1e9f, max_v = -1e9f;
    const float* pixels = train.images.data();
    for (size_t i = 0; i < train.images.size(); ++i) {
        min_v = std::min(min_v, pixels[i]);
        max_v = std::max(max_v, pixels[i]);
    }
    check(min_v >= 0.0f && max_v <= 1.0f, "the pixels end up normalised to [0, 1]");
    check(max_v > 0.9f, "and they reach the maximum (there are saturated pixels)");

    for (size_t label : train.labels) {
        if (label > 9) {
            check(false, "the labels fall outside 0-9");
            return;
        }
    }
    check(true, "every label is in the range 0-9");

    // MNIST's first digit is a 5: if the big-endian byte order were wrong, neither
    // the dimensions nor the labels would add up
    check(train.labels[0] == 5, "MNIST's first label is a 5");

    // max_samples recorta
    engine::data::Dataset small = engine::data::load_mnist_test(paths, 10);
    check(small.size() == 10, "max_samples limits how many samples are read");

    check_throws([&] { (void)engine::data::load_idx_images("no_existe.idx"); },
                 "a nonexistent file throws");
    check_throws([&] { (void)engine::data::load_idx_images(paths.train_labels); },
                 "reading labels as images throws (different magic)");
    check_throws([&] { (void)engine::data::find_mnist("directorio_inexistente"); },
                 "a directory with no data throws");
}

}  // namespace

void run_nn_tests() {
    test_nn_layers();
    test_softmax_and_losses();
    test_optimizers();
    test_end_to_end_training();
    test_activations();
    test_train_eval_and_dropout();
    test_named_parameters();
    test_clip_and_schedulers();
    test_serialization();
    test_idx_reader();
}
