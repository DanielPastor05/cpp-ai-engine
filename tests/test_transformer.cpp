#include "test_support.hpp"

using namespace testing;

namespace {

void test_layernorm_and_embedding() {
    section("transformer: LayerNorm y Embedding");

    engine::manual_seed(23);

    nn::LayerNorm norm(4);
    check(norm.num_parameters() == 8, "LayerNorm has gamma and beta (4 + 4)");

    Tensor x({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 10.0f, 10.0f, 10.0f}, false);
    Tensor normed = norm(x);
    check(normed.shape() == x.shape(), "LayerNorm preserves the shape");

    // First row: mean 0 and variance 1 after normalising
    float mean = 0.0f;
    for (size_t j = 0; j < 4; ++j) mean += normed.data()[j];
    mean /= 4.0f;
    float var = 0.0f;
    for (size_t j = 0; j < 4; ++j) {
        const float d = normed.data()[j] - mean;
        var += d * d;
    }
    var /= 4.0f;
    check_close(mean, 0.0f, "LayerNorm leaves mean 0", 1e-3f);
    check_close(var, 1.0f, "LayerNorm leaves variance 1", 1e-2f);

    // A constant row: with no variance, epsilon avoids dividing by zero
    check_close(normed.data()[4], 0.0f, "a constant row normalises to 0 without dividing by zero");

    // gamma y beta reescalan
    nn::LayerNorm scaled(4);
    scaled.gamma() = Tensor({4}, 2.0f, true);
    scaled.beta() = Tensor({4}, 5.0f, true);
    Tensor out2 = scaled(x);
    check_close(out2.data()[4], 5.0f, "beta shifts the output");

    check_throws([&] { (void)norm(Tensor({2, 3}, 1.0f)); },
                 "LayerNorm with a different last axis size throws");
    check_throws([&] { (void)nn::LayerNorm(0); }, "LayerNorm of size zero throws");

    Tensor gx = Tensor::randn({3, 4});
    Tensor w_norm = Tensor::randn({3, 4});
    check_gradient("gradient of LayerNorm with respect to the input", gx,
                   [&](Tensor& t) { return (norm(t) * w_norm).sum(); });
    {
        nn::LayerNorm n2(4);
        Tensor fixed = Tensor::randn({3, 4});
        Tensor w = Tensor::randn({3, 4});
        check_gradient("gradient of LayerNorm with respect to gamma", n2.gamma(),
                       [&](Tensor&) { return (n2(fixed) * w).sum(); });
        check_gradient("gradient of LayerNorm with respect to beta", n2.beta(),
                       [&](Tensor&) { return (n2(fixed) * w).sum(); });
    }

    // Embedding
    nn::Embedding emb(5, 3);
    check(emb.weight().shape() == std::vector<size_t>({5, 3}),
          "Embedding creates a (vocab, dim) table");

    Tensor ids({2, 3}, {0, 1, 2, 4, 4, 0}, false);
    Tensor vectors = emb(ids);
    check(vectors.shape() == std::vector<size_t>({2, 3, 3}), "Embedding da (batch, seq, dim)");
    check_close(vectors.data()[0], emb.weight().data()[0], "finds the right row for token 0");
    check_close(vectors.data()[9], emb.weight().data()[12], "finds the right row for token 4");

    check_throws([&] { (void)emb(Tensor({3}, 1.0f)); }, "Embedding with 1D indices throws");
    check_throws([&] { (void)emb(Tensor({1, 2}, {0.0f, 9.0f})); },
                 "a token outside the vocabulary throws");

    // A repeated token accumulates gradient in its row
    emb.zero_grad();
    emb(ids).sum().backward();
    check_close(emb.weight().grad().data()[12], 2.0f, "token 4 appears twice and accumulates 2");
    check_close(emb.weight().grad().data()[0], 2.0f, "token 0 also appears twice");
    check_close(emb.weight().grad().data()[9], 0.0f, "an absent token receives no gradient");
}

void test_attention() {
    section("transformer: attention");

    // Orthogonal keys: each query must recover its value
    Tensor Q({3, 3}, {20, 0, 0, 0, 20, 0, 0, 0, 20}, false);
    Tensor K({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, false);
    Tensor V({3, 2}, {10, 100, 20, 200, 30, 300}, false);

    Tensor weights;
    Tensor out = nn::scaled_dot_product_attention(Q, K, V, nullptr, &weights);
    check(out.shape() == std::vector<size_t>({3, 2}), "attention gives (seq, d_v)");
    check(weights.shape() == std::vector<size_t>({3, 3}), "the weights are (seq, seq)");
    check_close(out.data()[0], 10.0f, "query 0 recovers value 0", 0.5f);
    check_close(out.data()[3], 200.0f, "query 1 recovers value 1", 5.0f);

    float row_sum = weights.data()[0] + weights.data()[1] + weights.data()[2];
    check_close(row_sum, 1.0f, "each row of weights sums to 1");

    // Mascara causal
    Tensor mask = nn::causal_mask(4);
    check(mask.shape() == std::vector<size_t>({4, 4}), "causal_mask es (seq, seq)");
    check_close(mask.data()[0], 0.0f, "the diagonal is not masked");
    check(mask.data()[1] < -1e8f, "the upper part is masked");
    check_close(mask.data()[4], 0.0f, "the lower part is not masked");

    Tensor mq = Tensor::randn({4, 3});
    Tensor mw;
    nn::scaled_dot_product_attention(mq, mq, mq, &mask, &mw);
    check_close(mw.data()[0], 1.0f, "the first position attends only to itself");
    check_close(mw.data()[1], 0.0f, "a position does not attend to the future");
    check_close(mw.data()[4] + mw.data()[5], 1.0f, "the second position splits across 2 tokens");

    check_throws(
        [&] {
            (void)nn::scaled_dot_product_attention(Tensor({3, 4}, 1.0f), Tensor({3, 5}, 1.0f), V);
        },
        "query and key with different d_k throw");

    // Gradient of attention
    Tensor gq = Tensor::randn({2, 4, 3});
    Tensor k_fixed = Tensor::randn({2, 4, 3});
    Tensor v_fixed = Tensor::randn({2, 4, 3});
    Tensor w_attn = Tensor::randn({2, 4, 3});
    Tensor causal = nn::causal_mask(4);

    check_gradient("gradient of attention w.r.t. the query", gq, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(t, k_fixed, v_fixed) * w_attn).sum();
    });
    check_gradient("gradient of attention w.r.t. the values", v_fixed, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(gq, k_fixed, t) * w_attn).sum();
    });
    check_gradient("gradient of masked attention", gq, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(t, t, t, &causal) * w_attn).sum();
    });
}

void test_positional_encoding() {
    section("transformer: positional encoding");

    Tensor pe = nn::positional_encoding(4, 6);
    check(pe.shape() == std::vector<size_t>({4, 6}), "positional_encoding da (seq, d_model)");

    // Position 0: sin(0) = 0 at even indices, cos(0) = 1 at odd ones
    check_close(pe(0, 0), 0.0f, "PE[0] starts with sin(0) == 0");
    check_close(pe(0, 1), 1.0f, "PE[0] continues with cos(0) == 1");
    check_close(pe(1, 0), std::sin(1.0f), "PE[1][0] == sin(1)");
    check_close(pe(1, 1), std::cos(1.0f), "PE[1][1] == cos(1)");

    // The frequencies decrease: the last pair varies far less with position
    const float fast = std::fabs(pe(3, 0) - pe(0, 0));
    const float slow = std::fabs(pe(3, 4) - pe(0, 4));
    check(slow < fast, "the high dimensions use lower frequencies");

    // Determinista y sin gradiente
    Tensor pe2 = nn::positional_encoding(4, 6);
    check_close(pe2(2, 3), pe(2, 3), "positional_encoding es determinista");
    check(!pe.requires_grad(), "the positional encoding is not a trainable parameter");

    check_throws([&] { (void)nn::positional_encoding(0, 4); }, "a zero length throws");
}

void test_multihead_and_block() {
    section("transformer: MultiHeadAttention y TransformerBlock");

    engine::manual_seed(29);

    nn::MultiHeadAttention mha(8, 2);
    mha.keep_attention(true);
    Tensor x = Tensor::randn({2, 5, 8});
    Tensor out = mha(x);
    check(out.shape() == std::vector<size_t>({2, 5, 8}),
          "MultiHeadAttention preserves (B, S, d_model)");
    check(mha.num_parameters() == 4 * (8 * 8 + 8), "MHA has 4 projections of d_model x d_model");
    check(mha.last_attention().shape() == std::vector<size_t>({2, 2, 5, 5}),
          "the attention weights are (B, H, S, S)");
    check(!mha.last_attention().requires_grad(), "the saved weights are detached from the graph");

    // They are not kept by default: it is a (B, H, S, S) copy per step
    nn::MultiHeadAttention quiet(8, 2);
    quiet(x);
    check(quiet.last_attention().size() == 0,
          "without keep_attention the weight copy is not paid for");

    check_throws([&] { (void)nn::MultiHeadAttention(8, 3); },
                 "a d_model not divisible by the heads throws");
    check_throws([&] { (void)nn::MultiHeadAttention(8, 0); }, "cero cabezas throws");
    check_throws([&] { (void)mha(Tensor::randn({2, 5, 4})); }, "MHA with the wrong d_model throws");
    check_throws([&] { (void)mha(Tensor::randn({5, 8})); }, "MHA with a 2D input throws");

    // Each head attends separately: with 1 head the weights are (B,1,S,S)
    nn::MultiHeadAttention single(8, 1);
    single.keep_attention(true);
    single(x);
    check(single.last_attention().shape() == std::vector<size_t>({2, 1, 5, 5}),
          "with a single head the weights are (B, 1, S, S)");

    Tensor gx = Tensor::randn({2, 3, 8});
    Tensor w_mha = Tensor::randn({2, 3, 8});
    check_gradient("gradient of MultiHeadAttention", gx,
                   [&](Tensor& t) { return (mha(t) * w_mha).sum(); });

    // TransformerBlock
    nn::TransformerBlock block(8, 2, 16);
    Tensor block_out = block(x);
    check(block_out.shape() == x.shape(), "TransformerBlock preserves the shape");
    check(block.num_parameters() == 4 * (8 * 8 + 8)      // attention
                                        + 2 * (8 + 8)    // dos LayerNorm
                                        + (8 * 16 + 16)  // ff1
                                        + (16 * 8 + 8),  // ff2
          "TransformerBlock sums attention, normalisations and the dense net");

    check_throws([&] { (void)nn::TransformerBlock(8, 2, 0); }, "a zero hidden layer throws");

    Tensor gb = Tensor::randn({2, 3, 8});
    Tensor w_block = Tensor::randn({2, 3, 8});
    Tensor block_mask = nn::causal_mask(3);

    check_gradient("gradient of TransformerBlock", gb,
                   [&](Tensor& t) { return (block(t) * w_block).sum(); });

    // With a causal mask the gradient must still be correct
    check_gradient("gradient of TransformerBlock with a causal mask", gb,
                   [&](Tensor& t) { return (block.forward(t, &block_mask) * w_block).sum(); });
}

void test_transformer_training() {
    section("transformer: training");

    engine::manual_seed(41);

    // The smallest task that demands order: the label is the sequence's first token,
    // but every sequence contains the same two tokens.
    const size_t N = 40;
    const size_t seq = 4;
    Tensor X({N, seq}, 0.0f, false);
    std::vector<size_t> y(N, 0);

    for (size_t n = 0; n < N; ++n) {
        const size_t first = n % 2;
        X.data()[n * seq + 0] = static_cast<float>(first);
        X.data()[n * seq + 1] = static_cast<float>(1 - first);
        X.data()[n * seq + 2] = static_cast<float>(first);
        X.data()[n * seq + 3] = static_cast<float>(1 - first);
        y[n] = first;
    }

    nn::Embedding emb(2, 16);
    nn::TransformerBlock block(16, 2, 32);
    nn::Linear head(16, 2);
    Tensor pe = nn::positional_encoding(seq, 16);

    std::vector<Tensor> params;
    for (nn::Module* m : {static_cast<nn::Module*>(&emb), static_cast<nn::Module*>(&block),
                          static_cast<nn::Module*>(&head)}) {
        std::vector<Tensor> sub = m->parameters();
        params.insert(params.end(), sub.begin(), sub.end());
    }
    optim::Adam opt(params, 0.02f);

    auto forward = [&](const Tensor& ids) {
        Tensor h = block(emb(ids) + pe);
        Tensor first = h.permute({1, 0, 2}).select_rows({0}).reshape({ids.shape()[0], 16});
        return head(first);
    };

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 60; ++epoch) {
        opt.zero_grad();
        Tensor loss = nn::cross_entropy_loss(forward(X), y);
        loss.backward();
        opt.step();
        if (epoch == 0) first_loss = loss.data()[0];
        last_loss = loss.data()[0];
    }

    check(last_loss < first_loss, "the transformer's loss decreases");
    check_close(nn::accuracy(forward(X), y), 1.0f, "the transformer learns the task at 100%");
}

}  // namespace

void run_transformer_tests() {
    test_layernorm_and_embedding();
    test_attention();
    test_positional_encoding();
    test_multihead_and_block();
    test_transformer_training();
}
