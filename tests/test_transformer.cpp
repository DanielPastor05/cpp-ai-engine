#include "test_support.hpp"

using namespace testing;

namespace {


void test_layernorm_and_embedding() {
    section("transformer: LayerNorm y Embedding");

    engine::manual_seed(23);

    nn::LayerNorm norm(4);
    check(norm.num_parameters() == 8, "LayerNorm tiene gamma y beta (4 + 4)");

    Tensor x({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f,
                      10.0f, 10.0f, 10.0f, 10.0f}, false);
    Tensor normed = norm(x);
    check(normed.shape() == x.shape(), "LayerNorm conserva la forma");

    // Primera fila: media 0 y varianza 1 tras normalizar
    float mean = 0.0f;
    for (size_t j = 0; j < 4; ++j) mean += normed.data()[j];
    mean /= 4.0f;
    float var = 0.0f;
    for (size_t j = 0; j < 4; ++j) {
        const float d = normed.data()[j] - mean;
        var += d * d;
    }
    var /= 4.0f;
    check_close(mean, 0.0f, "LayerNorm deja media 0", 1e-3f);
    check_close(var, 1.0f, "LayerNorm deja varianza 1", 1e-2f);

    // Fila constante: sin varianza, epsilon evita dividir por cero
    check_close(normed.data()[4], 0.0f, "una fila constante se normaliza a 0 sin dividir por cero");

    // gamma y beta reescalan
    nn::LayerNorm scaled(4);
    scaled.gamma() = Tensor({4}, 2.0f, true);
    scaled.beta() = Tensor({4}, 5.0f, true);
    Tensor out2 = scaled(x);
    check_close(out2.data()[4], 5.0f, "beta desplaza la salida");

    check_throws([&] { norm(Tensor({2, 3}, 1.0f)); },
                 "LayerNorm con un ultimo eje de otro tamano lanza excepcion");
    check_throws([&] { nn::LayerNorm(0); }, "LayerNorm de tamano cero lanza excepcion");

    Tensor gx = Tensor::randn({3, 4});
    Tensor w_norm = Tensor::randn({3, 4});
    check_gradient("gradiente de LayerNorm respecto a la entrada", gx, [&](Tensor& t) {
        return (norm(t) * w_norm).sum();
    });
    {
        nn::LayerNorm n2(4);
        Tensor fixed = Tensor::randn({3, 4});
        Tensor w = Tensor::randn({3, 4});
        check_gradient("gradiente de LayerNorm respecto a gamma", n2.gamma(),
                       [&](Tensor&) { return (n2(fixed) * w).sum(); });
        check_gradient("gradiente de LayerNorm respecto a beta", n2.beta(),
                       [&](Tensor&) { return (n2(fixed) * w).sum(); });
    }

    // Embedding
    nn::Embedding emb(5, 3);
    check(emb.weight().shape() == std::vector<size_t>({5, 3}), "Embedding crea una tabla (vocab, dim)");

    Tensor ids({2, 3}, {0, 1, 2,
                        4, 4, 0}, false);
    Tensor vectors = emb(ids);
    check(vectors.shape() == std::vector<size_t>({2, 3, 3}), "Embedding da (batch, seq, dim)");
    check_close(vectors.data()[0], emb.weight().data()[0], "busca la fila correcta del token 0");
    check_close(vectors.data()[9], emb.weight().data()[12], "busca la fila correcta del token 4");

    check_throws([&] { emb(Tensor({3}, 1.0f)); }, "Embedding con indices 1D lanza excepcion");
    check_throws([&] { emb(Tensor({1, 2}, {0.0f, 9.0f})); },
                 "un token fuera del vocabulario lanza excepcion");

    // Un token repetido acumula gradiente en su fila
    emb.zero_grad();
    emb(ids).sum().backward();
    check_close(emb.weight().grad().data()[12], 2.0f, "el token 4 aparece 2 veces y acumula 2");
    check_close(emb.weight().grad().data()[0], 2.0f, "el token 0 tambien aparece 2 veces");
    check_close(emb.weight().grad().data()[9], 0.0f, "un token ausente no recibe gradiente");
}


void test_attention() {
    section("transformer: atencion");

    // Claves ortogonales: cada consulta debe recuperar su valor
    Tensor Q({3, 3}, {20, 0, 0,
                      0, 20, 0,
                      0, 0, 20}, false);
    Tensor K({3, 3}, {1, 0, 0,
                      0, 1, 0,
                      0, 0, 1}, false);
    Tensor V({3, 2}, {10, 100,
                      20, 200,
                      30, 300}, false);

    Tensor weights;
    Tensor out = nn::scaled_dot_product_attention(Q, K, V, nullptr, &weights);
    check(out.shape() == std::vector<size_t>({3, 2}), "la atencion da (seq, d_v)");
    check(weights.shape() == std::vector<size_t>({3, 3}), "los pesos son (seq, seq)");
    check_close(out.data()[0], 10.0f, "la consulta 0 recupera el valor 0", 0.5f);
    check_close(out.data()[3], 200.0f, "la consulta 1 recupera el valor 1", 5.0f);

    float row_sum = weights.data()[0] + weights.data()[1] + weights.data()[2];
    check_close(row_sum, 1.0f, "cada fila de pesos suma 1");

    // Mascara causal
    Tensor mask = nn::causal_mask(4);
    check(mask.shape() == std::vector<size_t>({4, 4}), "causal_mask es (seq, seq)");
    check_close(mask.data()[0], 0.0f, "la diagonal no esta enmascarada");
    check(mask.data()[1] < -1e8f, "la parte superior esta enmascarada");
    check_close(mask.data()[4], 0.0f, "la parte inferior no esta enmascarada");

    Tensor mq = Tensor::randn({4, 3});
    Tensor mw;
    nn::scaled_dot_product_attention(mq, mq, mq, &mask, &mw);
    check_close(mw.data()[0], 1.0f, "la primera posicion solo se atiende a si misma");
    check_close(mw.data()[1], 0.0f, "una posicion no atiende al futuro");
    check_close(mw.data()[4] + mw.data()[5], 1.0f, "la segunda posicion reparte entre 2 tokens");

    check_throws([&] {
        nn::scaled_dot_product_attention(Tensor({3, 4}, 1.0f), Tensor({3, 5}, 1.0f), V);
    }, "query y key con d_k distinto lanzan excepcion");

    // Gradiente de la atencion
    Tensor gq = Tensor::randn({2, 4, 3});
    Tensor k_fixed = Tensor::randn({2, 4, 3});
    Tensor v_fixed = Tensor::randn({2, 4, 3});
    Tensor w_attn = Tensor::randn({2, 4, 3});
    Tensor causal = nn::causal_mask(4);

    check_gradient("gradiente de la atencion respecto a la consulta", gq, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(t, k_fixed, v_fixed) * w_attn).sum();
    });
    check_gradient("gradiente de la atencion respecto a los valores", v_fixed, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(gq, k_fixed, t) * w_attn).sum();
    });
    check_gradient("gradiente de la atencion enmascarada", gq, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(t, t, t, &causal) * w_attn).sum();
    });
}


void test_positional_encoding() {
    section("transformer: codificacion posicional");

    Tensor pe = nn::positional_encoding(4, 6);
    check(pe.shape() == std::vector<size_t>({4, 6}), "positional_encoding da (seq, d_model)");

    // Posicion 0: sin(0) = 0 en los indices pares, cos(0) = 1 en los impares
    check_close(pe(0, 0), 0.0f, "PE[0] empieza con sin(0) == 0");
    check_close(pe(0, 1), 1.0f, "PE[0] sigue con cos(0) == 1");
    check_close(pe(1, 0), std::sin(1.0f), "PE[1][0] == sin(1)");
    check_close(pe(1, 1), std::cos(1.0f), "PE[1][1] == cos(1)");

    // Las frecuencias decrecen: la ultima pareja varia mucho menos con la posicion
    const float fast = std::fabs(pe(3, 0) - pe(0, 0));
    const float slow = std::fabs(pe(3, 4) - pe(0, 4));
    check(slow < fast, "las dimensiones altas usan frecuencias mas bajas");

    // Determinista y sin gradiente
    Tensor pe2 = nn::positional_encoding(4, 6);
    check_close(pe2(2, 3), pe(2, 3), "positional_encoding es determinista");
    check(!pe.requires_grad(), "la codificacion posicional no es un parametro entrenable");

    check_throws([&] { nn::positional_encoding(0, 4); }, "una longitud nula lanza excepcion");
}


void test_multihead_and_block() {
    section("transformer: MultiHeadAttention y TransformerBlock");

    engine::manual_seed(29);

    nn::MultiHeadAttention mha(8, 2);
    mha.keep_attention(true);
    Tensor x = Tensor::randn({2, 5, 8});
    Tensor out = mha(x);
    check(out.shape() == std::vector<size_t>({2, 5, 8}), "MultiHeadAttention conserva (B, S, d_model)");
    check(mha.num_parameters() == 4 * (8 * 8 + 8), "MHA tiene 4 proyecciones de d_model x d_model");
    check(mha.last_attention().shape() == std::vector<size_t>({2, 2, 5, 5}),
          "los pesos de atencion son (B, H, S, S)");
    check(!mha.last_attention().requires_grad(),
          "los pesos guardados estan desligados del grafo");

    // Por defecto no se guardan: es una copia de (B, H, S, S) por paso
    nn::MultiHeadAttention quiet(8, 2);
    quiet(x);
    check(quiet.last_attention().size() == 0,
          "sin keep_attention no se paga la copia de los pesos");

    check_throws([&] { nn::MultiHeadAttention(8, 3); },
                 "d_model no divisible entre las cabezas lanza excepcion");
    check_throws([&] { nn::MultiHeadAttention(8, 0); }, "cero cabezas lanza excepcion");
    check_throws([&] { mha(Tensor::randn({2, 5, 4})); },
                 "MHA con un d_model erroneo lanza excepcion");
    check_throws([&] { mha(Tensor::randn({5, 8})); }, "MHA con una entrada 2D lanza excepcion");

    // Cada cabeza atiende por separado: con 1 cabeza los pesos son (B,1,S,S)
    nn::MultiHeadAttention single(8, 1);
    single.keep_attention(true);
    single(x);
    check(single.last_attention().shape() == std::vector<size_t>({2, 1, 5, 5}),
          "con una sola cabeza los pesos son (B, 1, S, S)");

    Tensor gx = Tensor::randn({2, 3, 8});
    Tensor w_mha = Tensor::randn({2, 3, 8});
    check_gradient("gradiente de MultiHeadAttention", gx, [&](Tensor& t) {
        return (mha(t) * w_mha).sum();
    });

    // TransformerBlock
    nn::TransformerBlock block(8, 2, 16);
    Tensor block_out = block(x);
    check(block_out.shape() == x.shape(), "TransformerBlock conserva la forma");
    check(block.num_parameters() ==
              4 * (8 * 8 + 8)      // atencion
              + 2 * (8 + 8)        // dos LayerNorm
              + (8 * 16 + 16)      // ff1
              + (16 * 8 + 8),      // ff2
          "TransformerBlock suma atencion, normalizaciones y red densa");

    check_throws([&] { nn::TransformerBlock(8, 2, 0); },
                 "una capa oculta nula lanza excepcion");

    Tensor gb = Tensor::randn({2, 3, 8});
    Tensor w_block = Tensor::randn({2, 3, 8});
    Tensor block_mask = nn::causal_mask(3);

    check_gradient("gradiente de TransformerBlock", gb, [&](Tensor& t) {
        return (block(t) * w_block).sum();
    });

    // Con mascara causal el gradiente debe seguir siendo correcto
    check_gradient("gradiente de TransformerBlock con mascara causal", gb, [&](Tensor& t) {
        return (block.forward(t, &block_mask) * w_block).sum();
    });
}


void test_transformer_training() {
    section("transformer: entrenamiento");

    engine::manual_seed(41);

    // Tarea minima que exige orden: la etiqueta es el primer token de la
    // secuencia, pero todas las secuencias contienen los mismos dos tokens.
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
    for (nn::Module* m : {static_cast<nn::Module*>(&emb),
                          static_cast<nn::Module*>(&block),
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

    check(last_loss < first_loss, "la perdida del transformer disminuye");
    check_close(nn::accuracy(forward(X), y), 1.0f, "el transformer aprende la tarea al 100%");
}

} // namespace

void run_transformer_tests() {
    test_layernorm_and_embedding();
    test_attention();
    test_positional_encoding();
    test_multihead_and_block();
    test_transformer_training();
}
