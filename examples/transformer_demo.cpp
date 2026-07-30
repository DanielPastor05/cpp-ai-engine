#include "engine/tensor.hpp"
#include "engine/random.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/transformer.hpp"
#include "engine/optim.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;

// ---------------------------------------------------------
// Tarea: "which token comes after the marker?"
//
// Cada secuencia es [CLS] seguido de una permutación de los seis valores, con
// una MARCA insertada en una posición aleatoria. La etiqueta es el valor que
// aparece justo después de la marca.
//
// El detalle importante: cada valor aparece exactamente una vez en toda
// secuencia. El multiconjunto de tokens es siempre el mismo, así que saber
// *qué* tokens hay no aporta absolutamente nada — un modelo que promedie la
// secuencia está condenado al azar (1/6). La única información está en el
// orden, y hay que relacionar dos posiciones para extraerla.
// ---------------------------------------------------------
constexpr size_t kNumValues = 6;            // valores 0..5
constexpr size_t kMarker = kNumValues;      // id 6
constexpr size_t kCls = kNumValues + 1;     // id 7
constexpr size_t kVocab = kNumValues + 2;   // 8 tokens
constexpr size_t kSeqLen = kNumValues + 2;  // [CLS] + 6 valores + MARCA

void make_dataset(size_t num_samples, Tensor& ids, std::vector<size_t>& labels,
                  std::vector<size_t>* marker_positions = nullptr) {
    ids = Tensor({num_samples, kSeqLen}, 0.0f, false);
    labels.assign(num_samples, 0);
    if (marker_positions) marker_positions->assign(num_samples, 0);

    std::vector<size_t> values(kNumValues);
    std::iota(values.begin(), values.end(), 0);
    // La marca nunca va al final: siempre tiene un valor detrás
    std::uniform_int_distribution<size_t> slot(0, kNumValues - 1);

    for (size_t n = 0; n < num_samples; ++n) {
        std::shuffle(values.begin(), values.end(), engine::global_rng());
        const size_t m = slot(engine::global_rng());

        float* row = ids.data().data() + n * kSeqLen;
        row[0] = static_cast<float>(kCls);

        // Se copian los valores insertando la marca justo antes del m-esimo
        size_t out = 1;
        for (size_t i = 0; i < kNumValues; ++i) {
            if (i == m) row[out++] = static_cast<float>(kMarker);
            row[out++] = static_cast<float>(values[i]);
        }

        labels[n] = values[m];
        if (marker_positions) (*marker_positions)[n] = m + 1;  // posicion de la MARCA
    }
}

// ---------------------------------------------------------
// Modelo con atención
// ---------------------------------------------------------
// Dos bloques, no uno: la tarea es de dos saltos. Un bloque tiene que marcar
// cada posición con "the one before me is the MARKER" y el otro recoger esa posición
// desde el [CLS]. Con un solo bloque la consulta del [CLS] no puede depender
// de dónde está la marca, y el modelo se queda muy por debajo.
struct TransformerClassifier {
    nn::Embedding embedding;
    nn::TransformerBlock block1;
    nn::TransformerBlock block2;
    nn::LayerNorm norm;
    nn::Linear head;
    Tensor pos_encoding;

    TransformerClassifier(size_t d_model, size_t heads, size_t ff)
        : embedding(kVocab, d_model),
          block1(d_model, heads, ff),
          block2(d_model, heads, ff),
          norm(d_model),
          head(d_model, kNumValues),
          pos_encoding(nn::positional_encoding(kSeqLen, d_model)) {
        // Solo el segundo bloque guarda los pesos, y solo para inspeccionarlos
        // al final: durante el entrenamiento seria una copia inutil por paso.
        block2.attention().keep_attention(true);
    }

    Tensor forward(const Tensor& ids) {
        const size_t batch = ids.shape()[0];

        // Embeddings + codificación posicional. La suma difunde (S, D) sobre
        // el lote (B, S, D): sin ella la atención no distinguiría el orden.
        Tensor h = embedding(ids) + pos_encoding;
        h = block1(h);
        h = block2(h);
        h = norm(h);

        // Se clasifica desde la posición del [CLS], que ha ido recogiendo
        // información del resto de la secuencia a través de la atención.
        const size_t d_model = h.shape()[2];
        Tensor cls = h.permute({1, 0, 2}).select_rows({0}).reshape({batch, d_model});
        return head(cls);
    }

    std::vector<Tensor> parameters() {
        std::vector<Tensor> params;
        for (nn::Module* m : {static_cast<nn::Module*>(&embedding),
                              static_cast<nn::Module*>(&block1), static_cast<nn::Module*>(&block2),
                              static_cast<nn::Module*>(&norm), static_cast<nn::Module*>(&head)}) {
            std::vector<Tensor> sub = m->parameters();
            params.insert(params.end(), sub.begin(), sub.end());
        }
        return params;
    }

    size_t num_parameters() {
        size_t total = 0;
        for (const Tensor& p : parameters()) total += p.size();
        return total;
    }
};

// ---------------------------------------------------------
// Referencia sin atención: promedia los embeddings de la secuencia.
// Ve exactamente los mismos tokens, pero pierde toda la información de orden.
// ---------------------------------------------------------
struct MeanPoolClassifier {
    nn::Embedding embedding;
    nn::Linear hidden;
    nn::Linear head;
    Tensor averager;  // (S, 1) con 1/S en cada posición

    MeanPoolClassifier(size_t d_model, size_t hidden_size)
        : embedding(kVocab, d_model),
          hidden(d_model, hidden_size),
          head(hidden_size, kNumValues),
          averager({kSeqLen, 1}, 1.0f / static_cast<float>(kSeqLen), false) {}

    Tensor forward(const Tensor& ids) {
        const size_t batch = ids.shape()[0];
        Tensor h = embedding(ids);  // (B, S, D)
        const size_t d_model = h.shape()[2];

        // Promedio sobre la secuencia como producto matricial: (B, D, S) x (S, 1)
        Tensor pooled = h.permute({0, 2, 1}).matmul(averager).reshape({batch, d_model});
        return head(hidden(pooled).relu());
    }

    std::vector<Tensor> parameters() {
        std::vector<Tensor> params;
        for (nn::Module* m : {static_cast<nn::Module*>(&embedding),
                              static_cast<nn::Module*>(&hidden), static_cast<nn::Module*>(&head)}) {
            std::vector<Tensor> sub = m->parameters();
            params.insert(params.end(), sub.begin(), sub.end());
        }
        return params;
    }

    size_t num_parameters() {
        size_t total = 0;
        for (const Tensor& p : parameters()) total += p.size();
        return total;
    }
};

template <typename Model>
float train(const std::string& title, Model& model, optim::Optimizer& opt, const Tensor& X,
            const std::vector<size_t>& y, const Tensor& X_test, const std::vector<size_t>& y_test,
            int epochs, size_t batch_size) {
    std::cout << "--- " << title << " ---\n";

    const size_t N = X.shape()[0];
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), engine::global_rng());

        float epoch_loss = 0.0f;
        size_t batches = 0;

        for (size_t start = 0; start < N; start += batch_size) {
            const size_t end = std::min(start + batch_size, N);
            const std::vector<size_t> idx(order.begin() + start, order.begin() + end);

            std::vector<size_t> y_batch;
            y_batch.reserve(idx.size());
            for (size_t i : idx) y_batch.push_back(y[i]);

            opt.zero_grad();
            Tensor loss = nn::cross_entropy_loss(model.forward(X.select_rows(idx)), y_batch);
            loss.backward();
            opt.step();

            epoch_loss += loss.data()[0];
            ++batches;
        }

        if (epoch == 1 || epoch % 10 == 0) {
            engine::autograd::NoGradGuard no_grad;
            std::cout << "  Epoch " << std::setw(2) << epoch << " | Loss = " << std::fixed
                      << std::setprecision(4) << (epoch_loss / static_cast<float>(batches))
                      << " | Train = " << std::setprecision(2)
                      << (nn::accuracy(model.forward(X), y) * 100.0f) << "%"
                      << " | Test = " << (nn::accuracy(model.forward(X_test), y_test) * 100.0f)
                      << "%\n";
        }
    }

    engine::autograd::NoGradGuard no_grad;
    const float acc = nn::accuracy(model.forward(X_test), y_test);
    std::cout << "\n";
    return acc;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Phase 5: the Transformer architecture                  \n";
    std::cout << "====================================================\n\n";

    engine::manual_seed(2024);

    // ---------------------------------------------------------
    // 1. Atención por producto escalar escalado
    // ---------------------------------------------------------
    std::cout << "--- 1. Scaled Dot-Product Attention ---\n";
    // Tres claves ortogonales y tres consultas, cada una alineada con una de
    // ellas. La atención debe recuperar casi exactamente el valor asociado.
    Tensor Q({3, 3}, {8.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f, 8.0f}, false);
    Tensor K({3, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, false);
    Tensor V({3, 2}, {10.0f, 100.0f, 20.0f, 200.0f, 30.0f, 300.0f}, false);

    Tensor weights;
    Tensor attended = nn::scaled_dot_product_attention(Q, K, V, nullptr, &weights);
    weights.print("attention weights (each row sums to 1)");
    attended.print("output = weights x V");

    // ---------------------------------------------------------
    // 2. Máscara causal
    // ---------------------------------------------------------
    std::cout << "--- 2. Mascara causal ---\n";
    Tensor mask = nn::causal_mask(4);
    Tensor masked_weights;
    Tensor q4 = Tensor::randn({4, 3});
    nn::scaled_dot_product_attention(q4, q4, q4, &mask, &masked_weights);
    masked_weights.print("masked weights (triangular: nobody sees the future)");

    // ---------------------------------------------------------
    // 3. Codificación posicional
    // ---------------------------------------------------------
    std::cout << "--- 3. Codificacion posicional sinusoidal ---\n";
    Tensor pe = nn::positional_encoding(6, 8);
    std::cout << "positional_encoding(6, 8) -> " << pe.shape_str() << "\n";
    std::cout << "Each row encodes a position with sines and cosines of\n"
              << "different frequencies, so nearby positions have\n"
              << "codificaciones parecidas:\n";
    pe.print("PE");

    // ---------------------------------------------------------
    // 4. Dataset
    // ---------------------------------------------------------
    std::cout << "--- 4. Task: the token that follows the marker ---\n";
    Tensor X, X_test;
    std::vector<size_t> y, y_test, markers;
    make_dataset(1500, X, y, &markers);
    make_dataset(400, X_test, y_test);

    std::cout << "Vocabulario: valores 0-5, MARCA=" << kMarker << ", [CLS]=" << kCls << "\n";
    std::cout << "Train: " << X.shape_str() << "   Test: " << X_test.shape_str() << "\n\n";
    for (size_t n = 0; n < 3; ++n) {
        std::cout << "  Secuencia " << n << ": [";
        for (size_t s = 0; s < kSeqLen; ++s) {
            const size_t tok = static_cast<size_t>(X.data()[n * kSeqLen + s]);
            if (tok == kCls)
                std::cout << "CLS";
            else if (tok == kMarker)
                std::cout << "MARCA";
            else
                std::cout << tok;
            if (s + 1 < kSeqLen) std::cout << " ";
        }
        std::cout << "]  -> etiqueta " << y[n] << " (posicion " << markers[n] + 1 << ")\n";
    }
    std::cout << "\n";

    // ---------------------------------------------------------
    // 5. Entrenamiento
    // ---------------------------------------------------------
    engine::manual_seed(2024);
    TransformerClassifier transformer(32, 4, 64);
    std::cout << "Transformer: " << transformer.num_parameters() << " parametros\n";
    optim::Adam t_opt(transformer.parameters(), 0.003f);
    const float t_acc = train("Transformer (2 bloques, 4 cabezas)", transformer, t_opt, X, y,
                              X_test, y_test, 60, 32);

    engine::manual_seed(2024);
    MeanPoolClassifier baseline(32, 64);
    std::cout << "Reference with no attention: " << baseline.num_parameters() << " parametros\n";
    optim::Adam b_opt(baseline.parameters(), 0.003f);
    const float b_acc =
        train("Referencia: promedio de embeddings", baseline, b_opt, X, y, X_test, y_test, 60, 32);

    // ---------------------------------------------------------
    // 6. ¿Dónde mira el modelo?
    // ---------------------------------------------------------
    std::cout << "--- 6. Pesos de atencion aprendidos ---\n";
    {
        engine::autograd::NoGradGuard no_grad;
        Tensor one = X.select_rows({0});
        transformer.forward(one);
        const Tensor& attn = transformer.block2.attention().last_attention();

        std::cout << "Sequence 0, marker at position " << markers[0] << ", answer at "
                  << markers[0] + 1 << ".\n";
        std::cout << "[CLS] attention in the second block, over each position:\n";

        const size_t heads = attn.shape()[1];
        for (size_t h = 0; h < heads; ++h) {
            std::cout << "  Cabeza " << h << ": ";
            for (size_t s = 0; s < kSeqLen; ++s) {
                // attn es (B, H, S, S): la fila 0 es a qué atiende el [CLS]
                const float w = attn.data()[(h * kSeqLen + 0) * kSeqLen + s];
                std::cout << std::fixed << std::setprecision(2) << w;
                std::cout << (s == markers[0] + 1 ? "* " : "  ");
            }
            std::cout << "\n";
        }
        std::cout << "  (* marks the position holding the answer)\n\n";
    }

    // ---------------------------------------------------------
    // 7. Resumen
    // ---------------------------------------------------------
    std::cout << "--- 7. Resumen ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Transformer                : " << t_acc * 100.0f << "% on test\n";
    std::cout << "  Promedio de embeddings     : " << b_acc * 100.0f << "% on test\n";
    std::cout << "  Azar (1 de " << kNumValues
              << " valores)        : " << 100.0f / static_cast<float>(kNumValues) << "%\n\n";
    std::cout << "Every sequence contains exactly the same tokens, so\n"
              << "averaging them leaves no information at all: the baseline stays at\n"
              << "chance by construction. Attention, on the other hand, relates the\n"
              << "marker's position to the next one and solves the task.\n\n";

    std::cout << "Phase 5 (Transformer architecture) complete.\n";
    return 0;
}
