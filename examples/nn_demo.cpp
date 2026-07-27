#include "engine/tensor.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/optim.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;

// Genera el clásico conjunto "espiral": K brazos entrelazados que ningún
// clasificador lineal puede separar, así que obliga a la red a usar las capas
// ocultas y la no linealidad de ReLU.
void make_spiral(size_t points_per_class, size_t num_classes,
                 Tensor& X, std::vector<size_t>& y) {
    const size_t N = points_per_class * num_classes;
    X = Tensor({N, 2}, 0.0f, false);
    y.assign(N, 0);

    std::normal_distribution<float> noise(0.0f, 0.20f);

    size_t idx = 0;
    for (size_t c = 0; c < num_classes; ++c) {
        for (size_t i = 0; i < points_per_class; ++i) {
            const float r = static_cast<float>(i) / static_cast<float>(points_per_class);
            const float t = static_cast<float>(c) * 4.0f
                          + r * 4.0f
                          + noise(engine::global_rng());

            X.data()[idx * 2 + 0] = r * std::sin(t);
            X.data()[idx * 2 + 1] = r * std::cos(t);
            y[idx] = c;
            ++idx;
        }
    }
}

// Entrena un perceptrón multicapa y devuelve la exactitud final
float train(const std::string& title, optim::Optimizer& opt, nn::Sequential& model,
            const Tensor& X, const std::vector<size_t>& y, int epochs) {
    std::cout << "--- " << title << " ---\n";

    float acc = 0.0f;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // 1. Limpiar los gradientes de la iteración anterior
        opt.zero_grad();

        // 2. Forward pass
        Tensor logits = model(X);

        // 3. Pérdida
        Tensor loss = nn::cross_entropy_loss(logits, y);

        // 4. Backward pass
        loss.backward();

        // 5. Actualizar los pesos
        opt.step();

        acc = nn::accuracy(logits, y);
        if (epoch == 1 || epoch % 100 == 0) {
            std::cout << "  Epoch " << std::setw(4) << epoch
                      << " | Loss = " << std::fixed << std::setprecision(4) << loss.data()[0]
                      << " | Exactitud = " << std::setprecision(2) << (acc * 100.0f) << "%\n";
        }
    }
    std::cout << "\n";
    return acc;
}

// Entrenamiento por mini-lotes: en cada época se baraja el conjunto y se
// recorren trozos de `batch_size` filas. Es lo que hace "estocástico" al
// descenso de gradiente estocástico: cada paso usa una muestra distinta.
float train_minibatch(const std::string& title, optim::Optimizer& opt, nn::Sequential& model,
                      const Tensor& X, const std::vector<size_t>& y,
                      int epochs, size_t batch_size) {
    std::cout << "--- " << title << " ---\n";

    const size_t N = X.shape()[0];
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), engine::global_rng());

        float epoch_loss = 0.0f;
        size_t num_batches = 0;

        for (size_t start = 0; start < N; start += batch_size) {
            const size_t end = std::min(start + batch_size, N);
            const std::vector<size_t> idx(order.begin() + start, order.begin() + end);

            std::vector<size_t> y_batch;
            y_batch.reserve(idx.size());
            for (size_t i : idx) y_batch.push_back(y[i]);

            opt.zero_grad();
            Tensor logits = model(X.select_rows(idx));
            Tensor loss = nn::cross_entropy_loss(logits, y_batch);
            loss.backward();
            opt.step();

            epoch_loss += loss.data()[0];
            ++num_batches;
        }

        if (epoch == 1 || epoch % 20 == 0) {
            // La exactitud se mide sobre todo el conjunto, sin construir grafo
            engine::autograd::NoGradGuard no_grad;
            std::cout << "  Epoch " << std::setw(4) << epoch
                      << " | Loss media = " << std::fixed << std::setprecision(4)
                      << (epoch_loss / static_cast<float>(num_batches))
                      << " | Exactitud = " << std::setprecision(2)
                      << (nn::accuracy(model(X), y) * 100.0f) << "%\n";
        }
    }

    engine::autograd::NoGradGuard no_grad;
    const float acc = nn::accuracy(model(X), y);
    std::cout << "\n";
    return acc;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Fase 3: Capas (nn::Module) y Optimizadores        \n";
    std::cout << "====================================================\n\n";

    // Semilla fija para que el entrenamiento sea reproducible
    engine::manual_seed(42);

    // ---------------------------------------------------------
    // 1. Capa densa y difusión del sesgo
    // ---------------------------------------------------------
    std::cout << "--- 1. Capa Densa (Linear) ---\n";
    nn::Linear dense(3, 2);
    dense.bias() = Tensor({1, 2}, {0.5f, -0.5f}, true);

    Tensor batch({4, 3}, {1.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 1.0f,
                          1.0f, 1.0f, 1.0f}, false);

    Tensor dense_out = dense(batch);
    std::cout << dense.name() << " -> " << dense.num_parameters() << " parametros\n";
    std::cout << "Entrada " << batch.shape_str() << " -> Salida " << dense_out.shape_str()
              << " (el sesgo se difunde sobre las 4 filas del lote)\n\n";

    // ---------------------------------------------------------
    // 2. Softmax y entropía cruzada
    // ---------------------------------------------------------
    std::cout << "--- 2. Softmax y Entropia Cruzada ---\n";
    Tensor logits({2, 3}, {2.0f, 1.0f, 0.1f,
                           0.5f, 2.5f, 0.3f}, true);
    Tensor probs = logits.softmax();
    probs.print("softmax(logits) (cada fila suma 1)");

    std::vector<size_t> labels = {0, 1}; // ambas predicciones son correctas
    Tensor ce = nn::cross_entropy_loss(logits, labels);
    std::cout << "Entropia cruzada = " << std::fixed << std::setprecision(4) << ce.data()[0]
              << " (baja: el argmax coincide con la etiqueta)\n";
    std::cout << "Exactitud = " << nn::accuracy(logits, labels) * 100.0f << "%\n\n";

    // ---------------------------------------------------------
    // 3. Clasificación no lineal: espiral de 3 clases
    // ---------------------------------------------------------
    std::cout << "--- 3. Clasificacion de la espiral de 3 clases ---\n";
    const size_t num_classes = 3;
    const size_t points_per_class = 100;

    Tensor X;
    std::vector<size_t> y;
    make_spiral(points_per_class, num_classes, X, y);
    std::cout << "Conjunto de datos: " << X.shape_str() << ", "
              << num_classes << " clases entrelazadas (no separables linealmente)\n\n";

    // Referencia: un clasificador lineal (una sola capa densa, sin activación)
    nn::Sequential linear_model{ nn::make<nn::Linear>(2, num_classes) };
    optim::Adam linear_opt(linear_model.parameters(), 0.05f);
    float linear_acc = train("Referencia: clasificador lineal (Adam)", linear_opt, linear_model, X, y, 300);

    // Perceptrón multicapa: 2 -> 64 -> 32 -> 3
    nn::Sequential mlp{
        nn::make<nn::Linear>(2, 64),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(64, 32),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(32, num_classes)
    };
    mlp.summary();
    std::cout << "\n";

    optim::Adam adam(mlp.parameters(), 0.02f);
    float mlp_acc = train("MLP 2-64-32-3 con Adam", adam, mlp, X, y, 500);

    // El mismo modelo con SGD + momento, entrenado por mini-lotes: 50 épocas
    // de 10 pasos bastan donde el lote completo necesitaba 500 iteraciones.
    engine::manual_seed(42);
    nn::Sequential mlp_sgd{
        nn::make<nn::Linear>(2, 64),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(64, 32),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(32, num_classes)
    };
    optim::SGD sgd(mlp_sgd.parameters(), 0.1f, 0.9f);
    float sgd_acc = train_minibatch("MLP 2-64-32-3 con SGD por mini-lotes (lr=0.1, momento=0.9, lote=32)",
                                    sgd, mlp_sgd, X, y, 60, 32);

    // ---------------------------------------------------------
    // 4. Resumen
    // ---------------------------------------------------------
    std::cout << "--- 4. Resumen ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Clasificador lineal : " << linear_acc * 100.0f << "% de exactitud\n";
    std::cout << "  MLP + Adam          : " << mlp_acc * 100.0f << "% de exactitud\n";
    std::cout << "  MLP + SGD mini-lotes: " << sgd_acc * 100.0f << "% de exactitud\n";
    std::cout << "\nEl modelo lineal se estanca porque la espiral no es separable\n"
              << "linealmente; las capas ocultas con ReLU si logran separarla.\n\n";

    std::cout << "¡Fase 3 (Capas y Optimizadores) completada exitosamente!\n";
    return 0;
}
