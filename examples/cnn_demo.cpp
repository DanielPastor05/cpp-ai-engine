#include "engine/tensor.hpp"
#include "engine/random.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/conv.hpp"
#include "engine/optim.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;

constexpr size_t kImageSize = 12;   // imágenes de 12x12
constexpr size_t kNumClasses = 3;   // barra horizontal, barra vertical, cruz

// Dibuja una figura de la clase pedida en una imagen de un solo canal,
// en una posición aleatoria y sobre un fondo con ruido. La posición cambia
// en cada muestra a propósito: es lo que obliga al modelo a reconocer la
// forma en lugar de memorizar píxeles concretos, y donde la invariancia a la
// traslación de una convolución marca la diferencia frente a una capa densa.
void draw_shape(float* img, size_t label) {
    std::uniform_int_distribution<size_t> pos(2, kImageSize - 5);
    std::normal_distribution<float> noise(0.0f, 0.12f);

    for (size_t i = 0; i < kImageSize * kImageSize; ++i) {
        img[i] = noise(engine::global_rng());
    }

    const size_t r = pos(engine::global_rng());
    const size_t c = pos(engine::global_rng());
    const size_t len = 3;

    auto set = [&](size_t y, size_t x) {
        if (y < kImageSize && x < kImageSize) img[y * kImageSize + x] = 1.0f;
    };

    if (label == 0) {              // barra horizontal
        for (size_t k = 0; k < len; ++k) set(r, c + k);
    } else if (label == 1) {       // barra vertical
        for (size_t k = 0; k < len; ++k) set(r + k, c);
    } else {                       // cruz
        for (size_t k = 0; k < len; ++k) set(r + 1, c + k);
        for (size_t k = 0; k < len; ++k) set(r + k, c + 1);
    }
}

void make_dataset(size_t samples_per_class, Tensor& X, std::vector<size_t>& y) {
    const size_t N = samples_per_class * kNumClasses;
    X = Tensor({N, 1, kImageSize, kImageSize}, 0.0f, false);
    y.assign(N, 0);

    size_t idx = 0;
    for (size_t c = 0; c < kNumClasses; ++c) {
        for (size_t i = 0; i < samples_per_class; ++i) {
            draw_shape(X.data().data() + idx * kImageSize * kImageSize, c);
            y[idx] = c;
            ++idx;
        }
    }
}

void print_image(const Tensor& X, size_t index, size_t label) {
    static const char* names[] = {"barra horizontal", "barra vertical", "cruz"};
    std::cout << "  Muestra " << index << " (" << names[label] << "):\n";
    const float* img = X.data().data() + index * kImageSize * kImageSize;
    for (size_t r = 0; r < kImageSize; ++r) {
        std::cout << "    ";
        for (size_t c = 0; c < kImageSize; ++c) {
            const float v = img[r * kImageSize + c];
            std::cout << (v > 0.6f ? '#' : (v > 0.25f ? '+' : '.'));
        }
        std::cout << "\n";
    }
}

// Entrena por mini-lotes y devuelve la exactitud sobre el conjunto de prueba
float train(const std::string& title, nn::Sequential& model, optim::Optimizer& opt,
            const Tensor& X, const std::vector<size_t>& y,
            const Tensor& X_test, const std::vector<size_t>& y_test,
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
            Tensor loss = nn::cross_entropy_loss(model(X.select_rows(idx)), y_batch);
            loss.backward();
            opt.step();

            epoch_loss += loss.data()[0];
            ++batches;
        }

        engine::autograd::NoGradGuard no_grad;
        const float train_acc = nn::accuracy(model(X), y);
        const float test_acc = nn::accuracy(model(X_test), y_test);
        std::cout << "  Epoch " << std::setw(2) << epoch
                  << " | Loss = " << std::fixed << std::setprecision(4)
                  << (epoch_loss / static_cast<float>(batches))
                  << " | Entrenamiento = " << std::setprecision(2) << (train_acc * 100.0f) << "%"
                  << " | Prueba = " << (test_acc * 100.0f) << "%\n";
    }

    engine::autograd::NoGradGuard no_grad;
    const float acc = nn::accuracy(model(X_test), y_test);
    std::cout << "\n";
    return acc;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Fase 4: Redes Convolucionales (CNN)               \n";
    std::cout << "====================================================\n\n";

    engine::manual_seed(7);

    // ---------------------------------------------------------
    // 1. im2col: la convolución como producto matricial
    // ---------------------------------------------------------
    std::cout << "--- 1. Transformacion im2col ---\n";
    Tensor small({1, 1, 4, 4}, {1,  2,  3,  4,
                                5,  6,  7,  8,
                                9, 10, 11, 12,
                               13, 14, 15, 16}, false);

    nn::Window2d w3(3, 3, 1, 0);
    Tensor cols = nn::im2col(small, w3);
    std::cout << "Entrada " << small.shape_str() << " con kernel 3x3, paso 1, sin relleno\n";
    std::cout << "im2col -> " << cols.shape_str()
              << ": 4 ventanas (2x2 posiciones) aplanadas en filas de 9 valores\n";
    cols.print("columnas");

    Tensor restored = nn::col2im(cols, small.shape(), w3);
    std::cout << "col2im devuelve la forma original " << restored.shape_str()
              << ". No es la inversa exacta: suma los solapes, que es justo\n"
              << "lo que necesita la propagacion hacia atras.\n\n";

    // ---------------------------------------------------------
    // 2. Conjunto de datos
    // ---------------------------------------------------------
    std::cout << "--- 2. Conjunto de datos sintetico ---\n";
    Tensor X, X_test;
    std::vector<size_t> y, y_test;
    make_dataset(120, X, y);        // 360 imágenes de entrenamiento
    make_dataset(40, X_test, y_test); // 120 imágenes de prueba

    std::cout << "Entrenamiento: " << X.shape_str() << "   Prueba: " << X_test.shape_str() << "\n";
    std::cout << "Tres clases dibujadas en posiciones aleatorias, con ruido de fondo:\n\n";
    print_image(X, 0, y[0]);
    print_image(X, 200, y[200]);
    std::cout << "\n";

    // ---------------------------------------------------------
    // 3. CNN frente a un MLP con un número de parámetros parecido
    // ---------------------------------------------------------
    std::cout << "--- 3. Arquitecturas ---\n";

    nn::Sequential cnn{
        nn::make<nn::Conv2d>(1, 8, nn::Window2d(3, 3, 1, 1)),  // (N,1,12,12) -> (N,8,12,12)
        nn::make<nn::ReLU>(),
        nn::make<nn::MaxPool2d>(2, 2),                          // -> (N,8,6,6)
        nn::make<nn::Conv2d>(8, 16, nn::Window2d(3, 3, 1, 1)),  // -> (N,16,6,6)
        nn::make<nn::ReLU>(),
        nn::make<nn::MaxPool2d>(2, 2),                          // -> (N,16,3,3)
        nn::make<nn::Flatten>(),                                // -> (N,144)
        nn::make<nn::Linear>(144, kNumClasses)
    };
    std::cout << "CNN:\n";
    cnn.summary();

    engine::manual_seed(7);
    nn::Sequential mlp{
        nn::make<nn::Flatten>(),
        nn::make<nn::Linear>(kImageSize * kImageSize, 12),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(12, kNumClasses)
    };
    std::cout << "\nMLP de referencia (parametros comparables):\n";
    mlp.summary();
    std::cout << "\n";

    // ---------------------------------------------------------
    // 4. Entrenamiento
    // ---------------------------------------------------------
    engine::manual_seed(7);
    optim::Adam cnn_opt(cnn.parameters(), 0.01f);
    const float cnn_acc = train("CNN con Adam", cnn, cnn_opt, X, y, X_test, y_test, 8, 32);

    engine::manual_seed(7);
    optim::Adam mlp_opt(mlp.parameters(), 0.01f);
    const float mlp_acc = train("MLP con Adam", mlp, mlp_opt, X, y, X_test, y_test, 8, 32);

    // ---------------------------------------------------------
    // 5. Resumen
    // ---------------------------------------------------------
    std::cout << "--- 5. Resumen ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  CNN (" << cnn.num_parameters() << " parametros) : "
              << cnn_acc * 100.0f << "% sobre el conjunto de prueba\n";
    std::cout << "  MLP (" << mlp.num_parameters() << " parametros) : "
              << mlp_acc * 100.0f << "% sobre el conjunto de prueba\n\n";

    std::cout << "¡Fase 4 (Redes Convolucionales) completada exitosamente!\n";
    return 0;
}
