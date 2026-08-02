// MNIST: the engine's CNN on real data.
//
// The repository ships a 2,000 + 1,000 image subset so this works on a fresh
// clone. tools/download_mnist.sh fetches the full set (60,000 + 10,000) and the
// example detects it by itself.

#include "engine/tensor.hpp"
#include "engine/random.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/conv.hpp"
#include "engine/optim.hpp"
#include "engine/data.hpp"
#include "engine/serialize.hpp"
#include "engine/cuda.hpp"
#include "engine/parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;
namespace data = engine::data;

namespace {

constexpr size_t kNumClasses = 10;
const std::string kCheckpoint = "mnist_cnn.bin";

void print_digit(const Tensor& images, size_t index, size_t label) {
    std::cout << "  Label " << label << ":\n";
    const float* img = images.data().data() + index * 28 * 28;
    for (size_t r = 0; r < 28; r += 2) {  // every other row: terminal cells are tall
        std::cout << "    ";
        for (size_t c = 0; c < 28; ++c) {
            const float v = img[r * 28 + c];
            std::cout << (v > 0.6f ? '#' : (v > 0.25f ? '+' : (v > 0.05f ? '.' : ' ')));
        }
        std::cout << "\n";
    }
}

// Accuracy is computed in batches: with 10,000 images at once, the CNN's
// intermediate activations do not fit comfortably in memory.
float evaluate(nn::Sequential& model, const data::Dataset& set, size_t batch_size = 500) {
    engine::autograd::NoGradGuard no_grad;
    model.eval();

    size_t hits = 0;
    for (size_t start = 0; start < set.size(); start += batch_size) {
        const size_t end = std::min(start + batch_size, set.size());
        std::vector<size_t> idx(end - start);
        std::iota(idx.begin(), idx.end(), start);

        Tensor logits = model(set.images.select_rows(idx));
        std::vector<size_t> predicted = nn::argmax_rows(logits);
        for (size_t i = 0; i < predicted.size(); ++i) {
            if (predicted[i] == set.labels[start + i]) ++hits;
        }
    }
    model.train();
    return static_cast<float>(hits) / static_cast<float>(set.size());
}

void print_confusion(nn::Sequential& model, const data::Dataset& set) {
    engine::autograd::NoGradGuard no_grad;
    model.eval();

    size_t matrix[kNumClasses][kNumClasses] = {};
    for (size_t start = 0; start < set.size(); start += 500) {
        const size_t end = std::min(start + size_t(500), set.size());
        std::vector<size_t> idx(end - start);
        std::iota(idx.begin(), idx.end(), start);
        std::vector<size_t> predicted = nn::argmax_rows(model(set.images.select_rows(idx)));
        for (size_t i = 0; i < predicted.size(); ++i) {
            ++matrix[set.labels[start + i]][predicted[i]];
        }
    }
    model.train();

    std::cout << "Confusion matrix (rows = actual, columns = predicted):\n\n";
    std::cout << "        ";
    for (size_t c = 0; c < kNumClasses; ++c) std::cout << std::setw(5) << c;
    std::cout << "\n";
    for (size_t r = 0; r < kNumClasses; ++r) {
        std::cout << "    " << r << " | ";
        for (size_t c = 0; c < kNumClasses; ++c) {
            if (matrix[r][c] == 0)
                std::cout << std::setw(5) << ".";
            else
                std::cout << std::setw(5) << matrix[r][c];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

}  // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "  MNIST: the engine's CNN on real data              \n";
    std::cout << "====================================================\n\n";

    // Whoever reads the timing should know what ran where. The convolutions now go
    // through Tensor::matmul, so the conv -> relu -> pool -> conv chain stays on the
    // card end to end; the loss and the optimiser are still on the CPU and break it
    // once per step.
    if (engine::cuda::available()) {
        const engine::cuda::DeviceInfo gpu = engine::cuda::device_info();
        std::cout << "Backend: CUDA on " << gpu.name << " (cc " << gpu.compute_major << "."
                  << gpu.compute_minor << "), convolutions included.\n";
    } else {
        std::cout << "Backend: CPU, " << engine::parallel::num_threads() << " thread(s).\n";
    }
    std::cout << "\n";

    engine::manual_seed(42);

    // ---------------------------------------------------------
    // 1. Data
    // ---------------------------------------------------------
    data::MnistPaths paths;
    try {
        paths = data::find_mnist();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    data::Dataset train = data::load_mnist_train(paths);
    data::Dataset test = data::load_mnist_test(paths);

    std::cout << "--- 1. Dataset ---\n";
    std::cout << (paths.full ? "full MNIST" : "Repository subset") << ": " << train.size()
              << " training images, " << test.size() << " test, of " << train.images.shape_str()
              << "\n";
    if (!paths.full) {
        std::cout << "(run tools/download_mnist.sh for the full set)\n";
    }
    std::cout << "\n";
    print_digit(train.images, 0, train.labels[0]);
    std::cout << "\n";

    // ---------------------------------------------------------
    // 2. Model
    // ---------------------------------------------------------
    nn::Sequential model{
        nn::make<nn::Conv2d>(1, 16, nn::Window2d(3, 3, 1, 1)),  // (N,1,28,28) -> (N,16,28,28)
        nn::make<nn::ReLU>(),
        nn::make<nn::MaxPool2d>(2, 2),                           // -> (N,16,14,14)
        nn::make<nn::Conv2d>(16, 32, nn::Window2d(3, 3, 1, 1)),  // -> (N,32,14,14)
        nn::make<nn::ReLU>(),
        nn::make<nn::MaxPool2d>(2, 2),  // -> (N,32,7,7)
        nn::make<nn::Flatten>(),        // -> (N,1568)
        nn::make<nn::Dropout>(0.25f),
        nn::make<nn::Linear>(32 * 7 * 7, 128),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(128, kNumClasses)};

    std::cout << "--- 2. Architecture ---\n";
    model.summary();
    std::cout << "\n";

    // ---------------------------------------------------------
    // 3. Training
    // ---------------------------------------------------------
    const int epochs = paths.full ? 6 : 12;
    const size_t batch_size = 64;

    optim::Adam opt(model.parameters(), 0.001f);
    optim::CosineAnnealingLR scheduler(opt, static_cast<size_t>(epochs), 0.0001f);

    std::vector<size_t> order(train.size());
    std::iota(order.begin(), order.end(), 0);

    std::cout << "--- 3. Training (" << epochs << " epochs, batches of " << batch_size << ") ---\n";

    const auto started = std::chrono::steady_clock::now();
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), engine::global_rng());
        model.train();

        float epoch_loss = 0.0f;
        size_t batches = 0;

        for (size_t start = 0; start < train.size(); start += batch_size) {
            const size_t end = std::min(start + batch_size, train.size());
            const std::vector<size_t> idx(order.begin() + start, order.begin() + end);

            std::vector<size_t> y;
            y.reserve(idx.size());
            for (size_t i : idx) y.push_back(train.labels[i]);

            opt.zero_grad();
            Tensor loss = nn::cross_entropy_loss(model(train.images.select_rows(idx)), y);
            loss.backward();
            // Global-norm clipping: stops an outlier batch from taking a huge step
            optim::clip_grad_norm(model.parameters(), 5.0f);
            opt.step();

            epoch_loss += loss.data()[0];
            ++batches;
        }
        scheduler.step();

        const float test_acc = evaluate(model, test);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        std::cout << "  Epoch " << std::setw(2) << epoch << " | Loss = " << std::fixed
                  << std::setprecision(4) << (epoch_loss / static_cast<float>(batches))
                  << " | Test = " << std::setprecision(2) << (test_acc * 100.0f) << "%"
                  << " | lr = " << std::scientific << std::setprecision(1) << opt.learning_rate()
                  << std::defaultfloat << " | " << std::fixed << std::setprecision(1) << elapsed
                  << " s\n";
    }
    std::cout << "\n";

    // ---------------------------------------------------------
    // 4. Results
    // ---------------------------------------------------------
    std::cout << "--- 4. Results ---\n";
    const float final_acc = evaluate(model, test);
    std::cout << std::fixed << std::setprecision(2)
              << "Final accuracy on the test set: " << (final_acc * 100.0f) << "%\n\n";
    print_confusion(model, test);

    // ---------------------------------------------------------
    // 5. The trained model is saved and restored
    // ---------------------------------------------------------
    std::cout << "--- 5. Persistencia ---\n";
    engine::save_parameters(model, kCheckpoint);
    std::cout << "Pesos guardados en " << kCheckpoint << " (" << model.num_parameters()
              << " parametros).\n";

    // A fresh model with random weights, which then loads the saved ones
    engine::manual_seed(999);
    nn::Sequential restored{nn::make<nn::Conv2d>(1, 16, nn::Window2d(3, 3, 1, 1)),
                            nn::make<nn::ReLU>(),
                            nn::make<nn::MaxPool2d>(2, 2),
                            nn::make<nn::Conv2d>(16, 32, nn::Window2d(3, 3, 1, 1)),
                            nn::make<nn::ReLU>(),
                            nn::make<nn::MaxPool2d>(2, 2),
                            nn::make<nn::Flatten>(),
                            nn::make<nn::Dropout>(0.25f),
                            nn::make<nn::Linear>(32 * 7 * 7, 128),
                            nn::make<nn::ReLU>(),
                            nn::make<nn::Linear>(128, kNumClasses)};
    std::cout << "A fresh untrained model: " << std::setprecision(2)
              << (evaluate(restored, test) * 100.0f) << "%\n";

    engine::load_parameters(restored, kCheckpoint);
    std::cout << "The same one after loading the weights: " << (evaluate(restored, test) * 100.0f)
              << "%\n\n";

    std::cout << "Training on real data complete.\n";
    return 0;
}
