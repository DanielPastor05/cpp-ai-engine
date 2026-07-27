#include "engine/nn.hpp"
#include "engine/autograd.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// Module
// ---------------------------------------------------------

void Module::zero_grad() {
    for (Tensor& p : parameters()) {
        p.zero_grad();
    }
}

size_t Module::num_parameters() {
    size_t total = 0;
    for (const Tensor& p : parameters()) {
        total += p.size();
    }
    return total;
}

// ---------------------------------------------------------
// Linear
// ---------------------------------------------------------

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Linear requiere in_features y out_features mayores que cero.");
    }

    // Inicialización Xavier/Glorot uniforme: mantiene la varianza de las
    // activaciones estable a lo largo de la profundidad de la red.
    const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));

    autograd::NoGradGuard no_grad;
    weight_ = Tensor::rand({in_features, out_features}, -limit, limit, true);
    bias_ = Tensor({1, out_features}, 0.0f, use_bias);
}

Tensor Linear::forward(const Tensor& input) {
    if (input.ndim() < 2) {
        throw std::invalid_argument("Linear espera al menos 2 dimensiones (..., in_features), recibió " +
                                    input.shape_str() + ".");
    }
    if (input.shape().back() != in_features_) {
        throw std::invalid_argument("Linear con in_features=" + std::to_string(in_features_) +
                                    " recibió una entrada " + input.shape_str() + ".");
    }

    // La capa actúa sobre el último eje. Con entradas de más de 2 ejes —como
    // (batch, secuencia, d_model) en un Transformer— se aplanan los ejes
    // iniciales, se proyecta y se restaura la forma.
    const bool needs_reshape = input.ndim() > 2;
    Tensor flat = needs_reshape
                      ? input.reshape({input.size() / in_features_, in_features_})
                      : input;

    Tensor out = flat.matmul(weight_);
    if (use_bias_) {
        out = out + bias_; // difusión del vector fila sobre todo el lote
    }

    if (needs_reshape) {
        std::vector<size_t> out_shape = input.shape();
        out_shape.back() = out_features_;
        out = out.reshape(out_shape);
    }
    return out;
}

std::vector<Tensor> Linear::parameters() {
    if (use_bias_) return { weight_, bias_ };
    return { weight_ };
}

std::string Linear::name() const {
    return "Linear(" + std::to_string(in_features_) + " -> " + std::to_string(out_features_) +
           (use_bias_ ? ", bias=true)" : ", bias=false)");
}

// ---------------------------------------------------------
// Activaciones
// ---------------------------------------------------------

Tensor ReLU::forward(const Tensor& input) {
    return input.relu();
}

Tensor Softmax::forward(const Tensor& input) {
    return input.softmax();
}

// ---------------------------------------------------------
// Sequential
// ---------------------------------------------------------

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> layers)
    : layers_(layers) {
    for (const auto& layer : layers_) {
        if (!layer) throw std::invalid_argument("Sequential no admite capas nulas.");
    }
}

Sequential& Sequential::add(std::shared_ptr<Module> layer) {
    if (!layer) throw std::invalid_argument("Sequential no admite capas nulas.");
    layers_.push_back(std::move(layer));
    return *this;
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor out = input;
    for (const auto& layer : layers_) {
        out = layer->forward(out);
    }
    return out;
}

std::vector<Tensor> Sequential::parameters() {
    std::vector<Tensor> params;
    for (const auto& layer : layers_) {
        std::vector<Tensor> sub = layer->parameters();
        params.insert(params.end(), sub.begin(), sub.end());
    }
    return params;
}

void Sequential::summary() const {
    std::cout << "Sequential (" << layers_.size() << " capas)\n";
    size_t total = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        size_t n = layers_[i]->num_parameters();
        total += n;
        std::cout << "  [" << i << "] " << std::left << std::setw(32) << layers_[i]->name()
                  << std::right << std::setw(10) << n << " parametros\n";
    }
    std::cout << "  Total entrenable: " << total << " parametros\n";
}

// ---------------------------------------------------------
// Funciones de pérdida
// ---------------------------------------------------------

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    Tensor diff = prediction - target;
    return (diff * diff).mean();
}

Tensor cross_entropy_loss(const Tensor& logits, const std::vector<size_t>& targets) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("cross_entropy_loss espera logits 2D (batch, clases), recibió " +
                                    logits.shape_str() + ".");
    }
    const size_t N = logits.shape()[0];
    const size_t C = logits.shape()[1];

    if (targets.size() != N) {
        throw std::invalid_argument("cross_entropy_loss recibió " + std::to_string(targets.size()) +
                                    " etiquetas para un lote de " + std::to_string(N) + ".");
    }
    if (N == 0) {
        throw std::invalid_argument("cross_entropy_loss recibió un lote vacío.");
    }

    // Softmax estable por filas y pérdida media del lote en una sola pasada.
    Tensor probs(logits.shape(), 0.0f, false);
    float total_loss = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        if (targets[i] >= C) {
            throw std::out_of_range("Etiqueta " + std::to_string(targets[i]) +
                                    " fuera del rango de " + std::to_string(C) + " clases.");
        }
        const float* row = logits.data().data() + i * C;
        const float max_v = *std::max_element(row, row + C);

        float sum_exp = 0.0f;
        for (size_t j = 0; j < C; ++j) {
            const float e = std::exp(row[j] - max_v);
            probs.data()[i * C + j] = e;
            sum_exp += e;
        }
        for (size_t j = 0; j < C; ++j) {
            probs.data()[i * C + j] /= sum_exp;
        }

        // log-sum-exp desplazado: log p_y = (x_y - max) - log(sum exp(x - max))
        const float log_prob = (row[targets[i]] - max_v) - std::log(sum_exp);
        total_loss += -log_prob;
    }

    const bool req_g = logits.requires_grad() && autograd::grad_enabled();
    Tensor loss({1}, std::vector<float>{total_loss / static_cast<float>(N)}, req_g);

    if (req_g) {
        loss.get_impl()->parents = { logits.get_impl() };
        Tensor logits_copy = logits;

        loss.get_impl()->backward_fn =
            [logits_copy, probs, targets, N, C](const Tensor& grad_out) mutable {
                // dL/dlogits = (softmax(logits) - one_hot(y)) / N
                const float scale = grad_out.data()[0] / static_cast<float>(N);
                Tensor dX(logits_copy.shape(), 0.0f, false);
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < C; ++j) {
                        float g = probs.data()[i * C + j];
                        if (j == targets[i]) g -= 1.0f;
                        dX.data()[i * C + j] = g * scale;
                    }
                }
                logits_copy.add_grad(dX);
            };
    }
    return loss;
}

// ---------------------------------------------------------
// Métricas
// ---------------------------------------------------------

std::vector<size_t> argmax_rows(const Tensor& logits) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("argmax_rows espera un tensor 2D, recibió " + logits.shape_str() + ".");
    }
    const size_t N = logits.shape()[0];
    const size_t C = logits.shape()[1];

    std::vector<size_t> result(N, 0);
    for (size_t i = 0; i < N; ++i) {
        const float* row = logits.data().data() + i * C;
        result[i] = static_cast<size_t>(std::max_element(row, row + C) - row);
    }
    return result;
}

float accuracy(const Tensor& logits, const std::vector<size_t>& targets) {
    std::vector<size_t> predictions = argmax_rows(logits);
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("accuracy recibió un número de etiquetas distinto al de predicciones.");
    }
    if (targets.empty()) return 0.0f;

    size_t hits = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (predictions[i] == targets[i]) ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(targets.size());
}

} // namespace nn
} // namespace engine
