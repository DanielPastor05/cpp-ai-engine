#include "engine/nn.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"
#include "engine/random.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace engine {
namespace nn {

namespace {

// Constants of the GELU approximation. They live at file scope rather than
// inside the function: MSVC requires them captured explicitly if they are local
// and used from a lambda with no default capture, even as constant expressions.
constexpr float kGeluAlpha = 0.7978845608f;  // sqrt(2/pi)
constexpr float kGeluBeta = 0.044715f;

// Registers an element-wise activation whose derivative can be written as a
// function of the input and the output: d/dx = f(x, y).
template <typename Derivative>
Tensor unary_with_grad(const Tensor& input, Tensor out, Derivative derivative) {
    if (!autograd::grad_enabled() || !input.requires_grad()) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = {input.get_impl()};
    Tensor input_copy = input;
    Tensor saved = out.detach();

    out.get_impl()->backward_fn = [input_copy, saved, derivative](const Tensor& grad_out) mutable {
        Tensor dX(input_copy.shape(), 0.0f, false);
        const size_t n = dX.size();
        const float* ENGINE_RESTRICT g = grad_out.data().data();
        const float* ENGINE_RESTRICT x = input_copy.data().data();
        const float* ENGINE_RESTRICT y = saved.data().data();
        float* ENGINE_RESTRICT dx = dX.data().data();
        parallel::parallel_for(n, parallel::kElementsPerThread, [&](size_t from, size_t to) {
            for (size_t i = from; i < to; ++i) dx[i] = g[i] * derivative(x[i], y[i]);
        });
        input_copy.add_grad(dX);
    };
    return out;
}

}  // namespace

// ---------------------------------------------------------
// Module
// ---------------------------------------------------------

void Module::zero_grad() {
    for (Tensor& p : parameters()) {
        p.zero_grad();
    }
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters(const std::string& prefix) {
    // By default they are numbered in the order the layer declares them
    std::vector<std::pair<std::string, Tensor>> named;
    std::vector<Tensor> params = parameters();
    named.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        named.emplace_back(prefix + name() + "." + std::to_string(i), params[i]);
    }
    return named;
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
        throw std::invalid_argument(
            "Linear requires in_features and out_features greater than zero.");
    }

    // Uniform Xavier/Glorot initialisation: it keeps the variance of the
    // activations stable through the depth of the network.
    const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));

    autograd::NoGradGuard no_grad;
    weight_ = Tensor::rand({in_features, out_features}, -limit, limit, true);
    bias_ = Tensor({1, out_features}, 0.0f, use_bias);
}

Tensor Linear::forward(const Tensor& input) {
    if (input.ndim() < 2) {
        throw std::invalid_argument(
            "Linear expects at least 2 dimensions (..., in_features), received " +
            input.shape_str() + ".");
    }
    if (input.shape().back() != in_features_) {
        throw std::invalid_argument("Linear with in_features=" + std::to_string(in_features_) +
                                    " received an input " + input.shape_str() + ".");
    }

    // The layer acts on the last axis. With inputs of more than 2 axes -- like
    // (batch, sequence, d_model) in a Transformer -- the leading axes are
    // flattened, the projection is applied, and the shape is restored.
    const bool needs_reshape = input.ndim() > 2;
    Tensor flat =
        needs_reshape ? input.reshape({input.size() / in_features_, in_features_}) : input;

    Tensor out = flat.matmul(weight_);
    if (use_bias_) {
        out = out + bias_;  // broadcast the row vector over the batch
    }

    if (needs_reshape) {
        std::vector<size_t> out_shape = input.shape();
        out_shape.back() = out_features_;
        out = out.reshape(out_shape);
    }
    return out;
}

std::vector<Tensor> Linear::parameters() {
    if (use_bias_) return {weight_, bias_};
    return {weight_};
}

std::string Linear::name() const {
    return "Linear(" + std::to_string(in_features_) + " -> " + std::to_string(out_features_) +
           (use_bias_ ? ", bias=true)" : ", bias=false)");
}

// ---------------------------------------------------------
// Activations
// ---------------------------------------------------------

Tensor ReLU::forward(const Tensor& input) {
    return input.relu();
}

Tensor Softmax::forward(const Tensor& input) {
    return input.softmax();
}

Tensor Sigmoid::forward(const Tensor& input) {
    // sigma(x) = 1 / (1 + e^-x), computed stably at both extremes
    Tensor out(input.shape(), 0.0f, false);
    const size_t n = input.size();
    const float* ENGINE_RESTRICT x = input.data().data();
    float* ENGINE_RESTRICT y = out.data().data();
    parallel::parallel_for(n, parallel::kElementsPerThread, [&](size_t from, size_t to) {
        for (size_t i = from; i < to; ++i) {
            y[i] = (x[i] >= 0.0f) ? 1.0f / (1.0f + std::exp(-x[i]))
                                  : std::exp(x[i]) / (1.0f + std::exp(x[i]));
        }
    });
    return unary_with_grad(input, out, [](float, float v) { return v * (1.0f - v); });
}

Tensor Tanh::forward(const Tensor& input) {
    Tensor out(input.shape(), 0.0f, false);
    const size_t n = input.size();
    const float* ENGINE_RESTRICT x = input.data().data();
    float* ENGINE_RESTRICT y = out.data().data();
    parallel::parallel_for(n, parallel::kElementsPerThread, [&](size_t from, size_t to) {
        for (size_t i = from; i < to; ++i) y[i] = std::tanh(x[i]);
    });
    return unary_with_grad(input, out, [](float, float v) { return 1.0f - v * v; });
}

Tensor GELU::forward(const Tensor& input) {
    // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
    Tensor out(input.shape(), 0.0f, false);
    const size_t n = input.size();
    const float* ENGINE_RESTRICT x = input.data().data();
    float* ENGINE_RESTRICT y = out.data().data();
    parallel::parallel_for(n, parallel::kElementsPerThread, [&](size_t from, size_t to) {
        for (size_t i = from; i < to; ++i) {
            y[i] = 0.5f * x[i] *
                   (1.0f + std::tanh(kGeluAlpha * (x[i] + kGeluBeta * x[i] * x[i] * x[i])));
        }
    });
    return unary_with_grad(input, out, [](float x, float) {
        const float inner = kGeluAlpha * (x + kGeluBeta * x * x * x);
        const float t = std::tanh(inner);
        const float dinner = kGeluAlpha * (1.0f + 3.0f * kGeluBeta * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
    });
}

Dropout::Dropout(float p) : p_(p) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("Dropout needs a probability in [0, 1).");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    // At inference it is the identity: the network must be deterministic when evaluating
    if (!is_training() || p_ == 0.0f) return input;

    const float keep = 1.0f - p_;
    const float scale = 1.0f / keep;
    std::bernoulli_distribution alive(keep);

    // The mask is generated here and reused in the derivative: the gradient
    // has to zero exactly the same positions.
    auto mask = std::make_shared<std::vector<float>>(input.size(), 0.0f);
    Tensor out(input.shape(), 0.0f, false);
    // This loop is NOT split, even though it is the same element-wise shape as the
    // activations above. global_rng() is a shared, unsynchronised mt19937: putting
    // it through parallel_for introduces a silent race, and on top of that the
    // result would stop being reproducible, because each thread would consume
    // numbers in an order that depends on how the split falls. Parallelising it
    // would need one generator per thread, seeded deterministically, not a
    // parallel_for wrapped around it.
    const size_t n = input.size();
    const float* ENGINE_RESTRICT x = input.data().data();
    float* ENGINE_RESTRICT y = out.data().data();
    for (size_t i = 0; i < n; ++i) {
        (*mask)[i] = alive(global_rng()) ? scale : 0.0f;
        y[i] = x[i] * (*mask)[i];
    }

    if (!autograd::grad_enabled() || !input.requires_grad()) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = {input.get_impl()};
    Tensor input_copy = input;
    out.get_impl()->backward_fn = [input_copy, mask](const Tensor& grad_out) mutable {
        Tensor dX(input_copy.shape(), 0.0f, false);
        for (size_t i = 0; i < dX.size(); ++i) dX.data()[i] = grad_out.data()[i] * (*mask)[i];
        input_copy.add_grad(dX);
    };
    return out;
}

std::string Dropout::name() const {
    return "Dropout(p=" + std::to_string(p_).substr(0, 4) + ")";
}

// ---------------------------------------------------------
// Sequential
// ---------------------------------------------------------

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> layers) : layers_(layers) {
    for (const auto& layer : layers_) {
        if (!layer) throw std::invalid_argument("Sequential does not accept null layers.");
    }
}

Sequential& Sequential::add(std::shared_ptr<Module> layer) {
    if (!layer) throw std::invalid_argument("Sequential does not accept null layers.");
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

Module& Sequential::at(size_t index) {
    if (index >= layers_.size()) {
        throw std::out_of_range("Sequential: no layer at index " + std::to_string(index) + ".");
    }
    return *layers_[index];
}

void Sequential::train(bool mode) {
    Module::train(mode);
    for (const auto& layer : layers_) layer->train(mode);
}

std::vector<std::pair<std::string, Tensor>> Sequential::named_parameters(
    const std::string& prefix) {
    std::vector<std::pair<std::string, Tensor>> named;
    for (size_t i = 0; i < layers_.size(); ++i) {
        // The index goes in the name so that two identical layers do not collide
        std::vector<std::pair<std::string, Tensor>> sub =
            layers_[i]->named_parameters(prefix + std::to_string(i) + ".");
        named.insert(named.end(), sub.begin(), sub.end());
    }
    return named;
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
    std::cout << "Sequential (" << layers_.size() << " layers)\n";
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
// Loss functions
// ---------------------------------------------------------

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    Tensor diff = prediction - target;
    return (diff * diff).mean();
}

Tensor cross_entropy_loss(const Tensor& logits, const std::vector<size_t>& targets) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument(
            "cross_entropy_loss expects 2D logits (batch, classes), received " +
            logits.shape_str() + ".");
    }
    const size_t N = logits.shape()[0];
    const size_t C = logits.shape()[1];

    if (targets.size() != N) {
        throw std::invalid_argument("cross_entropy_loss received " +
                                    std::to_string(targets.size()) + " labels for a batch of " +
                                    std::to_string(N) + ".");
    }
    if (N == 0) {
        throw std::invalid_argument("cross_entropy_loss received an empty batch.");
    }

    // Row-stable softmax and the batch's mean loss in a single pass.
    Tensor probs(logits.shape(), 0.0f, false);
    float total_loss = 0.0f;

    // Serial for two reasons, either of which would be enough: total_loss is a
    // reduction over the rows, and the loop throws when a label falls out of range
    // -- crossing an exception out of a parallel region is a whole other problem.
    // The backward, which has neither, is split.
    //
    // ponytail: serial; separate validation from reduction if the profile asks.
    for (size_t i = 0; i < N; ++i) {
        if (targets[i] >= C) {
            throw std::out_of_range("Etiqueta " + std::to_string(targets[i]) +
                                    " outside the range of " + std::to_string(C) + " classes.");
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

        // shifted log-sum-exp: log p_y = (x_y - max) - log(sum exp(x - max))
        const float log_prob = (row[targets[i]] - max_v) - std::log(sum_exp);
        total_loss += -log_prob;
    }

    const bool req_g = logits.requires_grad() && autograd::grad_enabled();
    Tensor loss({1}, std::vector<float>{total_loss / static_cast<float>(N)}, req_g);

    if (req_g) {
        loss.get_impl()->parents = {logits.get_impl()};
        Tensor logits_copy = logits;

        // add_grad throws when the gradient's shape does not match: that is the
        // intended report rather than a fault to swallow here.
        loss.get_impl()->backward_fn = [logits_copy, probs, targets, N,
                                        C](const Tensor& grad_out) mutable {
            // dL/dlogits = (softmax(logits) - one_hot(y)) / N
            const float scale = grad_out.data()[0] / static_cast<float>(N);
            Tensor dX(logits_copy.shape(), 0.0f, false);
            // This one is split, unlike the forward: it is a per-element
            // write and accumulates nothing.
            const float* ENGINE_RESTRICT pr = probs.data().data();
            float* ENGINE_RESTRICT dx = dX.data().data();
            const size_t rows_per_thread =
                std::max<size_t>(1, parallel::kElementsPerThread / std::max<size_t>(1, C));
            parallel::parallel_for(N, rows_per_thread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) {
                    for (size_t j = 0; j < C; ++j) {
                        float g = pr[i * C + j];
                        if (j == targets[i]) g -= 1.0f;
                        dx[i * C + j] = g * scale;
                    }
                }
            });
            logits_copy.add_grad(dX);
        };
    }
    return loss;
}

// ---------------------------------------------------------
// Metrics
// ---------------------------------------------------------

std::vector<size_t> argmax_rows(const Tensor& logits) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("argmax_rows expects a 2D tensor, received " +
                                    logits.shape_str() + ".");
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
        throw std::invalid_argument(
            "accuracy received a different number of labels than predictions.");
    }
    if (targets.empty()) return 0.0f;

    size_t hits = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (predictions[i] == targets[i]) ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(targets.size());
}

}  // namespace nn
}  // namespace engine
