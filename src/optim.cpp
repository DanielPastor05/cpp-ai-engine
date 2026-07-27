#include "engine/optim.hpp"
#include "engine/autograd.hpp"

#include <cmath>
#include <stdexcept>

namespace engine {
namespace optim {

// ---------------------------------------------------------
// Optimizer
// ---------------------------------------------------------

Optimizer::Optimizer(std::vector<Tensor> parameters) : params_(std::move(parameters)) {}

void Optimizer::zero_grad() {
    for (Tensor& p : params_) {
        p.zero_grad();
    }
}

// ---------------------------------------------------------
// SGD
// ---------------------------------------------------------

SGD::SGD(std::vector<Tensor> parameters, float lr, float momentum, float weight_decay)
    : Optimizer(std::move(parameters)), momentum_(momentum), weight_decay_(weight_decay) {
    if (lr <= 0.0f) throw std::invalid_argument("SGD requiere un learning rate positivo.");
    lr_ = lr;

    if (momentum_ != 0.0f) {
        velocity_.reserve(params_.size());
        for (const Tensor& p : params_) {
            velocity_.emplace_back(p.size(), 0.0f);
        }
    }
}

void SGD::step() {
    // La actualización de pesos nunca debe formar parte del grafo
    autograd::NoGradGuard no_grad;

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.requires_grad() || !p.has_grad()) continue;

        // Se conserva el Tensor del gradiente en una variable: tomar la
        // referencia directamente de p.grad().data() apuntaría al interior de
        // un temporal.
        Tensor grad_tensor = p.grad();
        const std::vector<float>& g = grad_tensor.data();
        std::vector<float>& w = p.data();

        for (size_t j = 0; j < w.size(); ++j) {
            float grad = g[j];
            if (weight_decay_ != 0.0f) grad += weight_decay_ * w[j];

            if (momentum_ != 0.0f) {
                velocity_[i][j] = momentum_ * velocity_[i][j] + grad;
                grad = velocity_[i][j];
            }
            w[j] -= lr_ * grad;
        }
    }
}

// ---------------------------------------------------------
// Adam
// ---------------------------------------------------------

Adam::Adam(std::vector<Tensor> parameters, float lr, float beta1, float beta2,
           float eps, float weight_decay)
    : Optimizer(std::move(parameters)), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {
    if (lr <= 0.0f) throw std::invalid_argument("Adam requiere un learning rate positivo.");
    if (beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("Adam requiere beta1 y beta2 en el intervalo [0, 1).");
    }
    lr_ = lr;

    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (const Tensor& p : params_) {
        m_.emplace_back(p.size(), 0.0f);
        v_.emplace_back(p.size(), 0.0f);
    }
}

void Adam::step() {
    autograd::NoGradGuard no_grad;

    ++t_;
    // Corrección de sesgo: sin ella los primeros pasos serían demasiado pequeños,
    // porque m y v arrancan en cero.
    const float bias_c1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias_c2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.requires_grad() || !p.has_grad()) continue;

        // Se conserva el Tensor del gradiente en una variable: tomar la
        // referencia directamente de p.grad().data() apuntaría al interior de
        // un temporal.
        Tensor grad_tensor = p.grad();
        const std::vector<float>& g = grad_tensor.data();
        std::vector<float>& w = p.data();

        for (size_t j = 0; j < w.size(); ++j) {
            float grad = g[j];
            if (weight_decay_ != 0.0f) grad += weight_decay_ * w[j];

            m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * grad;
            v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * grad * grad;

            const float m_hat = m_[i][j] / bias_c1;
            const float v_hat = v_[i][j] / bias_c2;

            w[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

} // namespace optim
} // namespace engine
