#include "engine/optim.hpp"
#include "engine/autograd.hpp"

#include <algorithm>
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

// ---------------------------------------------------------
// Recorte de gradiente
// ---------------------------------------------------------

float clip_grad_norm(const std::vector<Tensor>& parameters, float max_norm) {
    if (max_norm <= 0.0f) {
        throw std::invalid_argument("clip_grad_norm necesita un max_norm positivo.");
    }
    autograd::NoGradGuard no_grad;

    // La norma es global, sobre todos los parámetros juntos: recortar cada uno
    // por separado cambiaría la dirección del paso, no solo su longitud.
    double total = 0.0;
    for (const Tensor& p : parameters) {
        if (!p.has_grad()) continue;
        for (float g : p.grad().data()) total += static_cast<double>(g) * g;
    }
    const float norm = static_cast<float>(std::sqrt(total));

    if (norm > max_norm) {
        const float scale = max_norm / (norm + 1e-6f);
        for (const Tensor& p : parameters) {
            if (!p.has_grad()) continue;
            Tensor g = p.grad();
            for (float& v : g.data()) v *= scale;
        }
    }
    return norm;
}

// ---------------------------------------------------------
// Planificadores
// ---------------------------------------------------------

Scheduler::Scheduler(Optimizer& optimizer)
    : optimizer_(optimizer), base_lr_(optimizer.learning_rate()) {}

void Scheduler::step() {
    ++epoch_;
    optimizer_.set_learning_rate(compute(epoch_));
}

StepLR::StepLR(Optimizer& optimizer, size_t step_size, float gamma)
    : Scheduler(optimizer), step_size_(step_size), gamma_(gamma) {
    if (step_size == 0) throw std::invalid_argument("StepLR necesita un step_size positivo.");
    if (gamma <= 0.0f) throw std::invalid_argument("StepLR necesita un gamma positivo.");
}

float StepLR::compute(size_t epoch) const {
    return base_lr_ * std::pow(gamma_, static_cast<float>(epoch / step_size_));
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, size_t total_epochs, float min_lr)
    : Scheduler(optimizer), total_epochs_(total_epochs), min_lr_(min_lr) {
    if (total_epochs == 0) {
        throw std::invalid_argument("CosineAnnealingLR necesita al menos una época.");
    }
    if (min_lr < 0.0f) throw std::invalid_argument("CosineAnnealingLR necesita min_lr >= 0.");
}

float CosineAnnealingLR::compute(size_t epoch) const {
    const float t = std::min(static_cast<float>(epoch) / static_cast<float>(total_epochs_), 1.0f);
    const float cosine = 0.5f * (1.0f + std::cos(3.14159265358979f * t));
    return min_lr_ + (base_lr_ - min_lr_) * cosine;
}

WarmupCosineLR::WarmupCosineLR(Optimizer& optimizer, size_t warmup_epochs,
                               size_t total_epochs, float min_lr)
    : Scheduler(optimizer), warmup_epochs_(warmup_epochs),
      total_epochs_(total_epochs), min_lr_(min_lr) {
    if (total_epochs == 0) {
        throw std::invalid_argument("WarmupCosineLR necesita al menos una época.");
    }
    if (warmup_epochs >= total_epochs) {
        throw std::invalid_argument("El calentamiento debe ser más corto que el total de épocas.");
    }
}

float WarmupCosineLR::compute(size_t epoch) const {
    if (epoch < warmup_epochs_) {
        // Rampa lineal repartida en warmup_epochs pasos: tras el último, la
        // época `warmup_epochs` ya cae en la rama del coseno con t = 0, que
        // vale exactamente el learning rate base.
        return base_lr_ * static_cast<float>(epoch) / static_cast<float>(warmup_epochs_);
    }
    const float t = static_cast<float>(epoch - warmup_epochs_) /
                    static_cast<float>(total_epochs_ - warmup_epochs_);
    const float cosine = 0.5f * (1.0f + std::cos(3.14159265358979f * std::min(t, 1.0f)));
    return min_lr_ + (base_lr_ - min_lr_) * cosine;
}

} // namespace optim
} // namespace engine
