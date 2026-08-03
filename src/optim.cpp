#include "engine/optim.hpp"
#include "engine/detail/cuda_ops.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"

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
    if (lr <= 0.0f) throw std::invalid_argument("SGD requires a positive learning rate.");
    lr_ = lr;

    if (momentum_ != 0.0f) {
        velocity_.reserve(params_.size());
        for (const Tensor& p : params_) {
            velocity_.emplace_back(p.size(), 0.0f);
        }
    }
}

void SGD::step() {
    // The weight update must never become part of the graph
    autograd::NoGradGuard no_grad;

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.requires_grad() || !p.has_grad()) continue;

        // The gradient Tensor is held in a variable: taking the reference straight
        // from p.grad().data() would point into a temporary.
        //
        Tensor grad_tensor = p.grad();
        Storage& param = p.storage();
        Storage& grad = grad_tensor.storage();
        Storage* vel = velocity_.empty() ? nullptr : &velocity_[i];

        // Offered to the device before anything reads .data(). That ordering is
        // the whole point: one .data() here pulls the entire model down, and the
        // next forward pushes it straight back up.
        if (cuda::ops::sgd_step(param, grad, vel, lr_, momentum_, weight_decay_)) continue;

        const std::vector<float>& g = grad.host();
        std::vector<float>& w = param.host_mut();

        // Each weight is updated with its own gradient and its own velocity: there is
        // no reduction, so splitting does not change the result.
        const float* ENGINE_RESTRICT gp = g.data();
        float* ENGINE_RESTRICT wp = w.data();
        float* ENGINE_RESTRICT vp =
            (vel != nullptr && momentum_ != 0.0f) ? vel->host_mut().data() : nullptr;
        parallel::parallel_for(w.size(), parallel::kElementsPerThread, [&](size_t from, size_t to) {
            for (size_t j = from; j < to; ++j) {
                float grad = gp[j];
                if (weight_decay_ != 0.0f) grad += weight_decay_ * wp[j];

                if (vp != nullptr) {
                    vp[j] = momentum_ * vp[j] + grad;
                    grad = vp[j];
                }
                wp[j] -= lr_ * grad;
            }
        });
    }
}

// ---------------------------------------------------------
// Adam
// ---------------------------------------------------------

Adam::Adam(std::vector<Tensor> parameters, float lr, float beta1, float beta2, float eps,
           float weight_decay)
    : Optimizer(std::move(parameters)),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay) {
    if (lr <= 0.0f) throw std::invalid_argument("Adam requires a positive learning rate.");
    if (beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("Adam requires beta1 and beta2 in the interval [0, 1).");
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
    // Bias correction: without it the first steps would be far too small, because m
    // and v start at zero.
    const float bias_c1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias_c2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.requires_grad() || !p.has_grad()) continue;

        // The gradient Tensor is held in a variable: taking the reference straight
        // from p.grad().data() would point into a temporary.
        //
        Tensor grad_tensor = p.grad();
        Storage& param = p.storage();
        Storage& grad = grad_tensor.storage();

        if (cuda::ops::adam_step(param, grad, m_[i], v_[i], lr_, beta1_, beta2_, eps_,
                                 weight_decay_, bias_c1, bias_c2)) {
            continue;
        }

        const std::vector<float>& g = grad.host();
        std::vector<float>& w = param.host_mut();

        // As in SGD: each weight carries its own moment and its own variance,
        // and nothing crosses between indices.
        const float* ENGINE_RESTRICT gp = g.data();
        float* ENGINE_RESTRICT wp = w.data();
        float* ENGINE_RESTRICT mp = m_[i].host_mut().data();
        float* ENGINE_RESTRICT vp = v_[i].host_mut().data();
        parallel::parallel_for(w.size(), parallel::kElementsPerThread, [&](size_t from, size_t to) {
            for (size_t j = from; j < to; ++j) {
                float grad = gp[j];
                if (weight_decay_ != 0.0f) grad += weight_decay_ * wp[j];

                mp[j] = beta1_ * mp[j] + (1.0f - beta1_) * grad;
                vp[j] = beta2_ * vp[j] + (1.0f - beta2_) * grad * grad;

                const float m_hat = mp[j] / bias_c1;
                const float v_hat = vp[j] / bias_c2;

                wp[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        });
    }
}

// ---------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------

float clip_grad_norm(const std::vector<Tensor>& parameters, float max_norm) {
    if (max_norm <= 0.0f) {
        throw std::invalid_argument("clip_grad_norm needs a positive max_norm.");
    }
    autograd::NoGradGuard no_grad;

    // The norm is global, over all parameters together: clipping each separately
    // would change the step's direction, not just its length.
    // Reduced on the device when the gradient is already there, per parameter,
    // and only the resulting double comes back. Reading .data() here instead is
    // what used to pull the entire model's gradients down every single step --
    // and, worse, left them host-resident, so the optimiser kernel that runs
    // immediately afterwards found nothing to work on and never fired at all.
    // One host-side reduction in the middle of a step poisons everything after
    // it.
    double total = 0.0;
    for (const Tensor& p : parameters) {
        if (!p.has_grad()) continue;
        Tensor g = p.grad();
        double partial = 0.0;
        if (cuda::ops::reduce_sum_squares(g.storage(), partial)) {
            total += partial;
            continue;
        }
        const float* ENGINE_RESTRICT gp = g.data();
        for (size_t i = 0; i < g.size(); ++i) total += static_cast<double>(gp[i]) * gp[i];
    }
    const float norm = static_cast<float>(std::sqrt(total));

    if (norm > max_norm) {
        const float scale = max_norm / (norm + 1e-6f);
        for (const Tensor& p : parameters) {
            if (!p.has_grad()) continue;
            Tensor g = p.grad();
            if (cuda::ops::scale_in_place(g.storage(), scale)) continue;
            float* ENGINE_RESTRICT gp = g.data();
            for (size_t i = 0; i < g.size(); ++i) gp[i] *= scale;
        }
    }
    return norm;
}

// ---------------------------------------------------------
// Schedulers
// ---------------------------------------------------------

Scheduler::Scheduler(Optimizer& optimizer)
    : optimizer_(optimizer), base_lr_(optimizer.learning_rate()) {}

void Scheduler::step() {
    ++epoch_;
    optimizer_.set_learning_rate(compute(epoch_));
}

StepLR::StepLR(Optimizer& optimizer, size_t step_size, float gamma)
    : Scheduler(optimizer), step_size_(step_size), gamma_(gamma) {
    if (step_size == 0) throw std::invalid_argument("StepLR needs a positive step_size.");
    if (gamma <= 0.0f) throw std::invalid_argument("StepLR needs a positive gamma.");
}

float StepLR::compute(size_t epoch) const {
    // The integer division is intentional: the learning rate drops in steps rather
    // than continuously. With step_size = 2, epochs 0 and 1 give step 0, epochs 2
    // and 3 step 1, and so on. Computed separately to make that explicit.
    const size_t completed_steps = epoch / step_size_;
    return base_lr_ * std::pow(gamma_, static_cast<float>(completed_steps));
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, size_t total_epochs, float min_lr)
    : Scheduler(optimizer), total_epochs_(total_epochs), min_lr_(min_lr) {
    if (total_epochs == 0) {
        throw std::invalid_argument("CosineAnnealingLR needs at least one epoch.");
    }
    if (min_lr < 0.0f) throw std::invalid_argument("CosineAnnealingLR needs min_lr >= 0.");
}

float CosineAnnealingLR::compute(size_t epoch) const {
    const float t = std::min(static_cast<float>(epoch) / static_cast<float>(total_epochs_), 1.0f);
    const float cosine = 0.5f * (1.0f + std::cos(3.14159265358979f * t));
    return min_lr_ + (base_lr_ - min_lr_) * cosine;
}

WarmupCosineLR::WarmupCosineLR(Optimizer& optimizer, size_t warmup_epochs, size_t total_epochs,
                               float min_lr)
    : Scheduler(optimizer),
      warmup_epochs_(warmup_epochs),
      total_epochs_(total_epochs),
      min_lr_(min_lr) {
    if (total_epochs == 0) {
        throw std::invalid_argument("WarmupCosineLR needs at least one epoch.");
    }
    if (warmup_epochs >= total_epochs) {
        throw std::invalid_argument("The warm-up must be shorter than the total number of epochs.");
    }
}

float WarmupCosineLR::compute(size_t epoch) const {
    if (epoch < warmup_epochs_) {
        // A linear ramp spread over warmup_epochs steps: after the last one, epoch
        // `warmup_epochs` already falls in the cosine branch with t = 0, which is
        // exactly the base learning rate.
        return base_lr_ * static_cast<float>(epoch) / static_cast<float>(warmup_epochs_);
    }
    const float t = static_cast<float>(epoch - warmup_epochs_) /
                    static_cast<float>(total_epochs_ - warmup_epochs_);
    const float cosine = 0.5f * (1.0f + std::cos(3.14159265358979f * std::min(t, 1.0f)));
    return min_lr_ + (base_lr_ - min_lr_) * cosine;
}

}  // namespace optim
}  // namespace engine
