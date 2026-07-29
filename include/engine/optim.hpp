#ifndef ENGINE_OPTIM_HPP
#define ENGINE_OPTIM_HPP

#include "engine/tensor.hpp"

#include <vector>

namespace engine {
namespace optim {

// ---------------------------------------------------------
// The interface every optimiser shares
// ---------------------------------------------------------
class Optimizer {
public:
    // Parameters are captured here and stored by value. Because Tensor is a
    // handle, writing to them updates the layer's weights; but if a parameter
    // is *reassigned* afterwards (layer.weight() = other_tensor), the optimiser
    // will still point at the old tensor. Set the weights up before
    // constructing the optimiser.
    explicit Optimizer(std::vector<Tensor> parameters);
    virtual ~Optimizer() = default;

    // Applies one update to every parameter that has an accumulated gradient
    virtual void step() = 0;

    // Zeroes the accumulated gradients (they must be cleared every iteration,
    // because add_grad accumulates rather than overwrites)
    void zero_grad();

    float learning_rate() const { return lr_; }
    void set_learning_rate(float lr) { lr_ = lr; }

    const std::vector<Tensor>& parameters() const { return params_; }

protected:
    std::vector<Tensor> params_;
    float lr_ = 0.01f;
};

// ---------------------------------------------------------
// Stochastic gradient descent, with optional momentum
//
//   v = momentum * v + g            (with weight_decay: g += wd * p)
//   p = p - lr * v
// ---------------------------------------------------------
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> parameters, float lr = 0.01f, float momentum = 0.0f,
        float weight_decay = 0.0f);

    void step() override;

private:
    float momentum_;
    float weight_decay_;
    std::vector<std::vector<float>> velocity_;
};

// ---------------------------------------------------------
// Adam (Adaptive Moment Estimation)
//
//   m = b1 * m + (1 - b1) * g          v = b2 * v + (1 - b2) * g^2
//   m_hat = m / (1 - b1^t)             v_hat = v / (1 - b2^t)
//   p = p - lr * m_hat / (sqrt(v_hat) + eps)
// ---------------------------------------------------------
class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> parameters, float lr = 0.001f, float beta1 = 0.9f,
         float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);

    void step() override;

    // Number of steps applied (used by the bias correction)
    size_t steps() const { return t_; }

private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    size_t t_ = 0;
    std::vector<std::vector<float>> m_;  // primer momento
    std::vector<std::vector<float>> v_;  // segundo momento
};

// ---------------------------------------------------------
// Gradient clipping by global norm
//
// If the L2 norm of all gradients taken together exceeds max_norm, they are all
// scaled by the same factor. It is what stops an outlier batch from taking a
// huge step and derailing training; with Transformers it is standard practice.
// Returns the norm as it was BEFORE clipping, so it can be monitored.
// ---------------------------------------------------------
float clip_grad_norm(const std::vector<Tensor>& parameters, float max_norm);

// ---------------------------------------------------------
// Learning-rate schedulers
//
// They wrap an optimiser and adjust its learning rate. The scheduler's step()
// is called once per epoch, after the optimisation steps.
// ---------------------------------------------------------
class Scheduler {
public:
    explicit Scheduler(Optimizer& optimizer);
    virtual ~Scheduler() = default;

    // Advances one epoch and applies the new learning rate
    void step();

    size_t epoch() const { return epoch_; }
    float base_learning_rate() const { return base_lr_; }

protected:
    // The learning rate for a given epoch (0 is the initial one)
    virtual float compute(size_t epoch) const = 0;

    Optimizer& optimizer_;
    float base_lr_;
    size_t epoch_ = 0;
};

// Multiplies the learning rate by gamma every step_size epochs.
class StepLR : public Scheduler {
public:
    StepLR(Optimizer& optimizer, size_t step_size, float gamma = 0.1f);

protected:
    float compute(size_t epoch) const override;

private:
    size_t step_size_;
    float gamma_;
};

// Decays along a cosine from the initial learning rate down to min_lr over
// total_epochs. It descends slowly at the start and at the end, which usually
// works better than a linear decay.
class CosineAnnealingLR : public Scheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, size_t total_epochs, float min_lr = 0.0f);

protected:
    float compute(size_t epoch) const override;

private:
    size_t total_epochs_;
    float min_lr_;
};

// Ramps up linearly from near zero over warmup_epochs and then decays along a
// cosine. It is the usual recipe for training Transformers.
class WarmupCosineLR : public Scheduler {
public:
    WarmupCosineLR(Optimizer& optimizer, size_t warmup_epochs, size_t total_epochs,
                   float min_lr = 0.0f);

protected:
    float compute(size_t epoch) const override;

private:
    size_t warmup_epochs_;
    size_t total_epochs_;
    float min_lr_;
};

}  // namespace optim
}  // namespace engine

#endif  // ENGINE_OPTIM_HPP
