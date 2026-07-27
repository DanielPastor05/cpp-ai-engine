#ifndef ENGINE_OPTIM_HPP
#define ENGINE_OPTIM_HPP

#include "engine/tensor.hpp"

#include <vector>

namespace engine {
namespace optim {

// ---------------------------------------------------------
// Interfaz común de los optimizadores
// ---------------------------------------------------------
class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> parameters);
    virtual ~Optimizer() = default;

    // Aplica una actualización a todos los parámetros con gradiente acumulado
    virtual void step() = 0;

    // Pone a cero los gradientes acumulados (deben limpiarse en cada iteración,
    // porque add_grad acumula en lugar de sobrescribir)
    void zero_grad();

    float learning_rate() const { return lr_; }
    void set_learning_rate(float lr) { lr_ = lr; }

    const std::vector<Tensor>& parameters() const { return params_; }

protected:
    std::vector<Tensor> params_;
    float lr_ = 0.01f;
};

// ---------------------------------------------------------
// Descenso de gradiente estocástico, con momento opcional
//
//   v = momentum * v + g            (con weight_decay: g += wd * p)
//   p = p - lr * v
// ---------------------------------------------------------
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> parameters, float lr = 0.01f,
        float momentum = 0.0f, float weight_decay = 0.0f);

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
    Adam(std::vector<Tensor> parameters, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f);

    void step() override;

    // Número de pasos aplicados (se usa en la corrección de sesgo)
    size_t steps() const { return t_; }

private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    size_t t_ = 0;
    std::vector<std::vector<float>> m_; // primer momento
    std::vector<std::vector<float>> v_; // segundo momento
};

} // namespace optim
} // namespace engine

#endif // ENGINE_OPTIM_HPP
