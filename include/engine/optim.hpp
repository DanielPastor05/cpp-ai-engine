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
    // Los parámetros se capturan aquí y se guardan por valor. Como Tensor es
    // un handle, escribir en ellos actualiza los pesos de la capa; pero si se
    // *reasigna* un parámetro después (capa.weight() = otro_tensor), el
    // optimizador seguirá apuntando al tensor viejo. Configura los pesos antes
    // de construir el optimizador.
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

// ---------------------------------------------------------
// Recorte de gradiente por norma global
//
// Si la norma L2 de todos los gradientes juntos supera max_norm, se escalan
// todos por el mismo factor. Es lo que evita que un lote atípico dé un paso
// enorme y descarrile el entrenamiento; en Transformers es práctica estándar.
// Devuelve la norma que había ANTES de recortar, para poder monitorizarla.
// ---------------------------------------------------------
float clip_grad_norm(const std::vector<Tensor>& parameters, float max_norm);

// ---------------------------------------------------------
// Planificadores de learning rate
//
// Envuelven un optimizador y le ajustan el learning rate. La llamada a step()
// del planificador se hace una vez por época, después de las de la optimización.
// ---------------------------------------------------------
class Scheduler {
public:
    explicit Scheduler(Optimizer& optimizer);
    virtual ~Scheduler() = default;

    // Avanza una época y aplica el nuevo learning rate
    void step();

    size_t epoch() const { return epoch_; }
    float base_learning_rate() const { return base_lr_; }

protected:
    // Learning rate que corresponde a la época dada (0 es la inicial)
    virtual float compute(size_t epoch) const = 0;

    Optimizer& optimizer_;
    float base_lr_;
    size_t epoch_ = 0;
};

// Multiplica el learning rate por gamma cada step_size épocas.
class StepLR : public Scheduler {
public:
    StepLR(Optimizer& optimizer, size_t step_size, float gamma = 0.1f);

protected:
    float compute(size_t epoch) const override;

private:
    size_t step_size_;
    float gamma_;
};

// Desciende siguiendo un coseno desde el learning rate inicial hasta min_lr a
// lo largo de total_epochs. Baja despacio al principio y al final, que es lo
// que suele funcionar mejor que un descenso lineal.
class CosineAnnealingLR : public Scheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, size_t total_epochs, float min_lr = 0.0f);

protected:
    float compute(size_t epoch) const override;

private:
    size_t total_epochs_;
    float min_lr_;
};

// Sube linealmente desde casi cero durante warmup_epochs y luego desciende en
// coseno. Es la receta habitual para entrenar Transformers.
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

} // namespace optim
} // namespace engine

#endif // ENGINE_OPTIM_HPP
