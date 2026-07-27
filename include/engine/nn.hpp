#ifndef ENGINE_NN_HPP
#define ENGINE_NN_HPP

#include "engine/tensor.hpp"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// Abstracción base de todas las capas
// ---------------------------------------------------------
class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    // Azúcar sintáctico: modulo(x) equivale a modulo.forward(x)
    Tensor operator()(const Tensor& input) { return forward(input); }

    // Parámetros entrenables de la capa (vacío para capas sin estado).
    // Los Tensor devueltos comparten la implementación interna, así que
    // escribir en ellos actualiza los pesos de la capa.
    virtual std::vector<Tensor> parameters() { return {}; }

    virtual std::string name() const { return "Module"; }

    // Pone a cero los gradientes acumulados de todos los parámetros
    void zero_grad();

    // Número total de escalares entrenables
    size_t num_parameters();
};

// ---------------------------------------------------------
// Capa densa (totalmente conectada): y = x · W + b
// ---------------------------------------------------------
class Linear : public Module {
public:
    // Inicialización Xavier/Glorot uniforme: U(-limit, limit) con
    // limit = sqrt(6 / (in_features + out_features)).
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    Tensor weight_; // (in_features, out_features)
    Tensor bias_;   // (1, out_features)
};

// ---------------------------------------------------------
// Activaciones (sin parámetros)
// ---------------------------------------------------------
class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "ReLU"; }
};

class Softmax : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Softmax"; }
};

// ---------------------------------------------------------
// Contenedor secuencial de capas
// ---------------------------------------------------------
class Sequential : public Module {
public:
    Sequential() = default;
    Sequential(std::initializer_list<std::shared_ptr<Module>> layers);

    Sequential& add(std::shared_ptr<Module> layer);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override { return "Sequential"; }

    void summary() const;
    size_t size() const { return layers_.size(); }

private:
    std::vector<std::shared_ptr<Module>> layers_;
};

// Ayudas de construcción
template <typename T, typename... Args>
std::shared_ptr<T> make(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

// ---------------------------------------------------------
// Funciones de pérdida
// ---------------------------------------------------------

// Error cuadrático medio: mean((pred - target)^2)
Tensor mse_loss(const Tensor& prediction, const Tensor& target);

// Entropía cruzada sobre logits sin normalizar, con las clases correctas dadas
// como índices enteros. Fusiona log-softmax y NLL en un solo nodo del grafo:
// es más estable numéricamente que encadenar Softmax + log, y su gradiente se
// reduce a (softmax(logits) - one_hot) / N.
//
//   logits  : (N, C) sin normalizar
//   targets : N índices de clase en [0, C)
Tensor cross_entropy_loss(const Tensor& logits, const std::vector<size_t>& targets);

class CrossEntropyLoss {
public:
    Tensor forward(const Tensor& logits, const std::vector<size_t>& targets) const {
        return cross_entropy_loss(logits, targets);
    }
    Tensor operator()(const Tensor& logits, const std::vector<size_t>& targets) const {
        return forward(logits, targets);
    }
};

// Índice de la clase de mayor puntuación por fila (para calcular la exactitud)
std::vector<size_t> argmax_rows(const Tensor& logits);
float accuracy(const Tensor& logits, const std::vector<size_t>& targets);

} // namespace nn
} // namespace engine

#endif // ENGINE_NN_HPP
