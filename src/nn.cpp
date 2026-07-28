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

// Constantes de la aproximación de GELU. Van en ámbito de fichero y no dentro
// de la función: MSVC exige capturarlas explícitamente si son locales y se usan
// desde una lambda sin captura por defecto, aunque sean expresiones constantes.
constexpr float kGeluAlpha = 0.7978845608f; // sqrt(2/pi)
constexpr float kGeluBeta = 0.044715f;

// Registra una activación elemento a elemento cuya derivada se puede escribir
// en función de la entrada y de la salida: d/dx = f(x, y).
template <typename Derivative>
Tensor unary_with_grad(const Tensor& input, Tensor out, Derivative derivative) {
    if (!autograd::grad_enabled() || !input.requires_grad()) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = { input.get_impl() };
    Tensor input_copy = input;
    Tensor saved = out.detach();

    out.get_impl()->backward_fn =
        [input_copy, saved, derivative](const Tensor& grad_out) mutable {
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

} // namespace

// ---------------------------------------------------------
// Module
// ---------------------------------------------------------

void Module::zero_grad() {
    for (Tensor& p : parameters()) {
        p.zero_grad();
    }
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters(const std::string& prefix) {
    // Por defecto se numeran en el orden en que los declara la capa
    std::vector<std::pair<std::string, Tensor>> named;
    std::vector<Tensor> params = parameters();
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

Tensor Sigmoid::forward(const Tensor& input) {
    // sigma(x) = 1 / (1 + e^-x), calculado de forma estable en ambos extremos
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
        throw std::invalid_argument("Dropout necesita una probabilidad en [0, 1).");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    // En inferencia es la identidad: la red debe ser determinista al evaluar
    if (!is_training() || p_ == 0.0f) return input;

    const float keep = 1.0f - p_;
    const float scale = 1.0f / keep;
    std::bernoulli_distribution alive(keep);

    // La máscara se genera aquí y se reutiliza en la derivada: el gradiente
    // tiene que anular exactamente las mismas posiciones.
    auto mask = std::make_shared<std::vector<float>>(input.size(), 0.0f);
    Tensor out(input.shape(), 0.0f, false);
    // Este bucle NO se reparte, aunque sea el mismo elemento a elemento que las
    // activaciones de arriba. global_rng() es un mt19937 compartido y sin
    // sincronizar: pasarlo por parallel_for mete una carrera silenciosa, y
    // encima el resultado dejaría de ser reproducible, porque cada hilo
    // consumiría números en un orden que depende de cómo caiga el reparto.
    // Para paralelizarlo haría falta un generador por hilo sembrado de forma
    // determinista, no un parallel_for alrededor.
    const size_t n = input.size();
    const float* ENGINE_RESTRICT x = input.data().data();
    float* ENGINE_RESTRICT y = out.data().data();
    for (size_t i = 0; i < n; ++i) {
        (*mask)[i] = alive(global_rng()) ? scale : 0.0f;
        y[i] = x[i] * (*mask)[i];
    }

    if (!autograd::grad_enabled() || !input.requires_grad()) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = { input.get_impl() };
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

Module& Sequential::at(size_t index) {
    if (index >= layers_.size()) {
        throw std::out_of_range("Sequential: no hay capa " + std::to_string(index) + ".");
    }
    return *layers_[index];
}

void Sequential::train(bool mode) {
    Module::train(mode);
    for (const auto& layer : layers_) layer->train(mode);
}

std::vector<std::pair<std::string, Tensor>> Sequential::named_parameters(const std::string& prefix) {
    std::vector<std::pair<std::string, Tensor>> named;
    for (size_t i = 0; i < layers_.size(); ++i) {
        // El índice va en el nombre para que dos capas iguales no colisionen
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

    // En serie por dos motivos, y basta cualquiera de los dos: total_loss es una
    // reducción sobre las filas, y el bucle lanza una excepción cuando una
    // etiqueta se sale de rango —cruzar una excepción desde dentro de una región
    // paralela es otro problema entero—. El backward, que no tiene ni lo uno ni
    // lo otro, sí se reparte.
    // ponytail: en serie; separar validación y reducción si el perfil lo pide.
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
                // Aquí sí se reparte, al revés que el forward: esto es una
                // escritura por elemento y no acumula nada.
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
