#include "engine/transformer.hpp"
#include "engine/autograd.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------

LayerNorm::LayerNorm(size_t normalized_size, float eps)
    : normalized_size_(normalized_size), eps_(eps) {
    if (normalized_size == 0) {
        throw std::invalid_argument("LayerNorm requiere un tamano mayor que cero.");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("LayerNorm requiere un epsilon positivo.");
    }

    autograd::NoGradGuard no_grad;
    // Identidad al inicio: sin escalar ni desplazar, la capa solo normaliza.
    gamma_ = Tensor({normalized_size}, 1.0f, true);
    beta_ = Tensor({normalized_size}, 0.0f, true);
}

Tensor LayerNorm::forward(const Tensor& input) {
    if (input.ndim() < 1 || input.shape().back() != normalized_size_) {
        throw std::invalid_argument("LayerNorm de tamano " + std::to_string(normalized_size_) +
                                    " recibió una entrada " + input.shape_str() + ".");
    }

    const size_t D = normalized_size_;
    const size_t rows = input.size() / D;
    const float inv_d = 1.0f / static_cast<float>(D);

    Tensor out(input.shape(), 0.0f, false);
    // xhat y 1/desviacion se guardan porque la derivada los necesita
    Tensor xhat(input.shape(), 0.0f, false);
    std::vector<float> inv_std(rows, 0.0f);

    const std::vector<float>& x = input.data();
    const std::vector<float>& g = gamma_.data();
    const std::vector<float>& b = beta_.data();

    for (size_t i = 0; i < rows; ++i) {
        const float* row = x.data() + i * D;

        float mean = 0.0f;
        for (size_t j = 0; j < D; ++j) mean += row[j];
        mean *= inv_d;

        float var = 0.0f;
        for (size_t j = 0; j < D; ++j) {
            const float d = row[j] - mean;
            var += d * d;
        }
        var *= inv_d;

        const float inv = 1.0f / std::sqrt(var + eps_);
        inv_std[i] = inv;

        for (size_t j = 0; j < D; ++j) {
            const float h = (row[j] - mean) * inv;
            xhat.data()[i * D + j] = h;
            out.data()[i * D + j] = g[j] * h + b[j];
        }
    }

    const bool req_g = autograd::grad_enabled() &&
                       (input.requires_grad() || gamma_.requires_grad() || beta_.requires_grad());
    if (!req_g) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = { input.get_impl(), gamma_.get_impl(), beta_.get_impl() };

    Tensor input_copy = input;
    Tensor gamma_copy = gamma_;
    Tensor beta_copy = beta_;

    out.get_impl()->backward_fn =
        [input_copy, gamma_copy, beta_copy, xhat, inv_std, rows, D, inv_d](const Tensor& grad_out) mutable {
            const std::vector<float>& dy = grad_out.data();
            const std::vector<float>& h = xhat.data();
            const std::vector<float>& g = gamma_copy.data();

            Tensor dgamma(gamma_copy.shape(), 0.0f, false);
            Tensor dbeta(beta_copy.shape(), 0.0f, false);
            Tensor dX(input_copy.shape(), 0.0f, false);

            for (size_t i = 0; i < rows; ++i) {
                // La media y la varianza dependen de todo el vector, así que la
                // derivada de cada componente arrastra dos términos de correción:
                // dx = (dxhat - media(dxhat) - xhat * media(dxhat * xhat)) / std
                float sum_dxhat = 0.0f;
                float sum_dxhat_h = 0.0f;
                for (size_t j = 0; j < D; ++j) {
                    const float dxhat = dy[i * D + j] * g[j];
                    sum_dxhat += dxhat;
                    sum_dxhat_h += dxhat * h[i * D + j];
                }

                for (size_t j = 0; j < D; ++j) {
                    const size_t k = i * D + j;
                    const float dxhat = dy[k] * g[j];
                    dX.data()[k] = inv_std[i] *
                                   (dxhat - inv_d * sum_dxhat - h[k] * inv_d * sum_dxhat_h);
                    dgamma.data()[j] += dy[k] * h[k];
                    dbeta.data()[j] += dy[k];
                }
            }

            if (input_copy.requires_grad()) input_copy.add_grad(dX);
            if (gamma_copy.requires_grad()) gamma_copy.add_grad(dgamma);
            if (beta_copy.requires_grad()) beta_copy.add_grad(dbeta);
        };

    return out;
}

std::vector<Tensor> LayerNorm::parameters() { return { gamma_, beta_ }; }

std::vector<std::pair<std::string, Tensor>> LayerNorm::named_parameters(const std::string& prefix) {
    return { {prefix + "layernorm.gamma", gamma_}, {prefix + "layernorm.beta", beta_} };
}

std::string LayerNorm::name() const {
    return "LayerNorm(" + std::to_string(normalized_size_) + ")";
}

// ---------------------------------------------------------
// Embedding
// ---------------------------------------------------------

Embedding::Embedding(size_t num_embeddings, size_t dim)
    : num_embeddings_(num_embeddings), dim_(dim) {
    if (num_embeddings == 0 || dim == 0) {
        throw std::invalid_argument("Embedding requiere un vocabulario y una dimension positivos.");
    }
    autograd::NoGradGuard no_grad;
    weight_ = Tensor::randn({num_embeddings, dim}, 0.0f, 0.02f, true);
}

Tensor Embedding::forward(const Tensor& ids) {
    if (ids.ndim() != 2) {
        throw std::invalid_argument("Embedding espera indices (batch, secuencia), recibió " +
                                    ids.shape_str() + ".");
    }
    const size_t batch = ids.shape()[0];
    const size_t seq = ids.shape()[1];

    std::vector<size_t> indices;
    indices.reserve(ids.size());
    for (float v : ids.data()) {
        const long long id = std::llround(v);
        if (id < 0 || static_cast<size_t>(id) >= num_embeddings_) {
            throw std::out_of_range("Embedding: el token " + std::to_string(id) +
                                    " esta fuera de un vocabulario de " +
                                    std::to_string(num_embeddings_) + ".");
        }
        indices.push_back(static_cast<size_t>(id));
    }

    // select_rows ya acumula el gradiente cuando un token se repite, que es
    // justo lo que necesita una tabla de embeddings.
    return weight_.select_rows(indices).reshape({batch, seq, dim_});
}

std::vector<Tensor> Embedding::parameters() { return { weight_ }; }

std::vector<std::pair<std::string, Tensor>> Embedding::named_parameters(const std::string& prefix) {
    return { {prefix + "embedding.weight", weight_} };
}

std::string Embedding::name() const {
    return "Embedding(" + std::to_string(num_embeddings_) + " x " + std::to_string(dim_) + ")";
}

// ---------------------------------------------------------
// Utilidades
// ---------------------------------------------------------

Tensor positional_encoding(size_t seq_len, size_t d_model) {
    if (seq_len == 0 || d_model == 0) {
        throw std::invalid_argument("positional_encoding requiere longitud y dimension positivas.");
    }

    Tensor pe({seq_len, d_model}, 0.0f, false);
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            // Cada par de dimensiones usa una frecuencia distinta, de modo que
            // la distancia relativa entre posiciones sea una función suave.
            const size_t pair = i / 2;
            const float exponent = 2.0f * static_cast<float>(pair) / static_cast<float>(d_model);
            const float angle = static_cast<float>(pos) / std::pow(10000.0f, exponent);
            pe.data()[pos * d_model + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    return pe;
}

Tensor causal_mask(size_t seq_len, float masked_value) {
    Tensor mask({seq_len, seq_len}, 0.0f, false);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            mask.data()[i * seq_len + j] = masked_value;
        }
    }
    return mask;
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key,
                                    const Tensor& value, const Tensor* mask,
                                    Tensor* attention_weights) {
    if (query.ndim() < 2 || key.ndim() < 2 || value.ndim() < 2) {
        throw std::invalid_argument("La atencion necesita al menos 2 dimensiones (seq, d_k).");
    }
    const size_t d_k = query.shape().back();
    if (key.shape().back() != d_k) {
        throw std::invalid_argument("Query y key deben compartir d_k: " + query.shape_str() +
                                    " y " + key.shape_str() + ".");
    }

    // Sin el escalado, el producto escalar crece con d_k, el softmax se satura
    // y los gradientes se desvanecen.
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    Tensor scores = query.matmul(key.transpose()) * scale;

    if (mask != nullptr) {
        scores = scores + *mask; // máscara aditiva, difundida sobre lote y cabezas
    }

    Tensor weights = scores.softmax();
    if (attention_weights != nullptr) {
        *attention_weights = weights.detach();
    }
    return weights.matmul(value);
}

// ---------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    : d_model_(d_model), num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : d_model / num_heads),
      w_query_(d_model, d_model), w_key_(d_model, d_model),
      w_value_(d_model, d_model), w_out_(d_model, d_model) {
    if (num_heads == 0) {
        throw std::invalid_argument("MultiHeadAttention requiere al menos una cabeza.");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model (" + std::to_string(d_model) +
                                    ") debe ser divisible entre num_heads (" +
                                    std::to_string(num_heads) + ").");
    }
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    return forward(input, nullptr);
}

Tensor MultiHeadAttention::forward(const Tensor& input, const Tensor* mask) {
    if (input.ndim() != 3 || input.shape()[2] != d_model_) {
        throw std::invalid_argument("MultiHeadAttention espera (batch, seq, " +
                                    std::to_string(d_model_) + "), recibió " +
                                    input.shape_str() + ".");
    }
    const size_t batch = input.shape()[0];
    const size_t seq = input.shape()[1];

    // (B, S, d_model) -> (B, S, H, head_dim) -> (B, H, S, head_dim).
    // La permutación es lo que deja a cada cabeza con su secuencia completa
    // contigua, para que el matmul por lotes opere cabeza a cabeza.
    auto split_heads = [&](const Tensor& projected) {
        return projected.reshape({batch, seq, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    };

    Tensor q = split_heads(w_query_(input));
    Tensor k = split_heads(w_key_(input));
    Tensor v = split_heads(w_value_(input));

    Tensor attended = scaled_dot_product_attention(
        q, k, v, mask, keep_attention_ ? &last_attention_ : nullptr);

    // Deshacer la separación en cabezas: (B, H, S, hd) -> (B, S, d_model)
    Tensor merged = attended.permute({0, 2, 1, 3}).reshape({batch, seq, d_model_});
    return w_out_(merged);
}

std::vector<Tensor> MultiHeadAttention::parameters() {
    std::vector<Tensor> params;
    for (Linear* layer : {&w_query_, &w_key_, &w_value_, &w_out_}) {
        std::vector<Tensor> sub = layer->parameters();
        params.insert(params.end(), sub.begin(), sub.end());
    }
    return params;
}

std::vector<std::pair<std::string, Tensor>> MultiHeadAttention::named_parameters(
    const std::string& prefix) {
    std::vector<std::pair<std::string, Tensor>> named;
    const char* tags[] = {"query", "key", "value", "out"};
    Linear* layers[] = {&w_query_, &w_key_, &w_value_, &w_out_};
    for (size_t i = 0; i < 4; ++i) {
        std::vector<std::pair<std::string, Tensor>> sub =
            layers[i]->named_parameters(prefix + "attn." + tags[i] + ".");
        named.insert(named.end(), sub.begin(), sub.end());
    }
    return named;
}

std::string MultiHeadAttention::name() const {
    return "MultiHeadAttention(d_model=" + std::to_string(d_model_) +
           ", heads=" + std::to_string(num_heads_) +
           ", head_dim=" + std::to_string(head_dim_) + ")";
}

// ---------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------

TransformerBlock::TransformerBlock(size_t d_model, size_t num_heads, size_t ff_hidden)
    : d_model_(d_model), ff_hidden_(ff_hidden),
      norm1_(d_model), norm2_(d_model),
      attention_(d_model, num_heads),
      ff1_(d_model, ff_hidden), ff2_(ff_hidden, d_model) {
    if (ff_hidden == 0) {
        throw std::invalid_argument("TransformerBlock requiere una capa oculta positiva.");
    }
}

Tensor TransformerBlock::forward(const Tensor& input) {
    return forward(input, nullptr);
}

Tensor TransformerBlock::forward(const Tensor& input, const Tensor* mask) {
    // Pre-norm: la conexión residual queda libre de normalizaciones, así que
    // el gradiente llega intacto a las capas de abajo.
    Tensor h = input + attention_.forward(norm1_(input), mask);
    return h + ff2_(ff1_(norm2_(h)).relu());
}

std::vector<Tensor> TransformerBlock::parameters() {
    std::vector<Tensor> params;
    for (Module* layer : {static_cast<Module*>(&norm1_), static_cast<Module*>(&norm2_),
                          static_cast<Module*>(&attention_), static_cast<Module*>(&ff1_),
                          static_cast<Module*>(&ff2_)}) {
        std::vector<Tensor> sub = layer->parameters();
        params.insert(params.end(), sub.begin(), sub.end());
    }
    return params;
}

std::vector<std::pair<std::string, Tensor>> TransformerBlock::named_parameters(
    const std::string& prefix) {
    std::vector<std::pair<std::string, Tensor>> named;
    const std::pair<const char*, Module*> parts[] = {
        {"norm1.", static_cast<Module*>(&norm1_)},
        {"norm2.", static_cast<Module*>(&norm2_)},
        {"", static_cast<Module*>(&attention_)},
        {"ff1.", static_cast<Module*>(&ff1_)},
        {"ff2.", static_cast<Module*>(&ff2_)},
    };
    for (const auto& part : parts) {
        std::vector<std::pair<std::string, Tensor>> sub =
            part.second->named_parameters(prefix + part.first);
        named.insert(named.end(), sub.begin(), sub.end());
    }
    return named;
}

void TransformerBlock::train(bool mode) {
    Module::train(mode);
    norm1_.train(mode);
    norm2_.train(mode);
    attention_.train(mode);
    ff1_.train(mode);
    ff2_.train(mode);
}

std::string TransformerBlock::name() const {
    return "TransformerBlock(d_model=" + std::to_string(d_model_) +
           ", ff=" + std::to_string(ff_hidden_) + ")";
}

} // namespace nn
} // namespace engine
