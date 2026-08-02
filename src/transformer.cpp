#include "engine/transformer.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/cuda_ops.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"

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
        throw std::invalid_argument("LayerNorm requires a size greater than zero.");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("LayerNorm requires a positive epsilon.");
    }

    autograd::NoGradGuard no_grad;
    // Identity at the start: with no scale or shift, the layer only normalises.
    gamma_ = Tensor({normalized_size}, 1.0f, true);
    beta_ = Tensor({normalized_size}, 0.0f, true);
}

namespace {

// The CPU forward, unchanged, lifted out so the dispatch above reads as one
// condition instead of an if wrapped around forty lines.
void layernorm_cpu(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor& out,
                   Tensor& xhat, Tensor& inv_std, size_t rows, size_t D, float inv_d, float eps) {
    const std::vector<float>& x = input.data();
    const std::vector<float>& g = gamma.data();
    const std::vector<float>& b = beta.data();

    // Each row is normalised with its own mean and variance, so no reduction
    // crosses a chunk boundary: splitting by row gives the same result bit for
    // bit with one thread or with eight.
    float* ENGINE_RESTRICT xhat_out = xhat.data().data();
    float* ENGINE_RESTRICT out_data = out.data().data();
    float* ENGINE_RESTRICT inv_out = inv_std.data().data();
    // The threshold is counted in rows, but the work is in the elements: a row
    // is D values and three passes over them.
    const size_t rows_per_thread =
        std::max<size_t>(1, parallel::kElementsPerThread / std::max<size_t>(1, D));
    parallel::parallel_for(rows, rows_per_thread, [&](size_t from, size_t to) {
        for (size_t i = from; i < to; ++i) {
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

            const float inv = 1.0f / std::sqrt(var + eps);
            inv_out[i] = inv;

            for (size_t j = 0; j < D; ++j) {
                const float h = (row[j] - mean) * inv;
                xhat_out[i * D + j] = h;
                out_data[i * D + j] = g[j] * h + b[j];
            }
        }
    });
}

}  // namespace

Tensor LayerNorm::forward(const Tensor& input) {
    if (input.ndim() < 1 || input.shape().back() != normalized_size_) {
        throw std::invalid_argument("LayerNorm of size " + std::to_string(normalized_size_) +
                                    " received an input " + input.shape_str() + ".");
    }

    const size_t D = normalized_size_;
    const size_t rows = input.size() / D;
    const float inv_d = 1.0f / static_cast<float>(D);

    Tensor out(input.shape(), 0.0f, false);
    // xhat and 1/deviation are saved because the derivative needs them. inv_std
    // is a Tensor and not a std::vector because the backward has a kernel now:
    // a host vector here would be a value the device path had to come down for,
    // and the chain would break in the middle of the block it was closed for.
    Tensor xhat(input.shape(), 0.0f, false);
    Tensor inv_std({rows}, 0.0f, false);

    // The device takes the whole row-normalisation in one pass, writing out,
    // xhat and inv_std together. Offered before any .data() call, because the
    // first accessor below would pull the input down and there would be nothing
    // resident left to work on.
    if (!cuda::ops::layernorm(input.get_impl()->storage, gamma_.get_impl()->storage,
                              beta_.get_impl()->storage, out.get_impl()->storage,
                              xhat.get_impl()->storage, inv_std.get_impl()->storage, rows, D,
                              eps_)) {
        layernorm_cpu(input, gamma_, beta_, out, xhat, inv_std, rows, D, inv_d, eps_);
    }

    const bool req_g = autograd::grad_enabled() &&
                       (input.requires_grad() || gamma_.requires_grad() || beta_.requires_grad());
    if (!req_g) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = {input.get_impl(), gamma_.get_impl(), beta_.get_impl()};

    Tensor input_copy = input;
    Tensor gamma_copy = gamma_;
    Tensor beta_copy = beta_;

    out.get_impl()->backward_fn = [input_copy, gamma_copy, beta_copy, xhat, inv_std, rows, D,
                                   inv_d](const Tensor& grad_out) mutable {
        Tensor dgamma(gamma_copy.shape(), 0.0f, false);
        Tensor dbeta(beta_copy.shape(), 0.0f, false);
        Tensor dX(input_copy.shape(), 0.0f, false);

        // Offered before a single accessor below, for the usual reason: the
        // first .data() would pull grad_out and xhat down and leave the device
        // path nothing resident to work on.
        if (cuda::ops::layernorm_backward(
                grad_out.get_impl()->storage, xhat.get_impl()->storage,
                gamma_copy.get_impl()->storage, inv_std.get_impl()->storage, dX.get_impl()->storage,
                dgamma.get_impl()->storage, dbeta.get_impl()->storage, rows, D)) {
            if (input_copy.requires_grad()) input_copy.add_grad(dX);
            if (gamma_copy.requires_grad()) gamma_copy.add_grad(dgamma);
            if (beta_copy.requires_grad()) beta_copy.add_grad(dbeta);
            return;
        }

        const std::vector<float>& dy = grad_out.data();
        const std::vector<float>& h = xhat.data();
        const std::vector<float>& g = gamma_copy.data();
        const std::vector<float>& inv = inv_std.data();

        // This loop stays serial deliberately, unlike the forward's. dX is
        // independent per row, but dgamma and dbeta accumulate **across** rows:
        // splitting it would need a per-thread partial and a fixed-order combine
        // so as not to lose the bit-for-bit identity the tests check. The device
        // path above does exactly that -- a fixed block count with private
        // partials, summed in index order -- and the same treatment here is the
        // remaining half of that idea.
        //
        // ponytail: serial reduction; per-thread partials if the profile flags it.
        for (size_t i = 0; i < rows; ++i) {
            // The mean and variance depend on the whole vector, so each
            // component's derivative drags two correction terms along:
            // dx = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / std
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
                dX.data()[k] = inv[i] * (dxhat - inv_d * sum_dxhat - h[k] * inv_d * sum_dxhat_h);
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

std::vector<Tensor> LayerNorm::parameters() {
    return {gamma_, beta_};
}

std::vector<std::pair<std::string, Tensor>> LayerNorm::named_parameters(const std::string& prefix) {
    return {{prefix + "layernorm.gamma", gamma_}, {prefix + "layernorm.beta", beta_}};
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
        throw std::invalid_argument("Embedding requires a positive vocabulary and dimension.");
    }
    autograd::NoGradGuard no_grad;
    weight_ = Tensor::randn({num_embeddings, dim}, 0.0f, 0.02f, true);
}

Tensor Embedding::forward(const Tensor& ids) {
    if (ids.ndim() != 2) {
        throw std::invalid_argument("Embedding expects indices (batch, sequence), received " +
                                    ids.shape_str() + ".");
    }
    const size_t batch = ids.shape()[0];
    const size_t seq = ids.shape()[1];

    std::vector<size_t> indices;
    indices.reserve(ids.size());
    for (float v : ids.data()) {
        const long long id = std::llround(v);
        if (id < 0 || static_cast<size_t>(id) >= num_embeddings_) {
            throw std::out_of_range("Embedding: token " + std::to_string(id) +
                                    " is outside a vocabulary of " +
                                    std::to_string(num_embeddings_) + ".");
        }
        indices.push_back(static_cast<size_t>(id));
    }

    // select_rows already accumulates the gradient when a token repeats, which is
    // exactly what an embedding table needs.
    return weight_.select_rows(indices).reshape({batch, seq, dim_});
}

std::vector<Tensor> Embedding::parameters() {
    return {weight_};
}

std::vector<std::pair<std::string, Tensor>> Embedding::named_parameters(const std::string& prefix) {
    return {{prefix + "embedding.weight", weight_}};
}

std::string Embedding::name() const {
    return "Embedding(" + std::to_string(num_embeddings_) + " x " + std::to_string(dim_) + ")";
}

// ---------------------------------------------------------
// Utilidades
// ---------------------------------------------------------

Tensor positional_encoding(size_t seq_len, size_t d_model) {
    if (seq_len == 0 || d_model == 0) {
        throw std::invalid_argument(
            "positional_encoding requires a positive length and dimension.");
    }

    Tensor pe({seq_len, d_model}, 0.0f, false);
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            // Each pair of dimensions uses a different frequency, so that the
            // relative distance between positions is a smooth function.
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

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    const Tensor* mask, Tensor* attention_weights) {
    if (query.ndim() < 2 || key.ndim() < 2 || value.ndim() < 2) {
        throw std::invalid_argument("Attention needs at least 2 dimensions (seq, d_k).");
    }
    const size_t d_k = query.shape().back();
    if (key.shape().back() != d_k) {
        throw std::invalid_argument("Query and key must share d_k: " + query.shape_str() + " and " +
                                    key.shape_str() + ".");
    }

    // Without the scaling the dot product grows with d_k, the softmax saturates
    // and the gradients vanish.
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    Tensor scores = query.matmul(key.transpose()) * scale;

    if (mask != nullptr) {
        scores = scores + *mask;  // additive mask, broadcast over batch and heads
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
    : d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : d_model / num_heads),
      w_query_(d_model, d_model),
      w_key_(d_model, d_model),
      w_value_(d_model, d_model),
      w_out_(d_model, d_model) {
    if (num_heads == 0) {
        throw std::invalid_argument("MultiHeadAttention requires at least one head.");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model (" + std::to_string(d_model) +
                                    ") must be divisible by num_heads (" +
                                    std::to_string(num_heads) + ").");
    }
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    return forward(input, nullptr);
}

Tensor MultiHeadAttention::forward(const Tensor& input, const Tensor* mask) {
    if (input.ndim() != 3 || input.shape()[2] != d_model_) {
        throw std::invalid_argument("MultiHeadAttention expects (batch, seq, " +
                                    std::to_string(d_model_) + "), received " + input.shape_str() +
                                    ".");
    }
    const size_t batch = input.shape()[0];
    const size_t seq = input.shape()[1];

    // (B, S, d_model) -> (B, S, H, head_dim) -> (B, H, S, head_dim).
    // The permutation is what leaves each head with its full sequence contiguous,
    // so that the batched matmul operates head by head.
    auto split_heads = [&](const Tensor& projected) {
        return projected.reshape({batch, seq, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    };

    Tensor q = split_heads(w_query_(input));
    Tensor k = split_heads(w_key_(input));
    Tensor v = split_heads(w_value_(input));

    Tensor attended =
        scaled_dot_product_attention(q, k, v, mask, keep_attention_ ? &last_attention_ : nullptr);

    // Undo the split into heads: (B, H, S, hd) -> (B, S, d_model)
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
           ", heads=" + std::to_string(num_heads_) + ", head_dim=" + std::to_string(head_dim_) +
           ")";
}

// ---------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------

TransformerBlock::TransformerBlock(size_t d_model, size_t num_heads, size_t ff_hidden)
    : d_model_(d_model),
      ff_hidden_(ff_hidden),
      norm1_(d_model),
      norm2_(d_model),
      attention_(d_model, num_heads),
      ff1_(d_model, ff_hidden),
      ff2_(ff_hidden, d_model) {
    if (ff_hidden == 0) {
        throw std::invalid_argument("TransformerBlock requires a positive hidden layer.");
    }
}

Tensor TransformerBlock::forward(const Tensor& input) {
    return forward(input, nullptr);
}

Tensor TransformerBlock::forward(const Tensor& input, const Tensor* mask) {
    // Pre-norm: the residual connection is left free of normalisation, so the
    // gradient reaches the layers below intact.
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
        {"norm1.", static_cast<Module*>(&norm1_)}, {"norm2.", static_cast<Module*>(&norm2_)},
        {"", static_cast<Module*>(&attention_)},   {"ff1.", static_cast<Module*>(&ff1_)},
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

}  // namespace nn
}  // namespace engine
