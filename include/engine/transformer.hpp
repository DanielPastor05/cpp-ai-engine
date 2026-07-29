#ifndef ENGINE_TRANSFORMER_HPP
#define ENGINE_TRANSFORMER_HPP

#include "engine/nn.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// Layer normalisation
//
// Normalises each vector along the last axis to mean 0 and variance 1, then
// rescales it with two learned parameters (gamma and beta). Unlike batch
// normalisation it mixes no information between examples, which is what makes
// it suitable for variable-length sequences.
// ---------------------------------------------------------
class LayerNorm : public Module {
public:
    explicit LayerNorm(size_t normalized_size, float eps = 1e-5f);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::vector<std::pair<std::string, Tensor>> named_parameters(
        const std::string& prefix = "") override;
    std::string name() const override;

    Tensor& gamma() { return gamma_; }
    Tensor& beta() { return beta_; }

private:
    size_t normalized_size_;
    float eps_;
    Tensor gamma_;  // learned scale (normalized_size)
    Tensor beta_;   // learned shift (normalized_size)
};

// ---------------------------------------------------------
// Embedding table: turns token indices into dense vectors.
// The input is a (batch, sequence) tensor whose values are indices; the output
// is (batch, sequence, dim).
// ---------------------------------------------------------
class Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t dim);

    Tensor forward(const Tensor& ids) override;
    std::vector<Tensor> parameters() override;
    std::vector<std::pair<std::string, Tensor>> named_parameters(
        const std::string& prefix = "") override;
    std::string name() const override;

    Tensor& weight() { return weight_; }
    size_t dim() const { return dim_; }

private:
    size_t num_embeddings_;
    size_t dim_;
    Tensor weight_;  // (num_embeddings, dim)
};

// ---------------------------------------------------------
// Utilities
// ---------------------------------------------------------

// Sinusoidal positional encoding (seq_len, d_model) from the original paper.
// Attention on its own cannot tell token order apart: without this, a sentence
// and a permutation of it would produce exactly the same output.
Tensor positional_encoding(size_t seq_len, size_t d_model);

// Additive causal mask (seq_len, seq_len): 0 where attending is allowed and a
// very negative value where it is not, so that softmax drives it to zero. It
// stops a position from seeing the future.
Tensor causal_mask(size_t seq_len, float masked_value = -1e9f);

// Scaled dot-product attention:
//   softmax(Q x K^T / sqrt(d_k) + mask) x V
// Q, K and V are (..., seq, d_k). Scaling by sqrt(d_k) stops the dot product
// from growing with the dimension and saturating the softmax.
// If attention_weights is passed, a copy of the attention weights detached from
// the graph (..., seq, seq) is left there.
Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    const Tensor* mask = nullptr,
                                    Tensor* attention_weights = nullptr);

// ---------------------------------------------------------
// Multi-head attention
//
// Projects the input into num_heads independent subspaces, applies attention
// in each and concatenates. Several heads make it possible to attend to
// different relationships at once, at the same cost as one wide head.
// ---------------------------------------------------------
class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads);

    // Self-attention: the (batch, seq, d_model) input acts as Q, K and V.
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, const Tensor* mask);

    std::vector<Tensor> parameters() override;
    std::vector<std::pair<std::string, Tensor>> named_parameters(
        const std::string& prefix = "") override;
    std::string name() const override;

    // The (batch, heads, seq, seq) attention weights from the last forward,
    // useful for inspecting what the model looks at. They have to be switched
    // on with keep_attention(true): saving them costs a (B, H, S, S) copy every
    // step, and during training nobody reads them.
    const Tensor& last_attention() const { return last_attention_; }
    void keep_attention(bool keep) { keep_attention_ = keep; }
    bool keeps_attention() const { return keep_attention_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    Linear w_query_;
    Linear w_key_;
    Linear w_value_;
    Linear w_out_;
    Tensor last_attention_;
    bool keep_attention_ = false;
};

// ---------------------------------------------------------
// Transformer encoder block (pre-norm)
//
//   x = x + Attention(LayerNorm(x))
//   x = x + FeedForward(LayerNorm(x))
//
// Normalisation happens before each sub-layer rather than after it as in the
// original paper: it leaves the residual connection free of normalisation and
// makes training considerably more stable without a learning-rate warm-up.
// ---------------------------------------------------------
class TransformerBlock : public Module {
public:
    TransformerBlock(size_t d_model, size_t num_heads, size_t ff_hidden);

    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, const Tensor* mask);

    std::vector<Tensor> parameters() override;
    std::vector<std::pair<std::string, Tensor>> named_parameters(
        const std::string& prefix = "") override;
    void train(bool mode = true) override;
    std::string name() const override;

    MultiHeadAttention& attention() { return attention_; }

private:
    size_t d_model_;
    size_t ff_hidden_;
    LayerNorm norm1_;
    LayerNorm norm2_;
    MultiHeadAttention attention_;
    Linear ff1_;
    Linear ff2_;
};

}  // namespace nn
}  // namespace engine

#endif  // ENGINE_TRANSFORMER_HPP
