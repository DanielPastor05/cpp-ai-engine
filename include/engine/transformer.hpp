#ifndef ENGINE_TRANSFORMER_HPP
#define ENGINE_TRANSFORMER_HPP

#include "engine/nn.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// Normalización por capa
//
// Normaliza cada vector del último eje a media 0 y varianza 1, y luego lo
// reescala con dos parámetros aprendidos (gamma y beta). A diferencia de la
// normalización por lotes, no mezcla información entre ejemplos, que es lo que
// la hace apta para secuencias de longitud variable.
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
    Tensor gamma_; // escala aprendida (normalized_size)
    Tensor beta_;  // desplazamiento aprendido (normalized_size)
};

// ---------------------------------------------------------
// Tabla de embeddings: convierte índices de token en vectores densos.
// La entrada es un tensor (batch, secuencia) cuyos valores son índices; la
// salida es (batch, secuencia, dim).
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
    Tensor weight_; // (num_embeddings, dim)
};

// ---------------------------------------------------------
// Utilidades
// ---------------------------------------------------------

// Codificación posicional sinusoidal (seq_len, d_model) del artículo original.
// La atención por sí sola no distingue el orden de los tokens: sin esto, una
// frase y su permutación producirían exactamente la misma salida.
Tensor positional_encoding(size_t seq_len, size_t d_model);

// Máscara causal aditiva (seq_len, seq_len): 0 donde se puede atender y un
// valor muy negativo donde no, de modo que el softmax lo lleve a cero. Impide
// que una posición vea el futuro.
Tensor causal_mask(size_t seq_len, float masked_value = -1e9f);

// Atención por producto escalar escalado:
//   softmax(Q x K^T / sqrt(d_k) + mask) x V
// Q, K y V son (..., seq, d_k). El escalado por sqrt(d_k) evita que el
// producto escalar crezca con la dimensión y sature el softmax.
// Si se pasa attention_weights, allí se deja una copia desligada del grafo con
// los pesos de atención (..., seq, seq).
Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key,
                                    const Tensor& value, const Tensor* mask = nullptr,
                                    Tensor* attention_weights = nullptr);

// ---------------------------------------------------------
// Atención multi-cabeza
//
// Proyecta la entrada en num_heads subespacios independientes, aplica atención
// en cada uno y concatena. Varias cabezas permiten atender a distintas
// relaciones a la vez con el mismo coste que una sola cabeza ancha.
// ---------------------------------------------------------
class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads);

    // Auto-atención: la entrada (batch, seq, d_model) hace de Q, K y V.
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, const Tensor* mask);

    std::vector<Tensor> parameters() override;
    std::vector<std::pair<std::string, Tensor>> named_parameters(
        const std::string& prefix = "") override;
    std::string name() const override;

    // Pesos de atención (batch, heads, seq, seq) del último forward, útiles
    // para inspeccionar en qué se fija el modelo. Hay que activarlos con
    // keep_attention(true): guardarlos cuesta una copia de (B, H, S, S) en
    // cada paso, y durante el entrenamiento no se miran.
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
// Bloque codificador Transformer (pre-norm)
//
//   x = x + Attention(LayerNorm(x))
//   x = x + FeedForward(LayerNorm(x))
//
// Se normaliza antes de cada subcapa, no después como en el artículo original:
// deja la conexión residual libre de normalizaciones y hace el entrenamiento
// bastante más estable sin necesidad de calentamiento del learning rate.
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

} // namespace nn
} // namespace engine

#endif // ENGINE_TRANSFORMER_HPP
