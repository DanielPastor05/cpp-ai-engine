#ifndef ENGINE_CONV_HPP
#define ENGINE_CONV_HPP

#include "engine/nn.hpp"

#include <vector>

namespace engine {
namespace nn {

// Geometría de una ventana deslizante (kernel, paso y relleno).
struct Window2d {
    size_t kernel_h;
    size_t kernel_w;
    size_t stride = 1;
    size_t padding = 0;

    Window2d(size_t k) : kernel_h(k), kernel_w(k) {}
    Window2d(size_t kh, size_t kw, size_t s = 1, size_t p = 0)
        : kernel_h(kh), kernel_w(kw), stride(s), padding(p) {}

    // Tamaño de salida: (dim + 2*padding - kernel) / stride + 1
    size_t out_h(size_t in_h) const;
    size_t out_w(size_t in_w) const;
    void validate(size_t in_h, size_t in_w) const;
};

// ---------------------------------------------------------
// im2col / col2im
//
// im2col aplana cada ventana del volumen de entrada en una fila, de modo que
// la convolución se reduce a una única multiplicación matricial. Es el truco
// que hace viable una CNN sin escribir siete bucles anidados.
//
//   entrada (N, C, H, W)  ->  columnas (N*oH*oW, C*kH*kW)
//
// col2im es su adjunto: reparte cada fila de vuelta a las posiciones del
// volumen original, sumando allí donde las ventanas se solapan. Ese solape es
// exactamente lo que hace que sea la derivada correcta de im2col.
// ---------------------------------------------------------
Tensor im2col(const Tensor& input, const Window2d& window);
Tensor col2im(const Tensor& cols, const std::vector<size_t>& input_shape, const Window2d& window);

// ---------------------------------------------------------
// Capas convolucionales
// ---------------------------------------------------------

// Convolución 2D sobre volúmenes (N, C, H, W).
// Los pesos tienen forma (out_channels, in_channels, kernel_h, kernel_w).
class Conv2d : public Module {
public:
    Conv2d(size_t in_channels, size_t out_channels, const Window2d& window, bool use_bias = true);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    const Window2d& window() const { return window_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    Window2d window_;
    bool use_bias_;
    Tensor weight_; // (out_channels, in_channels, kernel_h, kernel_w)
    Tensor bias_;   // (out_channels)
};

// Submuestreo por máximo. Propaga el gradiente solo a la posición ganadora de
// cada ventana, que es la única que influyó en la salida.
class MaxPool2d : public Module {
public:
    explicit MaxPool2d(const Window2d& window);
    MaxPool2d(size_t kernel, size_t stride);

    Tensor forward(const Tensor& input) override;
    std::string name() const override;

private:
    Window2d window_;
};

// Aplana todo menos el eje del lote: (N, C, H, W) -> (N, C*H*W).
// Es la bisagra entre la parte convolucional y las capas densas.
class Flatten : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Flatten"; }
};

} // namespace nn
} // namespace engine

#endif // ENGINE_CONV_HPP
