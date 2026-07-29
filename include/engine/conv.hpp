#ifndef ENGINE_CONV_HPP
#define ENGINE_CONV_HPP

#include "engine/nn.hpp"

#include <vector>

namespace engine {
namespace nn {

// The geometry of a sliding window (kernel, stride and padding).
struct Window2d {
    size_t kernel_h;
    size_t kernel_w;
    size_t stride = 1;
    size_t padding = 0;

    Window2d(size_t k) : kernel_h(k), kernel_w(k) {}
    Window2d(size_t kh, size_t kw, size_t s = 1, size_t p = 0)
        : kernel_h(kh), kernel_w(kw), stride(s), padding(p) {}

    // Output size: (dim + 2*padding - kernel) / stride + 1
    size_t out_h(size_t in_h) const;
    size_t out_w(size_t in_w) const;
    void validate(size_t in_h, size_t in_w) const;
};

// ---------------------------------------------------------
// im2col / col2im
//
// im2col flattens each window of the input volume into a row, so that the
// convolution reduces to a single matrix multiplication. It is the trick that
// makes a CNN viable without writing seven nested loops.
//
//   input (N, C, H, W)  ->  columns (N*oH*oW, C*kH*kW)
//
// col2im is its adjoint: it scatters each row back to the positions of the
// original volume, summing wherever the windows overlap. That overlap is
// exactly what makes it the correct derivative of im2col.
// ---------------------------------------------------------
Tensor im2col(const Tensor& input, const Window2d& window);
Tensor col2im(const Tensor& cols, const std::vector<size_t>& input_shape, const Window2d& window);

// ---------------------------------------------------------
// Convolutional layers
// ---------------------------------------------------------

// 2D convolution over (N, C, H, W) volumes.
// Weights have shape (out_channels, in_channels, kernel_h, kernel_w).
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

// Max pooling. Propagates the gradient only to each window's winning position,
// which is the only one that influenced the output.
class MaxPool2d : public Module {
public:
    explicit MaxPool2d(const Window2d& window);
    MaxPool2d(size_t kernel, size_t stride);

    Tensor forward(const Tensor& input) override;
    std::string name() const override;

private:
    Window2d window_;
};

// Flattens everything but the batch axis: (N, C, H, W) -> (N, C*H*W).
// It is the hinge between the convolutional part and the dense layers.
class Flatten : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Flatten"; }
};

} // namespace nn
} // namespace engine

#endif // ENGINE_CONV_HPP
