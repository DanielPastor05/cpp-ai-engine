#include "engine/conv.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/cuda_ops.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"

#include <limits>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// Window2d
// ---------------------------------------------------------

namespace {

// Same criterion as in the tensor: splitting costs about 8 us, so it only pays
// once there is a fair amount of work per region.
constexpr size_t kConvRowsPerThread = 4096;

size_t output_size(size_t in_size, size_t kernel, size_t stride, size_t padding) {
    const size_t padded = in_size + 2 * padding;
    if (kernel > padded) {
        throw std::invalid_argument("The kernel (" + std::to_string(kernel) +
                                    ") does not fit the padded input dimension (" +
                                    std::to_string(padded) + ").");
    }
    return (padded - kernel) / stride + 1;
}

}  // namespace

size_t Window2d::out_h(size_t in_h) const {
    return output_size(in_h, kernel_h, stride, padding);
}
size_t Window2d::out_w(size_t in_w) const {
    return output_size(in_w, kernel_w, stride, padding);
}

void Window2d::validate(size_t in_h, size_t in_w) const {
    if (kernel_h == 0 || kernel_w == 0) {
        throw std::invalid_argument("The kernel must have both dimensions greater than zero.");
    }
    if (stride == 0) {
        throw std::invalid_argument("The stride must be greater than zero.");
    }
    out_h(in_h);
    out_w(in_w);
}

// ---------------------------------------------------------
// im2col / col2im
// ---------------------------------------------------------

Tensor im2col(const Tensor& input, const Window2d& window) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("im2col expects a 4D volume (N, C, H, W), received " +
                                    input.shape_str() + ".");
    }
    const size_t N = input.shape()[0];
    const size_t C = input.shape()[1];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];

    window.validate(H, W);
    const size_t oH = window.out_h(H);
    const size_t oW = window.out_w(W);
    const size_t kH = window.kernel_h;
    const size_t kW = window.kernel_w;
    const size_t K = C * kH * kW;

    // The shape relation the matrix product downstream is written against: one
    // row per output position, one column per (channel, kernel row, kernel
    // column). col2im inverts it and test_conv asserts they are adjoint, but
    // neither notices a window that produced no positions at all.
    assert(oH > 0 && oW > 0 && "the window has to produce at least one position");
    assert(K == C * kH * kW && "one column per channel and kernel element");
    Tensor cols({N * oH * oW, K}, 0.0f, false);

    // It is offered to the device first. If the device takes it, the columns stay
    // up there and the product behind reads them without crossing PCIe: they are
    // kH*kW times the input, so uploading them costs more than multiplying them.
    if (cuda::ops::im2col(input.storage(), cols.storage(),
                          {N, C, H, W, kH, kW, window.stride, window.padding, oH, oW})) {
        return cols;
    }

    const float* src = input.data();
    float* dst = cols.data();

    // A single iteration writes each output row, so splitting by rows creates no
    // race.
    //
    // The bounds checks are hoisted, and that is the whole optimisation. Written
    // the obvious way this loop tests `h` inside the channel loop and `w` inside
    // the kernel-column loop, so a 16-channel 3x3 convolution re-derives and
    // re-checks the same three column coordinates 48 times per output row. It
    // measured 4.06 ms for the second convolution of the MNIST model -- 33% of
    // the layer, and 1.98 GB/s when the traffic involved is about 8 MB, which at
    // this machine's memory speed should take 0.27. Fifteen times off is not a
    // bandwidth problem; it is arithmetic per element.
    //
    // Neither coordinate depends on the loops it was nested in. `h` is a function
    // of `oh` and `i` alone, `w` of `ow` and `j` alone. Computing the valid range
    // of each once per output row leaves the inner loop a contiguous copy with no
    // branch in it, and the positions that fall in the padding keep the zero the
    // buffer was created with -- which is why cols is zero-filled and must stay
    // that way.
    const long long pad = static_cast<long long>(window.padding);
    parallel::parallel_for(N * oH * oW, kConvRowsPerThread, [&](size_t from, size_t to) {
        for (size_t row = from; row < to; ++row) {
            const size_t n = row / (oH * oW);
            const size_t oh = (row % (oH * oW)) / oW;
            const size_t ow = row % oW;

            // Columns: w = ow*stride + j - padding has to land inside [0, W).
            const long long w0 = static_cast<long long>(ow * window.stride) - pad;
            const size_t j_begin = w0 < 0 ? static_cast<size_t>(-w0) : 0;
            const long long j_limit = static_cast<long long>(W) - w0;
            const size_t j_end = j_limit <= 0 ? 0 : std::min(kW, static_cast<size_t>(j_limit));
            if (j_begin >= j_end) continue;  // the whole window is in the padding
            const size_t run = j_end - j_begin;
            const size_t w_begin = static_cast<size_t>(w0) + j_begin;

            // Rows: the same, for h.
            const long long h0 = static_cast<long long>(oh * window.stride) - pad;
            const size_t i_begin = h0 < 0 ? static_cast<size_t>(-h0) : 0;
            const long long i_limit = static_cast<long long>(H) - h0;
            const size_t i_end = i_limit <= 0 ? 0 : std::min(kH, static_cast<size_t>(i_limit));
            if (i_begin >= i_end) continue;

            float* ENGINE_RESTRICT drow = dst + row * K;
            for (size_t c = 0; c < C; ++c) {
                for (size_t i = i_begin; i < i_end; ++i) {
                    const size_t h = static_cast<size_t>(h0) + i;
                    const float* ENGINE_RESTRICT s = src + ((n * C + c) * H + h) * W + w_begin;
                    float* ENGINE_RESTRICT d = drow + (c * kH + i) * kW + j_begin;
                    for (size_t jj = 0; jj < run; ++jj) d[jj] = s[jj];
                }
            }
        }
    });
    return cols;
}

Tensor col2im(const Tensor& cols, const std::vector<size_t>& input_shape, const Window2d& window) {
    if (input_shape.size() != 4) {
        throw std::invalid_argument("col2im needs a 4D destination shape (N, C, H, W).");
    }
    const size_t N = input_shape[0];
    const size_t C = input_shape[1];
    const size_t H = input_shape[2];
    const size_t W = input_shape[3];

    window.validate(H, W);
    const size_t oH = window.out_h(H);
    const size_t oW = window.out_w(W);
    const size_t kH = window.kernel_h;
    const size_t kW = window.kernel_w;
    const size_t K = C * kH * kW;

    if (cols.ndim() != 2 || cols.shape()[0] != N * oH * oW || cols.shape()[1] != K) {
        throw std::invalid_argument("col2im expected columns (" + std::to_string(N * oH * oW) +
                                    ", " + std::to_string(K) + ") and received " +
                                    cols.shape_str() + ".");
    }

    Tensor out(input_shape, 0.0f, false);

    if (cuda::ops::col2im(cols.storage(), out.storage(),
                          {N, C, H, W, kH, kW, window.stride, window.padding, oH, oW})) {
        return out;
    }

    const float* src = cols.data();
    float* dst = out.data();

    // Splitting by rows is NOT possible here: two overlapping windows accumulate
    // into the same pixel. The split is by batch image, which are disjoint regions
    // of the output.
    parallel::parallel_for(N, 1, [&](size_t n_from, size_t n_to) {
        for (size_t n = n_from; n < n_to; ++n) {
            for (size_t oh = 0; oh < oH; ++oh) {
                for (size_t ow = 0; ow < oW; ++ow) {
                    const size_t row = (n * oH + oh) * oW + ow;
                    for (size_t c = 0; c < C; ++c) {
                        for (size_t i = 0; i < kH; ++i) {
                            const long long h = static_cast<long long>(oh * window.stride + i) -
                                                static_cast<long long>(window.padding);
                            if (h < 0 || static_cast<size_t>(h) >= H) continue;

                            for (size_t j = 0; j < kW; ++j) {
                                const long long w = static_cast<long long>(ow * window.stride + j) -
                                                    static_cast<long long>(window.padding);
                                if (w < 0 || static_cast<size_t>(w) >= W) continue;

                                const size_t k = (c * kH + i) * kW + j;
                                // Add, do not assign: with stride < kernel the
                                // windows overlap and several rows contribute
                                // to the same pixel.
                                dst[((n * C + c) * H + static_cast<size_t>(h)) * W +
                                    static_cast<size_t>(w)] += src[row * K + k];
                            }
                        }
                    }
                }
            }
        }
    });
    return out;
}

// ---------------------------------------------------------
// Conv2d
// ---------------------------------------------------------

namespace {

// im2col with its autograd node attached.
//
// It is the only derivative in this layer still written by hand, and it fits on
// one line because its adjoint already existed: col2im scatters each window's
// gradient back to the pixels that formed it and sums the overlaps, which is
// exactly what makes it correct.
//
// The public im2col() stays as it is: the tests use it on its own and it has no
// reason to build a graph.
Tensor im2col_node(const Tensor& input, const Window2d& window) {
    Tensor cols = im2col(input, window);
    if (!autograd::grad_enabled() || !input.requires_grad()) return cols;

    cols.set_requires_grad(true);
    cols.get_impl()->parents = {input.get_impl()};

    Tensor input_copy = input;
    const std::vector<size_t>& in_shape = input.shape();
    const Window2d win = window;

    // col2im throws on a shape it cannot reconcile, and that is the intended
    // behaviour: the exception propagates out of autograd::backward() to
    // whoever asked for the gradient, which is what the tests asserting a
    // throw on mismatched shapes rely on.
    cols.get_impl()->backward_fn = [input_copy, in_shape, win](const Tensor& grad_out) mutable {
        input_copy.add_grad(col2im(grad_out, in_shape, win));
    };
    return cols;
}

}  // namespace

Conv2d::Conv2d(size_t in_channels, size_t out_channels, const Window2d& window, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels), window_(window), use_bias_(use_bias) {
    if (in_channels == 0 || out_channels == 0) {
        throw std::invalid_argument(
            "Conv2d requires in_channels and out_channels greater than zero.");
    }
    if (window_.kernel_h == 0 || window_.kernel_w == 0 || window_.stride == 0) {
        throw std::invalid_argument("Conv2d requires a kernel and a stride greater than zero.");
    }

    // Xavier/Glorot with a convolution's fans:
    // fan_in = C*kH*kW (inputs per neuron), fan_out = outC*kH*kW.
    const size_t receptive = window_.kernel_h * window_.kernel_w;
    const float fan_in = static_cast<float>(in_channels * receptive);
    const float fan_out = static_cast<float>(out_channels * receptive);
    const float limit = std::sqrt(6.0f / (fan_in + fan_out));

    autograd::NoGradGuard no_grad;
    weight_ = Tensor::rand({out_channels, in_channels, window_.kernel_h, window_.kernel_w}, -limit,
                           limit, true);
    bias_ = Tensor({out_channels}, 0.0f, use_bias);
}

Tensor Conv2d::forward(const Tensor& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Conv2d expects a 4D volume (N, C, H, W), received " +
                                    input.shape_str() + ".");
    }
    if (input.shape()[1] != in_channels_) {
        throw std::invalid_argument("Conv2d with in_channels=" + std::to_string(in_channels_) +
                                    " received an input " + input.shape_str() + ".");
    }

    const size_t N = input.shape()[0];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    const size_t oH = window_.out_h(H);
    const size_t oW = window_.out_w(W);
    const size_t spatial = oH * oW;
    const size_t K = in_channels_ * window_.kernel_h * window_.kernel_w;

    // im2col reduces the convolution to a matrix product, and from there nothing
    // has to be written by hand: every operation below carries its own derivative
    // and its own kernel, so the whole convolution -- forward and backward -- goes
    // to the GPU and autograd derives the backward pass.
    //
    // This used to carry a product and a backward_fn of its own, with two parallel
    // passes over disjoint axes to avoid races. It worked, but by not going through
    // Tensor::matmul the backend never saw it: the engine had four tuned product
    // kernels and the convolutions touched none of them. That is what made MNIST
    // gain nothing from the card.
    //
    // There is a trade-off this reverses, and it is worth saying so rather than
    // letting it be discovered: the previous version did NOT keep the columns for
    // the backward -- they take kH*kW times the input -- and recomputed them with a
    // second im2col, trading 5% of time for an order of magnitude of memory.
    // matmul's backward captures `cols`, so now they do stay live between the
    // forward and the backward. It is unavoidable when composing rather than fusing
    // by hand, and it is what every framework does when it composes; in exchange the
    // whole hand-written backward disappears and the layer reaches the GPU.
    Tensor cols = im2col_node(input, window_);  // (N*oH*oW, K)

    // weight_ already stores outC contiguous rows of K values, so viewing it as
    // (outC, K) reinterprets rather than reorders it; the transpose is what leaves
    // it as (K, outC) to multiply from the right.
    Tensor out = cols.matmul(weight_.reshape({out_channels_, K}).transpose());
    if (use_bias_) out = out + bias_;  // broadcast of the channel vector across each row

    // The product comes out ordered (N, oH*oW, outC) and the layer returns
    // (N, outC, oH, oW): swapping the last two axes is all that is left.
    //
    // This used to say that permute({0,2,1}) beat transpose() by 5% here, because
    // transpose strided its writes while permute paid per-element index arithmetic
    // instead -- a measured choice between two variants of the same mistake. Both
    // were limited by memory access and neither blocked, and this swap was running
    // at 2.57 GB/s on a machine that does about 35: 46% of this layer, more than
    // im2col and the matrix product together. They now share one blocked
    // implementation and the choice no longer matters.
    return out.reshape({N, spatial, out_channels_})
        .permute({0, 2, 1})
        .reshape({N, out_channels_, oH, oW});
}

std::vector<Tensor> Conv2d::parameters() {
    if (use_bias_) return {weight_, bias_};
    return {weight_};
}

std::string Conv2d::name() const {
    return "Conv2d(" + std::to_string(in_channels_) + " -> " + std::to_string(out_channels_) +
           ", k=" + std::to_string(window_.kernel_h) + "x" + std::to_string(window_.kernel_w) +
           ", s=" + std::to_string(window_.stride) + ", p=" + std::to_string(window_.padding) + ")";
}

// ---------------------------------------------------------
// MaxPool2d
// ---------------------------------------------------------

MaxPool2d::MaxPool2d(const Window2d& window) : window_(window) {
    // With padding >= kernel there would be windows entirely inside the padded
    // region, with no real value to maximise: the output would be -infinity.
    if (window_.padding >= window_.kernel_h || window_.padding >= window_.kernel_w) {
        throw std::invalid_argument("MaxPool2d requires padding smaller than the kernel; with " +
                                    std::to_string(window_.padding) +
                                    " there would be windows with no real value at all.");
    }
}

MaxPool2d::MaxPool2d(size_t kernel, size_t stride) : window_(kernel, kernel, stride, 0) {}

Tensor MaxPool2d::forward(const Tensor& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("MaxPool2d expects a 4D volume (N, C, H, W), received " +
                                    input.shape_str() + ".");
    }
    const size_t N = input.shape()[0];
    const size_t C = input.shape()[1];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];

    window_.validate(H, W);
    const size_t oH = window_.out_h(H);
    const size_t oW = window_.out_w(W);

    Tensor out({N, C, oH, oW}, 0.0f, false);
    // The winning position of each window, in flat input indices: it is the only
    // thing the backward pass needs saved.
    //
    // It goes in a Tensor rather than a vector<size_t> so it can stay on the
    // device. With the index on the host this layer broke the chain right between
    // the two convolutions and forced the first one's whole output down and back
    // up again.
    Tensor argmax(out.shape(), 0.0f, false);

    const cuda::ops::WindowShape shape{
        N, C, H, W, window_.kernel_h, window_.kernel_w, window_.stride, window_.padding, oH, oW};

    if (!cuda::ops::maxpool(input.storage(), out.storage(), argmax.storage(), shape)) {
        const float* src = input.data();
        float* dst = out.data();
        float* am = argmax.data();

        // It was the only operation in this file not split across threads, with im2col
        // and col2im parallel right above it. Each (n, c) plane writes its own chunk
        // of the output and reads nothing from the others, so splitting by plane
        // crosses no boundary and gives the same result with one thread or with eight.
        //
        const size_t planes = N * C;
        const size_t work_per_plane = oH * oW * window_.kernel_h * window_.kernel_w;
        const size_t planes_per_thread =
            std::max<size_t>(1, parallel::kElementsPerThread / std::max<size_t>(1, work_per_plane));

        parallel::parallel_for(planes, planes_per_thread, [&](size_t from, size_t to) {
            for (size_t p = from; p < to; ++p) {
                const size_t n = p / C;
                const size_t c = p % C;
                for (size_t oh = 0; oh < oH; ++oh) {
                    for (size_t ow = 0; ow < oW; ++ow) {
                        float best = -std::numeric_limits<float>::infinity();
                        size_t best_idx = 0;
                        bool found = false;

                        for (size_t i = 0; i < window_.kernel_h; ++i) {
                            const long long h = static_cast<long long>(oh * window_.stride + i) -
                                                static_cast<long long>(window_.padding);
                            if (h < 0 || static_cast<size_t>(h) >= H) continue;

                            for (size_t j = 0; j < window_.kernel_w; ++j) {
                                const long long w =
                                    static_cast<long long>(ow * window_.stride + j) -
                                    static_cast<long long>(window_.padding);
                                if (w < 0 || static_cast<size_t>(w) >= W) continue;

                                const size_t idx = ((n * C + c) * H + static_cast<size_t>(h)) * W +
                                                   static_cast<size_t>(w);
                                if (!found || src[idx] > best) {
                                    best = src[idx];
                                    best_idx = idx;
                                    found = true;
                                }
                            }
                        }

                        const size_t out_idx = ((n * C + c) * oH + oh) * oW + ow;
                        dst[out_idx] = best;
                        am[out_idx] = static_cast<float>(best_idx);
                    }
                }
            }
        });
    }

    if (!autograd::grad_enabled() || !input.requires_grad()) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = {input.get_impl()};
    Tensor input_copy = input;

    out.get_impl()->backward_fn = [input_copy, argmax, shape](const Tensor& grad_out) mutable {
        // Only the maximum influenced the output, so only it receives gradient.
        Tensor dX(input_copy.shape(), 0.0f, false);
        if (!cuda::ops::maxpool_backward(argmax.storage(), grad_out.storage(), dX.storage(),
                                         shape)) {
            const float* ENGINE_RESTRICT a = argmax.data();
            const float* ENGINE_RESTRICT g = grad_out.data();
            float* ENGINE_RESTRICT d = dX.data();
            // The += is necessary: with a stride smaller than the kernel, two
            // overlapping windows may have chosen the same pixel.
            for (size_t i = 0; i < argmax.size(); ++i) {
                d[static_cast<size_t>(a[i])] += g[i];
            }
        }
        input_copy.add_grad(dX);
    };

    return out;
}

std::string MaxPool2d::name() const {
    return "MaxPool2d(k=" + std::to_string(window_.kernel_h) + "x" +
           std::to_string(window_.kernel_w) + ", s=" + std::to_string(window_.stride) + ")";
}

// ---------------------------------------------------------
// Flatten
// ---------------------------------------------------------

Tensor Flatten::forward(const Tensor& input) {
    if (input.ndim() < 2) {
        throw std::invalid_argument("Flatten expects at least 2 dimensions, received " +
                                    input.shape_str() + ".");
    }
    const size_t N = input.shape()[0];
    if (N == 0) {
        throw std::invalid_argument("Flatten received an empty batch.");
    }
    // reshape already carries its own derivative, so Flatten needs no node.
    return input.reshape({N, input.size() / N});
}

}  // namespace nn
}  // namespace engine
