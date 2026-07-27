#include "engine/conv.hpp"
#include "engine/autograd.hpp"
#include "engine/parallel.hpp"

#include <limits>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// Window2d
// ---------------------------------------------------------

namespace {

// Mismo criterio que en el tensor: repartir cuesta unos 8 us, así que solo
// compensa a partir de bastante trabajo por región.
constexpr size_t kConvRowsPerThread = 4096;

size_t output_size(size_t in_size, size_t kernel, size_t stride, size_t padding) {
    const size_t padded = in_size + 2 * padding;
    if (kernel > padded) {
        throw std::invalid_argument("El kernel (" + std::to_string(kernel) +
                                    ") no cabe en la dimension de entrada con relleno (" +
                                    std::to_string(padded) + ").");
    }
    return (padded - kernel) / stride + 1;
}

} // namespace

size_t Window2d::out_h(size_t in_h) const { return output_size(in_h, kernel_h, stride, padding); }
size_t Window2d::out_w(size_t in_w) const { return output_size(in_w, kernel_w, stride, padding); }

void Window2d::validate(size_t in_h, size_t in_w) const {
    if (kernel_h == 0 || kernel_w == 0) {
        throw std::invalid_argument("El kernel debe tener ambas dimensiones mayores que cero.");
    }
    if (stride == 0) {
        throw std::invalid_argument("El paso (stride) debe ser mayor que cero.");
    }
    out_h(in_h);
    out_w(in_w);
}

// ---------------------------------------------------------
// im2col / col2im
// ---------------------------------------------------------

Tensor im2col(const Tensor& input, const Window2d& window) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("im2col espera un volumen 4D (N, C, H, W), recibió " +
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

    Tensor cols({N * oH * oW, K}, 0.0f, false);
    const std::vector<float>& src = input.data();
    std::vector<float>& dst = cols.data();

    // Cada fila de salida la escribe una única iteración, así que repartir por
    // filas no crea ninguna carrera.
    parallel::parallel_for(N * oH * oW, kConvRowsPerThread, [&](size_t from, size_t to) {
    for (size_t row = from; row < to; ++row) {
        {
            {
                const size_t n = row / (oH * oW);
                const size_t oh = (row % (oH * oW)) / oW;
                const size_t ow = row % oW;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t i = 0; i < kH; ++i) {
                        // Coordenada en la imagen sin relleno; puede quedar fuera
                        const long long h = static_cast<long long>(oh * window.stride + i) -
                                            static_cast<long long>(window.padding);
                        if (h < 0 || static_cast<size_t>(h) >= H) continue;

                        for (size_t j = 0; j < kW; ++j) {
                            const long long w = static_cast<long long>(ow * window.stride + j) -
                                                static_cast<long long>(window.padding);
                            if (w < 0 || static_cast<size_t>(w) >= W) continue;

                            const size_t k = (c * kH + i) * kW + j;
                            dst[row * K + k] =
                                src[((n * C + c) * H + static_cast<size_t>(h)) * W +
                                    static_cast<size_t>(w)];
                        }
                    }
                }
            }
        }
    }
    });
    return cols;
}

Tensor col2im(const Tensor& cols, const std::vector<size_t>& input_shape, const Window2d& window) {
    if (input_shape.size() != 4) {
        throw std::invalid_argument("col2im necesita una forma de destino 4D (N, C, H, W).");
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
        throw std::invalid_argument("col2im esperaba columnas (" + std::to_string(N * oH * oW) +
                                    ", " + std::to_string(K) + ") y recibió " + cols.shape_str() + ".");
    }

    Tensor out(input_shape, 0.0f, false);
    const std::vector<float>& src = cols.data();
    std::vector<float>& dst = out.data();

    // Aquí NO se puede repartir por filas: dos ventanas solapadas acumulan en
    // el mismo píxel. Se reparte por imagen del lote, que son regiones
    // disjuntas de la salida.
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
                            // Suma, no asignación: con stride < kernel las
                            // ventanas se solapan y varias filas contribuyen
                            // al mismo píxel.
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

Conv2d::Conv2d(size_t in_channels, size_t out_channels, const Window2d& window, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels), window_(window), use_bias_(use_bias) {
    if (in_channels == 0 || out_channels == 0) {
        throw std::invalid_argument("Conv2d requiere in_channels y out_channels mayores que cero.");
    }
    if (window_.kernel_h == 0 || window_.kernel_w == 0 || window_.stride == 0) {
        throw std::invalid_argument("Conv2d requiere un kernel y un paso mayores que cero.");
    }

    // Xavier/Glorot con los abanicos de una convolución:
    // fan_in = C*kH*kW (entradas por neurona), fan_out = outC*kH*kW.
    const size_t receptive = window_.kernel_h * window_.kernel_w;
    const float fan_in = static_cast<float>(in_channels * receptive);
    const float fan_out = static_cast<float>(out_channels * receptive);
    const float limit = std::sqrt(6.0f / (fan_in + fan_out));

    autograd::NoGradGuard no_grad;
    weight_ = Tensor::rand({out_channels, in_channels, window_.kernel_h, window_.kernel_w},
                           -limit, limit, true);
    bias_ = Tensor({out_channels}, 0.0f, use_bias);
}

Tensor Conv2d::forward(const Tensor& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Conv2d espera un volumen 4D (N, C, H, W), recibió " +
                                    input.shape_str() + ".");
    }
    if (input.shape()[1] != in_channels_) {
        throw std::invalid_argument("Conv2d con in_channels=" + std::to_string(in_channels_) +
                                    " recibió una entrada " + input.shape_str() + ".");
    }

    const size_t N = input.shape()[0];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    const size_t oH = window_.out_h(H);
    const size_t oW = window_.out_w(W);
    const size_t spatial = oH * oW;
    const size_t M = N * spatial;
    const size_t K = in_channels_ * window_.kernel_h * window_.kernel_w;

    // im2col reduce la convolución a un producto matricial (M, K) x (K, outC).
    // Los pesos ya están guardados como outC filas contiguas de K valores, así
    // que el producto se hace leyendo cada fila directamente.
    //
    // Las columnas NO se guardan para el backward: ocupan kH*kW veces la
    // entrada (con un kernel 3x3, nueve veces), así que sale mucho más barato
    // guardar la entrada y recalcularlas. Es el mismo compromiso que hace
    // PyTorch: un 5% más de tiempo a cambio de un orden de magnitud de memoria.
    Tensor cols = im2col(input, window_);

    Tensor out({N, out_channels_, oH, oW}, 0.0f, false);
    {
        const std::vector<float>& c = cols.data();
        const std::vector<float>& w = weight_.data();
        const std::vector<float>& b = bias_.data();
        std::vector<float>& o = out.data();

        // Cada m escribe posiciones distintas de la salida: reparto directo.
        const size_t rows_per_thread = std::max<size_t>(1, 32768 / std::max<size_t>(1, K * out_channels_));
        parallel::parallel_for(M, rows_per_thread, [&](size_t from, size_t to) {
            for (size_t m = from; m < to; ++m) {
                const size_t n = m / spatial;
                const size_t p = m % spatial;
                const float* col_row = c.data() + m * K;

                for (size_t oc = 0; oc < out_channels_; ++oc) {
                    const float* w_row = w.data() + oc * K;
                    float acc = use_bias_ ? b[oc] : 0.0f;
                    for (size_t k = 0; k < K; ++k) acc += col_row[k] * w_row[k];
                    o[(n * out_channels_ + oc) * spatial + p] = acc;
                }
            }
        });
    }

    const bool req_g = autograd::grad_enabled() &&
                       (input.requires_grad() || weight_.requires_grad() ||
                        (use_bias_ && bias_.requires_grad()));
    if (!req_g) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = { input.get_impl(), weight_.get_impl() };
    if (use_bias_) out.get_impl()->parents.push_back(bias_.get_impl());

    Tensor input_copy = input;
    Tensor weight_copy = weight_;
    Tensor bias_copy = bias_;
    Window2d win = window_;
    const size_t out_channels = out_channels_;
    const bool use_bias = use_bias_;
    // La lambda la captura por valor, así que aquí basta con la referencia
    const std::vector<size_t>& in_shape = input.shape();

    out.get_impl()->backward_fn =
        [input_copy, weight_copy, bias_copy, win, in_shape,
         M, K, spatial, out_channels, use_bias](const Tensor& grad_out) mutable {
            const std::vector<float>& g = grad_out.data();
            const Tensor cols = im2col(input_copy, win); // recalculadas, no guardadas
            const std::vector<float>& c = cols.data();
            const std::vector<float>& w = weight_copy.data();

            Tensor dW(weight_copy.shape(), 0.0f, false);
            Tensor db(bias_copy.shape(), 0.0f, false);
            Tensor dcols(cols.shape(), 0.0f, false);

            std::vector<float>& dw = dW.data();
            std::vector<float>& dbv = db.data();
            std::vector<float>& dc = dcols.data();

            // Se recorre dos veces, no una, para poder repartir sin carreras.
            // En un solo recorrido sobre m, dW y db los escribirían todos los
            // hilos a la vez; separando las pasadas cada una tiene su propio
            // eje disjunto y el orden de acumulación sigue siendo fijo, así que
            // el resultado no depende del número de hilos.

            // Pasada 1, repartida por m: dcols = dOut x W^T. Cada m escribe su
            // propia fila de dcols.
            const size_t rows_per_thread =
                std::max<size_t>(1, 32768 / std::max<size_t>(1, K * out_channels));
            parallel::parallel_for(M, rows_per_thread, [&](size_t from, size_t to) {
                for (size_t m = from; m < to; ++m) {
                    const size_t n = m / spatial;
                    const size_t p = m % spatial;
                    float* dcol_row = dc.data() + m * K;

                    for (size_t oc = 0; oc < out_channels; ++oc) {
                        const float grad = g[(n * out_channels + oc) * spatial + p];
                        if (grad == 0.0f) continue;
                        const float* w_row = w.data() + oc * K;
                        for (size_t k = 0; k < K; ++k) dcol_row[k] += grad * w_row[k];
                    }
                }
            });

            // Pasada 2, repartida por canal de salida: dW = cols^T x dOut y
            // db = suma por canal. Cada oc escribe su propia fila de dW y su
            // propia entrada de db.
            parallel::parallel_for(out_channels, 1, [&](size_t oc_from, size_t oc_to) {
                for (size_t oc = oc_from; oc < oc_to; ++oc) {
                    float* dw_row = dw.data() + oc * K;
                    float bias_sum = 0.0f;

                    for (size_t m = 0; m < M; ++m) {
                        const size_t n = m / spatial;
                        const size_t p = m % spatial;
                        const float grad = g[(n * out_channels + oc) * spatial + p];
                        if (grad == 0.0f) continue;

                        const float* col_row = c.data() + m * K;
                        bias_sum += grad;
                        for (size_t k = 0; k < K; ++k) dw_row[k] += grad * col_row[k];
                    }
                    dbv[oc] = bias_sum;
                }
            });

            if (weight_copy.requires_grad()) weight_copy.add_grad(dW);
            if (use_bias && bias_copy.requires_grad()) bias_copy.add_grad(db);
            if (input_copy.requires_grad()) {
                // col2im es el adjunto de im2col: reparte el gradiente de cada
                // ventana a los píxeles que la formaron, sumando los solapes.
                input_copy.add_grad(col2im(dcols, in_shape, win));
            }
        };

    return out;
}

std::vector<Tensor> Conv2d::parameters() {
    if (use_bias_) return { weight_, bias_ };
    return { weight_ };
}

std::string Conv2d::name() const {
    return "Conv2d(" + std::to_string(in_channels_) + " -> " + std::to_string(out_channels_) +
           ", k=" + std::to_string(window_.kernel_h) + "x" + std::to_string(window_.kernel_w) +
           ", s=" + std::to_string(window_.stride) +
           ", p=" + std::to_string(window_.padding) + ")";
}

// ---------------------------------------------------------
// MaxPool2d
// ---------------------------------------------------------

MaxPool2d::MaxPool2d(const Window2d& window) : window_(window) {
    // Con relleno >= kernel habría ventanas enteramente dentro de la zona
    // rellenada, sin ningún valor real que maximizar: la salida sería -infinito.
    if (window_.padding >= window_.kernel_h || window_.padding >= window_.kernel_w) {
        throw std::invalid_argument(
            "MaxPool2d requiere un relleno menor que el kernel; con " +
            std::to_string(window_.padding) + " habría ventanas sin ningún valor real.");
    }
}

MaxPool2d::MaxPool2d(size_t kernel, size_t stride)
    : window_(kernel, kernel, stride, 0) {}

Tensor MaxPool2d::forward(const Tensor& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("MaxPool2d espera un volumen 4D (N, C, H, W), recibió " +
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
    // Posición ganadora de cada ventana, en índices planos de la entrada:
    // es lo único que hace falta guardar para la propagación hacia atrás.
    std::vector<size_t> argmax(out.size(), 0);

    const std::vector<float>& src = input.data();
    std::vector<float>& dst = out.data();

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
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
                            const long long w = static_cast<long long>(ow * window_.stride + j) -
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
                    argmax[out_idx] = best_idx;
                }
            }
        }
    }

    if (!autograd::grad_enabled() || !input.requires_grad()) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = { input.get_impl() };
    Tensor input_copy = input;

    out.get_impl()->backward_fn = [input_copy, argmax](const Tensor& grad_out) mutable {
        // Solo el máximo influyó en la salida, así que solo él recibe gradiente.
        Tensor dX(input_copy.shape(), 0.0f, false);
        for (size_t i = 0; i < argmax.size(); ++i) {
            dX.data()[argmax[i]] += grad_out.data()[i];
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
        throw std::invalid_argument("Flatten espera al menos 2 dimensiones, recibió " +
                                    input.shape_str() + ".");
    }
    const size_t N = input.shape()[0];
    if (N == 0) {
        throw std::invalid_argument("Flatten recibió un lote vacío.");
    }
    // reshape ya lleva su propia derivada, así que Flatten no necesita nodo propio.
    return input.reshape({N, input.size() / N});
}

} // namespace nn
} // namespace engine
