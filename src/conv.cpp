#include "engine/conv.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/cuda_ops.hpp"
#include "engine/detail/restrict.hpp"
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

    // Primero se ofrece al dispositivo. Si se hace cargo, las columnas se quedan
    // arriba y el producto que viene detrás las lee sin que crucen el PCIe: son
    // kH*kW veces la entrada, así que subirlas cuesta más que multiplicarlas.
    if (cuda::ops::im2col(input.get_impl()->storage, cols.get_impl()->storage,
                          {N, C, H, W, kH, kW, window.stride, window.padding, oH, oW})) {
        return cols;
    }

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

    if (cuda::ops::col2im(cols.get_impl()->storage, out.get_impl()->storage,
                          {N, C, H, W, kH, kW, window.stride, window.padding, oH, oW})) {
        return out;
    }

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

namespace {

// im2col con su nodo de autograd colgado.
//
// Es la única derivada de esta capa que sigue escrita a mano, y cabe en una
// línea porque su adjunto ya existía: col2im reparte el gradiente de cada
// ventana a los píxeles que la formaron y suma los solapes, que es exactamente
// lo que la hace correcta.
//
// La función pública im2col() se queda como está: las pruebas la usan suelta y
// no tiene por qué construir grafo.
Tensor im2col_node(const Tensor& input, const Window2d& window) {
    Tensor cols = im2col(input, window);
    if (!autograd::grad_enabled() || !input.requires_grad()) return cols;

    cols.set_requires_grad(true);
    cols.get_impl()->parents = { input.get_impl() };

    Tensor input_copy = input;
    const std::vector<size_t> in_shape = input.shape();
    const Window2d win = window;

    cols.get_impl()->backward_fn = [input_copy, in_shape, win](const Tensor& grad_out) mutable {
        input_copy.add_grad(col2im(grad_out, in_shape, win));
    };
    return cols;
}

} // namespace

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
    const size_t K = in_channels_ * window_.kernel_h * window_.kernel_w;

    // im2col reduce la convolución a un producto matricial, y a partir de ahí
    // no hay nada que escribir a mano: cada operación de las de abajo trae su
    // propia derivada y su propio kernel, así que la convolución entera —ida y
    // vuelta— se va a la GPU y el paso hacia atrás lo deduce autograd.
    //
    // Antes esto llevaba un producto y un backward_fn propios, con dos pasadas
    // paralelas sobre ejes disjuntos para no tener carreras. Funcionaba, pero
    // por no pasar por Tensor::matmul el backend no lo veía nunca: el motor
    // tenía cuatro kernels de producto afinados y las convoluciones no tocaban
    // ninguno. Es lo que hacía que MNIST no ganara nada con la tarjeta.
    //
    // Hay un compromiso que esto invierte, y conviene decirlo en vez de dejar
    // que se descubra: la versión anterior NO guardaba las columnas para el
    // backward —ocupan kH*kW veces la entrada— y las recalculaba con un segundo
    // im2col, cambiando un 5% de tiempo por un orden de magnitud de memoria. El
    // backward de matmul captura `cols`, así que ahora sí quedan vivas entre la
    // ida y la vuelta. Es inevitable al componer en lugar de fusionar a mano, y
    // es lo que hace cualquier framework cuando compone; a cambio desaparece el
    // backward propio entero y la capa entra en la GPU.
    Tensor cols = im2col_node(input, window_);   // (N*oH*oW, K)

    // weight_ ya guarda outC filas contiguas de K valores, así que verlo como
    // (outC, K) es reinterpretarlo y no reordenarlo; la transposición es la que
    // lo deja en (K, outC) para multiplicar por la derecha.
    Tensor out = cols.matmul(weight_.reshape({out_channels_, K}).transpose());
    if (use_bias_) out = out + bias_;  // difusión del vector de canales por fila

    // El producto sale ordenado (N, oH*oW, outC) y la capa devuelve
    // (N, outC, oH, oW): intercambiar los dos últimos ejes es todo lo que falta.
    //
    // permute({0,2,1}) y no transpose(), aunque sobre un tensor 3D hagan lo
    // mismo y transpose tenga bucle propio: medido, transpose sale un 5% peor
    // aquí. Escribe con paso y lee contiguo, y a este tamaño mandan los accesos
    // a memoria, no la aritmética de índices que permute hace por elemento.
    return out.reshape({N, spatial, out_channels_})
              .permute({0, 2, 1})
              .reshape({N, out_channels_, oH, oW});
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
    // Posición ganadora de cada ventana, en índices planos de la entrada: es lo
    // único que hace falta guardar para la propagación hacia atrás.
    //
    // Va en un Tensor y no en un vector<size_t> para que pueda quedarse en el
    // dispositivo. Con el índice en host, esta capa cortaba la cadena justo
    // entre las dos convoluciones y obligaba a bajar y volver a subir la salida
    // entera de la primera.
    Tensor argmax(out.shape(), 0.0f, false);

    const cuda::ops::WindowShape shape{N, C, H, W, window_.kernel_h, window_.kernel_w,
                                       window_.stride, window_.padding, oH, oW};

    if (!cuda::ops::maxpool(input.get_impl()->storage, out.get_impl()->storage,
                            argmax.get_impl()->storage, shape)) {
        const std::vector<float>& src = input.data();
        std::vector<float>& dst = out.data();
        float* am = argmax.data().data();

        // Era la única operación de este fichero sin repartir, teniendo im2col y
        // col2im paralelos justo encima. Cada plano (n, c) escribe su propio
        // trozo de la salida y no lee nada de los demás, así que el reparto por
        // planos no cruza ninguna frontera y da el mismo resultado con uno o con
        // ocho hilos.
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
                        am[out_idx] = static_cast<float>(best_idx);
                    }
                }
            }
        });
    }

    if (!autograd::grad_enabled() || !input.requires_grad()) return out;

    out.set_requires_grad(true);
    out.get_impl()->parents = { input.get_impl() };
    Tensor input_copy = input;

    out.get_impl()->backward_fn =
        [input_copy, argmax, shape](const Tensor& grad_out) mutable {
            // Solo el máximo influyó en la salida, así que solo él recibe gradiente.
            Tensor dX(input_copy.shape(), 0.0f, false);
            if (!cuda::ops::maxpool_backward(argmax.get_impl()->storage,
                                             grad_out.get_impl()->storage,
                                             dX.get_impl()->storage, shape)) {
                const float* ENGINE_RESTRICT a = argmax.data().data();
                const float* ENGINE_RESTRICT g = grad_out.data().data();
                float* ENGINE_RESTRICT d = dX.data().data();
                // El += es necesario: con paso menor que el kernel dos ventanas
                // solapadas pueden haber elegido el mismo píxel.
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
