#ifndef ENGINE_DETAIL_CUDA_OPS_HPP
#define ENGINE_DETAIL_CUDA_OPS_HPP

#include <cstddef>

#include "engine/detail/storage.hpp"

namespace engine {
namespace cuda {
namespace ops {

// ---------------------------------------------------------
// Puntos de entrada a los kernels.
//
// Todas devuelven **true si el trabajo se hizo en el dispositivo**. Devolver
// false significa «aquí no compensa» o «este binario no lleva CUDA», y el
// llamante sigue por el camino de CPU sin enterarse de nada más. Ese contrato
// es lo que mantiene src/tensor.cpp legible: una condición por operación, no
// dos implementaciones enredadas.
//
// Sin ENGINE_CUDA existen igualmente, implementadas en src/cuda_disabled.cpp
// devolviendo false, así que el despacho no necesita #ifdef.
// ---------------------------------------------------------

enum class Binary { Add, Sub, Mul, Div };

// out = a `op` b, con difusión por sufijo: b tiene `inner` elementos que se
// repiten `repeat` veces para cubrir a. Sin difusión, inner == a.size() y
// repeat == 1.
bool binary(Binary op, const Storage& a, const Storage& b, Storage& out,
            size_t inner, size_t repeat);

// out = a x b por lotes. Un operando no lotificado (a_batched/b_batched a
// false) se reutiliza para todas las matrices del lote, igual que en CPU.
bool matmul(const Storage& a, const Storage& b, Storage& out,
            size_t batch, size_t rows, size_t inner_dim, size_t cols,
            bool a_batched, bool b_batched);

bool relu(const Storage& x, Storage& out);
bool relu_backward(const Storage& x, const Storage& grad_out, Storage& out);

// El acumulador del backward: grad = g la primera vez, grad += g después.
//
// No puede ser binary(Add, grad, g, grad, ...) con la salida aliaseada a la
// entrada: esa ruta pide la salida con device_write(), que da el búfer por
// válido **sin subirlo**, y entonces el device() de la entrada ve que ya vale y
// tampoco sube. El acumulado se perdería. Encima el orden de evaluación de los
// argumentos del lanzamiento no está definido, así que dependería del
// compilador. De ahí que tenga kernel propio en lugar de reutilizar binary().
bool accumulate_grad(Storage& grad, const Storage& g, bool initialize);

// out = x * mul + add, en una sola pasada. Las dos operaciones con escalar del
// motor caen aquí: `t * k` es (k, 0) y `t + k` es (1, k). No parece gran cosa,
// pero sin kernel el escalado de la atención —un `* 1/sqrt(d_k)` entre dos
// matmul— bajaba a host el tensor entero y lo volvía a subir.
bool scalar(const Storage& x, Storage& out, float mul, float add);

// Reordenación de ejes: out[plano] = x[sum_d coord_d * src_strides[d]], donde
// las coordenadas se sacan de out_shape. Cubre permute() y transpose(), que es
// el caso particular de intercambiar los dos últimos ejes.
//
// Los dos vectores describen la salida sobre la memoria de la entrada, igual
// que en el camino de CPU: quien llama ya los tiene calculados.
bool permute(const Storage& x, Storage& out,
             const size_t* out_shape, const size_t* src_strides, size_t ndim);

// Suma sobre un eje, viendo el tensor como (outer, axis_len, inner). Es la
// misma descomposición que usa AxisView en src/tensor.cpp.
bool sum_axis(const Storage& x, Storage& out, size_t outer, size_t axis_len, size_t inner);

// Geometría de una ventana deslizante sobre un lote de volúmenes. Repite lo que
// ya dice nn::Window2d más las dimensiones del tensor, a propósito: así esta
// cabecera no depende de engine/conv.hpp y el backend sigue sin saber nada de
// las capas.
struct WindowShape {
    size_t batch, channels, height, width;
    size_t kernel_h, kernel_w, stride, padding;
    size_t out_h, out_w;
};

// im2col aplana cada ventana en una fila: (N,C,H,W) -> (N*oH*oW, C*kH*kW).
// col2im es su adjunto y suma donde las ventanas se solapan.
//
// Sin estos dos, dar kernel al producto de la convolución no sirve de nada: las
// columnas ocupan kH*kW veces la entrada, así que subirlas cuesta más que el
// propio producto. Medido en MNIST: 24,6 s con el producto en la GPU y las
// columnas construidas en host, contra 19,0 s haciéndolo todo en CPU.
bool im2col(const Storage& input, Storage& cols, const WindowShape& s);
bool col2im(const Storage& cols, Storage& input, const WindowShape& s);

// Submuestreo por máximo. `argmax` guarda el índice plano del píxel ganador de
// cada ventana, que es lo único que el paso hacia atrás necesita.
//
// Sin estos dos la cadena se corta justo **entre las dos convoluciones**: la
// salida de la primera baja a host para agruparse y vuelve a subir para la
// segunda. Con lotes de evaluación grandes son decenas de MB por pasada.
//
// El paso hacia atrás recorre la entrada, no la salida, igual que col2im: así
// cada píxel suma las ventanas que lo eligieron sin operaciones atómicas y con
// un orden de acumulación fijo, que es lo que mantiene el resultado
// reproducible.
//
// ponytail: el índice viaja en float, exacto hasta 2^24; por encima de eso el
// despacho se rechaza y lo hace la CPU. Un Storage de enteros sería lo suyo si
// algún día hace falta pasar de ahí.
bool maxpool(const Storage& input, Storage& out, Storage& argmax, const WindowShape& s);
bool maxpool_backward(const Storage& argmax, const Storage& grad_out, Storage& dx,
                      const WindowShape& s);

// Softmax sobre el último eje: `rows` filas de `cols` valores contiguos.
bool softmax(const Storage& x, Storage& out, size_t rows, size_t cols);
// y es la salida guardada del forward, no la entrada.
bool softmax_backward(const Storage& y, const Storage& grad_out, Storage& out,
                      size_t rows, size_t cols);

} // namespace ops
} // namespace cuda
} // namespace engine

#endif // ENGINE_DETAIL_CUDA_OPS_HPP
