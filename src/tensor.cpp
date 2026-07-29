#include "engine/tensor.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/restrict.hpp"
#include "engine/parallel.hpp"
#include "engine/detail/cuda_ops.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "engine/random.hpp"

namespace engine {

// ---------------------------------------------------------
// Generador aleatorio global
// ---------------------------------------------------------

std::mt19937& global_rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

void manual_seed(uint64_t seed) {
    global_rng().seed(static_cast<std::mt19937::result_type>(seed));
}

namespace {

// El grafo solo se construye si el tensor lo pide y el modo autograd está
// activo (durante el backward y dentro de los optimizadores está desactivado).
// Umbrales de reparto, calibrados con el coste medido de despachar una región
// paralela: unos 8 us con cuatro hilos en esta máquina. Por debajo de unos
// 100 us de trabajo, repartir sale más caro que calcular.
//
// El primer intento usó umbrales diez veces menores y los ejemplos se volvieron
// más LENTOS: el del Transformer pasó de 15,9 s a 22,6 s, porque encadena
// muchos productos pequeños y cada uno pagaba la sincronización sin ganar nada.

// Un producto de 1M de multiplicaciones y sumas son unos 130 us a 15 GFLOP/s.
constexpr size_t kMatmulParallelFloor = 1u << 20;
// Trozos de ~65k operaciones: bastantes para que el reparto dinámico equilibre,
// sin que el coste de pedir el siguiente trozo pese.
constexpr size_t kMatmulChunkWork = 1u << 16;

using parallel::kElementsPerThread;

// Devuelve 0 —«ejecuta en línea»— cuando el producto no da para repartir.
inline size_t matmul_rows_per_thread(size_t rows, size_t K, size_t N) {
    const size_t per_row = std::max<size_t>(1, K * N);
    if (rows * per_row < kMatmulParallelFloor) return 0;
    return std::max<size_t>(1, kMatmulChunkWork / per_row);
}

inline bool track(bool requires_grad) {
    return requires_grad && autograd::grad_enabled();
}

size_t product(const std::vector<size_t>& dims, size_t from = 0, size_t to = 0) {
    const size_t end = (to == 0) ? dims.size() : to;
    size_t total = 1;
    for (size_t i = from; i < end; ++i) total *= dims[i];
    return total;
}

// Difusión por sufijo: `other` se difunde sobre `base` si, tras descartar sus
// unos iniciales, su forma es un sufijo de la de `base`. Cubre el sesgo de una
// capa densa (1, N) sobre (M, N), la codificación posicional (S, D) sobre
// (B, S, D) y la máscara (S, S) sobre (B, H, S, S).
//
// Como el tensor es contiguo en orden C, la difusión sobre los ejes iniciales
// se reduce a repetir el bloque final: base[i] + other[i % inner].
struct BroadcastPlan {
    bool valid = false;
    size_t inner = 1;   // tamaño del bloque que se repite
    size_t repeat = 1;  // cuántas veces se repite
};

BroadcastPlan plan_broadcast(const std::vector<size_t>& base, const std::vector<size_t>& other) {
    BroadcastPlan plan;

    size_t first = 0;
    while (first < other.size() && other[first] == 1) ++first;
    const size_t core = other.size() - first;

    if (core > base.size()) return plan;

    const size_t offset = base.size() - core;
    for (size_t i = 0; i < core; ++i) {
        if (base[offset + i] != other[first + i]) return plan;
    }

    plan.valid = true;
    plan.inner = product(base, offset);
    // El número de repeticiones se deduce del total, no de un producto parcial:
    // con offset == 0 (mismo rango tras descartar unos iniciales) un producto
    // sobre el rango vacío se confundiría con el producto completo.
    plan.repeat = (plan.inner == 0) ? 0 : product(base) / plan.inner;
    return plan;
}

// Materializa el operando difundido con la forma de la base. Solo se usa en
// las derivadas, donde tener ambas formas iguales simplifica las fórmulas.
Tensor expand_operand(const Tensor& other, const std::vector<size_t>& base_shape,
                      size_t total, size_t inner) {
    // Los accesores se izan fuera del bucle: data() no es un puntero, es la
    // puerta al lado host, y llamarla por elemento comprueba la validez del
    // espejo del dispositivo una vez por valor copiado.
    Tensor full(base_shape, 0.0f, false);
    const float* ENGINE_RESTRICT src = other.data().data();
    float* ENGINE_RESTRICT dst = full.data().data();
    for (size_t i = 0; i < total; ++i) dst[i] = src[i % inner];
    return full;
}

// Suma los ejes iniciales de `full` hasta dejar la forma `target`. Es el
// adjunto de difundir un operando sobre un lote, y lo usan tanto la suma
// difundida como el matmul con un operando compartido.
Tensor fold_leading(const Tensor& full, const std::vector<size_t>& target) {
    const size_t inner = product(target);
    const size_t repeat = full.size() / inner;

    Tensor folded(target, 0.0f, false);
    const float* ENGINE_RESTRICT src = full.data().data();
    float* ENGINE_RESTRICT dst = folded.data().data();
    for (size_t r = 0; r < repeat; ++r) {
        for (size_t j = 0; j < inner; ++j) {
            dst[j] += src[r * inner + j];
        }
    }
    return folded;
}

// Un tensor nuevo con los mismos valores, otra forma y sin historial.
//
// La diferencia con Tensor(shape, data(), ...) es que ese data() **es una
// bajada a host**: el búfer se copia igual, pero pasando por el PCIe dos veces
// si el tensor vivía en la GPU. clone() copia conservando el lado en el que
// está. Lo usan reshape() y detach(), que son justo las dos que se aplican a
// salidas recién calculadas por un kernel.
Tensor clone_with_shape(const Storage& src, const std::vector<size_t>& shape, bool req_grad) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = src.clone();
    impl->shape = shape;
    impl->compute_strides();
    impl->requires_grad = req_grad;
    return Tensor::from_impl(impl);
}

// Ofrece la operación al backend CUDA. Devuelve true si la GPU se hizo cargo;
// false significa «no hay dispositivo» o «a este tamaño no compensa», y el
// llamante sigue por el camino de CPU. Sin CUDA la llamada es una función que
// devuelve false y el enlazador la elimina.
//
// Traducción de la difusión: el operando derecho tiene `inner` elementos que se
// repiten `repeat` veces, exactamente el mismo plan que usa el bucle de CPU.
bool offer_to_device(cuda::ops::Binary op, const Tensor& a, const Tensor& b, Tensor& out,
                     bool broadcast, const BroadcastPlan& plan) {
    const size_t inner = broadcast ? plan.inner : a.size();
    const size_t repeat = broadcast ? plan.repeat : 1;
    return cuda::ops::binary(op, a.get_impl()->storage, b.get_impl()->storage,
                             out.get_impl()->storage, inner, repeat);
}

} // namespace

// ---------------------------------------------------------
// Implementación de TensorImpl
// ---------------------------------------------------------

TensorImpl::TensorImpl(const std::vector<size_t>& s, float fill_val, bool req_grad)
    : shape(s), requires_grad(req_grad) {
    compute_strides();
    size_t total_elements = 1;
    for (size_t dim : shape) total_elements *= dim;
    storage.assign(total_elements, fill_val);
}

TensorImpl::TensorImpl(const std::vector<size_t>& s, const std::vector<float>& d, bool req_grad)
    : storage(d), shape(s), requires_grad(req_grad) {
    compute_strides();
    size_t total_elements = 1;
    for (size_t dim : shape) total_elements *= dim;
    if (storage.size() != total_elements) {
        throw std::invalid_argument("El número de elementos en data no coincide con la forma dada.");
    }
}

void TensorImpl::compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

size_t TensorImpl::get_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Número de índices incompatible con ndim.");
    }
    size_t flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("Índice fuera de rango.");
        }
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

// ---------------------------------------------------------
// Implementación de la clase wrapper Tensor
// ---------------------------------------------------------

Tensor::Tensor()
    : impl_(std::make_shared<TensorImpl>(std::vector<size_t>{0}, 0.0f, false)) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument("No se puede construir un Tensor sobre una implementación nula.");
    }
}

Tensor::Tensor(const std::vector<size_t>& shape, float fill_value, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, fill_value, requires_grad)) {}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, data, requires_grad)) {}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(std::move(impl));
}

// Métodos estáticos de fábrica
Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    return Tensor(shape, 0.0f, requires_grad);
}

Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    return Tensor(shape, 1.0f, requires_grad);
}

Tensor Tensor::rand(const std::vector<size_t>& shape, float min_val, float max_val, bool requires_grad) {
    Tensor t(shape, 0.0f, requires_grad);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dis(global_rng());
    }
    return t;
}

Tensor Tensor::randn(const std::vector<size_t>& shape, float mean, float stddev, bool requires_grad) {
    Tensor t(shape, 0.0f, requires_grad);
    std::normal_distribution<float> dis(mean, stddev);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dis(global_rng());
    }
    return t;
}

// Métodos Autograd y Gradientes
bool Tensor::requires_grad() const {
    return impl_->requires_grad;
}

void Tensor::set_requires_grad(bool req_grad) {
    impl_->requires_grad = req_grad;
}

Tensor Tensor::grad() const {
    if (!impl_->grad) {
        throw std::runtime_error("El tensor no tiene gradiente calculado.");
    }
    return Tensor(impl_->grad);
}

bool Tensor::has_grad() const {
    return impl_->grad != nullptr;
}

void Tensor::zero_grad() {
    // assign() y no host_mut() + fill(): host_mut() sincroniza, o sea que se
    // baja del dispositivo un gradiente que sólo vamos a machacar con ceros, una
    // vez por parámetro y por paso. assign() marca host como bueno y dispositivo
    // como obsoleto sin copiar nada, y conserva la reserva de GPU porque el
    // tamaño no cambia.
    if (impl_->grad) {
        impl_->grad->storage.assign(impl_->grad->storage.size(), 0.0f);
    }
}

void Tensor::add_grad(const Tensor& g) {
    if (!requires_grad()) return;
    if (g.shape() != shape()) {
        throw std::invalid_argument("Forma del gradiente " + g.shape_str() +
                                    " incompatible con la del tensor " + shape_str() + ".");
    }
    // Aquí se decidía toda la residencia del backward sin querer. Construir el
    // gradiente desde g.data() lo bajaba del dispositivo, y autograd se lo pasa
    // tal cual al siguiente backward_fn, que tenía que volver a subirlo: ni un
    // solo nodo se quedaba en GPU, daba igual el tamaño y daban igual los
    // umbrales. El forward lleva esta invariante bien resuelta desde la Fase 6;
    // esto es aplicarle al gradiente la misma.
    //
    // La rama que importa es la primera escritura, no el +=: autograd descarta
    // el gradiente de los nodos intermedios antes del recorrido y lo libera en
    // cuanto se consume, así que un nodo con un solo consumidor —el caso
    // normal— entra siempre por aquí y nunca por la acumulación.
    const bool initialize = !impl_->grad;
    // A ceros para que el camino de CPU sea un único bucle: sobre ceros, sumar
    // es asignar. Copiar el Storage de g no valdría de atajo, porque su
    // constructor de copia se lleva sólo el lado host y para eso sincroniza:
    // sería exactamente la bajada que estamos quitando.
    if (initialize) impl_->grad = std::make_shared<TensorImpl>(shape(), 0.0f, false);

    Storage& acc = impl_->grad->storage;
    if (!cuda::ops::accumulate_grad(acc, g.get_impl()->storage, initialize)) {
        // g.data() se queda dentro del if a propósito: fuera, la bajada volvería
        // por la puerta de atrás en el camino que sí acelera el dispositivo.
        std::vector<float>& dst = acc.host_mut();
        const float* ENGINE_RESTRICT src = g.data().data();
        float* ENGINE_RESTRICT out = dst.data();
        parallel::parallel_for(dst.size(), kElementsPerThread, [&](size_t from, size_t to) {
            for (size_t i = from; i < to; ++i) out[i] += src[i];
        });
    }
}

void Tensor::backward() {
    autograd::backward(*this);
}

void Tensor::backward(const Tensor& grad_output) {
    if (grad_output.shape() != shape()) {
        throw std::invalid_argument("Forma del gradiente inicial " + grad_output.shape_str() +
                                    " incompatible con la de la raíz " + shape_str() + ".");
    }
    impl_->grad = std::make_shared<TensorImpl>(shape(), grad_output.data(), false);
    autograd::backward(*this);
}

// Accessors y propiedades
size_t Tensor::get_flat_index(const std::vector<size_t>& indices) const {
    return impl_->get_flat_index(indices);
}

float& Tensor::operator()(const std::vector<size_t>& indices) {
    return impl_->storage[get_flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    return impl_->storage[get_flat_index(indices)];
}

float& Tensor::operator()(size_t row, size_t col) {
    return const_cast<float&>(static_cast<const Tensor&>(*this)(row, col));
}

const float& Tensor::operator()(size_t row, size_t col) const {
    if (ndim() != 2) {
        throw std::invalid_argument("La indexacion (fila, columna) requiere un tensor 2D, este es " +
                                    shape_str() + ".");
    }
    if (row >= shape()[0] || col >= shape()[1]) {
        throw std::out_of_range("Índice (" + std::to_string(row) + ", " + std::to_string(col) +
                                ") fuera de rango para un tensor " + shape_str() + ".");
    }
    return impl_->storage[row * shape()[1] + col];
}

float& Tensor::at(size_t flat_index) {
    return impl_->storage.at(flat_index);
}

const float& Tensor::at(size_t flat_index) const {
    return impl_->storage.at(flat_index);
}

const std::vector<size_t>& Tensor::shape() const { return impl_->shape; }
const std::vector<size_t>& Tensor::strides() const { return impl_->strides; }
// data() es la puerta al lado host. La sobrecarga mutable marca de paso el
// espejo del dispositivo como obsoleto, que es lo que mantiene coherentes las
// dos copias sin que ninguna operación tenga que acordarse de hacerlo.
const std::vector<float>& Tensor::data() const { return impl_->storage.host(); }
std::vector<float>& Tensor::data() { return impl_->storage.host_mut(); }
size_t Tensor::size() const { return impl_->storage.size(); }
size_t Tensor::ndim() const { return impl_->shape.size(); }

std::string Tensor::shape_str() const {
    std::string s = "(";
    for (size_t i = 0; i < shape().size(); ++i) {
        s += std::to_string(shape()[i]);
        if (i + 1 < shape().size()) s += ", ";
    }
    return s + ")";
}

Tensor Tensor::detach() const {
    // Aquí es donde se guarda la salida que el paso hacia atrás va a necesitar
    // —el jacobiano del softmax depende de ella—, así que pasaba una vez por
    // host por cada softmax del modelo aunque el tensor fuera a quedarse arriba.
    return clone_with_shape(impl_->storage, shape(), false);
}

// ---------------------------------------------------------
// Operadores Matemáticos con Registro Autograd
//
// Todas las lambdas reciben el gradiente de salida (grad_out) como parámetro
// y solo capturan los tensores de entrada, nunca el resultado: capturar el
// resultado formaría un ciclo de shared_ptr y el grafo nunca se liberaría.
// ---------------------------------------------------------

// Suma de tensores, con difusión por sufijo del operando derecho
Tensor Tensor::operator+(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Formas incompatibles para suma de tensores: " +
                                        shape_str() + " y " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Add, *this, other, res, broadcast, plan)) {
        // Los accesores se izan fuera del bucle: llamarlos por elemento impide
        // que el compilador vectorice.
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] + rhs[i];
            });
        } else {
            // Por bloques en lugar de con un módulo por elemento: el operando
            // difundido se repite tal cual, y así el bucle interno vectoriza.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] + rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy, broadcast, plan](const Tensor& grad_out) mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out);
            if (!other_copy.requires_grad()) return;
            if (!broadcast) {
                other_copy.add_grad(grad_out);
            } else {
                // El operando difundido se repitió `repeat` veces, así que su
                // gradiente es la suma de todas esas copias.
                Tensor db(other_copy.shape(), 0.0f, false);
                for (size_t r = 0; r < plan.repeat; ++r) {
                    for (size_t j = 0; j < plan.inner; ++j) {
                        db.data()[j] += grad_out.data()[r * plan.inner + j];
                    }
                }
                other_copy.add_grad(db);
            }
        };
    }
    return res;
}

// Resta de tensores, con difusión del operando derecho
Tensor Tensor::operator-(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Formas incompatibles para resta de tensores: " +
                                        shape_str() + " y " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Sub, *this, other, res, broadcast, plan)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] - rhs[i];
            });
        } else {
            // Por bloques en lugar de con un módulo por elemento: el operando
            // difundido se repite tal cual, y así el bucle interno vectoriza.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] - rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn =
            [self_copy, other_copy, broadcast](const Tensor& grad_out) mutable {
                if (self_copy.requires_grad()) self_copy.add_grad(grad_out);
                if (!other_copy.requires_grad()) return;
                // El operando difundido recoge la suma de todas sus copias.
                // La resta no necesita el plan: su derivada no depende de los
                // valores del operando, solo de su forma.
                Tensor d = grad_out * -1.0f;
                other_copy.add_grad(broadcast ? fold_leading(d, other_copy.shape()) : d);
            };
    }
    return res;
}

// Multiplicación elemento a elemento (Hadamard), con difusión del operando derecho
Tensor Tensor::operator*(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Formas incompatibles para multiplicación de tensores: " +
                                        shape_str() + " y " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Mul, *this, other, res, broadcast, plan)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] * rhs[i];
            });
        } else {
            // Por bloques en lugar de con un módulo por elemento: el operando
            // difundido se repite tal cual, y así el bucle interno vectoriza.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] * rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn =
            [self_copy, other_copy, broadcast, plan](const Tensor& grad_out) mutable {
                const Tensor rhs = broadcast
                    ? expand_operand(other_copy, self_copy.shape(), self_copy.size(), plan.inner)
                    : other_copy;
                if (self_copy.requires_grad()) self_copy.add_grad(grad_out * rhs);
                if (other_copy.requires_grad()) {
                    Tensor d = grad_out * self_copy;
                    other_copy.add_grad(broadcast ? fold_leading(d, other_copy.shape()) : d);
                }
            };
    }
    return res;
}

// División elemento a elemento, con difusión del operando derecho
Tensor Tensor::operator/(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    BroadcastPlan plan;
    if (broadcast) {
        plan = plan_broadcast(shape(), other.shape());
        if (!plan.valid) {
            throw std::invalid_argument("Formas incompatibles para división de tensores: " +
                                        shape_str() + " y " + other.shape_str() + ".");
        }
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!offer_to_device(cuda::ops::Binary::Div, *this, other, res, broadcast, plan)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        if (!broadcast) {
            parallel::parallel_for(n, kElementsPerThread, [&](size_t from, size_t to) {
                for (size_t i = from; i < to; ++i) out[i] = lhs[i] / rhs[i];
            });
        } else {
            // Por bloques en lugar de con un módulo por elemento: el operando
            // difundido se repite tal cual, y así el bucle interno vectoriza.
            const size_t inner = plan.inner;
            for (size_t r = 0; r < plan.repeat; ++r) {
                const float* ENGINE_RESTRICT l = lhs + r * inner;
                float* ENGINE_RESTRICT o = out + r * inner;
                for (size_t j = 0; j < inner; ++j) o[j] = l[j] / rhs[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn =
            [self_copy, other_copy, broadcast, plan](const Tensor& grad_out) mutable {
                const Tensor rhs = broadcast
                    ? expand_operand(other_copy, self_copy.shape(), self_copy.size(), plan.inner)
                    : other_copy;
                if (self_copy.requires_grad()) self_copy.add_grad(grad_out / rhs);
                if (other_copy.requires_grad()) {
                    // d/db (a/b) = -a/b^2
                    Tensor d = (grad_out * self_copy * -1.0f) / (rhs * rhs);
                    other_copy.add_grad(broadcast ? fold_leading(d, other_copy.shape()) : d);
                }
            };
    }
    return res;
}

// Suma con escalar
Tensor Tensor::operator+(float scalar) const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    // out = x * 1 + k. Los accesores se quedan dentro del else: llamarlos fuera
    // bajaria el tensor a host justo en el camino que no lo necesita.
    if (!cuda::ops::scalar(impl_->storage, res.impl_->storage, 1.0f, scalar)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] + scalar;
    }
    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out);
        };
    }
    return res;
}

// Resta con escalar
Tensor Tensor::operator-(float scalar) const {
    return (*this) + (-scalar);
}

// Multiplicación por escalar
Tensor Tensor::operator*(float scalar) const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    // out = x * k + 0. Es el escalado de la atencion, entre dos matmul que sí
    // tienen kernel: sin este, la cadena bajaba a host justo en medio.
    if (!cuda::ops::scalar(impl_->storage, res.impl_->storage, scalar, 0.0f)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] * scalar;
    }
    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, scalar](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out * scalar);
        };
    }
    return res;
}

// División por escalar
Tensor Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::invalid_argument("División por cero.");
    }
    return (*this) * (1.0f / scalar);
}

// Transposición: intercambia los dos últimos ejes.
// Para un tensor 2D es la transposición de toda la vida; para (B, ..., M, N)
// transpone cada matriz del lote, que es lo que necesita la atención.
Tensor Tensor::transpose() const {
    if (ndim() < 2) {
        throw std::invalid_argument("Transpose requiere al menos 2 dimensiones, este tensor es " +
                                    shape_str() + ".");
    }
    const size_t nd = ndim();
    const size_t rows = shape()[nd - 2];
    const size_t cols = shape()[nd - 1];
    const size_t batch = size() / (rows * cols);

    std::vector<size_t> out_shape = shape();
    std::swap(out_shape[nd - 2], out_shape[nd - 1]);

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);

    // Transponer es permutar intercambiando los dos ultimos ejes, asi que
    // comparte kernel con permute(): los strides de la entrada leidos en el
    // orden de la salida son toda la diferencia entre las dos operaciones.
    std::vector<size_t> src_strides = strides();
    std::swap(src_strides[nd - 2], src_strides[nd - 1]);

    if (!cuda::ops::permute(impl_->storage, res.impl_->storage,
                            out_shape.data(), src_strides.data(), nd)) {
        const float* ENGINE_RESTRICT src_base = data().data();
        float* ENGINE_RESTRICT dst_base = res.data().data();
        // Se reparte por fila de origen: cada una escribe una columna entera del
        // destino, y ninguna toca lo de las demás. El umbral se cuenta en filas
        // porque el trabajo de cada una son sus `cols` elementos.
        const size_t rows_per_thread =
            std::max<size_t>(1, kElementsPerThread / std::max<size_t>(1, cols));
        parallel::parallel_for(batch * rows, rows_per_thread, [&](size_t from, size_t to) {
            for (size_t r = from; r < to; ++r) {
                const size_t b = r / rows;
                const size_t i = r % rows;
                const float* src = src_base + b * rows * cols + i * cols;
                float* dst = dst_base + b * rows * cols + i;
                for (size_t j = 0; j < cols; ++j) dst[j * rows] = src[j];
            }
        });
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out.transpose());
        };
    }
    return res;
}

// Permutación general de ejes: order[i] indica qué eje de la entrada pasa a
// ocupar la posición i. Es lo que reordena (B, S, H, d) a (B, H, S, d) para
// que cada cabeza de atención opere sobre su propia secuencia.
Tensor Tensor::permute(const std::vector<size_t>& order) const {
    const size_t nd = ndim();
    if (order.size() != nd) {
        throw std::invalid_argument("permute necesita un orden de " + std::to_string(nd) +
                                    " ejes para un tensor " + shape_str() + ".");
    }
    std::vector<bool> seen(nd, false);
    for (size_t axis : order) {
        if (axis >= nd || seen[axis]) {
            throw std::invalid_argument("permute necesita una permutacion valida de los ejes.");
        }
        seen[axis] = true;
    }

    std::vector<size_t> out_shape(nd);
    for (size_t i = 0; i < nd; ++i) out_shape[i] = shape()[order[i]];

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);

    // Strides del tensor de salida expresados sobre la memoria de la entrada
    std::vector<size_t> src_strides(nd);
    for (size_t i = 0; i < nd; ++i) src_strides[i] = strides()[order[i]];

    if (!cuda::ops::permute(impl_->storage, res.impl_->storage,
                            out_shape.data(), src_strides.data(), nd)) {
        const float* ENGINE_RESTRICT src_data = data().data();
        float* ENGINE_RESTRICT dst_data = res.data().data();
        // El contador con acarreo es barato por elemento pero encadena cada uno
        // con el anterior, y esa dependencia es la que obligaba a recorrerlo en
        // serie. Basta con sembrarlo: cada trozo calcula las coordenadas de su
        // primer elemento —una división por eje, pagada una vez por trozo— y a
        // partir de ahí sigue con el acarreo de siempre.
        //
        // No se notaba mientras permute era cosa de la atención, con tensores de
        // unos miles de elementos. Conv2d permuta la salida entera de cada capa.
        parallel::parallel_for(size(), kElementsPerThread, [&](size_t from, size_t to) {
            std::vector<size_t> idx(nd, 0);
            size_t rem = from;
            for (size_t i = nd; i-- > 0;) {
                idx[i] = rem % out_shape[i];
                rem /= out_shape[i];
            }

            for (size_t flat = from; flat < to; ++flat) {
                size_t src = 0;
                for (size_t i = 0; i < nd; ++i) src += idx[i] * src_strides[i];
                dst_data[flat] = src_data[src];

                for (size_t i = nd; i-- > 0;) {
                    if (++idx[i] < out_shape[i]) break;
                    idx[i] = 0;
                }
            }
        });
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        // La derivada es la permutación inversa
        std::vector<size_t> inverse(nd);
        for (size_t i = 0; i < nd; ++i) inverse[order[i]] = i;

        res.impl_->backward_fn = [self_copy, inverse](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out.permute(inverse));
        };
    }
    return res;
}

// Multiplicación matricial, con lotes sobre los ejes iniciales.
// (M, K) x (K, N) -> (M, N) y (B..., M, K) x (B..., K, N) -> (B..., M, N).
Tensor Tensor::matmul(const Tensor& B) const {
    if (ndim() < 2 || B.ndim() < 2) {
        throw std::invalid_argument("MatMul requiere al menos 2 dimensiones en ambos operandos.");
    }
    const size_t nd_a = ndim();
    const size_t nd_b = B.ndim();

    const std::vector<size_t> batch_a(shape().begin(), shape().end() - 2);
    const std::vector<size_t> batch_b(B.shape().begin(), B.shape().end() - 2);

    // Un operando 2D se comparte con todas las matrices del lote del otro:
    // (B, M, K) x (K, N) aplica la misma matriz a cada elemento del lote.
    const bool a_batched = !batch_a.empty();
    const bool b_batched = !batch_b.empty();
    if (a_batched && b_batched && batch_a != batch_b) {
        throw std::invalid_argument("MatMul por lotes necesita ejes de lote identicos: " +
                                    shape_str() + " y " + B.shape_str() + ".");
    }
    const std::vector<size_t>& batch_dims = a_batched ? batch_a : batch_b;

    const size_t M = shape()[nd_a - 2];
    const size_t K = shape()[nd_a - 1];
    const size_t K2 = B.shape()[nd_b - 2];
    const size_t N = B.shape()[nd_b - 1];

    if (K != K2) {
        throw std::invalid_argument("Dimensiones incompatibles para MatMul: " +
                                    shape_str() + " y " + B.shape_str() + ".");
    }

    const size_t batch = product(batch_dims);
    std::vector<size_t> out_shape = batch_dims;
    out_shape.push_back(M);
    out_shape.push_back(N);

    bool req_g = track(requires_grad() || B.requires_grad());
    Tensor C(out_shape, 0.0f, req_g);

    // Un operando no lotificado usa paso 0: se reutiliza en cada iteración
    const size_t a_stride = a_batched ? M * K : 0;
    const size_t b_stride = b_batched ? K * N : 0;

    // Primero se ofrece al dispositivo. Si se hace cargo, C se queda residente
    // en la GPU y ni siquiera se baja a host: sólo la lectura de un valor
    // desde el programa provoca la copia de vuelta.
    if (!cuda::ops::matmul(impl_->storage, B.impl_->storage, C.impl_->storage,
                           batch, M, K, N, a_batched, b_batched)) {
        const std::vector<float>& a_data = data();
        const std::vector<float>& b_data = B.data();
        std::vector<float>& c_data = C.data();

        // Bucle en orden i -> k -> j: el recorrido de j es contiguo en memoria, de
        // modo que la fila de B se lee y la de C se escribe secuencialmente.
        //
        // Los punteros de fila se izan fuera del bucle interno y se marcan con
        // ENGINE_RESTRICT: sin esa promesa de no solapamiento el compilador no
        // puede vectorizar el acumulador.
        //
        // La comprobación de a_ik == 0 sí compensa, aunque impida vectorizar esa
        // rama. Sobre datos densos cuesta un 40%, pero las matrices que llegan
        // aquí a menudo son salidas de ReLU con la mitad de los valores en cero
        // exacto —la segunda capa densa de cada bloque Transformer, por ejemplo—
        // y ahí se salta la mitad del trabajo. Medido de las dos formas sobre el
        // ejemplo del Transformer: sin la rama 18,7 s, con ella 15,9 s.
        // El reparto va por filas de la salida: cada fila la calcula un único hilo
        // de principio a fin, así que el orden de acumulación no cambia y el
        // resultado es idéntico bit a bit sea cual sea el número de hilos.
        const size_t rows = batch * M;
        parallel::parallel_for(rows, matmul_rows_per_thread(rows, K, N),
                               [&](size_t from, size_t to) {
            for (size_t row = from; row < to; ++row) {
                const size_t b = row / M;
                const size_t i = row % M;

                const float* ENGINE_RESTRICT a_row = a_data.data() + b * a_stride + i * K;
                const float* ENGINE_RESTRICT bm = b_data.data() + b * b_stride;
                float* ENGINE_RESTRICT c_row = c_data.data() + b * M * N + i * N;

                for (size_t k = 0; k < K; ++k) {
                    const float a_ik = a_row[k];
                    if (a_ik == 0.0f) continue;
                    const float* ENGINE_RESTRICT b_row = bm + k * N;
                    for (size_t j = 0; j < N; ++j) {
                        c_row[j] += a_ik * b_row[j];
                    }
                }
            }
        });
    }

    if (req_g) {
        C.impl_->parents = { impl_, B.impl_ };
        Tensor self_copy = *this;
        Tensor B_copy = B;

        C.impl_->backward_fn = [self_copy, B_copy](const Tensor& grad_out) mutable {
            // Las mismas formulas que en 2D, aplicadas a cada matriz del lote:
            // dL/dA = dL/dC x B^T   y   dL/dB = A^T x dL/dC.
            // Si un operando se compartió con todo el lote, su gradiente es la
            // suma de las contribuciones de cada elemento.
            if (self_copy.requires_grad()) {
                Tensor dA = grad_out.matmul(B_copy.transpose());
                self_copy.add_grad(dA.shape() == self_copy.shape()
                                       ? dA
                                       : fold_leading(dA, self_copy.shape()));
            }
            if (B_copy.requires_grad()) {
                Tensor dB = self_copy.transpose().matmul(grad_out);
                B_copy.add_grad(dB.shape() == B_copy.shape()
                                    ? dB
                                    : fold_leading(dB, B_copy.shape()));
            }
        };
    }
    return C;
}

// Función de activación ReLU
Tensor Tensor::relu() const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    if (!cuda::ops::relu(impl_->storage, res.impl_->storage)) {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = std::max(0.0f, lhs[i]);
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;

        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            Tensor dX(self_copy.shape(), 0.0f, false);
            if (!cuda::ops::relu_backward(self_copy.get_impl()->storage,
                                          grad_out.get_impl()->storage,
                                          dX.get_impl()->storage)) {
                const size_t n = self_copy.size();
                const float* ENGINE_RESTRICT x = self_copy.data().data();
                const float* ENGINE_RESTRICT g = grad_out.data().data();
                float* ENGINE_RESTRICT out = dX.data().data();
                for (size_t i = 0; i < n; ++i) out[i] = (x[i] > 0.0f) ? g[i] : 0.0f;
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Softmax sobre el último eje (estable numéricamente: se resta el máximo).
Tensor Tensor::softmax() const {
    if (ndim() == 0 || size() == 0) {
        throw std::invalid_argument("Softmax necesita un tensor no vacio.");
    }
    // Siempre normaliza sobre el ultimo eje: para (N, C) son las filas y para
    // (B, H, S, S) son las puntuaciones de atencion de cada consulta.
    const size_t cols = shape().back();
    const size_t rows = size() / cols;

    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!cuda::ops::softmax(impl_->storage, res.impl_->storage, rows, cols)) {
        const float* ENGINE_RESTRICT src = data().data();
        float* ENGINE_RESTRICT dst = res.data().data();
        for (size_t i = 0; i < rows; ++i) {
            const float* row = src + i * cols;
            float max_v = *std::max_element(row, row + cols);
            float denom = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                float e = std::exp(row[j] - max_v);
                dst[i * cols + j] = e;
                denom += e;
            }
            for (size_t j = 0; j < cols; ++j) dst[i * cols + j] /= denom;
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        // El jacobiano del softmax depende de la salida, así que se guarda una
        // copia desligada del grafo (igual que hace PyTorch con save_for_backward).
        Tensor saved = res.detach();

        res.impl_->backward_fn = [self_copy, saved, rows, cols](const Tensor& grad_out) mutable {
            // dX_ij = y_ij * (dY_ij - sum_k dY_ik * y_ik)
            Tensor dX(self_copy.shape(), 0.0f, false);
            if (!cuda::ops::softmax_backward(saved.get_impl()->storage,
                                             grad_out.get_impl()->storage,
                                             dX.get_impl()->storage, rows, cols)) {
                const float* ENGINE_RESTRICT y = saved.data().data();
                const float* ENGINE_RESTRICT g = grad_out.data().data();
                float* ENGINE_RESTRICT out = dX.data().data();
                for (size_t i = 0; i < rows; ++i) {
                    float dot = 0.0f;
                    for (size_t j = 0; j < cols; ++j) dot += g[i * cols + j] * y[i * cols + j];
                    for (size_t j = 0; j < cols; ++j) {
                        out[i * cols + j] = y[i * cols + j] * (g[i * cols + j] - dot);
                    }
                }
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Reducción Suma a un escalar {1}
Tensor Tensor::sum() const {
    // El acumulador es double aunque los datos sean float, igual que en
    // clip_grad_norm. Sumar un millon de valores en float pierde los bits bajos
    // en cuanto el total crece frente a cada sumando, y de esta reduccion cuelga
    // mean(), o sea mse_loss y el gradiente inicial de cualquier backward.
    double total = 0.0;
    for (float v : data()) total += v;
    bool req_g = track(requires_grad());
    Tensor res({1}, std::vector<float>{static_cast<float>(total)}, req_g);

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;

        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            Tensor dX(self_copy.shape(), grad_out.data()[0], false);
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Reducción Media a un escalar {1}
Tensor Tensor::mean() const {
    if (size() == 0) {
        throw std::invalid_argument("No se puede calcular la media de un tensor vacío.");
    }
    Tensor s = sum();
    return s * (1.0f / static_cast<float>(size()));
}

// Reshape
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_total = 1;
    for (size_t dim : new_shape) new_total *= dim;
    if (new_total != size()) {
        throw std::invalid_argument("Total de elementos incompatibles para Reshape.");
    }
    bool req_g = track(requires_grad());

    // El tensor que se reinterpreta suele venir de un matmul y estar residente
    // en la GPU —split_heads, en la atención, es exactamente eso—, así que
    // construirlo desde data() costaba una bajada y la subida de la operación
    // siguiente, dos veces por proyección.
    //
    // ponytail: sigue siendo una copia del búfer entero, no una vista. Quitarla
    // del todo pide un Storage compartido con desplazamiento y forma propios, y
    // eso toca el grafo de autograd; esto se lleva la parte cara sin ese cambio.
    Tensor res = clone_with_shape(impl_->storage, new_shape, req_g);

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out.reshape(self_copy.shape()));
        };
    }
    return res;
}


// ---------------------------------------------------------
// Reducciones por eje
//
// Un tensor contiguo se ve, respecto a un eje, como un bloque
// (outer, axis_len, inner): outer son los ejes anteriores, inner los
// posteriores. Con eso, reducir es recorrer axis_len para cada (outer, inner).
// ---------------------------------------------------------

namespace {

struct AxisView {
    size_t outer;
    size_t axis_len;
    size_t inner;
};

AxisView axis_view(const std::vector<size_t>& shape, size_t axis) {
    AxisView v{1, shape[axis], 1};
    for (size_t i = 0; i < axis; ++i) v.outer *= shape[i];
    for (size_t i = axis + 1; i < shape.size(); ++i) v.inner *= shape[i];
    return v;
}

std::vector<size_t> reduced_shape(const std::vector<size_t>& shape, size_t axis, bool keepdim) {
    std::vector<size_t> out = shape;
    if (keepdim) {
        out[axis] = 1;
    } else {
        out.erase(out.begin() + static_cast<long>(axis));
        if (out.empty()) out.push_back(1); // reducir un tensor 1D deja un escalar {1}
    }
    return out;
}

void check_axis(size_t axis, size_t ndim, const std::string& what, const std::string& shape_txt) {
    if (axis >= ndim) {
        throw std::out_of_range(what + ": el eje " + std::to_string(axis) +
                                " no existe en un tensor " + shape_txt + ".");
    }
}

} // namespace

Tensor Tensor::sum(size_t axis, bool keepdim) const {
    check_axis(axis, ndim(), "sum", shape_str());
    const AxisView v = axis_view(shape(), axis);

    bool req_g = track(requires_grad());
    Tensor res(reduced_shape(shape(), axis, keepdim), 0.0f, req_g);

    if (!cuda::ops::sum_axis(impl_->storage, res.impl_->storage, v.outer, v.axis_len, v.inner)) {
        const float* ENGINE_RESTRICT src_base = data().data();
        float* ENGINE_RESTRICT dst_base = res.data().data();
        for (size_t o = 0; o < v.outer; ++o) {
            for (size_t a = 0; a < v.axis_len; ++a) {
                const float* src = src_base + (o * v.axis_len + a) * v.inner;
                float* dst = dst_base + o * v.inner;
                for (size_t i = 0; i < v.inner; ++i) dst[i] += src[i];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, v](const Tensor& grad_out) mutable {
            // Cada elemento contribuyó una vez, así que recibe el gradiente entero
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t o = 0; o < v.outer; ++o) {
                const float* g = grad_out.data().data() + o * v.inner;
                for (size_t a = 0; a < v.axis_len; ++a) {
                    float* d = dX.data().data() + (o * v.axis_len + a) * v.inner;
                    for (size_t i = 0; i < v.inner; ++i) d[i] = g[i];
                }
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

Tensor Tensor::mean(size_t axis, bool keepdim) const {
    check_axis(axis, ndim(), "mean", shape_str());
    const float n = static_cast<float>(shape()[axis]);
    return sum(axis, keepdim) * (1.0f / n);
}

Tensor Tensor::max(size_t axis, bool keepdim) const {
    check_axis(axis, ndim(), "max", shape_str());
    if (size() == 0) throw std::invalid_argument("max sobre un tensor vacío.");
    const AxisView v = axis_view(shape(), axis);

    bool req_g = track(requires_grad());
    Tensor res(reduced_shape(shape(), axis, keepdim), 0.0f, req_g);
    // Posición ganadora de cada reducción, para repartir el gradiente
    std::vector<size_t> argmax(res.size(), 0);

    for (size_t o = 0; o < v.outer; ++o) {
        for (size_t i = 0; i < v.inner; ++i) {
            float best = data()[o * v.axis_len * v.inner + i];
            size_t best_a = 0;
            for (size_t a = 1; a < v.axis_len; ++a) {
                const float value = data()[(o * v.axis_len + a) * v.inner + i];
                if (value > best) { best = value; best_a = a; }
            }
            res.data()[o * v.inner + i] = best;
            argmax[o * v.inner + i] = (o * v.axis_len + best_a) * v.inner + i;
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, argmax](const Tensor& grad_out) mutable {
            // Solo el máximo influyó en la salida
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t k = 0; k < argmax.size(); ++k) {
                dX.data()[argmax[k]] += grad_out.data()[k];
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// ---------------------------------------------------------
// Rebanado, concatenación y apilado
// ---------------------------------------------------------

Tensor Tensor::slice(size_t axis, size_t start, size_t count) const {
    check_axis(axis, ndim(), "slice", shape_str());
    if (count == 0) throw std::invalid_argument("slice necesita al menos un elemento.");
    if (start + count > shape()[axis]) {
        throw std::out_of_range("slice [" + std::to_string(start) + ", " +
                                std::to_string(start + count) + ") se sale del eje " +
                                std::to_string(axis) + " de un tensor " + shape_str() + ".");
    }

    const AxisView v = axis_view(shape(), axis);
    std::vector<size_t> out_shape = shape();
    out_shape[axis] = count;

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);

    for (size_t o = 0; o < v.outer; ++o) {
        const float* src = data().data() + (o * v.axis_len + start) * v.inner;
        float* dst = res.data().data() + o * count * v.inner;
        std::copy_n(src, count * v.inner, dst);
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy, v, start, count](const Tensor& grad_out) mutable {
            // El gradiente vuelve a su hueco; el resto del tensor recibe cero
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t o = 0; o < v.outer; ++o) {
                const float* g = grad_out.data().data() + o * count * v.inner;
                float* d = dX.data().data() + (o * v.axis_len + start) * v.inner;
                std::copy_n(g, count * v.inner, d);
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

Tensor Tensor::concat(const std::vector<Tensor>& parts, size_t axis) {
    if (parts.empty()) throw std::invalid_argument("concat necesita al menos un tensor.");
    const std::vector<size_t>& first = parts[0].shape();
    check_axis(axis, first.size(), "concat", parts[0].shape_str());

    size_t total_axis = 0;
    bool req_g = false;
    for (const Tensor& t : parts) {
        if (t.ndim() != first.size()) {
            throw std::invalid_argument("concat necesita el mismo número de ejes en todas las partes.");
        }
        for (size_t d = 0; d < first.size(); ++d) {
            if (d != axis && t.shape()[d] != first[d]) {
                throw std::invalid_argument("concat: las partes solo pueden diferir en el eje " +
                                            std::to_string(axis) + "; " + parts[0].shape_str() +
                                            " frente a " + t.shape_str() + ".");
            }
        }
        total_axis += t.shape()[axis];
        req_g = req_g || t.requires_grad();
    }
    req_g = track(req_g);

    std::vector<size_t> out_shape = first;
    out_shape[axis] = total_axis;
    Tensor res(out_shape, 0.0f, req_g);

    const AxisView out_view = axis_view(out_shape, axis);
    size_t offset = 0;
    for (const Tensor& t : parts) {
        const size_t len = t.shape()[axis];
        const AxisView v = axis_view(t.shape(), axis);
        for (size_t o = 0; o < v.outer; ++o) {
            std::copy_n(t.data().data() + o * len * v.inner, len * v.inner,
                        res.data().data() + (o * out_view.axis_len + offset) * v.inner);
        }
        offset += len;
    }

    if (req_g) {
        std::vector<std::shared_ptr<TensorImpl>> parents;
        parents.reserve(parts.size());
        std::vector<Tensor> copies = parts;
        for (const Tensor& t : parts) parents.push_back(t.get_impl());
        res.impl_->parents = parents;

        res.impl_->backward_fn = [copies, axis, out_view](const Tensor& grad_out) mutable {
            // Cada parte recibe la franja del gradiente que aportó
            size_t off = 0;
            for (Tensor& t : copies) {
                const size_t len = t.shape()[axis];
                const AxisView v = axis_view(t.shape(), axis);
                if (t.requires_grad()) {
                    Tensor d(t.shape(), 0.0f, false);
                    for (size_t o = 0; o < v.outer; ++o) {
                        std::copy_n(grad_out.data().data() + (o * out_view.axis_len + off) * v.inner,
                                    len * v.inner, d.data().data() + o * len * v.inner);
                    }
                    t.add_grad(d);
                }
                off += len;
            }
        };
    }
    return res;
}

Tensor Tensor::stack(const std::vector<Tensor>& parts, size_t axis) {
    if (parts.empty()) throw std::invalid_argument("stack necesita al menos un tensor.");
    if (axis > parts[0].ndim()) {
        throw std::out_of_range("stack: el eje " + std::to_string(axis) +
                                " no cabe en un tensor " + parts[0].shape_str() + ".");
    }
    // Apilar es concatenar tras insertar un eje de tamaño 1 en cada parte
    std::vector<Tensor> expanded;
    expanded.reserve(parts.size());
    for (const Tensor& t : parts) {
        if (t.shape() != parts[0].shape()) {
            throw std::invalid_argument("stack necesita que todas las partes tengan la misma forma: " +
                                        parts[0].shape_str() + " frente a " + t.shape_str() + ".");
        }
        std::vector<size_t> s = t.shape();
        s.insert(s.begin() + static_cast<long>(axis), 1);
        expanded.push_back(t.reshape(s));
    }
    return concat(expanded, axis);
}

// Selección de filas (recogida de un mini-lote)
Tensor Tensor::select_rows(const std::vector<size_t>& indices) const {
    if (ndim() < 2) {
        throw std::invalid_argument("select_rows requiere al menos 2 dimensiones, este tensor es " +
                                    shape_str() + ".");
    }
    if (indices.empty()) {
        throw std::invalid_argument("select_rows recibió una lista de índices vacía.");
    }

    const size_t rows = shape()[0];
    // Todo lo que hay tras el primer eje viaja junto: para (N, C, H, W) cada
    // "fila" es una imagen completa de C*H*W valores contiguos.
    const size_t row_size = (rows == 0) ? 0 : size() / rows;

    for (size_t idx : indices) {
        if (idx >= rows) {
            throw std::out_of_range("select_rows: la fila " + std::to_string(idx) +
                                    " no existe en un tensor " + shape_str() + ".");
        }
    }

    std::vector<size_t> out_shape = shape();
    out_shape[0] = indices.size();

    bool req_g = track(requires_grad());
    Tensor res(out_shape, 0.0f, req_g);
    for (size_t i = 0; i < indices.size(); ++i) {
        std::copy_n(data().begin() + indices[i] * row_size, row_size,
                    res.data().begin() + i * row_size);
    }

    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        std::vector<size_t> idx_copy = indices;

        res.impl_->backward_fn = [self_copy, idx_copy, row_size](const Tensor& grad_out) mutable {
            // Dispersión inversa: cada fila devuelve su gradiente a la fila de
            // origen. El += es necesario porque un índice puede repetirse.
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t i = 0; i < idx_copy.size(); ++i) {
                for (size_t j = 0; j < row_size; ++j) {
                    dX.data()[idx_copy[i] * row_size + j] += grad_out.data()[i * row_size + j];
                }
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// ---------------------------------------------------------
// Operadores con el escalar a la izquierda
// ---------------------------------------------------------

Tensor operator+(float scalar, const Tensor& t) { return t + scalar; }
Tensor operator*(float scalar, const Tensor& t) { return t * scalar; }
Tensor operator-(float scalar, const Tensor& t) { return (t * -1.0f) + scalar; }

// Impresión por consola
void Tensor::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " = ";
    }
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape().size(); ++i) {
        std::cout << shape()[i] << (i + 1 < shape().size() ? ", " : "");
    }
    std::cout << "], requires_grad=" << (requires_grad() ? "true" : "false") << ")\n";

    if (ndim() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape()[0]; ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data()[i];
            if (i + 1 < shape()[0]) std::cout << ", ";
        }
        std::cout << "]\n";
    } else if (ndim() == 2) {
        std::cout << "[\n";
        for (size_t i = 0; i < shape()[0]; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < shape()[1]; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << (*this)({i, j});
                if (j + 1 < shape()[1]) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "]\n";
    } else if (size() > 0) {
        std::cout << "[ ";
        for (size_t i = 0; i < size(); ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data()[i] << " ";
            if ((i + 1) % shape().back() == 0 && i + 1 < size()) std::cout << "\n  ";
        }
        std::cout << "]\n";
    } else {
        std::cout << "[]\n";
    }

    if (has_grad()) {
        std::cout << "  grad = \n";
        grad().print("  gradientes");
    }
    std::cout << "\n";
}

} // namespace engine
