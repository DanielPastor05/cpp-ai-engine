#include "engine/tensor.hpp"
#include "engine/autograd.hpp"
#include "engine/detail/restrict.hpp"

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

// Suma los ejes iniciales de `full` hasta dejar la forma `target`. Es el
// adjunto de difundir un operando sobre un lote, y lo usan tanto la suma
// difundida como el matmul con un operando compartido.
Tensor fold_leading(const Tensor& full, const std::vector<size_t>& target) {
    const size_t inner = product(target);
    const size_t repeat = full.size() / inner;

    Tensor folded(target, 0.0f, false);
    for (size_t r = 0; r < repeat; ++r) {
        for (size_t j = 0; j < inner; ++j) {
            folded.data()[j] += full.data()[r * inner + j];
        }
    }
    return folded;
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
    data.assign(total_elements, fill_val);
}

TensorImpl::TensorImpl(const std::vector<size_t>& s, const std::vector<float>& d, bool req_grad)
    : data(d), shape(s), requires_grad(req_grad) {
    compute_strides();
    size_t total_elements = 1;
    for (size_t dim : shape) total_elements *= dim;
    if (data.size() != total_elements) {
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
    if (impl_->grad) {
        std::fill(impl_->grad->data.begin(), impl_->grad->data.end(), 0.0f);
    }
}

void Tensor::add_grad(const Tensor& g) {
    if (!requires_grad()) return;
    if (g.shape() != shape()) {
        throw std::invalid_argument("Forma del gradiente " + g.shape_str() +
                                    " incompatible con la del tensor " + shape_str() + ".");
    }
    if (!impl_->grad) {
        impl_->grad = std::make_shared<TensorImpl>(shape(), g.data(), false);
    } else {
        for (size_t i = 0; i < impl_->grad->data.size(); ++i) {
            impl_->grad->data[i] += g.data()[i];
        }
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
    return impl_->data[get_flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    return impl_->data[get_flat_index(indices)];
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
    return impl_->data[row * shape()[1] + col];
}

float& Tensor::at(size_t flat_index) {
    return impl_->data.at(flat_index);
}

const float& Tensor::at(size_t flat_index) const {
    return impl_->data.at(flat_index);
}

const std::vector<size_t>& Tensor::shape() const { return impl_->shape; }
const std::vector<size_t>& Tensor::strides() const { return impl_->strides; }
const std::vector<float>& Tensor::data() const { return impl_->data; }
std::vector<float>& Tensor::data() { return impl_->data; }
size_t Tensor::size() const { return impl_->data.size(); }
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
    return Tensor(shape(), data(), false);
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

    // Los accesores se izan fuera del bucle: llamarlos por elemento impide
    // que el compilador vectorice.
    const size_t n = size();
    const float* ENGINE_RESTRICT lhs = data().data();
    const float* ENGINE_RESTRICT rhs = other.data().data();
    float* ENGINE_RESTRICT out = res.data().data();

    if (!broadcast) {
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] + rhs[i];
    } else {
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] + rhs[i % plan.inner];
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

// Resta de tensores
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Formas incompatibles para resta de tensores: " +
                                    shape_str() + " y " + other.shape_str() + ".");
    }
    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] - rhs[i];
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy](const Tensor& grad_out) mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out);
            if (other_copy.requires_grad()) other_copy.add_grad(grad_out * -1.0f);
        };
    }
    return res;
}

// Multiplicación elemento a elemento (Hadamard)
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Formas incompatibles para multiplicación de tensores: " +
                                    shape_str() + " y " + other.shape_str() + ".");
    }
    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] * rhs[i];
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy](const Tensor& grad_out) mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out * other_copy);
            if (other_copy.requires_grad()) other_copy.add_grad(grad_out * self_copy);
        };
    }
    return res;
}

// División elemento a elemento
Tensor Tensor::operator/(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Formas incompatibles para división de tensores: " +
                                    shape_str() + " y " + other.shape_str() + ".");
    }
    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    {
        const size_t n = size();
        const float* ENGINE_RESTRICT lhs = data().data();
        const float* ENGINE_RESTRICT rhs = other.data().data();
        float* ENGINE_RESTRICT out = res.data().data();
        for (size_t i = 0; i < n; ++i) out[i] = lhs[i] / rhs[i];
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy](const Tensor& grad_out) mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out / other_copy);
            if (other_copy.requires_grad()) {
                Tensor grad_b = (grad_out * self_copy * -1.0f) / (other_copy * other_copy);
                other_copy.add_grad(grad_b);
            }
        };
    }
    return res;
}

// Suma con escalar
Tensor Tensor::operator+(float scalar) const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    {
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
    {
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

    for (size_t b = 0; b < batch; ++b) {
        const float* src = data().data() + b * rows * cols;
        float* dst = res.data().data() + b * rows * cols;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
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

    std::vector<size_t> idx(nd, 0);
    for (size_t flat = 0; flat < size(); ++flat) {
        size_t src = 0;
        for (size_t i = 0; i < nd; ++i) src += idx[i] * src_strides[i];
        res.data()[flat] = data()[src];

        for (size_t i = nd; i-- > 0;) {
            if (++idx[i] < out_shape[i]) break;
            idx[i] = 0;
        }
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

    const std::vector<float>& a_data = data();
    const std::vector<float>& b_data = B.data();
    std::vector<float>& c_data = C.data();

    // Un operando no lotificado usa paso 0: se reutiliza en cada iteración
    const size_t a_stride = a_batched ? M * K : 0;
    const size_t b_stride = b_batched ? K * N : 0;

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
    for (size_t b = 0; b < batch; ++b) {
        const float* ENGINE_RESTRICT a = a_data.data() + b * a_stride;
        const float* ENGINE_RESTRICT bm = b_data.data() + b * b_stride;
        float* ENGINE_RESTRICT c = c_data.data() + b * M * N;

        for (size_t i = 0; i < M; ++i) {
            float* ENGINE_RESTRICT c_row = c + i * N;
            const float* ENGINE_RESTRICT a_row = a + i * K;

            for (size_t k = 0; k < K; ++k) {
                const float a_ik = a_row[k];
                if (a_ik == 0.0f) continue;
                const float* ENGINE_RESTRICT b_row = bm + k * N;
                for (size_t j = 0; j < N; ++j) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
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
    {
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
            for (size_t i = 0; i < self_copy.size(); ++i) {
                dX.data()[i] = (self_copy.data()[i] > 0.0f) ? grad_out.data()[i] : 0.0f;
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

    for (size_t i = 0; i < rows; ++i) {
        const float* row = data().data() + i * cols;
        float max_v = *std::max_element(row, row + cols);
        float denom = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float e = std::exp(row[j] - max_v);
            res.data()[i * cols + j] = e;
            denom += e;
        }
        for (size_t j = 0; j < cols; ++j) {
            res.data()[i * cols + j] /= denom;
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
            for (size_t i = 0; i < rows; ++i) {
                float dot = 0.0f;
                for (size_t j = 0; j < cols; ++j) {
                    dot += grad_out.data()[i * cols + j] * saved.data()[i * cols + j];
                }
                for (size_t j = 0; j < cols; ++j) {
                    dX.data()[i * cols + j] =
                        saved.data()[i * cols + j] * (grad_out.data()[i * cols + j] - dot);
                }
            }
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Reducción Suma a un escalar {1}
Tensor Tensor::sum() const {
    float total = 0.0f;
    for (float v : data()) total += v;
    bool req_g = track(requires_grad());
    Tensor res({1}, std::vector<float>{total}, req_g);

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
    Tensor res(new_shape, data(), req_g);
    if (req_g) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        res.impl_->backward_fn = [self_copy](const Tensor& grad_out) mutable {
            self_copy.add_grad(grad_out.reshape(self_copy.shape()));
        };
    }
    return res;
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
    std::cout << std::endl;
}

} // namespace engine
