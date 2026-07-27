#include "engine/tensor.hpp"
#include "engine/autograd.hpp"

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

// Devuelve true si `other` es un vector fila difundible sobre `base`:
// base (M, N) con other (1, N) o (N,).
bool is_row_broadcast(const std::vector<size_t>& base, const std::vector<size_t>& other) {
    if (base.size() != 2) return false;
    if (other.size() == 1) return other[0] == base[1];
    if (other.size() == 2) return other[0] == 1 && other[1] == base[1];
    return false;
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

// Suma de tensores (con difusión de vector fila para el sesgo)
Tensor Tensor::operator+(const Tensor& other) const {
    const bool broadcast = shape() != other.shape();
    if (broadcast && !is_row_broadcast(shape(), other.shape())) {
        throw std::invalid_argument("Formas incompatibles para suma de tensores: " +
                                    shape_str() + " y " + other.shape_str() + ".");
    }

    bool req_g = track(requires_grad() || other.requires_grad());
    Tensor res(shape(), 0.0f, req_g);

    if (!broadcast) {
        for (size_t i = 0; i < size(); ++i) {
            res.data()[i] = data()[i] + other.data()[i];
        }
    } else {
        const size_t rows = shape()[0];
        const size_t cols = shape()[1];
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                res.data()[i * cols + j] = data()[i * cols + j] + other.data()[j];
            }
        }
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;

        res.impl_->backward_fn = [self_copy, other_copy, broadcast](const Tensor& grad_out) mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(grad_out);
            if (!other_copy.requires_grad()) return;
            if (!broadcast) {
                other_copy.add_grad(grad_out);
            } else {
                // La difusión replica el vector fila en cada fila, así que su
                // gradiente es la suma por columnas del gradiente de salida.
                const size_t rows = grad_out.shape()[0];
                const size_t cols = grad_out.shape()[1];
                Tensor db(other_copy.shape(), 0.0f, false);
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        db.data()[j] += grad_out.data()[i * cols + j];
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

    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] - other.data()[i];
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

    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] * other.data()[i];
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

    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] / other.data()[i];
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
    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] + scalar;
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
    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] * scalar;
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

// Transposición 2D (Swap de filas y columnas)
Tensor Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::invalid_argument("Transpose solo esta implementado para tensores 2D (matrices).");
    }
    size_t rows = shape()[0];
    size_t cols = shape()[1];
    bool req_g = track(requires_grad());
    Tensor res({cols, rows}, 0.0f, req_g);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res.data()[j * rows + i] = data()[i * cols + j];
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

// Multiplicación Matricial (MatMul) 2D
Tensor Tensor::matmul(const Tensor& B) const {
    if (ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("MatMul solo esta implementado para tensores 2D.");
    }
    size_t M = shape()[0];
    size_t K = shape()[1];
    size_t K2 = B.shape()[0];
    size_t N = B.shape()[1];

    if (K != K2) {
        throw std::invalid_argument("Dimensiones incompatibles para MatMul: " +
                                    shape_str() + " y " + B.shape_str() + ".");
    }

    bool req_g = track(requires_grad() || B.requires_grad());
    Tensor C({M, N}, 0.0f, req_g);

    // Bucle optimizado para la memoria caché (i -> k -> j)
    const std::vector<float>& a_data = data();
    const std::vector<float>& b_data = B.data();
    std::vector<float>& c_data = C.data();
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = a_data[i * K + k];
            if (a_ik == 0.0f) continue;
            for (size_t j = 0; j < N; ++j) {
                c_data[i * N + j] += a_ik * b_data[k * N + j];
            }
        }
    }

    if (req_g) {
        C.impl_->parents = { impl_, B.impl_ };
        Tensor self_copy = *this;
        Tensor B_copy = B;

        C.impl_->backward_fn = [self_copy, B_copy](const Tensor& grad_out) mutable {
            // dL/dA = dL/dC x B^T
            if (self_copy.requires_grad()) {
                self_copy.add_grad(grad_out.matmul(B_copy.transpose()));
            }
            // dL/dB = A^T x dL/dC
            if (B_copy.requires_grad()) {
                B_copy.add_grad(self_copy.transpose().matmul(grad_out));
            }
        };
    }
    return C;
}

// Función de activación ReLU
Tensor Tensor::relu() const {
    bool req_g = track(requires_grad());
    Tensor res(shape(), 0.0f, req_g);
    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = std::max(0.0f, data()[i]);
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

// Softmax por filas (estable numéricamente: se resta el máximo de cada fila).
// Acepta un vector 1D (N,) o una matriz 2D (batch, clases).
Tensor Tensor::softmax() const {
    if (ndim() != 1 && ndim() != 2) {
        throw std::invalid_argument("Softmax solo esta implementado para tensores 1D o 2D.");
    }
    const size_t rows = (ndim() == 1) ? 1 : shape()[0];
    const size_t cols = (ndim() == 1) ? shape()[0] : shape()[1];

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
    Tensor res({1}, {total}, req_g);

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
