#include "engine/tensor.hpp"
#include "engine/autograd.hpp"

namespace engine {

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
    : shape(s), data(d), requires_grad(req_grad) {
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

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

Tensor::Tensor(const std::vector<size_t>& shape, float fill_value, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, fill_value, requires_grad)) {}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>(shape, data, requires_grad)) {}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(impl);
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dis(gen);
    }
    return t;
}

// Métodos Autograd y Gradientes
bool Tensor::requires_grad() const {
    return impl_ ? impl_->requires_grad : false;
}

void Tensor::set_requires_grad(bool req_grad) {
    if (impl_) impl_->requires_grad = req_grad;
}

Tensor Tensor::grad() const {
    if (!impl_ || !impl_->grad) {
        throw std::runtime_error("El tensor no tiene gradiente calculado.");
    }
    return Tensor(impl_->grad);
}

bool Tensor::has_grad() const {
    return impl_ && impl_->grad != nullptr;
}

void Tensor::zero_grad() {
    if (impl_ && impl_->grad) {
        std::fill(impl_->grad->data.begin(), impl_->grad->data.end(), 0.0f);
    }
}

void Tensor::add_grad(const Tensor& g) {
    if (!requires_grad()) return;
    if (!impl_->grad) {
        impl_->grad = std::make_shared<TensorImpl>(g.shape(), g.data(), false);
    } else {
        for (size_t i = 0; i < impl_->grad->data.size(); ++i) {
            impl_->grad->data[i] += g.data()[i];
        }
    }
}

void Tensor::backward() {
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

// ---------------------------------------------------------
// Operadores Matemáticos con Registro Autograd
// ---------------------------------------------------------

// Suma de tensores
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Formas incompatibles para suma de tensores.");
    }
    bool req_g = requires_grad() || other.requires_grad();
    Tensor res(shape(), 0.0f, req_g);

    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] + other.data()[i];
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;
        Tensor res_copy = res;

        res.impl_->backward_fn = [self_copy, other_copy, res_copy]() mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(res_copy.grad());
            if (other_copy.requires_grad()) other_copy.add_grad(res_copy.grad());
        };
    }
    return res;
}

// Resta de tensores
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Formas incompatibles para resta de tensores.");
    }
    bool req_g = requires_grad() || other.requires_grad();
    Tensor res(shape(), 0.0f, req_g);

    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] - other.data()[i];
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;
        Tensor res_copy = res;

        res.impl_->backward_fn = [self_copy, other_copy, res_copy]() mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(res_copy.grad());
            if (other_copy.requires_grad()) other_copy.add_grad(res_copy.grad() * -1.0f);
        };
    }
    return res;
}

// Multiplicación elemento a elemento (Hadamard)
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Formas incompatibles para multiplicación de tensores.");
    }
    bool req_g = requires_grad() || other.requires_grad();
    Tensor res(shape(), 0.0f, req_g);

    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] * other.data()[i];
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;
        Tensor res_copy = res;

        res.impl_->backward_fn = [self_copy, other_copy, res_copy]() mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(res_copy.grad() * other_copy);
            if (other_copy.requires_grad()) other_copy.add_grad(res_copy.grad() * self_copy);
        };
    }
    return res;
}

// División elemento a elemento
Tensor Tensor::operator/(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Formas incompatibles para división de tensores.");
    }
    bool req_g = requires_grad() || other.requires_grad();
    Tensor res(shape(), 0.0f, req_g);

    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] / other.data()[i];
    }

    if (req_g) {
        res.impl_->parents = { impl_, other.impl_ };
        Tensor self_copy = *this;
        Tensor other_copy = other;
        Tensor res_copy = res;

        res.impl_->backward_fn = [self_copy, other_copy, res_copy]() mutable {
            if (self_copy.requires_grad()) self_copy.add_grad(res_copy.grad() / other_copy);
            if (other_copy.requires_grad()) {
                Tensor grad_b = (res_copy.grad() * self_copy * -1.0f) / (other_copy * other_copy);
                other_copy.add_grad(grad_b);
            }
        };
    }
    return res;
}

// Suma con escalar
Tensor Tensor::operator+(float scalar) const {
    Tensor res(shape(), 0.0f, requires_grad());
    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] + scalar;
    }
    if (requires_grad()) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        Tensor res_copy = res;
        res.impl_->backward_fn = [self_copy, res_copy]() mutable {
            self_copy.add_grad(res_copy.grad());
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
    Tensor res(shape(), 0.0f, requires_grad());
    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = data()[i] * scalar;
    }
    if (requires_grad()) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        Tensor res_copy = res;
        res.impl_->backward_fn = [self_copy, res_copy, scalar]() mutable {
            self_copy.add_grad(res_copy.grad() * scalar);
        };
    }
    return res;
}

// División por escalar
Tensor Tensor::operator/(float scalar) const {
    return (*this) * (1.0f / scalar);
}

// Transposición 2D (Swap de filas y columnas)
Tensor Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::invalid_argument("Transpose solo esta implementado para tensores 2D (matrices).");
    }
    size_t rows = shape()[0];
    size_t cols = shape()[1];
    Tensor res({cols, rows}, 0.0f, requires_grad());

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res({j, i}) = (*this)({i, j});
        }
    }

    if (requires_grad()) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        Tensor res_copy = res;
        res.impl_->backward_fn = [self_copy, res_copy]() mutable {
            self_copy.add_grad(res_copy.grad().transpose());
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
        throw std::invalid_argument("Dimensiones incompatibles para MatMul: (" +
                                    std::to_string(M) + "x" + std::to_string(K) + ") y (" +
                                    std::to_string(K2) + "x" + std::to_string(N) + ")");
    }

    bool req_g = requires_grad() || B.requires_grad();
    Tensor C({M, N}, 0.0f, req_g);

    // Bucle optimizado para la memoria caché (i -> k -> j)
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = (*this)({i, k});
            for (size_t j = 0; j < N; ++j) {
                C({i, j}) += a_ik * B({k, j});
            }
        }
    }

    if (req_g) {
        C.impl_->parents = { impl_, B.impl_ };
        Tensor self_copy = *this;
        Tensor B_copy = B;
        Tensor C_copy = C;

        C.impl_->backward_fn = [self_copy, B_copy, C_copy]() mutable {
            // dL/dA = dL/dC x B^T
            if (self_copy.requires_grad()) {
                self_copy.add_grad(C_copy.grad().matmul(B_copy.transpose()));
            }
            // dL/dB = A^T x dL/dC
            if (B_copy.requires_grad()) {
                B_copy.add_grad(self_copy.transpose().matmul(C_copy.grad()));
            }
        };
    }
    return C;
}

// Función de activación ReLU
Tensor Tensor::relu() const {
    Tensor res(shape(), 0.0f, requires_grad());
    for (size_t i = 0; i < size(); ++i) {
        res.data()[i] = std::max(0.0f, data()[i]);
    }

    if (requires_grad()) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        Tensor res_copy = res;

        res.impl_->backward_fn = [self_copy, res_copy]() mutable {
            Tensor dX(self_copy.shape(), 0.0f, false);
            for (size_t i = 0; i < self_copy.size(); ++i) {
                dX.data()[i] = (self_copy.data()[i] > 0.0f) ? res_copy.grad().data()[i] : 0.0f;
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
    Tensor res({1}, {total}, requires_grad());

    if (requires_grad()) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        Tensor res_copy = res;

        res.impl_->backward_fn = [self_copy, res_copy]() mutable {
            float g = res_copy.grad().data()[0];
            Tensor dX(self_copy.shape(), g, false);
            self_copy.add_grad(dX);
        };
    }
    return res;
}

// Reducción Media a un escalar {1}
Tensor Tensor::mean() const {
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
    Tensor res(new_shape, data(), requires_grad());
    if (requires_grad()) {
        res.impl_->parents = { impl_ };
        Tensor self_copy = *this;
        Tensor res_copy = res;
        res.impl_->backward_fn = [self_copy, res_copy]() mutable {
            self_copy.add_grad(res_copy.grad().reshape(self_copy.shape()));
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
    } else {
        std::cout << "[ ";
        for (size_t i = 0; i < size(); ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data()[i] << " ";
            if ((i + 1) % shape().back() == 0 && i + 1 < size()) std::cout << "\n  ";
        }
        std::cout << "]\n";
    }

    if (has_grad()) {
        std::cout << "  grad = \n";
        grad().print("  gradientes");
    }
    std::cout << std::endl;
}

} // namespace engine
