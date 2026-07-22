#include "engine/tensor.hpp"

namespace engine {

// Constructor por defecto
Tensor::Tensor() : shape_({0}), strides_({0}) {}

// Constructor por forma y valor inicial
Tensor::Tensor(const std::vector<size_t>& shape, float fill_value)
    : shape_(shape) {
    compute_strides();
    size_t total_elements = 1;
    for (size_t dim : shape_) {
        total_elements *= dim;
    }
    data_.assign(total_elements, fill_value);
}

// Constructor desde un vector de datos plano
Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
    : shape_(shape), data_(data) {
    compute_strides();
    size_t total_elements = 1;
    for (size_t dim : shape_) {
        total_elements *= dim;
    }
    if (data_.size() != total_elements) {
        throw std::invalid_argument("El número de elementos en data no coincide con el producto de las dimensiones en shape.");
    }
}

// Cálculo del vector de pasos (strides) en orden C / row-major
void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;

    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

// Métodos estáticos de fábrica
Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::rand(const std::vector<size_t>& shape, float min_val, float max_val) {
    Tensor t(shape, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < t.data_.size(); ++i) {
        t.data_[i] = dis(gen);
    }
    return t;
}

// Mapeo de coordenadas multidimensionales a un índice unidimensional
size_t Tensor::get_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("El número de índices proporcionados no coincide con la dimensión (ndim) del Tensor.");
    }
    size_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Índice fuera de rango en la dimensión " + std::to_string(i));
        }
        flat_index += indices[i] * strides_[i];
    }
    return flat_index;
}

// Acceso con verificación de rango mediante coordenadas (i, j, k...)
float& Tensor::operator()(const std::vector<size_t>& indices) {
    return data_[get_flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    return data_[get_flat_index(indices)];
}

// Acceso directo mediante índice plano
float& Tensor::at(size_t flat_index) {
    if (flat_index >= data_.size()) {
        throw std::out_of_range("Índice plano fuera de rango.");
    }
    return data_[flat_index];
}

const float& Tensor::at(size_t flat_index) const {
    if (flat_index >= data_.size()) {
        throw std::out_of_range("Índice plano fuera de rango.");
    }
    return data_[flat_index];
}

// Suma elemento a elemento
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Las formas de los tensores deben ser idénticas para operaciones elemento a elemento.");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// Resta elemento a elemento
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Las formas de los tensores deben ser idénticas para operaciones elemento a elemento.");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

// Multiplicación elemento a elemento (Hadamard product)
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Las formas de los tensores deben ser idénticas para operaciones elemento a elemento.");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// División elemento a elemento
Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Las formas de los tensores deben ser idénticas para operaciones elemento a elemento.");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

// Suma con escalar
Tensor Tensor::operator+(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

// Multiplicación por escalar
Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

// Multiplicación matricial (MatMul) 2D optimizada para la memoria caché (i -> k -> j)
Tensor Tensor::matmul(const Tensor& B) const {
    if (ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("MatMul solo está implementado para tensores de 2 dimensiones (matrices).");
    }
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t K2 = B.shape_[0];
    size_t N = B.shape_[1];

    if (K != K2) {
        throw std::invalid_argument("Dimensiones incompatibles para MatMul: (" +
                                    std::to_string(M) + "x" + std::to_string(K) + ") y (" +
                                    std::to_string(K2) + "x" + std::to_string(N) + ")");
    }

    Tensor C({M, N}, 0.0f);

    // Bucle con acceso contiguo en memoria para maximizar aciertos en caché CPU
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = data_[i * strides_[0] + k * strides_[1]];
            for (size_t j = 0; j < N; ++j) {
                C.data_[i * C.strides_[0] + j * C.strides_[1]] += a_ik * B.data_[k * B.strides_[0] + j * B.strides_[1]];
            }
        }
    }
    return C;
}

// Función de activación ReLU
Tensor Tensor::relu() const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    return result;
}

// Reconfigurar forma sin copiar datos si el tamaño total se conserva
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_total = 1;
    for (size_t dim : new_shape) {
        new_total *= dim;
    }
    if (new_total != data_.size()) {
        throw std::invalid_argument("El tamaño total de elementos debe coincidir para hacer Reshape.");
    }
    return Tensor(new_shape, data_);
}

// Método de impresión estética por consola
void Tensor::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " = ";
    }
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    std::cout << "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        std::cout << strides_[i] << (i + 1 < strides_.size() ? ", " : "");
    }
    std::cout << "])\n";

    if (ndim() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape_[0]; ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data_[i];
            if (i + 1 < shape_[0]) std::cout << ", ";
        }
        std::cout << "]\n";
    } else if (ndim() == 2) {
        std::cout << "[\n";
        for (size_t i = 0; i < shape_[0]; ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < shape_[1]; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << (*this)({i, j});
                if (j + 1 < shape_[1]) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "]\n";
    } else {
        // Para tensores > 2D imprimimos el buffer ordenado
        std::cout << "[ ";
        for (size_t i = 0; i < data_.size(); ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data_[i] << " ";
            if ((i + 1) % shape_.back() == 0 && i + 1 < data_.size()) std::cout << "\n  ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

} // namespace engine
