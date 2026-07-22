#ifndef ENGINE_TENSOR_HPP
#define ENGINE_TENSOR_HPP

#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <random>
#include <string>
#include <numeric>

namespace engine {

class Tensor {
private:
    std::vector<float> data_;         // Buffer de memoria unidimensional contiguo en memoria (CPU)
    std::vector<size_t> shape_;       // Dimensiones del tensor (ej. [2, 3] para una matriz 2x3)
    std::vector<size_t> strides_;     // Pasos o zancadas para indexación multidimensional (row-major)

    // Calcula los strides en orden C (row-major) basándose en la forma (shape)
    void compute_strides();

public:
    // Constructores
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape, float fill_value = 0.0f);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);

    // Métodos estáticos de fábrica (Factory Methods)
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor rand(const std::vector<size_t>& shape, float min_val = -1.0f, float max_val = 1.0f);

    // Cálculo del índice plano 1D desde coordenadas multidimensionales
    size_t get_flat_index(const std::vector<size_t>& indices) const;

    // Operadores de acceso a elementos
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;
    
    float& at(size_t flat_index);
    const float& at(size_t flat_index) const;

    // Métodos de consulta de propiedades (Getters)
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }
    size_t size() const { return data_.size(); }
    size_t ndim() const { return shape_.size(); }

    // Operaciones matemáticas elemento a elemento (Element-wise)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float scalar) const;
    Tensor operator*(float scalar) const;

    // Multiplicación matricial (2D MatMul)
    Tensor matmul(const Tensor& other) const;

    // Funciones de activación (elemento a elemento)
    Tensor relu() const;

    // Reconfiguración de forma (Reshape)
    Tensor reshape(const std::vector<size_t>& new_shape) const;

    // Formateo e impresión por consola (PyTorch style)
    void print(const std::string& name = "") const;
};

} // namespace engine

#endif // ENGINE_TENSOR_HPP
