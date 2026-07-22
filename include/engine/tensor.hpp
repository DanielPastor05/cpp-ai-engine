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
#include <memory>
#include <functional>

namespace engine {

struct TensorImpl;

class Tensor {
private:
    std::shared_ptr<TensorImpl> impl_;

    // Constructor privado para envolver un TensorImpl existente
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

public:
    // Constructores públicos
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape, float fill_value = 0.0f, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, bool requires_grad = false);

    // Método estático auxiliar para envolver una implementación compartida
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    // Métodos estáticos de fábrica
    static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor rand(const std::vector<size_t>& shape, float min_val = -1.0f, float max_val = 1.0f, bool requires_grad = false);

    // Métodos Autograd y Gradientes
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    Tensor grad() const;
    bool has_grad() const;
    void zero_grad();
    void add_grad(const Tensor& g);
    void backward();

    // Obtener implementación compartida interna
    std::shared_ptr<TensorImpl> get_impl() const { return impl_; }

    // Indexación y propiedades
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;
    float& at(size_t flat_index);
    const float& at(size_t flat_index) const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    const std::vector<float>& data() const;
    std::vector<float>& data();
    size_t size() const;
    size_t ndim() const;

    // Operaciones matemáticas con soporte para Autograd
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor relu() const;
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor sum() const;
    Tensor mean() const;

    // Formateo e impresión
    void print(const std::string& name = "") const;
};

// Estructura interna para almacenar el estado y los nodos del grafo Autograd
struct TensorImpl {
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    bool requires_grad = false;
    std::shared_ptr<TensorImpl> grad = nullptr;

    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void()> backward_fn = nullptr;

    TensorImpl() = default;
    TensorImpl(const std::vector<size_t>& s, float fill_val = 0.0f, bool req_grad = false);
    TensorImpl(const std::vector<size_t>& s, const std::vector<float>& d, bool req_grad = false);

    void compute_strides();
    size_t get_flat_index(const std::vector<size_t>& indices) const;
};

} // namespace engine

#endif // ENGINE_TENSOR_HPP
