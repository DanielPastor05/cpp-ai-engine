#ifndef ENGINE_TENSOR_HPP
#define ENGINE_TENSOR_HPP

#include <algorithm>
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

// Fija la semilla del generador global usado por rand()/randn() y por la
// inicializacion de las capas, para poder reproducir un entrenamiento.
void manual_seed(uint64_t seed);
std::mt19937& global_rng();

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
    static Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float stddev = 1.0f, bool requires_grad = false);

    // Métodos Autograd y Gradientes
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    Tensor grad() const;
    bool has_grad() const;
    void zero_grad();
    void add_grad(const Tensor& g);
    void backward();
    void backward(const Tensor& grad_output);

    // Obtener implementación compartida interna
    std::shared_ptr<TensorImpl> get_impl() const { return impl_; }

    // Indexación y propiedades.
    // La sobrecarga con vector es genérica pero reserva memoria dinámica en
    // cada acceso; para recorrer una matriz conviene la sobrecarga (fila, col).
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& at(size_t flat_index);
    const float& at(size_t flat_index) const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    const std::vector<float>& data() const;
    std::vector<float>& data();
    size_t size() const;
    size_t ndim() const;
    std::string shape_str() const;

    // Operaciones matemáticas con soporte para Autograd
    // La suma admite difusión (broadcasting) de un vector fila (1, N) o (N,)
    // sobre una matriz (M, N), necesaria para el sesgo de las capas densas.
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
    Tensor softmax() const;
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor sum() const;
    Tensor mean() const;

    // Extrae los elementos indicados del primer eje para formar un mini-lote:
    // de (M, N) toma filas y de (N, C, H, W) toma imágenes completas. Es la
    // operación que permite el entrenamiento por mini-lotes. Los índices
    // pueden repetirse: su gradiente se acumula en el elemento de origen.
    Tensor select_rows(const std::vector<size_t>& indices) const;

    // Copia desligada del grafo (comparte forma y valores, no el historial)
    Tensor detach() const;

    // Formateo e impresión
    void print(const std::string& name = "") const;
};

// Operadores con el escalar a la izquierda, para poder escribir 2.0f * t
Tensor operator+(float scalar, const Tensor& t);
Tensor operator-(float scalar, const Tensor& t);
Tensor operator*(float scalar, const Tensor& t);

// Estructura interna para almacenar el estado y los nodos del grafo Autograd.
//
// Nota sobre la propiedad de la memoria: un nodo referencia a sus padres con
// shared_ptr (aristas hijo -> padre) y nunca a sus hijos, de modo que el grafo
// es acíclico también en el conteo de referencias. Por eso backward_fn recibe
// el gradiente de salida como argumento en lugar de capturar su propio tensor:
// capturarlo crearía un ciclo y el grafo jamás se liberaría.
struct TensorImpl {
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    bool requires_grad = false;
    std::shared_ptr<TensorImpl> grad = nullptr;

    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void(const Tensor&)> backward_fn = nullptr;

    TensorImpl() = default;
    TensorImpl(const std::vector<size_t>& s, float fill_val = 0.0f, bool req_grad = false);
    TensorImpl(const std::vector<size_t>& s, const std::vector<float>& d, bool req_grad = false);

    void compute_strides();
    size_t get_flat_index(const std::vector<size_t>& indices) const;
};

} // namespace engine

#endif // ENGINE_TENSOR_HPP
