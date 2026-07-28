#ifndef ENGINE_TENSOR_HPP
#define ENGINE_TENSOR_HPP

// Esta cabecera la incluye todo el motor, así que solo trae lo que la
// declaración de Tensor necesita. Lo demás vive donde se usa:
//   - TensorImpl (y <functional>) en engine/detail/tensor_impl.hpp
//   - global_rng() (y <random>) en engine/random.hpp
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/detail/tensor_impl.hpp"

namespace engine {

struct TensorImpl;

// Fija la semilla del generador global usado por rand()/randn() y por la
// inicializacion de las capas, para poder reproducir un entrenamiento.
// El generador en sí se declara en <engine/random.hpp>.
//
// Aviso: el generador es un único objeto compartido, sin protección frente a
// accesos concurrentes. El motor no usa hilos; si se añadieran, habría que
// darle uno por hilo o protegerlo.
void manual_seed(uint64_t seed);

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
    // Los cuatro operadores admiten difusión (broadcasting) por sufijo del
    // operando derecho: (N,) o (1, N) sobre (M, N) para el sesgo, (S, D) sobre
    // (B, S, D) para la codificación posicional, (S, S) sobre (B, H, S, S)
    // para la máscara, y {1} como escalar sobre cualquier forma.
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // Concatena a lo largo de un eje; el resto de dimensiones debe coincidir.
    static Tensor concat(const std::vector<Tensor>& parts, size_t axis);
    // Apila creando un eje nuevo en la posición indicada.
    static Tensor stack(const std::vector<Tensor>& parts, size_t axis = 0);

    // matmul admite lotes: (B..., M, K) x (B..., K, N) -> (B..., M, N).
    // transpose intercambia los dos últimos ejes; permute reordena todos.
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor permute(const std::vector<size_t>& order) const;
    Tensor relu() const;
    Tensor softmax() const;   // sobre el último eje
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    // Reducciones. Sin argumentos reducen a un escalar {1}; con un eje lo
    // eliminan (o lo dejan a 1 con keepdim), igual que en NumPy o PyTorch.
    Tensor sum() const;
    Tensor mean() const;
    Tensor sum(size_t axis, bool keepdim = false) const;
    Tensor mean(size_t axis, bool keepdim = false) const;
    Tensor max(size_t axis, bool keepdim = false) const;

    // Sub-tensor contiguo a lo largo de un eje: [start, start + count).
    Tensor slice(size_t axis, size_t start, size_t count) const;

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

} // namespace engine

#endif // ENGINE_TENSOR_HPP
