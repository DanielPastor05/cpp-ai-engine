#ifndef ENGINE_DATA_HPP
#define ENGINE_DATA_HPP

#include "engine/tensor.hpp"

#include <string>
#include <vector>

namespace engine {
namespace data {

// ---------------------------------------------------------
// Lector del formato IDX (el de MNIST)
//
//   magic       int32   0x00000803 imágenes, 0x00000801 etiquetas
//   dimensiones int32   una por eje
//   datos       uint8
//
// Ojo con el orden de bytes: IDX es **big-endian por especificación**, al
// contrario que el formato de pesos de engine/serialize.hpp, que exige
// little-endian. Aquí hay que intercambiar los bytes explícitamente en vez de
// leer los enteros directamente.
// ---------------------------------------------------------

// Imágenes normalizadas a [0, 1] con forma (N, 1, alto, ancho), lista para
// alimentar una Conv2d. Con max_samples > 0 se leen solo las primeras.
Tensor load_idx_images(const std::string& path, size_t max_samples = 0);

// Etiquetas como índices de clase.
std::vector<size_t> load_idx_labels(const std::string& path, size_t max_samples = 0);

// ---------------------------------------------------------
// MNIST
// ---------------------------------------------------------

struct Dataset {
    Tensor images;               // (N, 1, 28, 28)
    std::vector<size_t> labels;  // N
    size_t size() const { return labels.size(); }
};

struct MnistPaths {
    std::string train_images;
    std::string train_labels;
    std::string test_images;
    std::string test_labels;
    bool full = false; // true si es el conjunto completo, false si el subconjunto
};

// Busca MNIST en el directorio dado. Prefiere el conjunto completo
// (train-images-idx3-ubyte, descargado con tools/download_mnist.sh) y, si no
// está, cae al subconjunto que viene con el repositorio para que los ejemplos
// funcionen recién clonados. Lanza excepción si no encuentra ninguno.
MnistPaths find_mnist(const std::string& directory = "data/mnist");

Dataset load_mnist_train(const MnistPaths& paths, size_t max_samples = 0);
Dataset load_mnist_test(const MnistPaths& paths, size_t max_samples = 0);

} // namespace data
} // namespace engine

#endif // ENGINE_DATA_HPP
