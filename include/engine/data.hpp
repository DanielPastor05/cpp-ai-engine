#ifndef ENGINE_DATA_HPP
#define ENGINE_DATA_HPP

#include "engine/tensor.hpp"

#include <string>
#include <vector>

namespace engine {
namespace data {

// ---------------------------------------------------------
// Reader for the IDX format (the one MNIST uses)
//
//   magic       int32   0x00000803 images, 0x00000801 labels
//   dimensions  int32   one per axis
//   data        uint8
//
// Mind the byte order: IDX is **big-endian by specification**, unlike the
// weight format in engine/serialize.hpp, which requires little-endian. Here the
// bytes have to be swapped explicitly rather than reading the integers
// directly.
// ---------------------------------------------------------

// Images normalised to [0, 1] with shape (N, 1, height, width), ready to feed
// a Conv2d. With max_samples > 0 only the first ones are read.
Tensor load_idx_images(const std::string& path, size_t max_samples = 0);

// Labels as class indices.
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

// Looks for MNIST in the given directory. It prefers the full set
// (train-images-idx3-ubyte, downloaded with tools/download_mnist.sh) and, if
// that is not there, falls back to the subset shipped with the repository so
// the examples work on a fresh clone. Throws if it finds neither.
MnistPaths find_mnist(const std::string& directory = "data/mnist");

Dataset load_mnist_train(const MnistPaths& paths, size_t max_samples = 0);
Dataset load_mnist_test(const MnistPaths& paths, size_t max_samples = 0);

} // namespace data
} // namespace engine

#endif // ENGINE_DATA_HPP
