#include "engine/data.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace engine {
namespace data {

namespace {

// IDX stores its integers big-endian, so they have to be reordered by hand:
// reading them as-is would give nonsense on any x86 or ARM machine.
uint32_t read_be_uint32(std::istream& in, const std::string& path) {
    unsigned char bytes[4];
    in.read(reinterpret_cast<char*>(bytes), 4);
    if (!in) {
        throw std::runtime_error("Truncated IDX file: " + path);
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

std::ifstream open_idx(const std::string& path, uint32_t expected_magic, uint32_t& count,
                       std::vector<uint32_t>& dims) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open: " + path);
    }

    const uint32_t magic = read_be_uint32(in, path);
    if (magic != expected_magic) {
        throw std::runtime_error("'" + path + "' does not have the expected IDX magic (" +
                                 std::to_string(expected_magic) + ", encontrado " +
                                 std::to_string(magic) + ").");
    }

    // The magic number's last byte says how many axes there are
    const uint32_t ndim = magic & 0xFF;
    count = read_be_uint32(in, path);
    dims.clear();
    for (uint32_t d = 1; d < ndim; ++d) {
        dims.push_back(read_be_uint32(in, path));
    }
    return in;
}

bool file_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

}  // namespace

Tensor load_idx_images(const std::string& path, size_t max_samples) {
    uint32_t count = 0;
    std::vector<uint32_t> dims;
    std::ifstream in = open_idx(path, 0x00000803, count, dims);

    if (dims.size() != 2) {
        throw std::runtime_error("Expected 2D images in " + path + ".");
    }
    const size_t height = dims[0];
    const size_t width = dims[1];
    const size_t pixels = height * width;

    size_t n = count;
    if (max_samples > 0 && max_samples < n) n = max_samples;

    Tensor images({n, 1, height, width}, 0.0f, false);
    std::vector<unsigned char> buffer(pixels);

    for (size_t i = 0; i < n; ++i) {
        in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(pixels));
        if (!in) {
            throw std::runtime_error("Image file truncated at sample " + std::to_string(i) + ": " +
                                     path);
        }
        // Normalised to [0, 1]: with values 0-255 the first layer's gradients
        // would be two orders of magnitude larger than is reasonable.
        float* dst = images.data().data() + i * pixels;
        for (size_t p = 0; p < pixels; ++p) {
            dst[p] = static_cast<float>(buffer[p]) / 255.0f;
        }
    }
    return images;
}

std::vector<size_t> load_idx_labels(const std::string& path, size_t max_samples) {
    uint32_t count = 0;
    std::vector<uint32_t> dims;
    std::ifstream in = open_idx(path, 0x00000801, count, dims);

    size_t n = count;
    if (max_samples > 0 && max_samples < n) n = max_samples;

    std::vector<size_t> labels;
    labels.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        unsigned char value = 0;
        in.read(reinterpret_cast<char*>(&value), 1);
        if (!in) {
            throw std::runtime_error("Truncated label file at sample " + std::to_string(i) + ": " +
                                     path);
        }
        labels.push_back(static_cast<size_t>(value));
    }
    return labels;
}

MnistPaths find_mnist(const std::string& directory) {
    MnistPaths paths;

    // The full set, if it has been downloaded
    const std::string full_train_images = directory + "/train-images-idx3-ubyte";
    if (file_exists(full_train_images)) {
        paths.train_images = full_train_images;
        paths.train_labels = directory + "/train-labels-idx1-ubyte";
        paths.test_images = directory + "/t10k-images-idx3-ubyte";
        paths.test_labels = directory + "/t10k-labels-idx1-ubyte";
        paths.full = true;
        return paths;
    }

    // Otherwise, the subset shipped with the repository
    const std::string subset_train = directory + "/subset-train-images-idx3-ubyte";
    if (file_exists(subset_train)) {
        paths.train_images = subset_train;
        paths.train_labels = directory + "/subset-train-labels-idx1-ubyte";
        paths.test_images = directory + "/subset-test-images-idx3-ubyte";
        paths.test_labels = directory + "/subset-test-labels-idx1-ubyte";
        paths.full = false;
        return paths;
    }

    throw std::runtime_error("MNIST not found in '" + directory +
                             "'. Run tools/download_mnist.sh for the full set, or check "
                             "that the repository subset is still there.");
}

Dataset load_mnist_train(const MnistPaths& paths, size_t max_samples) {
    Dataset set;
    set.images = load_idx_images(paths.train_images, max_samples);
    set.labels = load_idx_labels(paths.train_labels, max_samples);
    return set;
}

Dataset load_mnist_test(const MnistPaths& paths, size_t max_samples) {
    Dataset set;
    set.images = load_idx_images(paths.test_images, max_samples);
    set.labels = load_idx_labels(paths.test_labels, max_samples);
    return set;
}

}  // namespace data
}  // namespace engine
