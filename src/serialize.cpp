#include "engine/serialize.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace engine {

namespace {

constexpr char kMagic[8] = {'C', 'P', 'P', 'A', 'I', 'E', 'N', 'G'};

bool is_little_endian() {
    const uint32_t probe = 1;
    unsigned char first;
    std::memcpy(&first, &probe, 1);
    return first == 1;
}

void require_little_endian() {
    // The format fixes little-endian. Rather than silently reading numbers
    // backwards, it is rejected explicitly.
    if (!is_little_endian()) {
        throw std::runtime_error("The weight format is little-endian and this machine is not.");
    }
}

template <typename T>
void write_raw(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_raw(std::istream& in, const std::string& path) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Truncated or corrupt weight file: " + path);
    }
    return value;
}

std::string read_string(std::istream& in, uint32_t length, const std::string& path) {
    std::string s(length, '\0');
    in.read(&s[0], static_cast<std::streamsize>(length));
    if (!in) {
        throw std::runtime_error("Truncated or corrupt weight file: " + path);
    }
    return s;
}

void check_unique_names(const std::vector<std::pair<std::string, Tensor>>& params) {
    std::set<std::string> seen;
    for (const auto& entry : params) {
        if (!seen.insert(entry.first).second) {
            throw std::runtime_error("Two parameters share the name '" + entry.first +
                                     "'; they could not be told apart on load.");
        }
    }
}

}  // namespace

// ---------------------------------------------------------
// Guardar
// ---------------------------------------------------------

void save_parameters(const std::vector<std::pair<std::string, Tensor>>& params,
                     const std::string& path) {
    require_little_endian();
    check_unique_names(params);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open for writing: " + path);
    }

    out.write(kMagic, sizeof(kMagic));
    write_raw<uint32_t>(out, kSerializationVersion);
    write_raw<uint32_t>(out, static_cast<uint32_t>(params.size()));

    for (const auto& entry : params) {
        const std::string& name = entry.first;
        const Tensor& tensor = entry.second;

        write_raw<uint32_t>(out, static_cast<uint32_t>(name.size()));
        out.write(name.data(), static_cast<std::streamsize>(name.size()));

        write_raw<uint32_t>(out, static_cast<uint32_t>(tensor.ndim()));
        for (size_t dim : tensor.shape()) write_raw<uint64_t>(out, static_cast<uint64_t>(dim));

        out.write(reinterpret_cast<const char*>(tensor.data()),
                  static_cast<std::streamsize>(tensor.size() * sizeof(float)));
    }

    if (!out) {
        throw std::runtime_error("Error writing the weights to: " + path);
    }
}

void save_parameters(nn::Module& model, const std::string& path) {
    save_parameters(model.named_parameters(), path);
}

// ---------------------------------------------------------
// Loading
// ---------------------------------------------------------
//
// Everything below treats the file as hostile, because it is the only input to
// this engine that does not come from its own API. A checkpoint is a thing
// people download.
//
// Three sizes in the header are attacker-controlled and used to allocate: the
// tensor count, the dimension count, and the name length. All three are 32-bit,
// so a crafted file can ask for four billion of anything. The defence is not a
// magic constant -- it is the file's own length: a claim that cannot fit in the
// bytes that exist is rejected before a byte is allocated for it.
//
// And the element count is a product of 64-bit dimensions accumulated into a
// size_t, which **overflowed silently**. Two dimensions of 2^32 wrap to zero,
// the tensor is allocated empty, and the shape it reports has nothing to do with
// the data it holds. That one was found by reading the loop with a fuzzer in
// mind rather than by the fuzzer, which is the argument for doing both.

namespace {

struct StoredTensor {
    std::vector<size_t> shape;
    std::vector<float> data;
};

std::vector<std::pair<std::string, StoredTensor>> read_file(const std::string& path) {
    require_little_endian();

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Could not open for reading: " + path);
    }
    // The bound every size below is checked against. A declared length larger
    // than the file cannot be honest, whatever else it is.
    const std::streamoff file_bytes = in.tellg();
    if (file_bytes < 0) {
        throw std::runtime_error("Could not measure: " + path);
    }
    const auto remaining = [&in, file_bytes]() -> uint64_t {
        const std::streamoff at = in.tellg();
        return (at < 0 || at > file_bytes) ? 0 : (uint64_t)(file_bytes - at);
    };
    in.seekg(0, std::ios::beg);

    char magic[sizeof(kMagic)];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("'" + path + "' is not a weight file from this engine.");
    }

    const uint32_t version = read_raw<uint32_t>(in, path);
    if (version != kSerializationVersion) {
        throw std::runtime_error("Format version " + std::to_string(version) +
                                 " not supported (this build understands version " +
                                 std::to_string(kSerializationVersion) + ").");
    }

    const uint32_t count = read_raw<uint32_t>(in, path);
    // Each tensor costs at least its name length, its dimension count and one
    // dimension: 16 bytes before any data. More than that many left is
    // impossible, so refuse before reserving.
    if ((uint64_t)count * 16u > remaining()) {
        throw std::runtime_error("'" + path + "' declares " + std::to_string(count) +
                                 " tensors, more than its " + std::to_string(file_bytes) +
                                 " bytes can hold.");
    }
    std::vector<std::pair<std::string, StoredTensor>> stored;
    stored.reserve(count);

    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t name_len = read_raw<uint32_t>(in, path);
        if (name_len > remaining()) {
            throw std::runtime_error("'" + path + "' declares a " + std::to_string(name_len) +
                                     "-byte tensor name with fewer bytes left than that.");
        }
        const std::string name = read_string(in, name_len, path);

        const uint32_t ndim = read_raw<uint32_t>(in, path);
        if ((uint64_t)ndim * sizeof(uint64_t) > remaining()) {
            throw std::runtime_error("'" + path + "' declares " + std::to_string(ndim) +
                                     " dimensions with fewer bytes left than that.");
        }
        StoredTensor entry;
        entry.shape.reserve(ndim);

        // Overflow-checked, because the unchecked version wrapped: with
        // dimensions of 2^32 the product became zero, the tensor came out empty,
        // and its shape described data it did not have.
        uint64_t total = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            const uint64_t dim = read_raw<uint64_t>(in, path);
            if (dim != 0 && total > UINT64_MAX / dim) {
                throw std::runtime_error("'" + path +
                                         "' declares a shape whose element count "
                                         "does not fit in 64 bits.");
            }
            entry.shape.push_back(static_cast<size_t>(dim));
            total *= dim;
        }
        // And the values themselves have to be in the file.
        if (total > remaining() / sizeof(float)) {
            throw std::runtime_error("'" + path + "' declares a tensor of " +
                                     std::to_string(total) + " values with only " +
                                     std::to_string(remaining()) + " bytes left.");
        }

        entry.data.resize(static_cast<size_t>(total));
        if (total > 0) {
            in.read(reinterpret_cast<char*>(entry.data.data()),
                    static_cast<std::streamsize>(total * sizeof(float)));
            if (!in) {
                std::string message = "Weight file truncated at '";
                message += name;
                message += "': ";
                message += path;
                throw std::runtime_error(message);
            }
        }
        stored.emplace_back(name, std::move(entry));
    }
    return stored;
}

}  // namespace

std::vector<std::pair<std::string, std::vector<size_t>>> inspect_parameters(
    const std::string& path) {
    std::vector<std::pair<std::string, std::vector<size_t>>> summary;
    for (auto& entry : read_file(path)) {
        summary.emplace_back(entry.first, entry.second.shape);
    }
    return summary;
}

std::vector<std::pair<std::string, Tensor>> load_tensors(const std::string& path) {
    std::vector<std::pair<std::string, Tensor>> tensors;
    for (auto& entry : read_file(path)) {
        tensors.emplace_back(entry.first, Tensor(entry.second.shape, entry.second.data, false));
    }
    return tensors;
}

size_t load_parameters(std::vector<std::pair<std::string, Tensor>>& params, const std::string& path,
                       bool strict) {
    check_unique_names(params);
    std::vector<std::pair<std::string, StoredTensor>> stored = read_file(path);

    size_t loaded = 0;
    std::set<std::string> used;

    for (auto& entry : params) {
        auto it = std::find_if(
            stored.begin(), stored.end(),
            [&](const std::pair<std::string, StoredTensor>& s) { return s.first == entry.first; });
        if (it == stored.end()) {
            if (strict) {
                throw std::runtime_error("The file does not contain the parameter '" + entry.first +
                                         "'.");
            }
            continue;
        }

        // The shape is always checked: loading weights from another model under
        // the same name would give a silently broken model.
        if (it->second.shape != entry.second.shape()) {
            throw std::runtime_error("The parameter '" + entry.first + "' has shape " +
                                     entry.second.shape_str() +
                                     " in the model, and a different one in '" + path + "'.");
        }

        // Assigning over the whole buffer is what this used to do, and it was
        // reachable from a file: nothing here proved the parsed vector was as
        // long as the shape it was checked against, so a truncated one
        // silently shortened a live tensor. Copying element by element into a
        // buffer that cannot be resized makes the length a precondition
        // instead of a consequence.
        if (it->second.data.size() != entry.second.size()) {
            throw std::runtime_error("The parameter '" + entry.first + "' declares shape " +
                                     entry.second.shape_str() + " but carries " +
                                     std::to_string(it->second.data.size()) + " values in '" +
                                     path + "'.");
        }
        std::copy_n(it->second.data.data(), entry.second.size(), entry.second.data());
        used.insert(entry.first);
        ++loaded;
    }

    if (strict && used.size() != stored.size()) {
        for (const auto& s : stored) {
            if (used.find(s.first) == used.end()) {
                throw std::runtime_error("The model has no parameter '" + s.first +
                                         "' that the file carries.");
            }
        }
    }
    return loaded;
}

size_t load_parameters(nn::Module& model, const std::string& path, bool strict) {
    std::vector<std::pair<std::string, Tensor>> params = model.named_parameters();
    return load_parameters(params, path, strict);
}

}  // namespace engine
