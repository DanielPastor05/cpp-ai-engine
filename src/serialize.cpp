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
        throw std::runtime_error("Fichero de pesos truncado o corrupto: " + path);
    }
    return value;
}

std::string read_string(std::istream& in, uint32_t length, const std::string& path) {
    std::string s(length, '\0');
    in.read(&s[0], static_cast<std::streamsize>(length));
    if (!in) {
        throw std::runtime_error("Fichero de pesos truncado o corrupto: " + path);
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

        out.write(reinterpret_cast<const char*>(tensor.data().data()),
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
// Cargar
// ---------------------------------------------------------

namespace {

struct StoredTensor {
    std::vector<size_t> shape;
    std::vector<float> data;
};

std::vector<std::pair<std::string, StoredTensor>> read_file(const std::string& path) {
    require_little_endian();

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open for reading: " + path);
    }

    char magic[sizeof(kMagic)];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("'" + path + "' no es un fichero de pesos de este motor.");
    }

    const uint32_t version = read_raw<uint32_t>(in, path);
    if (version != kSerializationVersion) {
        throw std::runtime_error("Format version " + std::to_string(version) +
                                 " not supported (this build understands version " +
                                 std::to_string(kSerializationVersion) + ").");
    }

    const uint32_t count = read_raw<uint32_t>(in, path);
    std::vector<std::pair<std::string, StoredTensor>> stored;
    stored.reserve(count);

    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t name_len = read_raw<uint32_t>(in, path);
        const std::string name = read_string(in, name_len, path);

        const uint32_t ndim = read_raw<uint32_t>(in, path);
        StoredTensor entry;
        entry.shape.reserve(ndim);
        size_t total = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            const uint64_t dim = read_raw<uint64_t>(in, path);
            entry.shape.push_back(static_cast<size_t>(dim));
            total *= static_cast<size_t>(dim);
        }

        entry.data.resize(total);
        if (total > 0) {
            in.read(reinterpret_cast<char*>(entry.data.data()),
                    static_cast<std::streamsize>(total * sizeof(float)));
            if (!in) {
                std::string message = "Fichero de pesos truncado en '";
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

        entry.second.data() = it->second.data;
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
