#ifndef ENGINE_SERIALIZE_HPP
#define ENGINE_SERIALIZE_HPP

#include "engine/nn.hpp"

#include <string>
#include <vector>

namespace engine {

// ---------------------------------------------------------
// Weight persistence
//
// A binary format of its own, deliberately simple and self-describing:
//
//   "CPPAIENG"      8 signature bytes
//   version         uint32
//   n_tensors       uint32
//   per tensor:
//     name_length   uint32
//     name          bytes
//     ndim          uint32
//     dimensions    uint64 x ndim
//     data          float32 x product(dimensions)
//
// Tensors are matched by name rather than by position, so adding a new layer at
// the end of a model does not invalidate an earlier file. Shapes are checked on
// load: a file from a different model is rejected instead of being misread.
//
// Files are little-endian; loading them on a big-endian machine is refused
// rather than returning nonsense.
// ---------------------------------------------------------

constexpr uint32_t kSerializationVersion = 1;

// Saves the model's parameters. Throws if the file cannot be written or if two
// parameters share a name.
void save_parameters(nn::Module& model, const std::string& path);

// Loads the parameters into the model. By default it requires the file and the
// model to carry exactly the same names; with strict=false the file may bring
// parameters the model does not have, and vice versa, returning how many were
// actually loaded.
size_t load_parameters(nn::Module& model, const std::string& path, bool strict = true);

// Overloads taking a list of named parameters, useful when the model is a
// composition of your own rather than a single Module.
void save_parameters(const std::vector<std::pair<std::string, Tensor>>& params,
                     const std::string& path);
size_t load_parameters(std::vector<std::pair<std::string, Tensor>>& params, const std::string& path,
                       bool strict = true);

// The names and shapes stored in a file, without touching any model.
// Useful for inspecting a checkpoint before loading it.
std::vector<std::pair<std::string, std::vector<size_t>>> inspect_parameters(
    const std::string& path);

// Reads every tensor into freshly created Tensors, with no need for an existing
// model carrying the right shapes. This is what makes it possible to load
// reference files generated outside the engine, where the shapes are not known
// in advance.
std::vector<std::pair<std::string, Tensor>> load_tensors(const std::string& path);

}  // namespace engine

#endif  // ENGINE_SERIALIZE_HPP
