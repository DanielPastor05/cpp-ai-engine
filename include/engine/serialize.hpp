#ifndef ENGINE_SERIALIZE_HPP
#define ENGINE_SERIALIZE_HPP

#include "engine/nn.hpp"

#include <string>
#include <vector>

namespace engine {

// ---------------------------------------------------------
// Persistencia de pesos
//
// Formato binario propio, deliberadamente simple y autodescriptivo:
//
//   "CPPAIENG"        8 bytes de firma
//   version           uint32
//   n_tensores        uint32
//   por cada tensor:
//     longitud_nombre uint32
//     nombre          bytes
//     ndim            uint32
//     dimensiones     uint64 x ndim
//     datos           float32 x producto(dimensiones)
//
// Los tensores se casan por nombre, no por posición, de modo que añadir una
// capa nueva al final de un modelo no invalida un fichero anterior. Las formas
// se comprueban al cargar: un fichero de un modelo distinto se rechaza en vez
// de leerse mal.
//
// Los ficheros son little-endian; se rechaza cargarlos en una máquina
// big-endian en lugar de devolver números sin sentido.
// ---------------------------------------------------------

constexpr uint32_t kSerializationVersion = 1;

// Guarda los parámetros del modelo. Lanza excepción si el fichero no se puede
// escribir o si dos parámetros comparten nombre.
void save_parameters(nn::Module& model, const std::string& path);

// Carga los parámetros en el modelo. Por defecto exige que el fichero y el
// modelo tengan exactamente los mismos nombres; con strict=false se permite
// que el fichero traiga parámetros que el modelo no tiene, y viceversa,
// devolviendo cuántos se cargaron de verdad.
size_t load_parameters(nn::Module& model, const std::string& path, bool strict = true);

// Sobrecargas para una lista de parámetros con nombre, útiles cuando el modelo
// es una composición propia y no un único Module.
void save_parameters(const std::vector<std::pair<std::string, Tensor>>& params,
                     const std::string& path);
size_t load_parameters(std::vector<std::pair<std::string, Tensor>>& params,
                       const std::string& path, bool strict = true);

// Nombres y formas guardados en un fichero, sin tocar ningún modelo.
// Útil para inspeccionar un checkpoint antes de cargarlo.
std::vector<std::pair<std::string, std::vector<size_t>>> inspect_parameters(const std::string& path);

// Lee todos los tensores creando Tensor nuevos, sin necesitar un modelo previo
// con las formas correctas. Es lo que permite cargar ficheros de referencia
// generados fuera del motor, donde las formas no se conocen de antemano.
std::vector<std::pair<std::string, Tensor>> load_tensors(const std::string& path);

} // namespace engine

#endif // ENGINE_SERIALIZE_HPP
