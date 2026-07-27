#ifndef ENGINE_RANDOM_HPP
#define ENGINE_RANDOM_HPP

// Generador aleatorio global del motor. Vive aparte de engine/tensor.hpp
// porque <random> es de las cabeceras estándar más caras de compilar y solo
// hace falta donde se generan números.
//
// Aviso: es un único objeto compartido y no está protegido frente a accesos
// concurrentes.

#include <random>

namespace engine {

std::mt19937& global_rng();

} // namespace engine

#endif // ENGINE_RANDOM_HPP
