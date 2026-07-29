#ifndef ENGINE_RANDOM_HPP
#define ENGINE_RANDOM_HPP

// The engine's global random generator. It lives apart from engine/tensor.hpp
// because <random> is one of the most expensive standard headers to compile and
// is only needed where numbers are actually generated.
//
// Warning: it is a single shared object with no protection against concurrent
// access.

#include <random>

namespace engine {

std::mt19937& global_rng();

}  // namespace engine

#endif  // ENGINE_RANDOM_HPP
