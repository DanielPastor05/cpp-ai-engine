#ifndef ENGINE_DETAIL_RESTRICT_HPP
#define ENGINE_DETAIL_RESTRICT_HPP

// `restrict` no es estándar en C++, pero los tres compiladores que soporta el
// proyecto lo ofrecen con otro nombre. Prometer que dos punteros no se solapan
// es lo que permite al compilador vectorizar los bucles de acumulación.
#if defined(_MSC_VER)
#  define ENGINE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define ENGINE_RESTRICT __restrict__
#else
#  define ENGINE_RESTRICT
#endif

#endif // ENGINE_DETAIL_RESTRICT_HPP
