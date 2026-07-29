#ifndef ENGINE_DETAIL_RESTRICT_HPP
#define ENGINE_DETAIL_RESTRICT_HPP

// `restrict` is not standard C++, but all three compilers the project supports
// offer it under another name. Promising that two pointers do not overlap is
// what lets the compiler vectorise the accumulation loops.
#if defined(_MSC_VER)
#define ENGINE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define ENGINE_RESTRICT __restrict__
#else
#define ENGINE_RESTRICT
#endif

#endif  // ENGINE_DETAIL_RESTRICT_HPP
