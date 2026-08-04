// Reading an environment variable, once, without a warning on every compiler.
//
// The engine takes five settings this way -- ENGINE_NUM_THREADS,
// ENGINE_BUFFER_POOL_MB, ENGINE_CUDA, ENGINE_CUDA_SYNC and the dispatch
// thresholds -- and every one of them used std::getenv directly. MSVC deprecates
// std::getenv in favour of _dupenv_s and emits C4996 for each call, so a clean
// clone built on Windows greeted the reader with deprecation warnings from a
// project that otherwise builds silently at /W4 and -Wall -Wextra.
//
// The deprecation is about thread safety: another thread calling setenv or
// _putenv while this one holds the returned pointer is a data race. That is a
// real hazard and it is not one this engine has -- every one of these is read
// once, into a function-local static, before any worker thread exists, and
// nothing in the library ever writes an environment variable. Suppressing the
// warning at the one place that knows that is honest; suppressing it project-wide
// with _CRT_SECURE_NO_WARNINGS would also silence the strcpy and sprintf
// diagnostics, which are not equally harmless.

#ifndef ENGINE_DETAIL_ENV_HPP
#define ENGINE_DETAIL_ENV_HPP

#include <cstdlib>
#include <optional>
#include <string>

namespace engine {
namespace detail {

// The variable's value, or empty if it is unset or set to the empty string --
// which the callers all treat the same way, so the distinction is not worth an
// extra state.
inline std::optional<std::string> env_var(const char* name) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // std::getenv; see the note above
#endif
    const char* raw = std::getenv(name);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (raw == nullptr || raw[0] == '\0') return std::nullopt;
    return std::string(raw);
}

}  // namespace detail
}  // namespace engine

#endif  // ENGINE_DETAIL_ENV_HPP
