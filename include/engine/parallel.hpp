#ifndef ENGINE_PARALLEL_HPP
#define ENGINE_PARALLEL_HPP

#include <cstddef>
#include <functional>

namespace engine {
namespace parallel {

// ---------------------------------------------------------
// Data parallelism over the hot loops.
//
// Threads are created once and reused: creating one costs tens of
// microseconds, more than many of the engine's operations, so doing it per
// operation would be a regression rather than an improvement.
//
// The split is **deterministic by construction**: each chunk of the range is
// always computed by a single thread, and no reduction ever crosses a chunk
// boundary. The result is identical bit for bit with one thread or with eight,
// which the tests check.
// ---------------------------------------------------------

// Split threshold for an element-wise loop.
//
// Such a loop is bound by memory bandwidth rather than arithmetic, so it scales
// worse and needs more work before it pays: measured, 1.77x on four threads
// against 3.08x for the matrix product.
//
// It lives here rather than in each .cpp because tensor.cpp, nn.cpp and
// transformer.cpp all use it: duplicated, it would get tuned in one place and
// not in the other two.
constexpr size_t kElementsPerThread = 1u << 17;

// Number of threads in use. Defaults to the machine's cores; can be set with
// the ENGINE_NUM_THREADS environment variable.
size_t num_threads();

// Resizes the pool. With 1, everything runs on the calling thread and none is
// created, which is what you want for measuring or debugging.
void set_num_threads(size_t threads);

// Runs body over [0, count) split into contiguous chunks.
//
// min_per_thread avoids splitting work that does not pay for it: below that
// size, synchronisation costs more than the computation and it runs inline.
// Nested calls also run inline, so that one parallel operation invoking another
// does not multiply threads.
void parallel_for(size_t count, size_t min_per_thread,
                  const std::function<void(size_t begin, size_t end)>& body);

// True if the current thread is already inside a parallel region.
bool inside_parallel_region();

} // namespace parallel
} // namespace engine

#endif // ENGINE_PARALLEL_HPP
