#include "engine/parallel.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace engine {
namespace parallel {

namespace {

// The calling thread joins in the work, so N threads means N-1 workers are
// spawned. This flag stops a nested parallel region from splitting again:
// multiplying threads inside a thread only adds contention.
thread_local bool g_inside_region = false;

size_t threads_from_environment() {
    const unsigned detected = std::thread::hardware_concurrency();
    const size_t fallback = detected > 0 ? detected : 1;

    const char* value = std::getenv("ENGINE_NUM_THREADS");
    if (value == nullptr) return fallback;

    try {
        const int parsed = std::stoi(value);
        return parsed > 0 ? static_cast<size_t>(parsed) : fallback;
    } catch (const std::exception&) {
        // An unreadable value is no reason to abort: the core count is used, exactly
        // as if the variable were not set.
        return fallback;
    }
}

// A simple pool of indexed tasks. Not a general pool: all it knows how to run
// is "this body, for these chunks", which is all the engine needs.
class Pool {
public:
    static Pool& instance() {
        static Pool pool;
        return pool;
    }

    size_t size() const { return workers_.size() + 1; }

    void resize(size_t threads) {
        if (threads == 0) threads = 1;
        if (threads == size()) return;

        stop_workers();
        for (size_t i = 0; i + 1 < threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    // Splits [0, count) into `chunks` contiguous pieces and waits for them.
    void run(size_t count, size_t chunks, const std::function<void(size_t, size_t)>& body) {
        std::unique_lock<std::mutex> lock(mutex_);

        body_ = &body;
        total_ = count;
        chunks_ = chunks;
        next_chunk_ = 0;
        remaining_ = chunks;
        error_ = nullptr;
        ++generation_;

        work_ready_.notify_all();
        lock.unlock();

        // The calling thread takes chunks like any other, instead of waiting
        run_chunks();

        lock.lock();
        work_done_.wait(lock, [this] { return remaining_ == 0; });
        std::exception_ptr error = error_;
        body_ = nullptr;
        lock.unlock();

        // An exception in a worker is rethrown on the thread that split the work
        if (error) std::rethrow_exception(error);
    }

private:
    Pool() { resize(threads_from_environment()); }
    ~Pool() { stop_workers(); }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        work_ready_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        workers_.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
    }

    void worker_loop() {
        g_inside_region = true;
        size_t seen_generation = 0;

        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [&] {
                    return stopping_ || (body_ != nullptr && generation_ != seen_generation);
                });
                if (stopping_) return;
                seen_generation = generation_;
            }
            run_chunks();
        }
    }

    // Dynamic scheduling: each thread grabs the next free chunk. With chunks of
    // uneven cost -- rows with more zeros in matmul, for instance -- it balances
    // better than assigning fixed chunks per thread.
    void run_chunks() {
        const bool was_inside = g_inside_region;
        g_inside_region = true;

        while (true) {
            size_t chunk = 0;
            const std::function<void(size_t, size_t)>* body = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (body_ == nullptr || next_chunk_ >= chunks_) break;
                chunk = next_chunk_++;
                body = body_;
            }

            const size_t begin = (total_ * chunk) / chunks_;
            const size_t end = (total_ * (chunk + 1)) / chunks_;

            try {
                if (end > begin) (*body)(begin, end);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--remaining_ == 0) work_done_.notify_all();
            }
        }

        g_inside_region = was_inside;
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;

    const std::function<void(size_t, size_t)>* body_ = nullptr;
    size_t total_ = 0;
    size_t chunks_ = 0;
    size_t next_chunk_ = 0;
    size_t remaining_ = 0;
    size_t generation_ = 0;
    bool stopping_ = false;
    std::exception_ptr error_;
};

}  // namespace

size_t num_threads() {
    return Pool::instance().size();
}

void set_num_threads(size_t threads) {
    Pool::instance().resize(threads);
}

bool inside_parallel_region() {
    return g_inside_region;
}

void parallel_for(size_t count, size_t min_per_thread,
                  const std::function<void(size_t, size_t)>& body) {
    if (count == 0) return;

    const size_t threads = Pool::instance().size();

    // Runs inline when splitting does not pay: a single thread, inside another
    // parallel region, or less work than the per-thread minimum.
    if (threads == 1 || g_inside_region || min_per_thread == 0 || count < min_per_thread * 2) {
        body(0, count);
        return;
    }

    // More chunks than threads so the dynamic scheduling can balance, but never
    // so many that each drops below the profitable minimum.
    const size_t by_size = count / min_per_thread;
    const size_t chunks = std::max<size_t>(2, std::min(threads * 4, by_size));

    Pool::instance().run(count, chunks, body);
}

}  // namespace parallel
}  // namespace engine
