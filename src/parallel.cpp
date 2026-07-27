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

// El hilo llamante participa en el trabajo, así que con N hilos se lanzan N-1
// trabajadores. Esta marca evita que una región paralela anidada vuelva a
// repartir: multiplicar hilos dentro de un hilo solo añade contención.
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
        // Un valor ilegible no es motivo para abortar: se usa el número de
        // núcleos, igual que si la variable no estuviera puesta.
        return fallback;
    }
}

// Pool sencillo de tareas indexadas. No es un pool general: solo sabe ejecutar
// «este cuerpo, para estos trozos», que es todo lo que el motor necesita.
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

    // Reparte [0, count) en `chunks` trozos contiguos y espera a que terminen.
    void run(size_t count, size_t chunks,
             const std::function<void(size_t, size_t)>& body) {
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

        // El hilo llamante toma trozos como cualquier otro, en vez de esperar
        run_chunks();

        lock.lock();
        work_done_.wait(lock, [this] { return remaining_ == 0; });
        std::exception_ptr error = error_;
        body_ = nullptr;
        lock.unlock();

        // Una excepción en un trabajador se relanza en el hilo que repartió
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

    // Reparto dinámico: cada hilo coge el siguiente trozo libre. Con trozos
    // de coste desigual —filas con más ceros en matmul, por ejemplo— reparte
    // mejor que asignar trozos fijos por hilo.
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

} // namespace

size_t num_threads() { return Pool::instance().size(); }

void set_num_threads(size_t threads) { Pool::instance().resize(threads); }

bool inside_parallel_region() { return g_inside_region; }

void parallel_for(size_t count, size_t min_per_thread,
                  const std::function<void(size_t, size_t)>& body) {
    if (count == 0) return;

    const size_t threads = Pool::instance().size();

    // Se ejecuta en línea cuando repartir no compensa: un solo hilo, dentro de
    // otra región paralela, o menos trabajo del mínimo por hilo.
    if (threads == 1 || g_inside_region || min_per_thread == 0 ||
        count < min_per_thread * 2) {
        body(0, count);
        return;
    }

    // Más trozos que hilos para que el reparto dinámico pueda equilibrar,
    // pero nunca tantos que cada uno baje del mínimo rentable.
    const size_t by_size = count / min_per_thread;
    const size_t chunks = std::max<size_t>(2, std::min(threads * 4, by_size));

    Pool::instance().run(count, chunks, body);
}

} // namespace parallel
} // namespace engine
