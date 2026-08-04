// Performance benchmark.
//
// It verifies nothing: it measures. It is in the repository so the figures in the
// README's performance notes are reproducible, and to catch regressions when the
// kernels are touched. It does not run in CI, because a shared runner's timings
// a shared runner are not comparable between runs.
//
//   cmake --build build --target bench && ./build/bench

#include "engine/tensor.hpp"
#include "engine/random.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/conv.hpp"
#include "engine/transformer.hpp"
#include "engine/optim.hpp"
#include "engine/parallel.hpp"
#include "engine/cuda.hpp"

#include <optional>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;

namespace {

// Runs the function until at least min_seconds have accumulated and returns the
// mean time per iteration, so that fast operations are not left at the mercy of
// the clock's resolution.
double time_op(const std::function<void()>& fn, double min_seconds = 0.3) {
    // A cold pass, so the first cache miss is not what gets measured
    fn();

    size_t reps = 0;
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    while (elapsed < min_seconds) {
        fn();
        ++reps;
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    return elapsed / static_cast<double>(reps);
}

void row(const std::string& label, double seconds, const std::string& note = "") {
    printf("  %-42s %9.3f ms   %s\n", label.c_str(), seconds * 1000.0, note.c_str());
}

void section(const std::string& title) {
    printf("\n=== %s ===\n", title.c_str());
}

std::string gflops(double seconds, double flops) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f GFLOP/s", flops / seconds / 1e9);
    return buf;
}

}  // namespace

int main() {
    engine::manual_seed(1);
    printf("cpp-ai-engine benchmark\n");

    section("matmul");
    {
        struct Case {
            size_t M, K, N;
        };
        const Case cases[] = {{64, 64, 64}, {256, 256, 256}, {512, 128, 512}};
        for (const Case& c : cases) {
            Tensor A = Tensor::randn({c.M, c.K});
            Tensor B = Tensor::randn({c.K, c.N});
            const double t = time_op([&] { Tensor C = A.matmul(B); });
            row(std::to_string(c.M) + "x" + std::to_string(c.K) + "x" + std::to_string(c.N), t,
                gflops(t, 2.0 * c.M * c.K * c.N));
        }
        // Batched, as in attention
        Tensor Q = Tensor::randn({32, 4, 16, 16});
        Tensor K = Tensor::randn({32, 4, 16, 16});
        const double t = time_op([&] { Tensor C = Q.matmul(K); });
        row("batched (32, 4, 16, 16)", t, gflops(t, 2.0 * 32 * 4 * 16 * 16 * 16));
    }

    section("element-wise operations (1M values)");
    {
        Tensor A = Tensor::randn({1000, 1000});
        Tensor B = Tensor::randn({1000, 1000});
        Tensor row_vec = Tensor::randn({1000});
        row("add", time_op([&] { Tensor C = A + B; }));
        row("multiply", time_op([&] { Tensor C = A * B; }));
        row("broadcast addition", time_op([&] { Tensor C = A + row_vec; }));
        row("relu", time_op([&] { Tensor C = A.relu(); }));
        row("softmax (row-wise)", time_op([&] { Tensor C = A.softmax(); }));
        row("transpose", time_op([&] { Tensor C = A.transpose(); }));
        row("sum(axis=0)", time_op([&] { Tensor C = A.sum(0); }));
    }

    section("permute");
    {
        Tensor T = Tensor::randn({32, 8, 16, 16});
        row("(32,8,16,16) -> {0,2,1,3}", time_op([&] { Tensor C = T.permute({0, 2, 1, 3}); }));
    }

    section("layers (forward + backward step)");
    {
        nn::Linear dense(512, 512);
        Tensor x = Tensor::randn({64, 512}, 0.0f, 1.0f, true);
        row("Linear(512,512) batch 64", time_op([&] {
                dense.zero_grad();
                dense(x).sum().backward();
            }));

        nn::Conv2d conv(8, 16, nn::Window2d(3, 3, 1, 1));
        Tensor img = Tensor::randn({16, 8, 16, 16}, 0.0f, 1.0f, true);
        row("Conv2d(8->16, 3x3) batch 16 of 16x16", time_op([&] {
                conv.zero_grad();
                conv(img).sum().backward();
            }));

        nn::MaxPool2d pool(2, 2);
        row("MaxPool2d 2x2", time_op([&] {
                Tensor out = pool(img);
                out.sum().backward();
            }));

        nn::LayerNorm norm(256);
        Tensor seq = Tensor::randn({32, 16, 256}, 0.0f, 1.0f, true);
        row("LayerNorm(256) over (32,16,256)", time_op([&] {
                norm.zero_grad();
                norm(seq).sum().backward();
            }));

        nn::TransformerBlock block(128, 4, 256);
        Tensor tokens = Tensor::randn({16, 32, 128}, 0.0f, 1.0f, true);
        row("TransformerBlock(128, 4 cabezas) (16,32,128)", time_op([&] {
                block.zero_grad();
                block(tokens).sum().backward();
            }));
    }

    section("a full training iteration");
    {
        nn::Sequential mlp{nn::make<nn::Linear>(128, 256), nn::make<nn::ReLU>(),
                           nn::make<nn::Linear>(256, 10)};
        optim::Adam opt(mlp.parameters(), 0.001f);
        Tensor X = Tensor::randn({128, 128});
        std::vector<size_t> y(128, 3);

        row("MLP 128-256-10, batch 128", time_op([&] {
                opt.zero_grad();
                nn::cross_entropy_loss(mlp(X), y).backward();
                opt.step();
            }));
    }

    section("inference against training");
    {
        nn::TransformerBlock block(128, 4, 256);
        Tensor tokens = Tensor::randn({16, 32, 128}, 0.0f, 1.0f, true);
        const double with_graph = time_op([&] { Tensor out = block(tokens); });
        const double without = time_op([&] {
            engine::autograd::NoGradGuard no_grad;
            Tensor out = block(tokens);
        });
        row("TransformerBlock construyendo grafo", with_graph);
        row("TransformerBlock bajo NoGradGuard", without,
            "x" + std::to_string(with_graph / without).substr(0, 4) + " mas rapido");
    }

    section("scaling with thread count");
    {
        // The split is by output row, so the result is identical bit for bit whatever
        // the thread count.
        Tensor A = Tensor::randn({512, 512});
        Tensor B = Tensor::randn({512, 512});
        Tensor E1 = Tensor::randn({2000, 2000});
        Tensor E2 = Tensor::randn({2000, 2000});

        const size_t original = engine::parallel::num_threads();
        double base_mm = 0.0;
        double base_add = 0.0;

        for (size_t threads = 1; threads <= original; ++threads) {
            engine::parallel::set_num_threads(threads);
            const double mm = time_op([&] { Tensor C = A.matmul(B); });
            const double add = time_op([&] { Tensor C = E1 + E2; });
            if (threads == 1) {
                base_mm = mm;
                base_add = add;
            }

            char note[96];
            snprintf(note, sizeof(note), "matmul %.2fx   add %.2fx  (%s)", base_mm / mm,
                     base_add / add, gflops(mm, 2.0 * 512 * 512 * 512).c_str());
            row(std::to_string(threads) + " thread(s): matmul 512^3", mm, note);
        }
        engine::parallel::set_num_threads(original);
        printf(
            "\n  Addition scales worse: it is limited by the bandwidth of\n"
            "  memory rather than by the arithmetic.\n");
    }

    section("CPU frente a GPU");
    {
        namespace cuda = engine::cuda;

        if (!cuda::available()) {
            printf(
                "  Built without CUDA, or no usable device.\n"
                "  Rebuild with: cmake -B build-cuda -S . -DENGINE_CUDA=ON\n");
        } else {
            const std::optional<cuda::DeviceInfo> device = cuda::device_info();
            printf("  Device: %s (cc %d.%d, %d SM, %zu MiB)\n\n", device->name.c_str(),
                   device->compute_major, device->compute_minor, device->multiprocessors,
                   device->total_memory >> 20);

            // No threshold: the point here is to measure where the crossover is, not apply it.
            const size_t saved_flops = cuda::min_matmul_flops();
            const size_t saved_elems = cuda::min_elementwise_elements();
            cuda::set_thresholds(0, 0);

            printf("  matmul NxNxN, with the operands already resident on the device:\n");
            printf("  %-10s %12s %12s %10s %14s\n", "N", "CPU (ms)", "GPU (ms)", "speedup",
                   "GPU GFLOP/s");

            const size_t sizes[] = {64, 128, 256, 512, 1024, 2048};
            for (size_t n : sizes) {
                Tensor A = Tensor::randn({n, n});
                Tensor B = Tensor::randn({n, n});
                const double flops = 2.0 * (double)n * n * n;

                cuda::set_enabled(false);
                const double cpu = time_op([&] { Tensor C = A.matmul(B); });

                // The first pass uploads A and B; from then on they stay up there, so what is
                // measured is the kernel and not PCIe.
                cuda::set_enabled(true);
                const double gpu = time_op([&] {
                    Tensor C = A.matmul(B);
                    cuda::synchronize();
                });

                printf("  %-10zu %12.3f %12.3f %9.2fx %14.1f\n", n, cpu * 1000.0, gpu * 1000.0,
                       cpu / gpu, flops / gpu / 1e9);
            }

            // The number that really matters: what crossing PCIe costs.
            // A CPU/GPU table that hides this inside the total says nothing, because in a
            // real engine the transfer dominates long before the arithmetic does.
            //
            printf("\n  Cost of the host <-> device transfers:\n");
            {
                const size_t n = 1024;
                Tensor A = Tensor::randn({n, n});
                Tensor B = Tensor::randn({n, n});
                cuda::set_enabled(true);

                // Case 1: the data is already up there and the result stays.
                Tensor warm = A.matmul(B);
                cuda::synchronize();
                cuda::reset_transfer_stats();
                const double resident = time_op([&] {
                    Tensor C = A.matmul(B);
                    cuda::synchronize();
                });
                const cuda::TransferStats resident_stats = cuda::transfer_stats();

                // Case 2: the operands are touched on the host between iterations, which is what
                // happens in a training loop where the optimiser and the loss are still on the
                // CPU. Each iteration goes back
                // a subir y a bajar.
                cuda::reset_transfer_stats();
                const double round_trip = time_op([&] {
                    A.data()[0] = A.data()[0];  // invalidates the device mirror
                    B.data()[0] = B.data()[0];
                    Tensor C = A.matmul(B);
                    (void)C.data()[0];  // forces the result back down
                });
                const cuda::TransferStats trip_stats = cuda::transfer_stats();

                const double mib = (double)(n * n * sizeof(float)) / (1024.0 * 1024.0);
                printf("    matmul 1024^3, data resident         %8.3f ms  (%zu up, %zu down)\n",
                       resident * 1000.0, resident_stats.to_device_count,
                       resident_stats.to_host_count);
                printf("    matmul 1024^3, full round trip       %8.3f ms  (%zu up, %zu down)\n",
                       round_trip * 1000.0, trip_stats.to_device_count, trip_stats.to_host_count);
                printf("    round-trip penalty                   %8.2fx\n", round_trip / resident);
                if (trip_stats.to_device_seconds > 0.0) {
                    printf(
                        "    H2D bandwidth                        %8.2f GB/s (%.1f MiB per "
                        "matrix)\n",
                        (double)trip_stats.to_device_bytes / trip_stats.to_device_seconds / 1e9,
                        mib);
                }
                if (trip_stats.to_host_seconds > 0.0) {
                    printf("    D2H bandwidth                        %8.2f GB/s\n",
                           (double)trip_stats.to_host_bytes / trip_stats.to_host_seconds / 1e9);
                }
                printf(
                    "\n    The download includes waiting for the kernel: cudaMemcpy synchronises.\n"
                    "    That is what should be measured: the real cost of reading a result.\n");
            }

            printf("\n  Element-wise operations (resident):\n");
            for (size_t n : {size_t{1} << 18, size_t{1} << 22, size_t{1} << 24}) {
                Tensor A = Tensor::randn({n});
                Tensor B = Tensor::randn({n});
                cuda::set_enabled(false);
                const double cpu = time_op([&] { Tensor C = A + B; });
                cuda::set_enabled(true);
                const double gpu = time_op([&] {
                    Tensor C = A + B;
                    cuda::synchronize();
                });
                printf("    sum of %10zu values    CPU %8.3f ms   GPU %8.3f ms   %5.2fx\n", n,
                       cpu * 1000.0, gpu * 1000.0, cpu / gpu);
            }

            cuda::set_thresholds(saved_flops, saved_elems);
            cuda::set_enabled(true);
            printf(
                "\n  Dispatch thresholds in use: %zu operations for matmul,\n"
                "  %zu elements for the element-wise operations.\n"
                "  Set them with ENGINE_CUDA_MIN_FLOPS and ENGINE_CUDA_MIN_ELEMENTS.\n",
                cuda::min_matmul_flops(), cuda::min_elementwise_elements());
        }
    }

    printf("\n");
    return 0;
}
