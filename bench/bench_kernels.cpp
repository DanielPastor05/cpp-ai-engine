// What each kernel costs the scheduler, and how much of the card it actually uses.
//
// docs/CUDA.md carried a roofline for exactly one operation -- the 4096-cubed
// matrix product -- and nothing for the other twenty-five kernels. That gap is
// how three of them shared a parallelism bug for weeks: each was correct, none
// had a number beside it, and "33 GB/s on a card that does 400" was only ever
// computed for permute_gather after a profiler pointed at it.
//
// Two halves, and they answer different questions.
//
// The occupancy table is static: registers per thread and shared memory per
// block decide how many blocks fit on an SM, which decides how much latency the
// hardware can hide. It comes from cudaFuncGetAttributes and
// cudaOccupancyMaxActiveBlocksPerMultiprocessor, both plain runtime calls --
// worth saying because `ncu` reports the same thing and wants administrator
// rights for its performance counters, which is why this project had no
// occupancy analysis at all.
//
// The bandwidth table is measured: each operation runs on a shape big enough to
// be memory bound, timed with CUDA events, and the bytes it must move are
// counted by hand. A kernel near peak bandwidth is done; one far below it is
// either uncoalesced or short of parallelism, and the two look identical in a
// wall-clock number.
//
// Build with -DENGINE_CUDA=ON. Without a device it says so and exits 0.

#include "engine/cuda.hpp"
#include "engine/random.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using engine::Tensor;
namespace cuda = engine::cuda;

namespace {

// Median of several timed repetitions, after a warm-up. The median and not the
// minimum: for a bandwidth figure the typical pass is what a training loop
// actually gets, and the fastest one flatters a cold cache.
double time_ms(const std::function<void()>& body, int reps = 20) {
    for (int i = 0; i < 3; ++i) body();
    cuda::synchronize();

    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        const auto start = std::chrono::steady_clock::now();
        body();
        cuda::synchronize();
        samples.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// A case is measured at two sizes so the two costs can be told apart.
//
// The first version of this timed one large shape and divided bytes by time,
// and every element-wise operation came out at 9-13% of peak -- which would mean
// six badly written kernels. They are not: the timed region is a whole Tensor
// operation, so it also contains allocating the output, the Storage bookkeeping
// and a synchronise, and those cost the same whatever the size.
//
// Two points separate them. Time is (fixed + bytes / bandwidth), so running the
// same operation at n and at n/8 gives a slope and an intercept: the slope is
// what the kernel achieves, the intercept is what the engine charges per
// operation regardless. Both numbers matter and they are different problems --
// one is a kernel to fix, the other is the allocator.
struct Case {
    std::string name;
    double bytes_big;
    double bytes_small;
    std::function<void()> run_big;
    std::function<void()> run_small;
};

void bandwidth_table(double peak_gbs) {
    engine::manual_seed(7);

    const size_t n = 1u << 24;  // 16.7M values, 64 MiB per buffer
    const size_t small = n / 8;
    Tensor a = Tensor::randn({n}), b = Tensor::randn({n});
    Tensor a_s = Tensor::randn({small}), b_s = Tensor::randn({small});
    Tensor rows = Tensor::randn({4096, 4096}), rows_s = Tensor::randn({1448, 1448});
    Tensor cube = Tensor::randn({64, 4096, 64}), cube_s = Tensor::randn({32, 2048, 64});

    // Touch each on the device first: dispatch decides on residency, and a
    // first-touch upload inside a timed region would be measuring PCIe.
    for (Tensor* t : {&a, &b, &a_s, &b_s, &rows, &rows_s, &cube, &cube_s}) (void)(*t + 0.0f);
    cuda::synchronize();

    const double f = sizeof(float);
    const std::vector<Case> cases = {
        {"add (elementwise)", 3.0 * n * f, 3.0 * small * f,
         [&] { Tensor c = a + b; }, [&] { Tensor c = a_s + b_s; }},
        {"scalar affine", 2.0 * n * f, 2.0 * small * f,
         [&] { Tensor c = a * 2.0f; }, [&] { Tensor c = a_s * 2.0f; }},
        {"relu", 2.0 * n * f, 2.0 * small * f,
         [&] { Tensor c = a.relu(); }, [&] { Tensor c = a_s.relu(); }},
        {"transpose (last two axes)", 2.0 * rows.size() * f, 2.0 * rows_s.size() * f,
         [&] { Tensor c = rows.transpose(); }, [&] { Tensor c = rows_s.transpose(); }},
        {"permute (0,2,1)", 2.0 * cube.size() * f, 2.0 * cube_s.size() * f,
         [&] { Tensor c = cube.permute({0, 2, 1}); },
         [&] { Tensor c = cube_s.permute({0, 2, 1}); }},
        {"sum over axis 0", 1.0 * rows.size() * f, 1.0 * rows_s.size() * f,
         [&] { Tensor c = rows.sum(0); }, [&] { Tensor c = rows_s.sum(0); }},
        {"sum to scalar", 1.0 * n * f, 1.0 * small * f,
         [&] { Tensor s = a.sum(); }, [&] { Tensor s = a_s.sum(); }},
    };

    printf("Bandwidth against size, %.0f GB/s theoretical peak\n\n", peak_gbs);
    printf("  %-26s %14s %14s %10s\n", "operation", "small", "large", "large/small");
    printf("  %-26s %14s %14s %10s\n", "", "GB/s", "GB/s", "per byte");
    for (const Case& c : cases) {
        const double big = time_ms(c.run_big);
        const double sml = time_ms(c.run_small);
        const double gbs_big = c.bytes_big / (big * 1e-3) / 1e9;
        const double gbs_sml = c.bytes_small / (sml * 1e-3) / 1e9;
        // Time per byte at each size. Equal means the cost is proportional to
        // the work, which is what a bandwidth number assumes.
        const double ratio = (big / c.bytes_big) / (sml / c.bytes_small);
        printf("  %-26s %9.1f (%3.0f%%) %9.1f (%3.0f%%) %9.2fx\n", c.name.c_str(), gbs_sml,
               gbs_sml / peak_gbs * 100.0, gbs_big, gbs_big / peak_gbs * 100.0, ratio);
    }
    printf(
        "\n  The last column is the honest one, and it is why there is no single\n"
        "  bandwidth figure here. A kernel whose cost is proportional to its work\n"
        "  reads 1.00x: the reductions do, and they reach a third to a half of\n"
        "  peak. The element-wise rows do not -- eight times the data costs far\n"
        "  more than eight times the time -- so neither their large nor their\n"
        "  small figure is a bandwidth, and quoting either as one would be wrong.\n"
        "\n"
        "  An earlier version of this table did exactly that, then a two-point fit\n"
        "  to separate a fixed cost from a slope, which produced negative fixed\n"
        "  costs: the model was linear and the measurement is not. What breaks the\n"
        "  proportionality has not been identified yet -- a 64 MiB output allocated\n"
        "  per call is the first suspect, and the tensor pool in the roadmap is\n"
        "  where that gets tested. Until then this is a shape, not a number.\n\n");
}

void occupancy_table() {
    const std::vector<cuda::KernelOccupancy> rows = cuda::kernel_occupancy();
    if (rows.empty()) {
        printf("No occupancy data: built without CUDA, or no device.\n");
        return;
    }

    printf("Occupancy ceiling per kernel\n\n");
    printf("  %-24s %8s %6s %8s %8s %8s  %s\n", "kernel", "threads", "regs", "smem",
           "blocks", "occup.", "limited by");
    for (const cuda::KernelOccupancy& k : rows) {
        printf("  %-24s %8d %6d %7zuB %8d %7.0f%%  %s\n", k.name.c_str(), k.block_threads,
               k.registers, k.shared_bytes, k.blocks_per_sm, k.occupancy * 100.0, k.limiter);
    }
    printf(
        "\n  This is the ceiling, not what a given launch achieves: it says how many\n"
        "  blocks *fit*, not how many were resident. A kernel below its own ceiling\n"
        "  is a grid problem; a low ceiling is a register or shared-memory problem.\n\n");
}

}  // namespace

int main() {
    if (!cuda::available()) {
        printf(
            "Built without CUDA, or no usable device.\n"
            "  cmake -B build-cuda -S . -DENGINE_CUDA=ON\n");
        return 0;
    }

    const cuda::DeviceInfo info = cuda::device_info();
    printf("Device: %s (cc %d.%d, %d SM)\n", info.name.c_str(), info.compute_major,
           info.compute_minor, info.multiprocessors);
    printf("Peak: %.0f GFLOP/s fp32, %.0f GB/s\n\n", cuda::peak_fp32_gflops(),
           cuda::peak_bandwidth_gbs());

    // Nothing goes to the CPU here: the point is to measure the kernels, even on
    // shapes where dispatching would not normally be worth it.
    cuda::set_thresholds(0, 0);

    occupancy_table();
    bandwidth_table(cuda::peak_bandwidth_gbs());
    return 0;
}
