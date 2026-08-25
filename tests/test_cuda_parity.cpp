// CPU / GPU parity.
//
// Each case computes the same expression twice over exactly the same data -- once
// with the backend off and once with it on -- and compares. It is the only way to
// check a kernel that is worth anything: kernels do not fail by returning an
// error, they fail by returning plausible numbers.
//
// The comparison is to a tolerance rather than exact, and that is not a
// concession. The device compiler fuses multiply and add into one FMA, which
// rounds once where the CPU rounds twice; the result differs in the last bit and
// accumulates with K. Demanding bit-identical results between CPU and GPU would
// be demanding that the GPU compute worse.
//
// Without ENGINE_CUDA the file compiles to a note that there is nothing to
// check, so the suite stays a single one and CI needs no GPU.

#include "test_support.hpp"

#include "engine/cuda.hpp"
#include "engine/detail/cuda_ops.hpp"
#include "engine/transformer.hpp"

#ifdef ENGINE_CUDA

#include <optional>
#include <algorithm>
#include <cmath>

namespace {

namespace cuda = engine::cuda;

// Turns a flat index into coordinates, so a failure says "row 128, column 0"
// rather than "element 16384".
std::string position_of(size_t flat, const std::vector<size_t>& shape) {
    if (shape.empty()) return "(scalar)";
    std::vector<size_t> coords(shape.size());
    for (size_t axis = shape.size(); axis-- > 0;) {
        coords[axis] = (shape[axis] == 0) ? 0 : flat % shape[axis];
        flat = (shape[axis] == 0) ? 0 : flat / shape[axis];
    }
    std::string out = "(";
    for (size_t i = 0; i < coords.size(); ++i) {
        out += std::to_string(coords[i]);
        if (i + 1 < coords.size()) out += ", ";
    }
    return out + ")";
}

// Evaluates the same function with the backend off and on, and compares.
//
// When it fails it prints considerably more than the maximum error, for a
// practical reason: these kernels cannot run in the environment they are written
// in, so each debugging iteration costs a whole round trip to a machine with a
// GPU. The message has to carry everything needed to locate the fault in one go.
//
//
// The **distribution** of the differences says more than their size:
//   - a few at the end      -> edge handling
//   - all of them          -> the formula, not the indices
//   - at regular intervals -> a stride or a transposition in the wrong place
void compare(const std::string& what, const std::function<Tensor()>& compute, float tol = 1e-5f) {
    cuda::set_enabled(false);
    const Tensor on_cpu = compute();
    const std::vector<float> want = on_cpu.to_vector();
    const std::vector<size_t> shape = on_cpu.shape();

    cuda::set_enabled(true);
    const size_t launched_before = cuda::kernels_launched();
    const std::vector<float> got = compute().to_vector();
    const size_t launched = cuda::kernels_launched() - launched_before;

    ++testing::g_checks;

    // Without this the test is a trap. The engine falls back to the CPU path when a
    // kernel cannot be launched -- correct in production -- so the comparison would
    // be CPU against CPU and would pass with exactly zero error without ever touching
    // the device. It really happened: the whole section came out green at 0.00e+00
    // while not one kernel ever ran.
    if (launched == 0) {
        ++testing::g_failures;
        std::cout << "  [FAIL] " << what
                  << " (no kernel was launched: this compared CPU against CPU)\n";
        return;
    }

    if (got.size() != want.size()) {
        ++testing::g_failures;
        std::cout << "  [FAIL] " << what << " (different sizes: " << got.size() << " against "
                  << want.size() << ")\n";
        return;
    }

    size_t differing = 0;
    size_t first = want.size();
    size_t last = 0;
    size_t worst_at = 0;
    float worst = 0.0f;

    for (size_t i = 0; i < want.size(); ++i) {
        const float scale = std::max(1.0f, std::fabs(want[i]));
        const float error = std::fabs(got[i] - want[i]) / scale;
        if (error > worst) {
            worst = error;
            worst_at = i;
        }
        if (error > tol) {
            ++differing;
            if (first == want.size()) first = i;
            last = i;
        }
    }

    if (differing == 0) {
        std::cout << "  [ ok ] " << what << " (max relative error " << std::scientific
                  << std::setprecision(2) << worst << std::defaultfloat << ")\n";
        return;
    }

    ++testing::g_failures;
    std::cout << "  [FAIL] " << what << "\n"
              << "         max relative error " << std::scientific << std::setprecision(3) << worst
              << " > " << tol << std::defaultfloat << "\n"
              << "         " << differing << " of " << want.size() << " elements differ, from "
              << first << " to " << last << "\n"
              << "         worst at " << worst_at << " = " << position_of(worst_at, shape)
              << ": expected " << want[worst_at] << ", got " << got[worst_at] << "\n";

    if (differing == want.size()) {
        std::cout << "         all differ: points at the formula, not the indices\n";
    } else if (first > 0 && last + 1 == want.size()) {
        std::cout << "         the tail differs: points at the edge handling\n";
    } else if (differing * 4 < want.size()) {
        std::cout << "         a scattered minority differs: points at a stride or a"
                     " transposition\n";
    }
}

}  // namespace

void run_cuda_parity_tests() {
    testing::section("CPU / GPU parity (Phase 6)");

    if (!cuda::available()) {
        std::cout << "  (built with CUDA but no device present: skipped)\n";
        return;
    }

    // The early return above already established there is a device, so the
    // optional is engaged; value() rather than * says so and would throw
    // rather than read past nothing if that ever stopped being true.
    const cuda::DeviceInfo info = cuda::device_info().value();
    // All three versions, and in this order, because the mismatch that breaks
    // execution is toolkit > driver. Printing only the runtime hides it: it follows
    // the driver, so on a machine with driver 13.2 and toolkit 13.3 this line said
    // "13.2 and 13.2" and everything looked consistent.
    const int built = cuda::compiled_version();
    const int rt = cuda::runtime_version();
    const int drv = cuda::driver_version();
    std::cout << "  Device: " << info.name << " (cc " << info.compute_major << "."
              << info.compute_minor << ", " << info.multiprocessors << " SM, "
              << (info.total_memory >> 20) << " MiB)\n"
              << "  Built with CUDA " << built / 1000 << "." << (built % 1000) / 10
              << ", runtime CUDA " << rt / 1000 << "." << (rt % 1000) / 10 << " loaded"
              << ", driver CUDA " << drv / 1000 << "." << (drv % 1000) / 10 << "\n";
    if (drv > 0 && built > drv) {
        std::cout << "  Warning: the toolkit is ahead of the driver. If a kernel does not\n"
                     "         start, that is the first thing to look at.\n";
    }

    cuda::reset_kernel_counters();

    // The normal thresholds would send everything here to the CPU, which is the
    // opposite of what a parity test wants: the point is to exercise small shapes
    // with remainders, where the tile edges are what fail. They are zeroed for the
    // duration of the test and restored at the end.
    //
    const size_t saved_flops = cuda::min_matmul_flops();
    const size_t saved_elements = cuda::min_elementwise_elements();
    const size_t saved_layernorm = cuda::min_layernorm_elements();
    cuda::set_thresholds(0, 0, 0);

    engine::manual_seed(20260728);

    // --- the matrix product, all four variants ---
    //
    // Each kernel is checked separately, over the same shapes. That is what really
    // protects this work: the register-tiled kernel operates on 128x128 blocks, so
    // its failures appear precisely on the shapes that are not a multiple of that,
    // and only there. Testing only the default variant would leave the other three
    // without a net.
    {
        const cuda::MatmulKernel variants[] = {
            cuda::MatmulKernel::Naive,
            cuda::MatmulKernel::Tiled,
            cuda::MatmulKernel::RegisterTiled,
            cuda::MatmulKernel::Vectorized,
        };

        struct Case {
            size_t M, K, N;
        };
        const Case cases[] = {
            {1, 1, 1},        // el caso degenerado
            {17, 23, 31},     // below a single tile
            {32, 32, 32},     // exactly one 32-wide tile
            {33, 65, 129},    // remainders on all three axes
            {127, 128, 129},  // around the 128 block
            {128, 128, 128},  // exactamente un bloque
            {129, 256, 257},  // more than one block, with a remainder
            {256, 260, 256},  // K a multiple of 4 but not 8: partial K tile
        };

        for (cuda::MatmulKernel variant : variants) {
            cuda::set_matmul_kernel(variant);
            const std::string tag = std::string("[") + cuda::matmul_kernel_name(variant) + "] ";

            for (const Case& c : cases) {
                Tensor A = Tensor::randn({c.M, c.K});
                Tensor B = Tensor::randn({c.K, c.N});
                compare(tag + "matmul " + std::to_string(c.M) + "x" + std::to_string(c.K) + "x" +
                            std::to_string(c.N),
                        [&] { return A.matmul(B); });
            }

            Tensor QB = Tensor::randn({4, 3, 17, 23});
            Tensor KB = Tensor::randn({4, 3, 23, 11});
            compare(tag + "batched matmul (4,3,17,23) x (4,3,23,11)",
                    [&] { return QB.matmul(KB); });

            // A 2D operand shared with the whole batch: this checks the stride of 0.
            Tensor X = Tensor::randn({8, 12, 40});
            Tensor W = Tensor::randn({40, 20});
            compare(tag + "matmul with a shared operand (8,12,40) x (40,20)",
                    [&] { return X.matmul(W); });
        }

        // A shape whose K is not a multiple of 4: asking for the vectorised variant has
        // to degrade to the register-tiled one, not read misaligned float4s.
        // A misaligned read raises no error; it returns a different value.
        cuda::set_matmul_kernel(cuda::MatmulKernel::Vectorized);
        Tensor A = Tensor::randn({131, 133});
        Tensor B = Tensor::randn({133, 135});
        compare("[vectorized] degrades with K and N unaligned (133, 135)",
                [&] { return A.matmul(B); });

        cuda::set_matmul_kernel(cuda::MatmulKernel::Auto);
    }

    // --- matmul with beta: accumulate instead of overwrite ---
    //
    // This case protects one line of the dispatch. With beta == 0 the output is
    // taken with device_write(), which reserves the buffer **without uploading**
    // because the kernel overwrites every element. With beta != 0 the kernel
    // reads what is there, so the host side has to be uploaded first and the
    // accessor must be device_mut().
    //
    // Getting that wrong loses the accumulated value silently, and it is
    // invisible onto a zero destination: reading an unuploaded buffer that
    // happens to hold zeros gives the same answer as reading a correctly
    // uploaded one. So the destination here is **not** zero, and it is written
    // on the host, which means the only way the device can see those values is
    // by uploading them.
    {
        struct BetaCase {
            size_t M, K, N;
        };
        const BetaCase beta_cases[] = {{17, 23, 31}, {64, 64, 64}, {129, 96, 130}};
        for (const BetaCase& c : beta_cases) {
            Tensor A(std::vector<size_t>{c.M, c.K}, 0.0f, false);
            Tensor B(std::vector<size_t>{c.K, c.N}, 0.0f, false);
            for (size_t i = 0; i < A.size(); ++i) A.data()[i] = 0.1f * (float)(i % 13) - 0.6f;
            for (size_t i = 0; i < B.size(); ++i) B.data()[i] = 0.2f * (float)(i % 7) - 0.7f;

            const std::string tag =
                std::to_string(c.M) + "x" + std::to_string(c.K) + "x" + std::to_string(c.N);
            compare("matmul beta=1 onto a non-zero destination " + tag, [&] {
                Tensor out(std::vector<size_t>{c.M, c.N}, 0.0f, false);
                for (size_t i = 0; i < out.size(); ++i)
                    out.data()[i] = 0.5f * (float)(i % 5) - 1.0f;

                if (!engine::cuda::ops::matmul(A.storage(), B.storage(), out.storage(), 1, c.M, c.K,
                                               c.N, true, true, 1.0f)) {
                    // What accumulation means, on the side that has no kernel.
                    const Tensor product = A.matmul(B);
                    for (size_t i = 0; i < out.size(); ++i) out.data()[i] += product.data()[i];
                }
                return out;
            });
        }
    }

    // --- split-K: a tall thin product, which is a convolution's weight gradient ---
    //
    // Auto picks the tiled kernel whenever either side is under 128, and when the
    // output is also small enough to fit in a handful of tiles it cuts K into
    // slices and sums the partials. That branch is invisible to the caller and
    // fires on no other shape in this file, so without these cases it would be
    // dead code that happens to run in the demos.
    //
    // The shapes are MNIST's, one per convolution: cols^T x dout for a 3x3 kernel
    // over 28x28 and over 14x14 with a batch of 8. K has to clear 4096 and the
    // output has to stay under eight 32x32 tiles for the path to be taken.
    {
        // The tolerance has to grow with K, and the reason is the **CPU** side:
        // it sums K products serially in float, so its error walks as sqrt(K)
        // steps of size u against a result of size sqrt(K), which lands at about
        // K*u relative. The GPU is the more accurate of the two here -- tiles and
        // slices are partial sums, which is shorter chains -- so this bound
        // measures how far the reference has drifted, not the kernel.
        //
        // At 1e-5 flat, 144x1568x32 failed at 2.2e-05 and that shape does not even
        // take the split-K branch. The bound was wrong, not the code.
        auto long_k_tolerance = [](size_t k) {
            return std::max(1e-5f, 2.0f * static_cast<float>(k) / 16777216.0f);
        };

        struct SplitCase {
            size_t M, K, N;
        };
        const SplitCase cases[] = {
            {9, 6272, 16},     // conv1: 3x3x1 columns, 16 channels out
            {144, 1568, 32},   // conv2, K just under the threshold: the plain kernel
            {144, 12544, 32},  // conv2 at a batch that crosses it
            {7, 5000, 5},      // K not a multiple of the tile, output not either
        };
        for (const SplitCase& c : cases) {
            Tensor A = Tensor::randn({c.M, c.K});
            Tensor B = Tensor::randn({c.K, c.N});
            compare(
                "[auto] split-K " + std::to_string(c.M) + "x" + std::to_string(c.K) + "x" +
                    std::to_string(c.N),
                [&] { return A.matmul(B); }, long_k_tolerance(c.K));
        }

        // A loosened tolerance is a weaker test, so this is the one that carries
        // the weight -- the same trick the tf32 section uses further down, and for
        // the same reason: it is what catches an index that is wrong rather than
        // a sum that is imprecise.
        //
        // Products of small integers are exact in fp32 and stay exact for K of
        // them, so a split whose chunk boundaries are off, or whose partials are
        // summed with one missing, cannot hide behind rounding. It has to match
        // the CPU bit for bit.
        {
            bool exact = true;
            for (const SplitCase& c : cases) {
                Tensor A(std::vector<size_t>{c.M, c.K}, 0.0f, false);
                Tensor B(std::vector<size_t>{c.K, c.N}, 0.0f, false);
                for (size_t i = 0; i < A.size(); ++i) A.data()[i] = (float)((int)(i % 5) - 2);
                for (size_t i = 0; i < B.size(); ++i) B.data()[i] = (float)((int)(i % 3) - 1);

                cuda::set_enabled(false);
                const std::vector<float> want = A.matmul(B).to_vector();
                cuda::set_enabled(true);
                const std::vector<float> got = A.matmul(B).to_vector();
                for (size_t i = 0; i < want.size() && exact; ++i) exact = want[i] == got[i];
                if (!exact) break;
            }
            testing::check(exact, exact ? "split-K matches the CPU exactly on integer inputs, so "
                                          "the chunk boundaries and the partial sum are right"
                                        : "split-K disagrees with the CPU on integer inputs: the "
                                          "tolerance cases above were hiding an indexing error");
        }
    }

    // --- the tensor-core kernel, at a tolerance that says what it costs ---
    //
    // tf32 keeps fp32's exponent and cuts the mantissa from 23 bits to 10, so a
    // single product carries about 2^-11 of relative error where the fp32 kernels
    // carry 2^-24. Comparing it at the 1e-5 the other four are held to would fail
    // for the right reason and tell nobody anything; comparing it at a tolerance
    // chosen to pass would be worse.
    //
    // The bound grows with K, because the error does: tf32's unit roundoff is
    // 2^-11, and a dot product of K terms with independent roundings accumulates
    // about sqrt(K) of them. 4 * sqrt(K) * 2^-11 leaves a factor of two of margin
    // over what is actually observed, and it is a formula rather than a constant
    // so that a change of shape cannot quietly move inside it.
    //
    // The first version of this used a flat 5e-3 and failed on four of six cases,
    // which was the bound being wrong rather than the kernel: compare() reported
    // "a scattered minority differs" every time -- 1 element of 256, 171 of 16384
    // -- and always on outputs whose true value sits near zero, where the
    // max(1, |want|) scale floor leaves the absolute error undivided. An indexing
    // error looks nothing like that; it differs everywhere.
    //
    // A loosened tolerance is still a weaker test, so the exact case below carries
    // the real weight.
    if (cuda::tensor_cores_available()) {
        cuda::set_matmul_kernel(cuda::MatmulKernel::TensorCore);

        auto tf32_tolerance = [](size_t k) {
            return 4.0f * std::sqrt(static_cast<float>(k)) / 2048.0f;
        };

        struct Case {
            size_t M, K, N;
        };
        const Case cases[] = {
            {16, 16, 16},     // exactly one wmma fragment
            {64, 32, 64},     // exactly one block tile
            {65, 33, 67},     // remainders on all three axes: the zero padding
            {128, 128, 128},  // several block tiles
            {129, 96, 130},   // several, with remainders
        };
        for (const Case& c : cases) {
            Tensor A = Tensor::randn({c.M, c.K});
            Tensor B = Tensor::randn({c.K, c.N});
            compare(
                "[tensorcore] tf32 matmul " + std::to_string(c.M) + "x" + std::to_string(c.K) +
                    "x" + std::to_string(c.N),
                [&] { return A.matmul(B); }, tf32_tolerance(c.K));
        }

        Tensor QB = Tensor::randn({3, 2, 40, 24});
        Tensor KB = Tensor::randn({3, 2, 24, 40});
        compare(
            "[tensorcore] batched (3,2,40,24) x (3,2,24,40)", [&] { return QB.matmul(KB); },
            tf32_tolerance(24));

        // --- and now the test that actually pins the indices down ---
        //
        // Every case above is a tolerance, and a tolerance can be widened until
        // anything passes. This one cannot be.
        //
        // tf32 has 10 explicit mantissa bits, so it represents integers up to
        // 2048 **exactly**. Feed the kernel small integers, keep every partial sum
        // under that, and the mantissa cut costs nothing: the tf32 product is then
        // required to equal the fp32 product bit for bit. Any error in the
        // fragment indices, the shared-memory staging, the zero padding at the
        // edges or the warp quadrant mapping survives no tolerance at all here.
        {
            struct Exact {
                size_t M, K, N;
            };
            const Exact shapes[] = {{16, 16, 16}, {64, 32, 64}, {65, 33, 67}, {129, 96, 130}};
            bool all_exact = true;
            size_t worst_shape = 0;

            for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
                const Exact& e = shapes[s];
                Tensor A(std::vector<size_t>{e.M, e.K}, 0.0f, false);
                Tensor B(std::vector<size_t>{e.K, e.N}, 0.0f, false);
                // Integers in [-4, 4]: products fit in 4 bits, and K terms of them
                // stay far inside both tf32's exact-integer range and fp32's.
                for (size_t i = 0; i < A.size(); ++i) {
                    A.data()[i] = static_cast<float>((int)(i % 9) - 4);
                }
                for (size_t i = 0; i < B.size(); ++i) {
                    B.data()[i] = static_cast<float>((int)(i % 7) - 3);
                }

                cuda::set_enabled(false);
                const std::vector<float> want = A.matmul(B).to_vector();
                cuda::set_enabled(true);
                cuda::set_matmul_kernel(cuda::MatmulKernel::TensorCore);
                const std::vector<float> got = A.matmul(B).to_vector();

                for (size_t i = 0; i < want.size(); ++i) {
                    if (want[i] != got[i]) {
                        all_exact = false;
                        worst_shape = s;
                        break;
                    }
                }
                if (!all_exact) break;
            }
            testing::check(all_exact, all_exact
                                          ? "tf32 matches fp32 exactly on integer inputs, so the "
                                            "fragment indices are right"
                                          : "tf32 disagrees with fp32 on integer inputs at shape " +
                                                std::to_string(worst_shape) +
                                                ": the tolerance cases above were hiding an "
                                                "indexing error");
        }

        // And the point of the whole exercise: tf32 is genuinely less precise, so
        // the same product under the fp32 kernels must be *visibly* tighter. A
        // kernel that quietly fell back to fp32 would pass every case above.
        Tensor A = Tensor::randn({256, 256});
        Tensor B = Tensor::randn({256, 256});
        cuda::set_enabled(false);
        const std::vector<float> want = A.matmul(B).to_vector();

        cuda::set_enabled(true);
        cuda::set_matmul_kernel(cuda::MatmulKernel::TensorCore);
        const std::vector<float> tf32 = A.matmul(B).to_vector();
        cuda::set_matmul_kernel(cuda::MatmulKernel::Vectorized);
        const std::vector<float> fp32 = A.matmul(B).to_vector();

        auto worst_against_cpu = [&want](const std::vector<float>& got) {
            double worst = 0.0;
            for (size_t i = 0; i < want.size(); ++i) {
                const double scale = std::max(1.0, std::fabs((double)want[i]));
                worst = std::max(worst, std::fabs((double)got[i] - want[i]) / scale);
            }
            return worst;
        };
        const double tf32_err = worst_against_cpu(tf32);
        const double fp32_err = worst_against_cpu(fp32);

        std::cout << "  tf32 error " << std::scientific << std::setprecision(2) << tf32_err
                  << " against fp32 error " << fp32_err << std::defaultfloat << " on 256x256\n";
        testing::check(tf32_err > fp32_err * 10.0,
                       "the tensor-core kernel really is running in tf32, not fp32");

        // --- and the same in fp16, which is the one with the throughput ---
        //
        // Same 10 mantissa bits as tf32, so the same derived bound applies: the
        // rounding per product is comparable and it is the *range* that differs,
        // 5 exponent bits against fp32's 8. Unit-normal inputs never approach
        // either end of that, which is exactly why a benchmark cannot stand in
        // for a training loop here.
        cuda::set_matmul_kernel(cuda::MatmulKernel::TensorCoreFp16);
        for (const Case& c : cases) {
            Tensor P = Tensor::randn({c.M, c.K});
            Tensor Q = Tensor::randn({c.K, c.N});
            compare(
                "[fp16] tensor-core matmul " + std::to_string(c.M) + "x" + std::to_string(c.K) +
                    "x" + std::to_string(c.N),
                [&] { return P.matmul(Q); }, tf32_tolerance(c.K));
        }

        // The case that carries the weight, for the reason the tf32 one does:
        // products of small integers are exact in fp16 too -- its mantissa holds
        // integers up to 2048 exactly, and K of them accumulate in fp32 -- so a
        // fragment index that is wrong cannot hide behind rounding. This is the
        // test that caught store_matrix_sync's leading-dimension requirement in
        // the tf32 kernel, and the fp16 kernel inherits that store verbatim.
        {
            // Same shapes as the tf32 exact case, redeclared because that one
            // scoped them to its own block. 65x33x67 is the one that matters:
            // an N of 67 is not a multiple of four, which is the store's
            // requirement, so it exercises the edge path both kernels share.
            struct ExactCase {
                size_t M, K, N;
            };
            const ExactCase fp16_shapes[] = {
                {16, 16, 16}, {64, 32, 64}, {65, 33, 67}, {129, 96, 130}};

            bool exact = true;
            for (const ExactCase& e : fp16_shapes) {
                Tensor P(std::vector<size_t>{e.M, e.K}, 0.0f, false);
                Tensor Q(std::vector<size_t>{e.K, e.N}, 0.0f, false);
                for (size_t i = 0; i < P.size(); ++i) P.data()[i] = (float)((int)(i % 9) - 4);
                for (size_t i = 0; i < Q.size(); ++i) Q.data()[i] = (float)((int)(i % 7) - 3);

                cuda::set_enabled(false);
                const std::vector<float> want_i = P.matmul(Q).to_vector();
                cuda::set_enabled(true);
                cuda::set_matmul_kernel(cuda::MatmulKernel::TensorCoreFp16);
                const std::vector<float> got_i = P.matmul(Q).to_vector();
                for (size_t i = 0; i < want_i.size() && exact; ++i) exact = want_i[i] == got_i[i];
                if (!exact) break;
            }
            testing::check(exact, exact ? "fp16 matches fp32 exactly on integer inputs, so the "
                                          "fragment indices and the store are right"
                                        : "fp16 disagrees with fp32 on integer inputs: the "
                                          "tolerance cases above were hiding an indexing error");
        }

        cuda::set_matmul_kernel(cuda::MatmulKernel::Auto);
    }

    // --- element-wise operations ---
    {
        Tensor A = Tensor::randn({64, 40});
        Tensor B = Tensor::randn({64, 40});
        compare("addition", [&] { return A + B; });
        compare("subtraction", [&] { return A - B; });
        compare("multiplication", [&] { return A * B; });

        // The divisor is kept away from zero: dividing by a near-zero value makes the
        // result depend on the denominator's last bit, which would measure the data
        // rather than the kernel.
        Tensor D = Tensor::rand({64, 40}, 1.0f, 2.0f);
        compare("division", [&] { return A / D; });

        // Broadcasting: a dense layer's bias and a positional encoding.
        Tensor bias = Tensor::randn({40});
        compare("broadcast addition (64,40) + (40)", [&] { return A + bias; });

        Tensor seq = Tensor::randn({6, 9, 16});
        Tensor pe = Tensor::randn({9, 16});
        compare("broadcast addition (6,9,16) + (9,16)", [&] { return seq + pe; });
        compare("broadcast multiplication (6,9,16) * (9,16)", [&] { return seq * pe; });
    }

    // --- activations ---
    {
        Tensor X = Tensor::randn({48, 33});
        compare("relu", [&] { return X.relu(); });

        // cols not a multiple of the reduction's block size, so that the block-stride
        // loop has to go round more than once.
        Tensor S = Tensor::randn({20, 300});
        compare("softmax over the last axis (20,300)", [&] { return S.softmax(); });

        Tensor A4 = Tensor::randn({3, 4, 7, 19});
        compare("softmax over (3,4,7,19)", [&] { return A4.softmax(); });
    }

    // --- scalars, axis reordering and reductions ---
    //
    // All three broke the residency chain. Between two matmuls that do have kernels,
    // a `* scale` or a permute without one pulled the whole tensor down to host and
    // forced the next operation to upload it again -- which is exactly what attention
    // does, twice per projection.
    {
        Tensor A = Tensor::randn({64, 40});
        compare("scalar multiplication", [&] { return A * 2.5f; });
        compare("scalar addition", [&] { return A + 1.5f; });
        compare("scalar division", [&] { return A / 4.0f; });

        compare("transpose (64,40)", [&] { return A.transpose(); });

        // Batched, it transposes each matrix in the batch, which is what is done to key
        // before attention's product.
        Tensor B4 = Tensor::randn({3, 5, 7, 11});
        compare("batched transpose (3,5,7,11)", [&] { return B4.transpose(); });

        // Attention's permutation: (B, S, H, d) -> (B, H, S, d).
        compare("permute (0,2,1,3) over (3,5,7,11)", [&] { return B4.permute({0, 2, 1, 3}); });
        // And one that moves the first axis to the end, so that no stride stays where it
        // was: a kernel that confused the axis order would still get attention's
        // permutation right, since that one leaves two in place.
        compare("permute (3,1,2,0) over (3,5,7,11)", [&] { return B4.permute({3, 1, 2, 0}); });
        // There and back. The second permute reads what the first left on the device, so
        // it also checks the chaining.
        compare("permute and its inverse over (3,5,7,11)",
                [&] { return B4.permute({0, 2, 1, 3}).permute({0, 2, 1, 3}); });

        Tensor R = Tensor::randn({6, 9, 16});
        compare("sum along axis 0 of (6,9,16)", [&] { return R.sum(0); });
        compare("sum along axis 1 of (6,9,16)", [&] { return R.sum(1); });
        // The last axis leaves inner == 1: the thread walks contiguous values rather than
        // jumping, the opposite access pattern to the other two.
        compare("sum along axis 2 of (6,9,16)", [&] { return R.sum(2); });
        compare("sum with keepdim along axis 1", [&] { return R.sum(1, true); });
        compare("mean along axis 1 of (6,9,16)", [&] { return R.mean(1); });
    }

    // --- gradients ---
    {
        Tensor X = Tensor::randn({40, 24}, 0.0f, 1.0f, true);
        compare("gradient of relu", [&] {
            X.zero_grad();
            Tensor y = X.relu();
            (y * y).sum().backward();
            return X.grad();
        });

        Tensor S = Tensor::randn({16, 50}, 0.0f, 1.0f, true);
        compare("gradient of softmax", [&] {
            S.zero_grad();
            Tensor y = S.softmax();
            // A non-uniform upstream gradient: with every weight equal, an indexing error
            // would cancel itself out.
            Tensor w = Tensor::zeros({16, 50});
            for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.1f * static_cast<float>(i % 7);
            (y * w).sum().backward();
            return S.grad();
        });

        // --- LayerNorm, at shapes that actually reach the kernels ---
        //
        // These are the only cases in this file that have to be *large*. Every
        // other dispatch takes small shapes because tile edges are where kernels
        // fail; LayerNorm carries a measured floor of 2^15 elements, below which
        // its four-call backward loses to one pass on the host, so a 17x23 case
        // would test the CPU path twice and pass at exactly zero error.
        //
        // The first version of this suite did precisely that: the kernels were
        // written, the tests were green, and the launch counter never moved off
        // 852. compare() refuses a case where nothing launched, which is what
        // turned that from a silent pass into a failure.
        {
            // 512 x 64 = 32 768, exactly the floor: the smallest shape that dispatches.
            nn::LayerNorm ln(64);
            Tensor X = Tensor::randn({512, 64}, 0.0f, 1.0f, true);
            compare("LayerNorm forward (512x64)", [&] { return ln.forward(X); });

            // A non-uniform upstream gradient, for the reason the softmax case
            // gives: with every weight equal, an indexing error cancels itself.
            compare("LayerNorm gradient of the input", [&] {
                X.zero_grad();
                Tensor w = Tensor::zeros(X.shape());
                for (size_t i = 0; i < w.size(); ++i) {
                    w.data()[i] = 0.1f * static_cast<float>(i % 7) - 0.2f;
                }
                (ln.forward(X) * w).sum().backward();
                return X.grad();
            });

            // dgamma and dbeta are the ones that accumulate across rows, which
            // is the whole reason this kernel needed a two-stage reduction
            // rather than an atomicAdd. They get their own cases because an
            // error there would not show in the input's gradient at all.
            compare("LayerNorm gradient of gamma", [&] {
                ln.parameters()[0].zero_grad();
                Tensor w = Tensor::zeros(X.shape());
                for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.3f - 0.05f * (i % 11);
                (ln.forward(X) * w).sum().backward();
                return ln.parameters()[0].grad();
            });
            compare("LayerNorm gradient of beta", [&] {
                ln.parameters()[1].zero_grad();
                Tensor w = Tensor::zeros(X.shape());
                for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.05f * (i % 13) - 0.3f;
                (ln.forward(X) * w).sum().backward();
                return ln.parameters()[1].grad();
            });

            // Same input twice on the device: the partials are summed in index
            // order, so two runs must agree bit for bit. An atomicAdd here would
            // pass every tolerance case above and fail this one.
            {
                cuda::set_enabled(true);
                ln.parameters()[0].zero_grad();
                Tensor w = Tensor::zeros(X.shape());
                for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.2f * (i % 5) - 0.1f;
                (ln.forward(X) * w).sum().backward();
                const std::vector<float> first = ln.parameters()[0].grad().to_vector();

                ln.parameters()[0].zero_grad();
                (ln.forward(X) * w).sum().backward();
                const std::vector<float> second = ln.parameters()[0].grad().to_vector();

                bool identical = first.size() == second.size();
                for (size_t i = 0; i < first.size() && identical; ++i) {
                    identical = first[i] == second[i];
                }
                testing::check(identical,
                               identical
                                   ? "LayerNorm's dgamma is bit-identical across runs, so the "
                                     "partials combine in a fixed order"
                                   : "LayerNorm's dgamma changed between two identical runs: the "
                                     "cross-row reduction depends on scheduling");
            }

            // And below the floor, on a tensor that is already on the card.
            //
            // The floor exists to decide whether the work is worth a PCIe
            // crossing. Once x is resident that question has no content, and
            // refusing costs the very round trip the floor was meant to avoid:
            // the host path pulls x down and the next operation puts it back.
            // Every other dispatch in this file already reasons that way -- this
            // is the case that holds LayerNorm to it.
            //
            // The floor is settable, and setting it reaches the dispatch.
            //
            // This is the number that decides whether a small forward chains
            // across the card or comes home at every normalisation, so a build
            // where the setter compiles but does not bind would be the quiet
            // kind of broken: everything still correct, half the speed, and no
            // test red. Both directions, because a threshold that can only be
            // lowered is only half a knob.
            {
                const size_t saved = cuda::min_layernorm_elements();
                Tensor small = Tensor::randn({16, 16}, 0.0f, 1.0f, false);  // 256 elements
                nn::LayerNorm ln16(16);

                cuda::set_thresholds(cuda::min_matmul_flops(), cuda::min_elementwise_elements(),
                                     1u << 15);
                const size_t before_high = cuda::kernels_launched();
                Tensor high = ln16.forward(small);
                testing::check(cuda::kernels_launched() == before_high,
                               "256 elements stays on the host under the default floor");

                cuda::set_thresholds(cuda::min_matmul_flops(), cuda::min_elementwise_elements(),
                                     64);
                const size_t before_low = cuda::kernels_launched();
                Tensor low = ln16.forward(small);
                testing::check(cuda::kernels_launched() > before_low,
                               "and dispatches once the floor is lowered under it");

                // Same answer either way: the threshold decides where the work
                // runs, never what it computes. One check over all 256, because
                // 256 green lines for one property is not 256 properties.
                float worst = 0.0f;
                for (size_t i = 0; i < high.size(); ++i) {
                    worst = std::max(worst, std::abs(high.data()[i] - low.data()[i]));
                }
                testing::check(worst < 1e-5f, "the host and device paths agree (max difference " +
                                                  std::to_string(worst) + ")");
                cuda::set_thresholds(cuda::min_matmul_flops(), cuda::min_elementwise_elements(),
                                     saved);
            }

            // What the clause does *not* do, measured after it landed: it does
            // not speed up a two-block Transformer at batches of one to five.
            // Those still cross PCIe five times per forward, once for each of
            // the four LayerNorms plus the initial upload, because the input is
            // never resident by the time a LayerNorm sees it -- the embedding
            // output starts on the host and reshape and transpose put it back.
            //
            // The halving that was originally credited to this clause came from
            // lowering kMinLayerNormElements instead, which dispatches whether
            // or not x is resident and pays an upload to hold the chain
            // afterwards. Two different changes. This one is the consistent
            // rule; the floor is the lever, and it is still a constexpr.
            {
                const size_t before = cuda::kernels_launched();

                // 64x1024 by 1024x64: enough work to dispatch on its own, and an
                // output of 4 096 elements, an eighth of the LayerNorm floor.
                Tensor wide = Tensor::randn({64, 1024}, 0.0f, 0.05f, false);
                Tensor tall = Tensor::randn({1024, 64}, 0.0f, 0.05f, false);
                Tensor resident = wide.matmul(tall);

                const size_t after_matmul = cuda::kernels_launched();
                testing::check(after_matmul > before,
                               "the matmul that leaves the tensor on the device ran there");

                nn::LayerNorm small(64);
                Tensor normed = small.forward(resident);
                testing::check(normed.shape() == std::vector<size_t>({64, 64}),
                               "LayerNorm below the floor returns the right shape");
                testing::check(cuda::kernels_launched() > after_matmul,
                               "LayerNorm dispatches on a resident input of 4 096 elements, "
                               "against a floor of " +
                                   std::to_string(1u << 15));
            }
        }

        Tensor A = Tensor::randn({30, 40}, 0.0f, 1.0f, true);
        Tensor B = Tensor::randn({40, 25}, 0.0f, 1.0f, true);
        compare("gradient of matmul with respect to A", [&] {
            A.zero_grad();
            B.zero_grad();
            A.matmul(B).sum().backward();
            return A.grad();
        });
    }

    // --- a whole network ---
    //
    // The per-operation tests can all pass and the model still give something else:
    // it takes one operation leaving a tensor on the wrong side and another reading
    // it without synchronising. This case chains attention, normalisation, two dense
    // layers and their backward pass, which is where it would show.
    {
        engine::manual_seed(7);
        nn::TransformerBlock block(32, 4, 64);
        Tensor tokens = Tensor::randn({4, 12, 32}, 0.0f, 1.0f, true);

        compare("TransformerBlock: output", [&] { return block(tokens); }, 1e-4f);

        compare(
            "TransformerBlock: gradient of the input",
            [&] {
                block.zero_grad();
                tokens.zero_grad();
                block(tokens).sum().backward();
                return tokens.grad();
            },
            1e-4f);

        compare(
            "TransformerBlock: gradient of the first parameter",
            [&] {
                block.zero_grad();
                tokens.zero_grad();
                block(tokens).sum().backward();
                return block.parameters()[0].grad();
            },
            1e-4f);
    }

    // --- transfer accounting ---
    //
    // It does not check a number, it checks the residency model: a tensor computed on
    // the GPU does not come down to host until somebody reads its values.
    {
        cuda::set_enabled(true);
        Tensor A = Tensor::randn({64, 64});
        Tensor B = Tensor::randn({64, 64});
        A.data();  // force both onto the host before counting starts
        B.data();

        cuda::reset_transfer_stats();
        Tensor C = A.matmul(B);
        const cuda::TransferStats after_kernel = cuda::transfer_stats();
        testing::check(after_kernel.to_device_count == 2,
                       "matmul uploads both operands and nothing else");
        testing::check(after_kernel.to_host_count == 0,
                       "the result stays on the device until it is read");

        volatile float first = C.data()[0];
        (void)first;
        const cuda::TransferStats after_read = cuda::transfer_stats();
        testing::check(after_read.to_host_count == 1,
                       "reading the result triggers exactly one download");

        // And a chain of operations does not re-upload what is already up there.
        cuda::reset_transfer_stats();
        Tensor D = A.matmul(B).relu();
        (void)D.data()[0];
        const cuda::TransferStats chained = cuda::transfer_stats();
        testing::check(chained.to_device_count == 0,
                       "a second operation does not re-upload already resident operands");
    }

    // --- a whole convolution ---
    //
    // Conv2d is now im2col, a product, the broadcast bias addition, a permutation and
    // two reshapes, and all six have kernels: it is the longest chain checked here.
    // im2col's and col2im's indices are also the most tangled in the backend --
    // padding, stride and overlapping windows -- and an error there does not raise a
    // fault, it returns a different number.
    {
        engine::manual_seed(99);
        nn::Conv2d conv(3, 5, nn::Window2d(3, 3, 1, 1));
        Tensor images = Tensor::randn({2, 3, 9, 9}, 0.0f, 1.0f, true);

        compare("Conv2d 3->5 (k3, s1, p1): output", [&] { return conv(images); }, 1e-4f);

        compare(
            "Conv2d: gradient of the input",
            [&] {
                conv.zero_grad();
                images.zero_grad();
                conv(images).sum().backward();
                return images.grad();
            },
            1e-4f);

        compare(
            "Conv2d: gradient of the kernel",
            [&] {
                conv.zero_grad();
                images.zero_grad();
                conv(images).sum().backward();
                return conv.weight().grad();
            },
            1e-4f);

        // With stride 2 and no padding the windows stop overlapping and pixels appear
        // that none of them covers. That is exactly where a col2im with the window range
        // solved wrongly fails: the stride-1 case would forgive it.
        nn::Conv2d strided(2, 3, nn::Window2d(3, 3, 2, 0));
        Tensor small = Tensor::randn({2, 2, 8, 8}, 0.0f, 1.0f, true);

        compare(
            "Conv2d with stride 2 and no padding: output", [&] { return strided(small); }, 1e-4f);
        compare(
            "Conv2d with stride 2: gradient of the input",
            [&] {
                strided.zero_grad();
                small.zero_grad();
                strided(small).sum().backward();
                return small.grad();
            },
            1e-4f);

        // MaxPool2d with **overlapping** windows (kernel 3, stride 2): two windows can
        // choose the same pixel, and that is the case that forces accumulation in the
        // gradient. It is also where a backward written with atomics would stop giving
        // the same result twice in a row.
        nn::MaxPool2d pool(nn::Window2d(3, 3, 2, 0));
        Tensor plane = Tensor::randn({2, 3, 9, 9}, 0.0f, 1.0f, true);

        compare("MaxPool2d 3x3 step 2 (solapado): output", [&] { return pool(plane); });
        compare("MaxPool2d overlapping: gradient of the input", [&] {
            plane.zero_grad();
            Tensor y = pool(plane);
            // A non-uniform upstream gradient: with every weight equal, a wrong distribution
            // would cancel itself out and the test would pass.
            Tensor w = Tensor::zeros(y.shape());
            for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.1f * static_cast<float>(i % 7);
            (y * w).sum().backward();
            return plane.grad();
        });
    }

    // --- the optimisers ---
    //
    // The largest transfer saving in the engine, and for that reason the easiest
    // to get subtly wrong. The parameter is updated **in place**, and the
    // optimiser's own state -- velocity for SGD, the two moments for Adam --
    // has to follow it onto the device. A step that reads a stale velocity
    // still produces a perfectly plausible number.
    //
    // Several steps, not one: momentum and Adam's bias correction both only
    // diverge from a single step after the state has had a chance to be wrong.
    {
        const size_t n = 96;

        auto seeded_weight = [n] {
            Tensor w(std::vector<size_t>{n, n}, 0.0f, true);
            for (size_t i = 0; i < w.size(); ++i) {
                w.data()[i] = 0.01f * static_cast<float>((int)(i % 17) - 8);
            }
            return w;
        };

        auto run_sgd = [&](bool on) {
            cuda::set_enabled(on);
            Tensor w = seeded_weight();
            Tensor x = seeded_weight();
            optim::SGD opt({w}, 0.05f, 0.9f);
            for (int step = 0; step < 5; ++step) {
                opt.zero_grad();
                w.matmul(x).relu().sum().backward();
                opt.step();
            }
            // to_vector, not data(): `auto` on the lambda's return type used to
            // deduce a vector because data() handed back a reference to one, and
            // the copy happened by accident. A pointer would deduce a pointer,
            // into a buffer this lambda destroys on the next line.
            return w.to_vector();
        };

        auto run_adam = [&](bool on) {
            cuda::set_enabled(on);
            Tensor w = seeded_weight();
            Tensor x = seeded_weight();
            optim::Adam opt({w}, 0.01f);
            for (int step = 0; step < 5; ++step) {
                opt.zero_grad();
                w.matmul(x).relu().sum().backward();
                opt.step();
            }
            // to_vector, not data(): `auto` on the lambda's return type used to
            // deduce a vector because data() handed back a reference to one, and
            // the copy happened by accident. A pointer would deduce a pointer,
            // into a buffer this lambda destroys on the next line.
            return w.to_vector();
        };

        for (int which = 0; which < 2; ++which) {
            const std::string name = which == 0 ? "SGD with momentum" : "Adam";
            const std::vector<float> want = which == 0 ? run_sgd(false) : run_adam(false);

            const size_t before = cuda::kernels_launched();
            const std::vector<float> got = which == 0 ? run_sgd(true) : run_adam(true);
            const size_t launched = cuda::kernels_launched() - before;

            ++testing::g_checks;
            if (launched == 0) {
                ++testing::g_failures;
                std::cout << "  [FAIL] " << name
                          << ": five steps launched no kernel, so this compared CPU to CPU\n";
                continue;
            }

            double worst = 0.0;
            for (size_t i = 0; i < want.size(); ++i) {
                const double scale = std::max(1.0, std::fabs((double)want[i]));
                worst = std::max(worst, std::fabs((double)got[i] - want[i]) / scale);
            }
            // Five steps of accumulated FMA differences, not one, so the bound is
            // looser than the single-operation cases above and deliberately so.
            if (worst <= 1e-4) {
                std::cout << "  [ ok ] " << name << ": 5 steps agree to " << std::scientific
                          << std::setprecision(2) << worst << std::defaultfloat << "\n";
            } else {
                ++testing::g_failures;
                std::cout << "  [FAIL] " << name << ": 5 steps drift to " << std::scientific
                          << std::setprecision(3) << worst << std::defaultfloat << "\n";
            }
        }
        cuda::set_enabled(true);
    }

    // --- reshape stays on the device ---
    //
    // reshape copies the whole buffer by definition -- it is not a view -- but
    // copying it **through host** cost one download and one upload per call.
    // Attention does two per projection, and the input to both comes from a matmul,
    // so it was resident. Now the copy never leaves the card.
    {
        cuda::set_enabled(true);
        Tensor A = Tensor::randn({64, 64});
        Tensor B = Tensor::randn({64, 64});
        A.data();
        B.data();

        cuda::reset_transfer_stats();
        Tensor flat = A.matmul(B).reshape({16, 256});
        const cuda::TransferStats after = cuda::transfer_stats();
        testing::check(after.to_host_count == 0,
                       "reshape of a resident tensor does not pull it down to host");
        testing::check(flat.storage().resident_on_device(),
                       "and the reshape's result is still on the device");

        // The values have to match, of course: a botched D2D copy would give a resident
        // tensor full of garbage, which is worse than not accelerating anything.
        //
        const std::vector<float> reshaped = flat.to_vector();
        const std::vector<float> original = A.matmul(B).to_vector();
        bool same = reshaped.size() == original.size();
        for (size_t i = 0; same && i < reshaped.size(); ++i) same = (reshaped[i] == original[i]);
        testing::check(same, "and the values survive the device-to-device copy");
    }

    // --- backward residency ---
    //
    // The other half of the residency model, and the half that was missing: the
    // accumulated gradient does not come down to host either. It always used to,
    // because add_grad built the gradient from g.data(); that forced the next
    // backward_fn to upload it again and no node stayed on the GPU.
    //
    // The invariant is checked directly rather than a counter, because the invariant
    // is what is actually meant to hold.
    {
        cuda::set_enabled(true);
        Tensor A = Tensor::randn({64, 64}, 0.0f, 1.0f, true);
        Tensor B = Tensor::randn({64, 64}, 0.0f, 1.0f, true);
        A.data();
        B.data();

        cuda::reset_transfer_stats();
        A.matmul(B).relu().sum().backward();

        testing::check(A.grad().storage().resident_on_device(),
                       "the accumulated gradient stays on the device");
        testing::check(B.grad().storage().resident_on_device(),
                       "the second operand's gradient too");
    }

    // --- exactness of the accumulation branch ---
    //
    // A tensor consumed by two branches makes add_grad take the += path, which the
    // normal case never exercises: autograd frees intermediate nodes' gradients as
    // soon as they are consumed, so almost everything goes through the first write.
    //
    //
    // The comparison is for **exact** equality, not compare()'s tolerance: a bare
    // addition has no multiply to fuse into an FMA, so here the GPU and the CPU agree
    // bit for bit. If they ever stop, the kernel has stopped being an addition and
    // that is worth knowing.
    {
        engine::manual_seed(4242);
        const Tensor X = Tensor::randn({128, 128});
        const Tensor Wa = Tensor::randn({128, 128});
        const Tensor Wb = Tensor::randn({128, 128});

        std::vector<float> want;
        std::vector<float> got;
        for (const bool on : {false, true}) {
            cuda::set_enabled(on);
            // A new tensor each time round, not `Tensor x = X`: Tensor is a handle over a
            // shared TensorImpl, so copying it would share the gradient and the second pass
            // would accumulate on top of the first.
            Tensor x(X.shape(), X.to_vector(), true);
            ((x.relu() * Wa) + (x.relu() * Wb)).sum().backward();
            (on ? got : want) = x.grad().to_vector();
        }
        cuda::set_enabled(true);

        bool identical = want.size() == got.size();
        for (size_t i = 0; identical && i < want.size(); ++i) {
            identical = (want[i] == got[i]);
        }
        testing::check(identical, "accumulating two gradients gives the same bit on CPU and GPU");
    }

    // --- the cost of a full step, in transfers ---
    //
    // Not a check: the figure that justifies the change -- how many times a whole
    // training step brings something down to host. It is reported rather than
    // asserted because it depends on how many operations have kernels, and that list
    // grows; what it must not do is rise.
    {
        cuda::set_enabled(true);
        engine::manual_seed(7);
        nn::TransformerBlock block(32, 4, 64);
        Tensor tokens = Tensor::randn({4, 12, 32}, 0.0f, 1.0f, true);
        block(tokens).sum().backward();  // calienta: reserva parametros y espejos

        block.zero_grad();
        tokens.zero_grad();
        cuda::reset_transfer_stats();
        block(tokens).sum().backward();
        const cuda::TransferStats step = cuda::transfer_stats();
        std::cout << "  One TransformerBlock step: " << step.to_host_count << " downloads and "
                  << step.to_device_count << " uploads\n";
    }

    // And the same count over a chain that only uses operations with kernels.
    //
    // Comparing the two figures maps what is left to do: in the TransformerBlock the
    // gradient residency shows less than in a pure matmul chain, because reshape and
    // transpose still bring the tensor down.
    //
    // LayerNorm was on that list until it started honouring residency, and the count
    // here went *up* when it did -- 15 downloads and 6 uploads became 18 and 5. At
    // this deliberately small case (4x12x32 = 1 536 elements) it now computes on the
    // device and the reshape after it has no kernel, so the result comes straight
    // back down where before it never went up.
    //
    // Which is the honest summary of the clause so far: it makes the rule the same
    // everywhere, and on today's graphs it mostly has nowhere to fire, because the
    // operations around a LayerNorm bring the tensor home anyway. It becomes worth
    // something when reshape and transpose stop doing that. Training is untouched
    // either way -- 400 steps of charlm_demo run 7.8 s against 8.0 s, with its
    // LayerNorms far above the floor in both builds.
    {
        cuda::set_enabled(true);
        engine::manual_seed(11);
        Tensor h0 = Tensor::randn({256, 256}, 0.0f, 0.05f, true);
        Tensor W = Tensor::randn({256, 256}, 0.0f, 0.05f, true);
        h0.data();
        W.data();

        cuda::reset_transfer_stats();
        Tensor h = h0;
        for (int i = 0; i < 4; ++i) h = h.matmul(W).relu();
        h.sum().backward();
        const cuda::TransferStats chain = cuda::transfer_stats();
        std::cout << "  Four matmul+relu layers:     " << chain.to_host_count << " downloads and "
                  << chain.to_device_count << " uploads\n";
    }

    // A summary at the end: how many kernels actually ran. If this line says zero,
    // everything above was computed on the CPU and means nothing.
    testing::check(cuda::kernels_failed() == 0,
                   "no kernel failed to launch (" + std::to_string(cuda::kernels_launched()) +
                       " launched, " + std::to_string(cuda::kernels_failed()) + " failed)");

    cuda::set_thresholds(saved_flops, saved_elements, saved_layernorm);
    cuda::set_enabled(true);
}

#else  // !ENGINE_CUDA

void run_cuda_parity_tests() {
    testing::section("CPU / GPU parity (Phase 6)");
    std::cout << "  (built without CUDA: -DENGINE_CUDA=ON to include these tests)\n";
}

#endif  // ENGINE_CUDA
