// A CPU simulation of the register-tiled kernel's index arithmetic.
//
//
// The problem it solves: CUDA kernels can only run where there is a card, and CI
// has none. The CUDA job compiles, which catches syntax errors and nothing else
// -- an indexing error compiles perfectly happily and shows up weeks later as
// wrong results.
//
// So this reproduces the kernel's structure with loops: block grid, 256 threads,
// shared memory, barriers, and **the same index expressions** as
// src/cuda/kernels.cu. It then compares against a reference matrix product. It
// runs on any machine, on all four platforms, on every push.
//
//
// The price is that the expressions are written twice and have to be maintained
// in parallel. That is accepted knowingly: the alternative was having no check on
// the indices at all until reaching a machine with a GPU, and by then the error
// is already committed.
//
// What this does NOT check: anything that depends on the hardware. Races between
// threads, real shared-memory coherence, float4 load alignment and performance
// itself only show on the device, and tests/test_cuda_parity.cpp covers those.
//

#include "test_support.hpp"

#include <cmath>
#include <vector>

namespace {

// These must match the constants in src/cuda/kernels.cu.
constexpr int kBM = 128, kBN = 128, kBK = 8, kTM = 8, kTN = 8;
constexpr int kRegBlock = (kBM / kTM) * (kBN / kTN);

// Reproduces a full kernel launch over a 2D matrix.
void simulate_kernel(const float* A, const float* B, float* C, int M, int K, int N, bool use_vec4) {
    const int grid_x = (N + kBN - 1) / kBN;
    const int grid_y = (M + kBM - 1) / kBM;

    for (int by = 0; by < grid_y; ++by) {
        for (int bx = 0; bx < grid_x; ++bx) {
            const int block_row = by * kBM;
            const int block_col = bx * kBN;

            std::vector<float> acc(static_cast<size_t>(kRegBlock) * kTM * kTN, 0.0f);
            float As[kBK][kBM];
            float Bs[kBK][kBN];

            for (int k_base = 0; k_base < K; k_base += kBK) {
                // Load phase. All 256 threads are walked before moving on to the
                // arithmetic, which is what the __syncthreads() does.
                for (int tid = 0; tid < kRegBlock; ++tid) {
                    if (use_vec4) {
                        const int a_r = tid / (kBK / 4);
                        const int a_c = (tid % (kBK / 4)) * 4;
                        const int g_row = block_row + a_r;
                        const int g_col = k_base + a_c;
                        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        if (g_row < M && g_col < K) {
                            for (int t = 0; t < 4; ++t) {
                                v[t] = A[static_cast<size_t>(g_row) * K + g_col + t];
                            }
                        }
                        for (int t = 0; t < 4; ++t) As[a_c + t][a_r] = v[t];

                        const int b_r = tid / (kBN / 4);
                        const int b_c = (tid % (kBN / 4)) * 4;
                        const int gb_row = k_base + b_r;
                        const int gb_col = block_col + b_c;
                        float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        if (gb_row < K && gb_col < N) {
                            for (int t = 0; t < 4; ++t) {
                                w[t] = B[static_cast<size_t>(gb_row) * N + gb_col + t];
                            }
                        }
                        for (int t = 0; t < 4; ++t) Bs[b_r][b_c + t] = w[t];
                    } else {
                        const int a_r = tid / kBK;
                        const int a_c = tid % kBK;
                        for (int off = 0; off < kBM; off += kRegBlock / kBK) {
                            const int g_row = block_row + a_r + off;
                            const int g_col = k_base + a_c;
                            As[a_c][a_r + off] = (g_row < M && g_col < K)
                                                     ? A[static_cast<size_t>(g_row) * K + g_col]
                                                     : 0.0f;
                        }
                        const int b_r = tid / kBN;
                        const int b_c = tid % kBN;
                        for (int off = 0; off < kBK; off += kRegBlock / kBN) {
                            const int g_row = k_base + b_r + off;
                            const int g_col = block_col + b_c;
                            Bs[b_r + off][b_c] = (g_row < K && g_col < N)
                                                     ? B[static_cast<size_t>(g_row) * N + g_col]
                                                     : 0.0f;
                        }
                    }
                }

                // Compute phase, after the barrier.
                for (int tid = 0; tid < kRegBlock; ++tid) {
                    const int t_row = tid / (kBN / kTN);
                    const int t_col = tid % (kBN / kTN);
                    float* a = &acc[static_cast<size_t>(tid) * kTM * kTN];
                    for (int dot = 0; dot < kBK; ++dot) {
                        float reg_m[kTM];
                        float reg_n[kTN];
                        for (int i = 0; i < kTM; ++i) reg_m[i] = As[dot][t_row * kTM + i];
                        for (int j = 0; j < kTN; ++j) reg_n[j] = Bs[dot][t_col * kTN + j];
                        for (int i = 0; i < kTM; ++i) {
                            for (int j = 0; j < kTN; ++j) a[i * kTN + j] += reg_m[i] * reg_n[j];
                        }
                    }
                }
            }

            for (int tid = 0; tid < kRegBlock; ++tid) {
                const int t_row = tid / (kBN / kTN);
                const int t_col = tid % (kBN / kTN);
                const float* a = &acc[static_cast<size_t>(tid) * kTM * kTN];
                for (int i = 0; i < kTM; ++i) {
                    const int row = block_row + t_row * kTM + i;
                    if (row >= M) continue;
                    for (int j = 0; j < kTN; ++j) {
                        const int col = block_col + t_col * kTN + j;
                        if (col < N) C[static_cast<size_t>(row) * N + col] = a[i * kTN + j];
                    }
                }
            }
        }
    }
}

void check_shape(int M, int K, int N, bool use_vec4) {
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    // Deterministic and varied values: if the kernel read the wrong position,
    // constant data would make it look fine.
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>((i * 37 % 19)) * 0.1f - 0.9f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>((i * 53 % 23)) * 0.1f - 1.1f;

    // The output starts with a sentinel: if the kernel left any position unwritten,
    // the error would be enormous rather than slipping past unnoticed.
    std::vector<float> got(static_cast<size_t>(M) * N, -12345.0f);
    std::vector<float> want(static_cast<size_t>(M) * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = A[static_cast<size_t>(i) * K + k];
            for (int j = 0; j < N; ++j) {
                want[static_cast<size_t>(i) * N + j] += a * B[static_cast<size_t>(k) * N + j];
            }
        }
    }

    simulate_kernel(A.data(), B.data(), got.data(), M, K, N, use_vec4);

    float worst = 0.0f;
    for (size_t i = 0; i < want.size(); ++i) {
        const float scale = std::fmax(1.0f, std::fabs(want[i]));
        worst = std::fmax(worst, std::fabs(got[i] - want[i]) / scale);
    }

    const std::string what = std::string(use_vec4 ? "vectorized" : "register ") + " " +
                             std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
    testing::check(worst < 1e-5f, "indices de " + what);
}

}  // namespace

void run_cuda_indexing_tests() {
    testing::section("Register-tiled kernel indices (simulated on the CPU)");

    struct Case {
        int M, K, N;
    };
    // The interesting sizes are the ones that do not fit: the block is 128x128 and
    // the step over K is 8, so the failures live precisely in the remainders.
    const Case cases[] = {
        {1, 1, 1},                       // degenerado
        {17, 23, 31},                    // far smaller than one block
        {32, 32, 32},    {33, 65, 129},  // remainders on all three axes
        {127, 128, 129},                 // one below and one above the block
        {128, 128, 128},                 // exactamente un bloque
        {129, 256, 257},                 // more than one block, with a remainder
        {256, 260, 256},                 // K a multiple of 4 but not 8: last tile partial
        {131, 133, 135},                 // nada alineado
        {200, 8, 200},                   // a single tile of K
        {130, 4, 130},                   // K smaller than the step
    };

    for (const Case& c : cases) {
        check_shape(c.M, c.K, c.N, false);
        // The vectorised variant is only dispatched with K and N multiples of 4.
        if (c.K % 4 == 0 && c.N % 4 == 0) check_shape(c.M, c.K, c.N, true);
    }
}
