// Simulación en CPU de la aritmética de índices del kernel con teselado de
// registros.
//
// El problema que resuelve: los kernels de CUDA sólo se pueden ejecutar donde
// hay una tarjeta, y CI no tiene ninguna. El job de CUDA compila, que detecta
// errores de sintaxis, y nada más — un error de indexación pasa la compilación
// sin inmutarse y aparece semanas después como resultados incorrectos.
//
// Así que esto reproduce la estructura del kernel con bucles: rejilla de
// bloques, 256 hilos, memoria compartida, barreras, y **las mismas expresiones
// de índice** que src/cuda/kernels.cu. Después compara contra un producto de
// matrices de referencia. Corre en cualquier máquina, en las cuatro
// plataformas, en cada push.
//
// El precio es que las expresiones están escritas dos veces y hay que
// mantenerlas en paralelo. Se asume a conciencia: la alternativa era no tener
// ninguna comprobación de los índices hasta llegar a una máquina con GPU, y
// para eso el error ya está commiteado.
//
// Lo que esto NO comprueba: nada que dependa del hardware. Las carreras entre
// hilos, la coherencia de la memoria compartida real, la alineación de las
// lecturas float4 o el propio rendimiento sólo se ven en el dispositivo, y de
// eso se encarga tests/test_cuda_parity.cpp.

#include "test_support.hpp"

#include <cmath>
#include <vector>

namespace {

// Deben coincidir con las constantes de src/cuda/kernels.cu.
constexpr int kBM = 128, kBN = 128, kBK = 8, kTM = 8, kTN = 8;
constexpr int kRegBlock = (kBM / kTM) * (kBN / kTN);

// Reproduce un lanzamiento completo del kernel sobre una matriz 2D.
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
                // Fase de carga. Se recorren los 256 hilos enteros antes de
                // pasar al cálculo, que es lo que hace el __syncthreads().
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

                // Fase de cálculo, tras la barrera.
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
    // Valores deterministas y variados: si el kernel leyera la posición
    // equivocada, con datos constantes daría igual.
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>((i * 37 % 19)) * 0.1f - 0.9f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>((i * 53 % 23)) * 0.1f - 1.1f;

    // La salida arranca con un centinela: si el kernel dejara alguna posición
    // sin escribir, el error sería enorme en lugar de pasar desapercibido.
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
    testing::section("Indices del kernel con teselado de registros (simulados en CPU)");

    struct Case {
        int M, K, N;
    };
    // Los tamaños interesantes son los que no encajan: el bloque es de 128x128
    // y el paso sobre K es de 8, así que los fallos viven justo en los restos.
    const Case cases[] = {
        {1, 1, 1},                       // degenerado
        {17, 23, 31},                    // mucho menor que un bloque
        {32, 32, 32},    {33, 65, 129},  // resto en los tres ejes
        {127, 128, 129},                 // uno por debajo y uno por encima del bloque
        {128, 128, 128},                 // exactamente un bloque
        {129, 256, 257},                 // más de un bloque, con resto
        {256, 260, 256},                 // K múltiplo de 4 pero no de 8: última tesela parcial
        {131, 133, 135},                 // nada alineado
        {200, 8, 200},                   // una sola tesela de K
        {130, 4, 130},                   // K menor que el paso
    };

    for (const Case& c : cases) {
        check_shape(c.M, c.K, c.N, false);
        // La variante vectorizada sólo se despacha con K y N múltiplos de 4.
        if (c.K % 4 == 0 && c.N % 4 == 0) check_shape(c.M, c.K, c.N, true);
    }
}
