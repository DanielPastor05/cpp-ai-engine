// Kernels CUDA del motor.
//
// Cubren lo que el perfilado señaló como caliente: el producto de matrices
// (53% del tiempo del ejemplo del Transformer), las operaciones elemento a
// elemento, ReLU y softmax. El resto sigue en CPU, y eso está bien: portar una
// operación que no domina el perfil sólo añade transferencias.
//
// Cada punto de entrada devuelve false cuando decide no hacerse cargo, y el
// llamante sigue por CPU. Las condiciones de rechazo son de dos tipos:
//   - no compensa (tamaño por debajo del umbral medido), o
//   - no cabe en la geometría de lanzamiento (lotes o dimensiones enormes).
// Las segundas no deberían ocurrir con formas realistas, pero devolver false
// es preferible a lanzar una malla inválida y obtener basura.

#include "engine/cuda.hpp"
#include "engine/detail/cuda_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>

namespace engine {
namespace cuda {
namespace ops {

namespace {

constexpr int kTile = 32;       // lado del bloque compartido del matmul
constexpr int kBlock = 256;     // hilos por bloque en los kernels 1D
constexpr int kReduceBlock = 256; // potencia de dos: lo exige la reducción

// Geometría del matmul con teselado de registros. Los cinco números están
// atados entre sí y no se pueden tocar por separado:
//   (kBM / kTM) * (kBN / kTN) == kRegBlock   -> cada hilo, su cuadrado de salida
//   kBM * kBK == kBN * kBK == 1024           -> una tesela cabe en 4 pasadas de
//                                               carga escalar, o 1 vectorizada
constexpr int kBM = 128;        // filas de la salida por bloque
constexpr int kBN = 128;        // columnas de la salida por bloque
constexpr int kBK = 8;          // paso sobre K
constexpr int kTM = 8;          // filas de la salida por hilo
constexpr int kTN = 8;          // columnas de la salida por hilo
constexpr int kRegBlock = (kBM / kTM) * (kBN / kTN);  // 256 hilos

static_assert(kRegBlock == 256, "los índices de carga suponen 256 hilos por bloque");
static_assert(kBM * kBK == 1024 && kBN * kBK == 1024,
              "cada tesela debe ser de 1024 valores para que el reparto cuadre");

// Límite de gridDim.y y gridDim.z en todas las arquitecturas soportadas.
constexpr size_t kMaxGridYZ = 65535;

constexpr size_t kMaxInt = static_cast<size_t>(std::numeric_limits<int>::max());

// Un kernel que falla al lanzarse no aborta el programa: se informa y se
// devuelve false, de modo que el resultado lo calcula la CPU. Un motor que se
// cae porque la GPU está ocupada es peor que uno que va más lento.
//
// Hay que deshacer además el device_write() de la salida. Si no, el camino de
// CPU pediría el búfer de host, Storage se lo bajaría del dispositivo sin
// inicializar, y matmul —que acumula sobre una salida que supone a cero—
// devolvería basura en lugar de un resultado correcto más lento.
bool launched_ok(const char* what, Storage& out) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    out.revert_device_write();
    std::fprintf(stderr, "engine: el kernel %s no se pudo lanzar (%s); se calcula en CPU\n",
                 what, cudaGetErrorString(status));
    return false;
}

// ---------------------------------------------------------
// Operaciones elemento a elemento
// ---------------------------------------------------------

template <int Op>
__device__ inline float apply(float x, float y) {
    if (Op == 0) return x + y;
    if (Op == 1) return x - y;
    if (Op == 2) return x * y;
    return x / y;
}

template <int Op>
__global__ void binary_contiguous(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out,
                                  long long n) {
    // Bucle con paso de malla: así el número de bloques no depende de n y
    // nunca se desborda gridDim.x.
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = apply<Op>(a[i], b[i]);
    }
}

// Difusión por sufijo. Igual que en CPU, no se calcula ningún módulo: el eje y
// de la malla recorre las repeticiones y el eje x el bloque que se repite.
// Un módulo de 64 bits por elemento costaría aquí bastante más que en CPU.
template <int Op>
__global__ void binary_broadcast(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ out,
                                 long long inner) {
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= inner) return;
    const long long i = (long long)blockIdx.y * inner + j;
    out[i] = apply<Op>(a[i], b[j]);
}

__global__ void relu_forward(const float* __restrict__ x, float* __restrict__ out, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = fmaxf(0.0f, x[i]);
    }
}

__global__ void relu_grad(const float* __restrict__ x, const float* __restrict__ g,
                          float* __restrict__ out, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = (x[i] > 0.0f) ? g[i] : 0.0f;
    }
}

// ---------------------------------------------------------
// Producto de matrices con teselas en memoria compartida
// ---------------------------------------------------------
//
// Cada bloque calcula una tesela de 32x32 de la salida. La tesela recorre K:
// en cada paso los 1024 hilos cargan una tesela de A y otra de B a memoria
// compartida y luego cada hilo hace sus 32 productos leyendo de ahí.
//
// El motivo es el tráfico a memoria global: sin teselas, cada elemento de A se
// lee N veces y cada uno de B, M veces. Con teselas de lado T cada uno se lee
// N/T y M/T veces, o sea 32 veces menos.
//
// El orden de acumulación es fijo (k ascendente, igual que en CPU), así que el
// resultado es reproducible de una ejecución a otra. No es idéntico bit a bit
// al de la CPU, y eso es esperable: el compilador de dispositivo funde
// multiplicación y suma en una sola instrucción FMA, que redondea una vez en
// lugar de dos. Por eso la prueba de paridad compara con tolerancia y no con
// igualdad exacta.
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N,
                             long long a_stride, long long b_stride) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;

    float acc = 0.0f;
    const int tiles = (K + kTile - 1) / kTile;

    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * kTile + threadIdx.x;
        const int b_row = t * kTile + threadIdx.y;

        // Los bordes se rellenan con ceros en lugar de acortar el bucle: así
        // todos los hilos del bloque llegan a los mismos __syncthreads(), que
        // es obligatorio para que la barrera sea válida.
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[(long long)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[(long long)b_row * N + col] : 0.0f;

        __syncthreads();

        // As[ty][k] es una difusión dentro del warp y Bs[k][tx] recorre bancos
        // consecutivos: ninguno de los dos accesos genera conflictos, así que
        // la memoria compartida no necesita relleno.
        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(long long)row * N + col] = acc;
    }
}

// ---------------------------------------------------------
// Producto de matrices sin memoria compartida
// ---------------------------------------------------------
//
// Existe para tener una referencia inferior honesta en la tabla del banco de
// pruebas. Cada hilo lee sus dos operandos directamente de memoria global, así
// que cada elemento de A se lee N veces y cada uno de B, M veces.
__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N,
                             long long a_stride, long long b_stride) {
    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[(long long)row * K + k] * B[(long long)k * N + col];
    }
    C[(long long)row * N + col] = acc;
}

// ---------------------------------------------------------
// Producto de matrices con teselado de registros
// ---------------------------------------------------------
//
// Éste es el salto de verdad, y conviene entender **por qué** el kernel de
// teselas de arriba se queda corto. No es la ocupación, y no es el tráfico a
// memoria global: eso ya lo arreglaron las teselas. Es la intensidad
// aritmética *a nivel de registro*.
//
// En matmul_tiled, cada hilo por cada paso de K hace:
//     1 FMA   contra   2 lecturas de memoria compartida
// Esa proporción de 1:2 es la que manda, porque las unidades de carga se
// saturan mucho antes que las de cálculo. Da igual cuántos warps haya en vuelo.
//
// La solución es que cada hilo calcule un **bloque** de resultados en lugar de
// uno solo. Con TM x TN = 8 x 8 salidas vivas en registros, por cada paso de K
// un hilo lee 8 valores de A y 8 de B, y con ellos hace 64 productos:
//     64 FMA   contra   16 lecturas de memoria compartida
// La proporción pasa de 1:2 a 4:1 — ocho veces mejor. Los 64 acumuladores
// viven en registros y no se tocan hasta el final.
//
// Geometría: bloques de 128x128 de la salida, 256 hilos, paso BK=8 sobre K.
// La memoria compartida son 2 x 8 x 128 x 4 = 8 KB por bloque.
//
// `As` se guarda **transpuesta** (índice [k][m] en vez de [m][k]) para que las
// ocho lecturas de A de cada hilo sean contiguas en el eje m; sin eso irían con
// paso K y cada una tocaría un banco distinto.
//
// Sobre la ocupación: con 64 acumuladores más los registros de trabajo, esto
// gasta del orden de 80-100 registros por hilo y baja la ocupación a la mitad
// respecto al kernel de teselas. **Es intencionado.** El paralelismo a nivel de
// instrucción dentro de cada hilo compensa de sobra tener menos warps: es el
// compromiso clásico de este kernel, y es justo lo que se ve al perfilarlo.
//
// UseVector4 elige cómo se cargan las teselas desde memoria global. La versión
// vectorizada mueve 16 bytes por instrucción en lugar de 4, lo que reduce el
// número de instrucciones de carga en el camino global -> compartida. Exige que
// K y N sean múltiplos de 4 para que las direcciones estén alineadas, cosa que
// comprueba el lado del host antes de elegir esta variante.
template <bool UseVector4>
__global__ void matmul_register_tiled(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int K, int N,
                                      long long a_stride, long long b_stride) {
    // Sin atributo de alineación a propósito: nada lee estas dos matrices en
    // bloques de 16 bytes. La carga vectorizada actúa sobre memoria **global**,
    // que es donde está la ganancia; lo que se escribe y se lee aquí es
    // escalar, así que no hay ningún requisito de alineación que cumplir. Es
    // además lo portable, porque __align__ se escribe distinto según el
    // compilador de host.
    __shared__ float As[kBK][kBM];  // transpuesta: [k][m]
    __shared__ float Bs[kBK][kBN];

    const long long batch = blockIdx.z;
    A += batch * a_stride;
    B += batch * b_stride;
    C += batch * (long long)M * N;

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;

    // Cada hilo se queda con un cuadrado de kTM x kTN de la tesela de salida.
    const int thread_row = threadIdx.x / (kBN / kTN);  // [0, 16)
    const int thread_col = threadIdx.x % (kBN / kTN);  // [0, 16)

    float acc[kTM][kTN];
    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        #pragma unroll
        for (int j = 0; j < kTN; ++j) acc[i][j] = 0.0f;
    }

    // Índices de carga, distintos de los de cálculo: para traer las teselas
    // interesa que los hilos consecutivos lean posiciones consecutivas, no que
    // cada uno lea lo que luego va a usar.
    const int a_load_row_v = threadIdx.x / (kBK / 4);   // [0, 128) con float4
    const int a_load_col_v = (threadIdx.x % (kBK / 4)) * 4;
    const int b_load_row_v = threadIdx.x / (kBN / 4);   // [0, 8) con float4
    const int b_load_col_v = (threadIdx.x % (kBN / 4)) * 4;

    const int a_load_row_s = threadIdx.x / kBK;         // [0, 32) escalar
    const int a_load_col_s = threadIdx.x % kBK;
    const int b_load_row_s = threadIdx.x / kBN;         // [0, 2) escalar
    const int b_load_col_s = threadIdx.x % kBN;

    for (int k_base = 0; k_base < K; k_base += kBK) {
        if (UseVector4) {
            // Una sola instrucción de 16 bytes por hilo cubre la tesela de A:
            // 128 filas x 8 columnas = 1024 valores = 256 float4 = 256 hilos.
            {
                const int g_row = block_row + a_load_row_v;
                const int g_col = k_base + a_load_col_v;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                // K es múltiplo de 4 y g_col también, así que el float4 entra
                // entero o no entra: no hay resto parcial que tratar.
                if (g_row < M && g_col < K) {
                    v = *reinterpret_cast<const float4*>(A + (long long)g_row * K + g_col);
                }
                // El almacenamiento sí es escalar, porque va transpuesto.
                As[a_load_col_v + 0][a_load_row_v] = v.x;
                As[a_load_col_v + 1][a_load_row_v] = v.y;
                As[a_load_col_v + 2][a_load_row_v] = v.z;
                As[a_load_col_v + 3][a_load_row_v] = v.w;
            }
            {
                const int g_row = k_base + b_load_row_v;
                const int g_col = block_col + b_load_col_v;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (g_row < K && g_col < N) {
                    v = *reinterpret_cast<const float4*>(B + (long long)g_row * N + g_col);
                }
                Bs[b_load_row_v][b_load_col_v + 0] = v.x;
                Bs[b_load_row_v][b_load_col_v + 1] = v.y;
                Bs[b_load_row_v][b_load_col_v + 2] = v.z;
                Bs[b_load_row_v][b_load_col_v + 3] = v.w;
            }
        } else {
            // Sin vectorizar hacen falta cuatro pasadas por tesela: 256 hilos
            // contra 1024 valores.
            #pragma unroll
            for (int off = 0; off < kBM; off += 256 / kBK) {
                const int g_row = block_row + a_load_row_s + off;
                const int g_col = k_base + a_load_col_s;
                As[a_load_col_s][a_load_row_s + off] =
                    (g_row < M && g_col < K) ? A[(long long)g_row * K + g_col] : 0.0f;
            }
            #pragma unroll
            for (int off = 0; off < kBK; off += 256 / kBN) {
                const int g_row = k_base + b_load_row_s + off;
                const int g_col = block_col + b_load_col_s;
                Bs[b_load_row_s + off][b_load_col_s] =
                    (g_row < K && g_col < N) ? B[(long long)g_row * N + g_col] : 0.0f;
            }
        }

        __syncthreads();

        // El bucle caliente. Las lecturas de compartida se izan a registros
        // antes de los productos: sin eso el compilador volvería a leer de
        // compartida dentro del doble bucle y se perdería toda la ventaja.
        #pragma unroll
        for (int dot = 0; dot < kBK; ++dot) {
            float reg_m[kTM];
            float reg_n[kTN];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) reg_m[i] = As[dot][thread_row * kTM + i];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) reg_n[j] = Bs[dot][thread_col * kTN + j];

            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                #pragma unroll
                for (int j = 0; j < kTN; ++j) acc[i][j] += reg_m[i] * reg_n[j];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int row = block_row + thread_row * kTM + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int col = block_col + thread_col * kTN + j;
            if (col < N) C[(long long)row * N + col] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------
// Softmax sobre el último eje
// ---------------------------------------------------------
//
// Un bloque por fila. Dos reducciones sobre memoria compartida: primero el
// máximo (para restarlo y que la exponencial no se desborde, igual que en CPU)
// y después la suma.
//
// Se usa expf y no __expf: la versión rápida ahorra unos ciclos pero pierde
// precisión, y estos valores se comparan contra PyTorch en la prueba de
// referencia. La exponencial no es el cuello de botella aquí.
__global__ void softmax_rows(const float* __restrict__ x, float* __restrict__ y, int cols) {
    __shared__ float shared[kReduceBlock];

    const long long row = blockIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) local_max = fmaxf(local_max, xr[j]);
    shared[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    const float max_v = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        const float e = expf(xr[j] - max_v);
        yr[j] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float denom = shared[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) yr[j] /= denom;
}

// dX_ij = y_ij * (dY_ij - sum_k dY_ik * y_ik)
__global__ void softmax_rows_grad(const float* __restrict__ y, const float* __restrict__ g,
                                  float* __restrict__ out, int cols) {
    __shared__ float shared[kReduceBlock];

    const long long row = blockIdx.x;
    const float* yr = y + row * cols;
    const float* gr = g + row * cols;
    float* orow = out + row * cols;
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) local += gr[j] * yr[j];
    shared[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float dot = shared[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) orow[j] = yr[j] * (gr[j] - dot);
}

// ---------------------------------------------------------
// Ayudas de despacho
// ---------------------------------------------------------

// Número de bloques para un kernel con paso de malla: suficientes para llenar
// el dispositivo, sin pasarse.
int grid_for(long long n) {
    const long long want = (n + kBlock - 1) / kBlock;
    const long long cap = 65535 * 16;
    return (int)(want < cap ? want : cap);
}

bool elementwise_worth_it(size_t n) {
    return enabled() && n > 0 && n >= min_elementwise_elements();
}

template <int Op>
bool launch_binary(const Storage& a, const Storage& b, Storage& out,
                   size_t inner, size_t repeat) {
    const size_t n = a.size();
    if (repeat <= 1) {
        binary_contiguous<Op><<<grid_for((long long)n), kBlock>>>(
            a.device(), b.device(), out.device_write(), (long long)n);
        return launched_ok("binary_contiguous", out);
    }
    if (repeat > kMaxGridYZ) return false;
    const dim3 grid((unsigned)((inner + kBlock - 1) / kBlock), (unsigned)repeat);
    binary_broadcast<Op><<<grid, kBlock>>>(
        a.device(), b.device(), out.device_write(), (long long)inner);
    return launched_ok("binary_broadcast", out);
}

} // namespace

bool binary(Binary op, const Storage& a, const Storage& b, Storage& out,
            size_t inner, size_t repeat) {
    if (!elementwise_worth_it(a.size())) return false;
    if (inner == 0 || repeat == 0) return false;
    if (inner * repeat != a.size() || out.size() != a.size()) return false;
    if (b.size() < inner) return false;
    // Con difusión la malla se organiza por repeticiones, y eso limita cuántas
    // caben; por encima de ese punto lo hace la CPU.
    if (repeat > 1 && repeat > kMaxGridYZ) return false;

    switch (op) {
        case Binary::Add: return launch_binary<0>(a, b, out, inner, repeat);
        case Binary::Sub: return launch_binary<1>(a, b, out, inner, repeat);
        case Binary::Mul: return launch_binary<2>(a, b, out, inner, repeat);
        case Binary::Div: return launch_binary<3>(a, b, out, inner, repeat);
    }
    return false;
}

bool matmul(const Storage& a, const Storage& b, Storage& out,
            size_t batch, size_t rows, size_t inner_dim, size_t cols,
            bool a_batched, bool b_batched) {
    if (!enabled()) return false;
    if (batch == 0 || rows == 0 || inner_dim == 0 || cols == 0) return false;

    // El umbral es el trabajo total del lote: un lote de matrices pequeñas sí
    // puede merecer la pena aunque cada una por separado no lo mereciera.
    const double flops = 2.0 * (double)batch * rows * inner_dim * cols;
    if (flops < (double)min_matmul_flops()) return false;

    if (rows > kMaxInt || inner_dim > kMaxInt || cols > kMaxInt) return false;
    if (batch > kMaxGridYZ) return false;
    if (out.size() != batch * rows * cols) return false;

    // Paso 0 en el operando no lotificado: la misma matriz para todo el lote,
    // exactamente igual que en el camino de CPU.
    const long long a_stride = a_batched ? (long long)rows * inner_dim : 0;
    const long long b_stride = b_batched ? (long long)inner_dim * cols : 0;

    const int M = (int)rows;
    const int K = (int)inner_dim;
    const int N = (int)cols;

    MatmulKernel choice = resolve_matmul_kernel(rows, inner_dim, cols);

    // Si alguien pidió a mano la variante vectorizada sobre una forma que no
    // cumple la alineación, se degrada en silencio a la de registros en lugar
    // de calcular mal. Una lectura float4 desalineada no da un error: da otro
    // valor, que es bastante peor.
    if (choice == MatmulKernel::Vectorized && (K % 4 != 0 || N % 4 != 0)) {
        choice = MatmulKernel::RegisterTiled;
    }

    if (choice == MatmulKernel::RegisterTiled || choice == MatmulKernel::Vectorized) {
        if ((rows + kBM - 1) / kBM > kMaxGridYZ) return false;
        const dim3 grid((unsigned)((cols + kBN - 1) / kBN),
                        (unsigned)((rows + kBM - 1) / kBM),
                        (unsigned)batch);
        if (choice == MatmulKernel::Vectorized) {
            matmul_register_tiled<true><<<grid, kRegBlock>>>(
                a.device(), b.device(), out.device_write(), M, K, N, a_stride, b_stride);
            return launched_ok("matmul_vectorized", out);
        }
        matmul_register_tiled<false><<<grid, kRegBlock>>>(
            a.device(), b.device(), out.device_write(), M, K, N, a_stride, b_stride);
        return launched_ok("matmul_register_tiled", out);
    }

    if ((rows + kTile - 1) / kTile > kMaxGridYZ) return false;
    const dim3 block(kTile, kTile);
    const dim3 grid((unsigned)((cols + kTile - 1) / kTile),
                    (unsigned)((rows + kTile - 1) / kTile),
                    (unsigned)batch);

    if (choice == MatmulKernel::Naive) {
        matmul_naive<<<grid, block>>>(a.device(), b.device(), out.device_write(),
                                      M, K, N, a_stride, b_stride);
        return launched_ok("matmul_naive", out);
    }

    matmul_tiled<<<grid, block>>>(a.device(), b.device(), out.device_write(),
                                  M, K, N, a_stride, b_stride);
    return launched_ok("matmul_tiled", out);
}

bool relu(const Storage& x, Storage& out) {
    if (!elementwise_worth_it(x.size())) return false;
    if (out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    relu_forward<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), n);
    return launched_ok("relu_forward", out);
}

bool relu_backward(const Storage& x, const Storage& grad_out, Storage& out) {
    if (!elementwise_worth_it(x.size())) return false;
    if (grad_out.size() != x.size() || out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    relu_grad<<<grid_for(n), kBlock>>>(x.device(), grad_out.device(), out.device_write(), n);
    return launched_ok("relu_grad", out);
}

bool softmax(const Storage& x, Storage& out, size_t rows, size_t cols) {
    if (!elementwise_worth_it(x.size())) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != x.size() || out.size() != x.size()) return false;
    if (rows > (size_t)std::numeric_limits<int>::max()) return false;

    softmax_rows<<<(unsigned)rows, kReduceBlock>>>(x.device(), out.device_write(), (int)cols);
    return launched_ok("softmax_rows", out);
}

bool softmax_backward(const Storage& y, const Storage& grad_out, Storage& out,
                      size_t rows, size_t cols) {
    if (!elementwise_worth_it(y.size())) return false;
    if (rows == 0 || cols == 0 || cols > kMaxInt) return false;
    if (rows * cols != y.size() || grad_out.size() != y.size() || out.size() != y.size()) {
        return false;
    }
    if (rows > (size_t)std::numeric_limits<int>::max()) return false;

    softmax_rows_grad<<<(unsigned)rows, kReduceBlock>>>(
        y.device(), grad_out.device(), out.device_write(), (int)cols);
    return launched_ok("softmax_rows_grad", out);
}

} // namespace ops
} // namespace cuda
} // namespace engine
