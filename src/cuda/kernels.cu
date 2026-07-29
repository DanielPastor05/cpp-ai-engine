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
#include <cstdlib>
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

// Ejes que admite el kernel de permutacion. Cuatro son los que usa el motor
// —(B, H, S, d) en la atencion, (N, C, H, W) en la convolucion—; ocho da margen
// sin que el plan deje de caber holgado en el espacio de argumentos del kernel.
constexpr int kMaxPermuteDims = 8;

// cudaGetLastError() solo ve fallos **de lanzamiento**. Si el fallo ocurre
// dentro del cuerpo del kernel —un acceso fuera de rango, por ejemplo— el error
// se queda pegado al contexto y aflora en la siguiente copia, que suele estar
// tres operaciones mas alla y no tiene ninguna culpa.
//
// Sincronizar aqui lo ataja en el lanzamiento culpable, pero es una barrera por
// kernel y el motor lanza cientos por paso: seria pagar en produccion por un
// diagnostico que solo interesa mientras se persigue un fallo. De ahi la
// variable de entorno, apagada por defecto.
//
//   ENGINE_CUDA_SYNC=1 .\build-cuda\Release\test_engine.exe
bool sync_after_launch() {
    static const bool on = [] {
        const char* raw = std::getenv("ENGINE_CUDA_SYNC");
        return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    return on;
}

// Un kernel que falla al lanzarse no aborta el programa: se informa y se
// devuelve false, de modo que el resultado lo calcula la CPU. Un motor que se
// cae porque la GPU está ocupada es peor que uno que va más lento.
//
// Va aparte de launched_ok() porque no todo lanzamiento fallido se deshace
// igual: quien pidió la salida con device_write() tiene que revertir, y quien
// la pidió con device_mut() no debe hacerlo. Ver accumulate_grad().
bool launch_ok(const char* what) {
    cudaError_t status = cudaGetLastError();
    // Con ENGINE_CUDA_SYNC el error de ejecucion del kernel se recoge aqui, en
    // el lanzamiento que lo provoco, en lugar de en la siguiente copia. El
    // camino de recuperacion es el mismo: se informa y se calcula en CPU.
    if (status == cudaSuccess && sync_after_launch()) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        detail::note_kernel_launched();
        return true;
    }

    detail::note_kernel_failed();

    // Sólo el primero, y con el diagnóstico completo. La versión anterior
    // imprimía una línea por lanzamiento fallido: en una suite de pruebas eso
    // son cientos de líneas identicas que tapan la salida real y no dicen la
    // causa. El resto se cuenta y se consulta con cuda::kernels_failed().
    static bool reported = false;
    if (!reported) {
        reported = true;
        // La comparacion util es la del toolkit que compilo esto contra el
        // driver. runtime_version() no sirve aqui: sigue al driver, asi que
        // nunca sale por delante y la pista no se imprimiria jamas.
        const int built = compiled_version();
        const int drv = driver_version();
        std::fprintf(stderr,
                     "\nengine: el kernel %s no se pudo lanzar (%s).\n"
                     "  Se calcula en CPU, asi que los resultados son correctos pero lentos.\n"
                     "  Compilado con CUDA %d.%d, driver instalado CUDA %d.%d.\n",
                     what, cudaGetErrorString(status),
                     built / 1000, (built % 1000) / 10, drv / 1000, (drv % 1000) / 10);

        // Las dos pistas son independientes y pueden darse a la vez: un binario
        // con la arquitectura equivocada cae a compilar el PTX, y es entonces
        // cuando un driver mas antiguo que el toolkit remata el fallo. Encadenar
        // las con else-if contaria media historia.
        if (drv > 0 && built > drv) {
            std::fprintf(stderr,
                         "  El driver es mas antiguo que el toolkit: actualiza el driver de\n"
                         "  NVIDIA, o compila con una version de CUDA que el driver admita.\n");
        }
        if (status == cudaErrorUnsupportedPtxVersion ||
            status == cudaErrorNoKernelImageForDevice) {
            const DeviceInfo info = device_info();
            std::fprintf(stderr,
                         "  El binario no lleva codigo nativo para esta tarjeta (cc %d.%d).\n"
                         "  Reconfigura con -DCMAKE_CUDA_ARCHITECTURES=%d%d\n",
                         info.compute_major, info.compute_minor,
                         info.compute_major, info.compute_minor);
        }
        std::fprintf(stderr, "  Los siguientes fallos no se repiten aqui;"
                             " se cuentan en cuda::kernels_failed().\n\n");
    }
    return false;
}

// Lo mismo, para la salida que se pidió con device_write().
//
// Hay que deshacer ese device_write(). Si no, el camino de CPU pediría el búfer
// de host, Storage se lo bajaría del dispositivo sin inicializar, y matmul —que
// acumula sobre una salida que supone a cero— devolvería basura en lugar de un
// resultado correcto más lento.
bool launched_ok(const char* what, Storage& out) {
    if (launch_ok(what)) return true;
    out.revert_device_write();
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

// El acumulador del backward: out = g la primera vez, out += g después.
//
// initialize es uniforme en toda la malla —lo decide el llamante mirando si el
// tensor ya tenía gradiente—, así que la rama no divide ningún warp.
__global__ void grad_accumulate(const float* __restrict__ g, float* __restrict__ out,
                                long long n, bool initialize) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = initialize ? g[i] : out[i] + g[i];
    }
}

// out = x * mul + add. Una sola forma para las dos operaciones con escalar:
// multiplicar es add = 0 y sumar es mul = 1. Con esos valores el producto o la
// suma sobran, y da igual: el kernel esta limitado por la memoria, no por las
// dos operaciones de coma flotante que hace por elemento.
__global__ void scalar_affine(const float* __restrict__ x, float* __restrict__ out,
                              float mul, float add, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = x[i] * mul + add;
    }
}

// ---------------------------------------------------------
// Reordenacion de ejes
// ---------------------------------------------------------
//
// Un plano de la salida se traduce a coordenadas y de ahi a un desplazamiento
// sobre la entrada, exactamente igual que en el camino de CPU. La escritura es
// contigua y la lectura salta: es la orientacion correcta de las dos, porque
// una escritura dispersa serializa el ancho de banda mucho peor que una lectura
// dispersa, que al menos se solapa con el resto del warp.
//
// El plan viaja por valor. Los argumentos de un kernel van a memoria constante,
// asi que las ocho parejas se leen de ahi y no hacen falta ni reserva ni copia
// aparte para pasarlos.
//
// ponytail: division entera por elemento y por eje; si el perfil la senala, se
// sustituye por el truco de multiplicar por el reciproco magico.
struct PermutePlan {
    int shape[kMaxPermuteDims];
    long long stride[kMaxPermuteDims];
};

__global__ void permute_gather(const float* __restrict__ x, float* __restrict__ out,
                               PermutePlan plan, int nd, long long n) {
    const long long grid_stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        long long rem = i;
        long long src = 0;
        // De atras hacia delante: el ultimo eje es el que varia mas rapido.
        for (int d = nd - 1; d >= 0; --d) {
            const long long extent = plan.shape[d];
            src += (rem % extent) * plan.stride[d];
            rem /= extent;
        }
        out[i] = x[src];
    }
}

// Suma sobre un eje visto como (outer, axis_len, inner): un hilo por elemento
// de la salida, que recorre el eje.
//
// El orden de acumulacion es el mismo que en CPU —el eje de menor a mayor— y
// tampoco hay ninguna multiplicacion que el compilador pueda fundir en un FMA,
// asi que este kernel si coincide bit a bit con el camino de CPU.
__global__ void sum_over_axis(const float* __restrict__ x, float* __restrict__ out,
                              long long axis_len, long long inner, long long n) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long o = i / inner;
        const long long j = i % inner;
        const float* base = x + o * axis_len * inner + j;
        float acc = 0.0f;
        for (long long a = 0; a < axis_len; ++a) acc += base[a * inner];
        out[i] = acc;
    }
}

// ---------------------------------------------------------
// im2col / col2im
// ---------------------------------------------------------
//
// Los dos recorren la salida, no la entrada, y esa es toda la diferencia entre
// que hagan falta operaciones atomicas o no.
//
// im2col escribe cada elemento de las columnas una sola vez, asi que sale
// directo. col2im es su adjunto: varias ventanas solapadas contribuyen al mismo
// pixel, y recorrer las ventanas obligaria a sumar con atomicas —lentas y, peor
// aun, con un orden de acumulacion que cambia de una ejecucion a otra—. Se
// invierte el recorrido: un hilo por pixel de **entrada**, que busca las
// ventanas que lo cubren. Sin atomicas, y con el mismo orden que la CPU.

struct WindowDims {
    int channels, height, width;
    int kernel_h, kernel_w, stride, padding;
    int out_h, out_w;
};

__global__ void im2col_gather(const float* __restrict__ input, float* __restrict__ cols,
                              WindowDims d, long long n) {
    const long long K = (long long)d.channels * d.kernel_h * d.kernel_w;
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long row = i / K;
        const long long k = i % K;

        const long long b = row / spatial;
        const long long oh = (row % spatial) / d.out_w;
        const long long ow = row % d.out_w;

        const long long c = k / ((long long)d.kernel_h * d.kernel_w);
        const long long r = k % ((long long)d.kernel_h * d.kernel_w);
        const long long ki = r / d.kernel_w;
        const long long kj = r % d.kernel_w;

        const long long h = oh * d.stride + ki - d.padding;
        const long long w = ow * d.stride + kj - d.padding;

        // El cero se escribe explicitamente. El camino de CPU se apoya en que el
        // tensor nace a ceros y se salta las posiciones de relleno; aqui la
        // salida se pidio con device_write(), que no sube nada, asi que saltarse
        // una posicion la dejaria con lo que hubiera antes en ese bufer.
        float value = 0.0f;
        if (h >= 0 && h < d.height && w >= 0 && w < d.width) {
            value = input[((b * d.channels + c) * d.height + h) * d.width + w];
        }
        cols[i] = value;
    }
}

__global__ void col2im_scatter(const float* __restrict__ cols, float* __restrict__ input,
                               WindowDims d, long long n) {
    const long long K = (long long)d.channels * d.kernel_h * d.kernel_w;
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long w = i % d.width;
        const long long h = (i / d.width) % d.height;
        const long long c = (i / ((long long)d.width * d.height)) % d.channels;
        const long long b = i / ((long long)d.width * d.height * d.channels);

        // Ventanas que cubren este pixel: de h + padding = oh*stride + ki con
        // 0 <= ki < kH se despeja el rango de oh, y dentro de el **toda** oh es
        // valida, asi que no hace falta comprobar divisibilidad.
        const long long hp = h + d.padding;
        const long long wp = w + d.padding;
        long long oh_lo = (hp - d.kernel_h + d.stride) / d.stride;   // techo de (hp-kH+1)/stride
        long long ow_lo = (wp - d.kernel_w + d.stride) / d.stride;
        if (oh_lo < 0) oh_lo = 0;
        if (ow_lo < 0) ow_lo = 0;
        const long long oh_hi = min(hp / d.stride, (long long)d.out_h - 1);
        const long long ow_hi = min(wp / d.stride, (long long)d.out_w - 1);

        // oh y ow ascendentes: el mismo orden en que acumula el bucle de CPU, de
        // modo que la suma se hace con los mismos redondeos.
        float acc = 0.0f;
        for (long long oh = oh_lo; oh <= oh_hi; ++oh) {
            const long long ki = hp - oh * d.stride;
            for (long long ow = ow_lo; ow <= ow_hi; ++ow) {
                const long long kj = wp - ow * d.stride;
                const long long row = (b * d.out_h + oh) * d.out_w + ow;
                const long long k = (c * d.kernel_h + ki) * d.kernel_w + kj;
                acc += cols[row * K + k];
            }
        }
        input[i] = acc;
    }
}

// ---------------------------------------------------------
// Submuestreo por maximo
// ---------------------------------------------------------
//
// El forward guarda, junto al maximo, el indice plano del pixel que lo gano:
// es lo unico que el paso hacia atras necesita, y con el criterio de desempate
// del camino de CPU —gana el primero en orden de recorrido, porque la
// comparacion es estricta—.
//
// El backward recorre la **entrada**, como col2im y por el mismo motivo: dos
// ventanas solapadas pueden haber elegido el mismo pixel, y recorrer las
// ventanas obligaria a sumar con atomicas. Asi cada pixel busca las ventanas
// que lo cubren, se queda con las que lo eligieron, y suma en un orden fijo.

__global__ void maxpool_windows(const float* __restrict__ input, float* __restrict__ out,
                                float* __restrict__ argmax, WindowDims d, long long n) {
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long ow = i % d.out_w;
        const long long oh = (i / d.out_w) % d.out_h;
        const long long c = (i / spatial) % d.channels;
        const long long b = i / (spatial * d.channels);

        float best = -INFINITY;
        long long best_idx = 0;
        bool found = false;

        for (int ki = 0; ki < d.kernel_h; ++ki) {
            const long long h = oh * d.stride + ki - d.padding;
            if (h < 0 || h >= d.height) continue;
            for (int kj = 0; kj < d.kernel_w; ++kj) {
                const long long w = ow * d.stride + kj - d.padding;
                if (w < 0 || w >= d.width) continue;

                const long long idx = ((b * d.channels + c) * d.height + h) * d.width + w;
                const float v = input[idx];
                // Estricto, no >=: en un empate gana el primero, igual que en CPU.
                if (!found || v > best) { best = v; best_idx = idx; found = true; }
            }
        }
        out[i] = best;
        argmax[i] = (float)best_idx;
    }
}

__global__ void maxpool_windows_grad(const float* __restrict__ argmax,
                                     const float* __restrict__ g,
                                     float* __restrict__ dx, WindowDims d, long long n) {
    const long long spatial = (long long)d.out_h * d.out_w;
    const long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += grid_stride) {
        const long long w = i % d.width;
        const long long h = (i / d.width) % d.height;
        const long long c = (i / ((long long)d.width * d.height)) % d.channels;
        const long long b = i / ((long long)d.width * d.height * d.channels);

        const long long hp = h + d.padding;
        const long long wp = w + d.padding;
        long long oh_lo = (hp - d.kernel_h + d.stride) / d.stride;
        long long ow_lo = (wp - d.kernel_w + d.stride) / d.stride;
        if (oh_lo < 0) oh_lo = 0;
        if (ow_lo < 0) ow_lo = 0;
        const long long oh_hi = min(hp / d.stride, (long long)d.out_h - 1);
        const long long ow_hi = min(wp / d.stride, (long long)d.out_w - 1);

        float acc = 0.0f;
        for (long long oh = oh_lo; oh <= oh_hi; ++oh) {
            for (long long ow = ow_lo; ow <= ow_hi; ++ow) {
                const long long out_idx = ((b * d.channels + c) * d.out_h + oh) * d.out_w + ow;
                // Solo suma si esta ventana eligio precisamente este pixel.
                if ((long long)argmax[out_idx] == i) acc += g[out_idx];
            }
        }
        dx[i] = acc;
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

// Traduce la geometria a los enteros del kernel, o dice que no cabe. Los dos
// puntos de entrada de la convolucion comparten estas comprobaciones.
bool window_dims(const WindowShape& s, WindowDims& d) {
    const size_t all[] = {s.batch, s.channels, s.height, s.width, s.kernel_h,
                          s.kernel_w, s.stride, s.padding, s.out_h, s.out_w};
    for (size_t v : all) {
        if (v > kMaxInt) return false;
    }
    if (s.batch == 0 || s.channels == 0 || s.height == 0 || s.width == 0) return false;
    if (s.kernel_h == 0 || s.kernel_w == 0 || s.stride == 0) return false;
    if (s.out_h == 0 || s.out_w == 0) return false;

    d.channels = (int)s.channels;
    d.height = (int)s.height;
    d.width = (int)s.width;
    d.kernel_h = (int)s.kernel_h;
    d.kernel_w = (int)s.kernel_w;
    d.stride = (int)s.stride;
    d.padding = (int)s.padding;
    d.out_h = (int)s.out_h;
    d.out_w = (int)s.out_w;
    return true;
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

bool scalar(const Storage& x, Storage& out, float mul, float add) {
    if (!elementwise_worth_it(x.size())) return false;
    if (out.size() != x.size()) return false;
    const long long n = (long long)x.size();
    scalar_affine<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), mul, add, n);
    return launched_ok("scalar_affine", out);
}

bool permute(const Storage& x, Storage& out,
             const size_t* out_shape, const size_t* src_strides, size_t ndim) {
    if (!elementwise_worth_it(x.size())) return false;
    if (ndim == 0 || ndim > (size_t)kMaxPermuteDims) return false;
    if (out.size() != x.size()) return false;

    PermutePlan plan{};
    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) {
        if (out_shape[d] == 0 || out_shape[d] > kMaxInt) return false;
        plan.shape[d] = (int)out_shape[d];
        plan.stride[d] = (long long)src_strides[d];
        total *= out_shape[d];
    }
    // La forma de salida tiene que cubrir la entrada entera: si no cuadra, el
    // kernel leeria fuera del bufer en lugar de dar un resultado equivocado.
    if (total != x.size()) return false;

    const long long n = (long long)x.size();
    permute_gather<<<grid_for(n), kBlock>>>(x.device(), out.device_write(), plan, (int)ndim, n);
    return launched_ok("permute_gather", out);
}

bool sum_axis(const Storage& x, Storage& out, size_t outer, size_t axis_len, size_t inner) {
    if (!elementwise_worth_it(x.size())) return false;
    if (outer == 0 || axis_len == 0 || inner == 0) return false;
    if (outer * axis_len * inner != x.size()) return false;
    if (out.size() != outer * inner) return false;

    const long long n = (long long)(outer * inner);
    sum_over_axis<<<grid_for(n), kBlock>>>(x.device(), out.device_write(),
                                           (long long)axis_len, (long long)inner, n);
    return launched_ok("sum_over_axis", out);
}

bool im2col(const Storage& input, Storage& cols, const WindowShape& s) {
    if (!elementwise_worth_it(cols.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    const size_t K = s.channels * s.kernel_h * s.kernel_w;
    if (input.size() != s.batch * s.channels * s.height * s.width) return false;
    if (cols.size() != s.batch * s.out_h * s.out_w * K) return false;

    const long long n = (long long)cols.size();
    im2col_gather<<<grid_for(n), kBlock>>>(input.device(), cols.device_write(), d, n);
    return launched_ok("im2col_gather", cols);
}

bool col2im(const Storage& cols, Storage& input, const WindowShape& s) {
    // El trabajo se mide por las columnas, que es el lado grande, aunque la
    // malla recorra la entrada: con kernel 3x3 son nueve veces mas valores.
    if (!elementwise_worth_it(cols.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    const size_t K = s.channels * s.kernel_h * s.kernel_w;
    if (input.size() != s.batch * s.channels * s.height * s.width) return false;
    if (cols.size() != s.batch * s.out_h * s.out_w * K) return false;

    const long long n = (long long)input.size();
    col2im_scatter<<<grid_for(n), kBlock>>>(cols.device(), input.device_write(), d, n);
    return launched_ok("col2im_scatter", input);
}

bool maxpool(const Storage& input, Storage& out, Storage& argmax, const WindowShape& s) {
    if (!elementwise_worth_it(input.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    if (input.size() != s.batch * s.channels * s.height * s.width) return false;
    if (out.size() != s.batch * s.channels * s.out_h * s.out_w) return false;
    if (argmax.size() != out.size()) return false;
    // El indice viaja en float, que solo representa enteros exactos hasta 2^24.
    // Por encima de ese tamano el redondeo elegiria otro pixel, asi que se
    // rechaza el despacho y lo hace la CPU.
    if (input.size() > (size_t{1} << 24)) return false;

    const long long n = (long long)out.size();
    maxpool_windows<<<grid_for(n), kBlock>>>(input.device(), out.device_write(),
                                             argmax.device_write(), d, n);
    // Dos salidas, y las dos se pidieron con device_write(): si el lanzamiento
    // falla hay que deshacer las dos, o el camino de CPU bajaria a host un bufer
    // sin inicializar.
    if (launch_ok("maxpool_windows")) return true;
    out.revert_device_write();
    argmax.revert_device_write();
    return false;
}

bool maxpool_backward(const Storage& argmax, const Storage& grad_out, Storage& dx,
                      const WindowShape& s) {
    if (!elementwise_worth_it(dx.size())) return false;

    WindowDims d{};
    if (!window_dims(s, d)) return false;

    if (dx.size() != s.batch * s.channels * s.height * s.width) return false;
    if (grad_out.size() != s.batch * s.channels * s.out_h * s.out_w) return false;
    if (argmax.size() != grad_out.size()) return false;
    if (dx.size() > (size_t{1} << 24)) return false;

    const long long n = (long long)dx.size();
    maxpool_windows_grad<<<grid_for(n), kBlock>>>(argmax.device(), grad_out.device(),
                                                  dx.device_write(), d, n);
    return launched_ok("maxpool_windows_grad", dx);
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

bool accumulate_grad(Storage& grad, const Storage& g, bool initialize) {
    if (!elementwise_worth_it(g.size())) return false;
    if (grad.size() != g.size()) return false;
    // Sólo si no cuesta ni una transferencia. El gradiente acaba leyéndose en
    // host —el optimizador va por CPU—, así que subir algo para sumarlo aquí
    // pagaría el viaje dos veces y saldría peor que no acelerar nada. Al
    // inicializar, el destino se escribe entero y no hay nada que subir.
    if (!g.resident_on_device()) return false;
    if (!initialize && !grad.resident_on_device()) return false;

    const long long n = (long long)g.size();
    float* out = initialize ? grad.device_write() : grad.device_mut();
    grad_accumulate<<<grid_for(n), kBlock>>>(g.device(), out, n, initialize);
    if (launch_ok("grad_accumulate")) return true;

    // Asimetría deliberada, y por eso este no usa launched_ok(). device_write()
    // dio el búfer por válido sin subirlo, así que si el kernel no salió hay que
    // deshacerlo. device_mut() sí subió: el dispositivo tiene el dato bueno y el
    // host está marcado obsoleto, de modo que el camino de CPU lo bajará intacto.
    // Revertir ahí ascendería a válido un host que no lo está.
    if (initialize) grad.revert_device_write();
    return false;
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
