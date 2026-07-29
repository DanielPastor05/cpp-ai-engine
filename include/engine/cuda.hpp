#ifndef ENGINE_CUDA_HPP
#define ENGINE_CUDA_HPP

#include <cstddef>
#include <string>

namespace engine {
namespace cuda {

// ---------------------------------------------------------
// Backend CUDA.
//
// Esta cabecera existe siempre. Si el motor se compiló sin CUDA, available()
// devuelve false, todo se ejecuta en CPU y el programa que la incluye compila
// igual: no hay que envolver nada en #ifdef para usar la librería.
// ---------------------------------------------------------

// True si el motor se compiló con -DENGINE_CUDA y hay un dispositivo
// utilizable. La primera llamada interroga al runtime; las siguientes
// devuelven el valor memorizado.
bool available();

// Interruptor en tiempo de ejecución. Arranca siguiendo a available(), y se
// puede apagar con ENGINE_CUDA=0 en el entorno o con set_enabled(false). Las
// pruebas de paridad lo usan para calcular la misma expresión de las dos
// formas y compararlas.
bool enabled();
void set_enabled(bool on);

struct DeviceInfo {
    std::string name = "sin dispositivo";
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessors = 0;
    size_t total_memory = 0;
};
DeviceInfo device_info();

// Espera a que terminen los kernels lanzados. Sólo hace falta para medir: la
// corrección la garantizan las copias, que son sincronizantes.
void synchronize();

// Versiones de CUDA en juego, en el formato del toolkit (12040 = 12.4).
//
// Son tres y no dos, y la distinción importa: si el toolkit con el que se
// compiló va por delante del driver instalado, el binario no lleva código
// nativo utilizable para la tarjeta, el driver cae a compilar el PTX en tiempo
// de ejecución y ahí falla con «unsupported toolchain». Es la causa más común
// de que un backend recién compilado no arranque.
//
// Para detectarlo hay que comparar compiled_version() con driver_version().
// runtime_version() **no** vale para eso: cudaRuntimeGetVersion() sigue al
// driver, así que devuelve 13.2 en una máquina con driver 13.2 aunque el
// runtime enlazado venga del toolkit 13.3. Medido, no supuesto.
int compiled_version();   // CUDART_VERSION: el toolkit que compiló esto
int runtime_version();    // lo que informa el runtime en ejecución
int driver_version();     // el driver instalado

// ---------------------------------------------------------
// Contadores de lanzamiento.
//
// El motor cae al camino de CPU cuando un kernel no se puede lanzar, que es lo
// correcto en producción y una trampa en las pruebas: la paridad compararía
// CPU contra CPU y pasaría con error cero sin haber tocado el dispositivo.
// Una prueba en verde que no ejercitó nada es peor que una en rojo.
// ---------------------------------------------------------
size_t kernels_launched();
size_t kernels_failed();
void reset_kernel_counters();

// ---------------------------------------------------------
// Techo teórico del dispositivo.
//
// Sin esto, decir «el kernel hace 4 TFLOP/s» no significa nada: la cifra que
// importa es qué fracción del pico alcanza, porque es la que dice si queda
// trabajo por hacer. Se deduce de cudaDeviceProp, así que no hay que buscar
// las especificaciones de cada tarjeta a mano.
// ---------------------------------------------------------

// SMs x núcleos por SM x 2 (una FMA cuentan como dos operaciones) x reloj.
double peak_fp32_gflops();
// Reloj de memoria x 2 (doble tasa) x anchura del bus.
double peak_bandwidth_gbs();

// ---------------------------------------------------------
// Variantes del producto de matrices.
//
// El motor lleva cuatro kernels para la misma operación, del más ingenuo al
// más afinado. No es indecisión: la progresión **es** el resultado. Tenerlas
// todas vivas permite medirlas una contra otra en la misma máquina y, sobre
// todo, comprobar la paridad de cada una por separado — un kernel con teselas
// de 128x128 falla justo en las formas con resto, y sin poder seleccionarlo
// no habría manera de dejarlo escrito en una prueba.
// ---------------------------------------------------------
enum class MatmulKernel {
    Auto,          // elige según la forma y la alineación
    Naive,         // sin memoria compartida; referencia inferior
    Tiled,         // teselas 32x32 en compartida, un resultado por hilo
    RegisterTiled, // 8x8 resultados por hilo en registros
    Vectorized     // igual, con cargas float4
};

MatmulKernel matmul_kernel();
void set_matmul_kernel(MatmulKernel kernel);
const char* matmul_kernel_name(MatmulKernel kernel);

// Qué variante resolvería `Auto` para una forma concreta. La usan el banco de
// pruebas y la documentación; no hace falta para calcular.
MatmulKernel resolve_matmul_kernel(size_t rows, size_t inner_dim, size_t cols);

// ---------------------------------------------------------
// Umbrales de despacho.
//
// Lanzar un kernel cuesta unos microsegundos, así que por debajo de cierto
// tamaño la GPU pierde contra un solo núcleo de CPU. Estos dos números
// deciden cuándo merece la pena; se pueden fijar con ENGINE_CUDA_MIN_FLOPS y
// ENGINE_CUDA_MIN_ELEMENTS para barrerlos sin recompilar.
// ---------------------------------------------------------

// Mínimo de operaciones (2*M*K*N) para mandar un producto de matrices a la GPU.
size_t min_matmul_flops();
// Mínimo de elementos para mandar una operación elemento a elemento.
size_t min_elementwise_elements();
void set_thresholds(size_t matmul_flops, size_t elementwise_elements);

// ---------------------------------------------------------
// Contabilidad de transferencias host <-> dispositivo.
//
// Se mide aparte del tiempo de kernel a propósito. En un motor real el enlace
// PCIe es el cuello de botella mucho antes que el cálculo, y una tabla CPU/GPU
// que esconda ese coste dentro del total no dice nada útil.
// ---------------------------------------------------------
struct TransferStats {
    size_t to_device_bytes = 0;
    size_t to_host_bytes = 0;
    size_t to_device_count = 0;
    size_t to_host_count = 0;
    double to_device_seconds = 0.0;
    double to_host_seconds = 0.0;
};
TransferStats transfer_stats();
void reset_transfer_stats();

namespace detail {

// Primitivas de memoria que usa Storage. No forman parte de la API pública:
// se declaran aquí para que engine/detail/storage.hpp no tenga que incluir las
// cabeceras de CUDA, que arrastrarían nvcc a todas las unidades de traducción.
// Contabilidad de lanzamientos, que lleva el despachador de kernels.
void note_kernel_launched();
void note_kernel_failed();

float* device_alloc(size_t elements);
void device_free(float* ptr);
void copy_to_device(float* dst, const float* src, size_t elements);
void copy_to_host(float* dst, const float* src, size_t elements);
// Duplica un búfer sin salir del dispositivo. No entra en TransferStats a
// propósito: esas cifras miden el PCIe, y meter aquí una copia que no lo cruza
// haría que el número dejara de significar lo que dice medir.
void copy_device_to_device(float* dst, const float* src, size_t elements);

} // namespace detail

} // namespace cuda
} // namespace engine

#endif // ENGINE_CUDA_HPP
