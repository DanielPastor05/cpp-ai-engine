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
float* device_alloc(size_t elements);
void device_free(float* ptr);
void copy_to_device(float* dst, const float* src, size_t elements);
void copy_to_host(float* dst, const float* src, size_t elements);

} // namespace detail

} // namespace cuda
} // namespace engine

#endif // ENGINE_CUDA_HPP
