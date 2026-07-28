#ifndef ENGINE_PARALLEL_HPP
#define ENGINE_PARALLEL_HPP

#include <cstddef>
#include <functional>

namespace engine {
namespace parallel {

// ---------------------------------------------------------
// Paralelismo de datos sobre los bucles calientes.
//
// Los hilos se crean una sola vez y se reutilizan: crear uno cuesta unas
// decenas de microsegundos, más que muchas de las operaciones del motor, así
// que hacerlo por operación sería una regresión y no una mejora.
//
// El reparto es **determinista por construcción**: cada trozo del rango lo
// calcula siempre un único hilo, y ninguna reducción cruza la frontera entre
// trozos. El resultado es idéntico bit a bit con uno o con ocho hilos, cosa
// que las pruebas comprueban.
// ---------------------------------------------------------

// Umbral de reparto para un bucle elemento a elemento.
//
// Una operación así está limitada por el ancho de banda de memoria y no por el
// cálculo, así que escala peor y necesita más trabajo para merecer la pena:
// medido, 1,77x con cuatro hilos frente a 3,08x del producto de matrices.
//
// Vive aquí y no en cada .cpp porque lo usan tensor.cpp, nn.cpp y
// transformer.cpp: repetido, se ajustaría en un sitio y no en los otros dos.
constexpr size_t kElementsPerThread = 1u << 17;

// Número de hilos en uso. Por defecto, los núcleos de la máquina; se puede
// fijar con la variable de entorno ENGINE_NUM_THREADS.
size_t num_threads();

// Cambia el tamaño del pool. Con 1 todo se ejecuta en el hilo llamante y no
// se crea ninguno, que es lo que conviene para medir o depurar.
void set_num_threads(size_t threads);

// Ejecuta body sobre [0, count) repartido en trozos contiguos.
//
// min_per_thread evita repartir trabajo que no compensa: por debajo de ese
// tamaño, la sincronización cuesta más que el cálculo y se ejecuta en línea.
// Las llamadas anidadas también se ejecutan en línea, para no multiplicar
// hilos cuando una operación paralela invoca a otra.
void parallel_for(size_t count, size_t min_per_thread,
                  const std::function<void(size_t begin, size_t end)>& body);

// True si el hilo actual ya está dentro de una región paralela.
bool inside_parallel_region();

} // namespace parallel
} // namespace engine

#endif // ENGINE_PARALLEL_HPP
