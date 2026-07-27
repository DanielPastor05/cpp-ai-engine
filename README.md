# C++ AI Engine desde Cero

[![CI](https://github.com/DanielPastor05/cpp-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielPastor05/cpp-ai-engine/actions/workflows/ci.yml)

Un motor de Inteligencia Artificial y Aprendizaje Profundo (Deep Learning) avanzado construido desde cero en **C++17**, sin dependencias externas, con backend de aceleración GPU en **CUDA** planificado.

## 🚀 Hoja de Ruta del Proyecto

- [x] **Fase 1: Librería de Tensores (CPU)**
  - Arreglo plano contiguo en memoria (`data`).
  - Cálculo de pasos/zancadas (*strides*) en orden C (*row-major*).
  - Indexación multidimensional a plano 1D.
  - Operaciones elemento a elemento (Suma, Resta, Multiplicación Hadamard, División, ReLU).
  - Multiplicación Matricial (`MatMul`) optimizada para la caché CPU ($i \to k \to j$).
- [x] **Fase 2: Motor de Autograd**
  - Grafos dirigidos acíclicos (DAG) de cómputo.
  - Diferenciación automática en modo inverso (*Reverse-mode Automatic Differentiation*).
  - Algoritmo de ordenamiento topológico (*Topological Sort*) iterativo para `.backward()`.
  - Modo sin gradiente (`NoGradGuard`) para inferencia y actualización de pesos.
- [x] **Fase 3: Capas y Optimizadores**
  - Abstracción `nn::Module` (`Linear`, `ReLU`, `Softmax`, `Sequential`).
  - Pérdidas `cross_entropy_loss` (log-softmax + NLL fusionados) y `mse_loss`.
  - Optimizadores `SGD` (con momento y *weight decay*) y `Adam` (con corrección de sesgo).
  - Entrenamiento por mini-lotes mediante `Tensor::select_rows`.
  - Inicialización Xavier/Glorot y semilla reproducible (`engine::manual_seed`).
- [ ] **Fase 4: Redes Convolucionales (CNN)**
  - Transformación `im2col` / `col2im`.
  - Capas `Conv2d`, `MaxPool2d` y `Flatten`.
- [ ] **Fase 5: Arquitectura Transformer**
  - *Scaled Dot-Product Attention* y *Multi-Head Attention (MHA)*.
  - Codificación posicional y `LayerNorm`.
- [ ] **Fase 6: Backend GPU & CUDA**
  - Gestión de memoria Host/Device (`cudaMalloc`, `cudaMemcpy`).
  - Custom CUDA Kernels y optimización con *Shared Memory Tiling*.

---

## 📁 Estructura del Proyecto

```
include/engine/
  tensor.hpp      Tensor (handle) + TensorImpl (nodo del grafo)
  autograd.hpp    backward(), grad_enabled(), NoGradGuard
  nn.hpp          Module, Linear, ReLU, Softmax, Sequential, pérdidas, métricas
  optim.hpp       Optimizer, SGD, Adam
src/              Implementación de la librería
examples/         main.cpp (Fase 1), autograd_demo.cpp (Fase 2), nn_demo.cpp (Fase 3)
tests/            Suite de pruebas con verificación numérica de gradientes
.github/workflows/ci.yml   Compila y ejecuta las pruebas en GCC, Clang y MSVC
```

---

## 🛠️ Compilación y Ejecución

### Requisitos Previos
- Compilador C++17 (MSVC, GCC o Clang)
- CMake (versión 3.18 o superior)

### Comandos de Construcción
```bash
# 1. Configurar CMake
cmake -B build -S .

# 2. Compilar el proyecto
cmake --build build

# 3. Ejecutar los ejemplos
./build/cpp_ai_engine   # Fase 1: tensores, strides y MatMul
./build/autograd_demo   # Fase 2: backpropagation y regresión lineal
./build/nn_demo         # Fase 3: MLP que clasifica una espiral de 3 clases

# 4. Ejecutar las pruebas
ctest --test-dir build --output-on-failure
# o directamente:
./build/test_engine
```

---

## 📖 Uso

### Tensores y autograd

```cpp
#include "engine/tensor.hpp"
using engine::Tensor;

Tensor a({1}, {2.0f}, /*requires_grad=*/true);
Tensor b({1}, {3.0f}, true);

Tensor L = (a * b) + a.relu();   // L = a*b + relu(a)
L.backward();                    // recorre el DAG en orden topológico inverso

a.grad().data()[0];              // 4.0  ->  dL/da = b + 1
b.grad().data()[0];              // 2.0  ->  dL/db = a
```

`backward()` implícito solo se permite sobre un escalar. Para una raíz no escalar
hay que dar el gradiente inicial: `y.backward(Tensor(y.shape(), 1.0f))`.

Para evaluar sin construir el grafo (inferencia, actualización manual de pesos):

```cpp
#include "engine/autograd.hpp"
{
    engine::autograd::NoGradGuard no_grad;
    Tensor logits = model(X);   // no se registran nodos ni funciones de gradiente
}
```

### Entrenar una red

```cpp
#include "engine/nn.hpp"
#include "engine/optim.hpp"

namespace nn = engine::nn;
namespace optim = engine::optim;

engine::manual_seed(42);   // entrenamiento reproducible

nn::Sequential model{
    nn::make<nn::Linear>(2, 64),
    nn::make<nn::ReLU>(),
    nn::make<nn::Linear>(64, 3)
};

optim::Adam opt(model.parameters(), 0.02f);

for (int epoch = 0; epoch < 500; ++epoch) {
    opt.zero_grad();                                   // los gradientes se acumulan
    Tensor logits = model(X);                          // (N, 3) sin normalizar
    Tensor loss = nn::cross_entropy_loss(logits, y);   // y: índices de clase
    loss.backward();
    opt.step();
}

float acc = nn::accuracy(model(X), y);
```

El ejemplo `nn_demo` entrena exactamente este esquema sobre una espiral de 3
clases entrelazadas y compara el resultado con un clasificador lineal:

```
Clasificador lineal : 52.67% de exactitud
MLP + Adam          : 99.33% de exactitud
MLP + SGD mini-lotes: 98.00% de exactitud
```

### Mini-lotes

`Tensor::select_rows` recoge un subconjunto de filas, que es lo que hace
"estocástico" al descenso de gradiente estocástico: cada paso usa una muestra
distinta en lugar del conjunto entero.

```cpp
std::vector<size_t> order(N);
std::iota(order.begin(), order.end(), 0);
std::shuffle(order.begin(), order.end(), engine::global_rng());

for (size_t start = 0; start < N; start += batch_size) {
    const size_t end = std::min(start + batch_size, N);
    const std::vector<size_t> idx(order.begin() + start, order.begin() + end);

    opt.zero_grad();
    Tensor loss = nn::cross_entropy_loss(model(X.select_rows(idx)), labels_de(idx));
    loss.backward();
    opt.step();
}
```

---

## 🧠 Notas de Diseño

**`Tensor` es un handle, `TensorImpl` es el nodo.** `Tensor` guarda solo un
`shared_ptr<TensorImpl>`, así que copiarlo comparte datos, gradientes e
historial. Eso permite que los parámetros devueltos por `Module::parameters()`
sean copias que aun así actualizan los pesos de la capa.

**El grafo es acíclico también en el conteo de referencias.** Cada nodo apunta a
sus padres con `shared_ptr` y nunca a sus hijos. Por eso `backward_fn` recibe el
gradiente de salida como argumento en lugar de capturar su propio tensor:
capturarlo formaría un ciclo de `shared_ptr` y el grafo no se liberaría jamás.

**El backward no construye grafo.** `autograd::backward()` activa internamente un
`NoGradGuard`, de modo que las operaciones sobre gradientes (transposiciones y
`matmul` de la regla de la cadena) no dejan atrás un grafo de segundo orden.

**El ordenamiento topológico es iterativo.** Una versión recursiva agota la pila
del proceso en grafos profundos.

**Estabilidad numérica.** `softmax` resta el máximo de cada fila, y
`cross_entropy_loss` fusiona log-softmax y NLL en un solo nodo cuyo gradiente se
reduce a `(softmax(logits) - one_hot) / N`.

**Solo las hojas acumulan gradiente.** `add_grad` suma en lugar de sobrescribir,
así que hay que llamar a `zero_grad()` en cada iteración (igual que en PyTorch).
Los nodos intermedios, en cambio, se reinician al empezar cada `backward()`: su
gradiente es un valor temporal del recorrido, y conservarlo haría que un segundo
`backward()` sobre el mismo grafo propagase la suma de ambos recorridos.

---

## ✅ Pruebas

`tests/test_engine.cpp` cubre tensores, autograd, capas y optimizadores con 103
comprobaciones, y se ejecutan en CI sobre GCC, Clang y MSVC. El grueso de la verificación de autograd es una **comprobación
numérica de gradientes** por diferencias centradas, que compara cada derivada
analítica con `(f(x+h) - f(x-h)) / 2h`; también se verifica que los nodos
intermedios del grafo se liberen al salir de ámbito y que un MLP resuelva el XOR
donde un modelo lineal no puede.
