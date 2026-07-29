# C++ AI Engine desde Cero

> Esta es la version en espanol. La principal, mas breve y actualizada, esta en
> **[README.md](README.md)** (ingles).

[![CI](https://github.com/DanielPastor05/cpp-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielPastor05/cpp-ai-engine/actions/workflows/ci.yml)

Un motor de Inteligencia Artificial y Aprendizaje Profundo (Deep Learning) avanzado construido desde cero en **C++17**, sin dependencias externas, con backend opcional de aceleración GPU en **CUDA** (apagado por defecto: sin toolkit ni tarjeta el motor compila y pasa las pruebas igual).

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
- [x] **Fase 4: Redes Convolucionales (CNN)**
  - Transformación `im2col` / `col2im` (col2im es el adjunto exacto de im2col).
  - Capas `Conv2d`, `MaxPool2d` y `Flatten`, con relleno y paso configurables.
  - Mini-lotes sobre volúmenes 4D `(N, C, H, W)`.
- [x] **Fase 5: Arquitectura Transformer**
  - *Scaled Dot-Product Attention* con máscara causal y *Multi-Head Attention*.
  - Codificación posicional sinusoidal, `LayerNorm` y `Embedding`.
  - `TransformerBlock` pre-norm con conexiones residuales.
  - Soporte N-dimensional en el tensor: `permute`, `matmul` por lotes,
    `softmax` sobre el último eje y difusión por sufijo.
- [x] **Fase 6: Backend GPU & CUDA**
  - `Storage`: búfer de host y espejo en el dispositivo con banderas de
    validez, de modo que una cadena de operaciones se queda en la GPU y sólo
    baja cuando el programa lee un valor.
  - Kernels propios para `matmul` (con teselas en memoria compartida), las
    operaciones elemento a elemento, ReLU y softmax.
  - Contabilidad de las transferencias host↔dispositivo, medida e informada
    aparte del tiempo de kernel.
  - Pruebas de paridad CPU/GPU sobre los mismos datos, hasta un
    `TransformerBlock` completo con su paso hacia atrás.
  - Detalle completo en **[docs/CUDA.md](docs/CUDA.md)**.

  ```bash
  cmake -B build-cuda -S . -DENGINE_CUDA=ON
  cmake --build build-cuda --parallel
  ctest --test-dir build-cuda --output-on-failure
  ```

  El backend está apagado por defecto: el motor tiene que seguir compilando y
  pasando las pruebas en una máquina sin toolkit ni tarjeta, que es lo que hay
  en CI.

---

## 📁 Estructura del Proyecto

```
include/engine/
  tensor.hpp      Tensor (handle) + TensorImpl (nodo del grafo)
  autograd.hpp    backward(), grad_enabled(), NoGradGuard
  nn.hpp          Module, Linear, ReLU, Softmax, Sequential, pérdidas, métricas
  conv.hpp        Window2d, im2col/col2im, Conv2d, MaxPool2d, Flatten
  transformer.hpp LayerNorm, Embedding, atención, MultiHeadAttention, TransformerBlock
  optim.hpp       Optimizer, SGD, Adam, recorte de gradiente, planificadores de lr
  serialize.hpp   Guardar y cargar pesos
  cuda.hpp        Backend GPU: disponibilidad, umbrales, contadores, transferencias
  data.hpp        Carga de MNIST y utilidades de conjuntos de datos
  parallel.hpp    parallel_for con reparto determinista y el umbral por hilo
  random.hpp      global_rng() (aparte, porque <random> es cara de compilar)
  detail/         TensorImpl, Storage y las declaraciones de los kernels
src/              Implementación de la librería (src/cuda/ los kernels)
examples/         main.cpp (F1), autograd_demo.cpp (F2), nn_demo.cpp (F3),
                  cnn_demo.cpp (F4), transformer_demo.cpp (F5), mnist_demo.cpp
tests/            Suite de pruebas con verificación numérica de gradientes,
                  repartida por áreas (tensor, autograd, nn, conv, transformer,
                  referencia contra PyTorch, índices de los kernels y paridad
                  CPU/GPU)
bench/            Banco de pruebas de rendimiento (no se ejecuta en CI)
.github/workflows/ci.yml   GCC, Clang, AppleClang y MSVC; ASan+UBSan;
                  ThreadSanitizer; Debug con _GLIBCXX_ASSERTIONS; y un trabajo
                  que compila y prueba el backend CUDA en un runner sin GPU
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

# 2. Compilar el proyecto (--parallel importa: la suite está repartida
#    en varias unidades de traducción precisamente para aprovecharlo)
cmake --build build --parallel

# 3. Ejecutar los ejemplos
./build/cpp_ai_engine   # Fase 1: tensores, strides y MatMul
./build/autograd_demo   # Fase 2: backpropagation y regresión lineal
./build/nn_demo         # Fase 3: MLP que clasifica una espiral de 3 clases
./build/cnn_demo        # Fase 4: CNN que clasifica formas en imágenes
./build/transformer_demo # Fase 5: Transformer sobre una tarea que exige orden

# 4. Ejecutar las pruebas
ctest --test-dir build --output-on-failure
# o directamente:
./build/test_engine

# 5. Medir el rendimiento
cmake --build build --target bench --parallel && ./build/bench
```

### Usar la librería desde otro proyecto

```bash
cmake --install build --prefix /donde/quieras
```

```cmake
find_package(cpp_ai_engine REQUIRED)
target_link_libraries(mi_app PRIVATE engine::engine)
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

### Redes convolucionales

```cpp
#include "engine/conv.hpp"

nn::Sequential cnn{
    nn::make<nn::Conv2d>(1, 8, nn::Window2d(3, 3, 1, 1)),   // (N,1,12,12) -> (N,8,12,12)
    nn::make<nn::ReLU>(),
    nn::make<nn::MaxPool2d>(2, 2),                          // -> (N,8,6,6)
    nn::make<nn::Conv2d>(8, 16, nn::Window2d(3, 3, 1, 1)),  // -> (N,16,6,6)
    nn::make<nn::ReLU>(),
    nn::make<nn::MaxPool2d>(2, 2),                          // -> (N,16,3,3)
    nn::make<nn::Flatten>(),                                // -> (N,144)
    nn::make<nn::Linear>(144, 3)
};
```

`Window2d(kernel_h, kernel_w, stride, padding)` describe la ventana deslizante;
el tamaño de salida es `(dim + 2*padding - kernel) / stride + 1`.

`cnn_demo` clasifica tres formas (barra horizontal, barra vertical y cruz)
dibujadas en posiciones aleatorias sobre fondo con ruido. La posición varía a
propósito: es donde la invariancia a la traslación de una convolución se
separa de una capa densa, con un número de parámetros comparable.

```
CNN (1683 parametros) : 100.00% sobre el conjunto de prueba
MLP (1779 parametros) :  73.33% sobre el conjunto de prueba
```

### Transformer

```cpp
#include "engine/transformer.hpp"

nn::Embedding embedding(vocab, 32);
nn::TransformerBlock block(/*d_model=*/32, /*heads=*/4, /*ff_hidden=*/64);
Tensor pe = nn::positional_encoding(seq_len, 32);

Tensor h = embedding(ids) + pe;   // (B, S, 32); la suma difunde (S, 32)
h = block(h);                     // atención + red densa, con residuales

// Con máscara causal, para que ninguna posición vea el futuro:
Tensor mask = nn::causal_mask(seq_len);
h = block.forward(h, &mask);
```

`transformer_demo` resuelve la tarea *«¿qué token viene después de la marca?»*.
Cada secuencia contiene una permutación de los seis valores más una marca, así
que **el multiconjunto de tokens es siempre el mismo**: promediar la secuencia
no deja ninguna información y un modelo sin atención está en el azar por
construcción.

```
Transformer (2 bloques)   : 99.25% sobre prueba
Promedio de embeddings    : 17.50% sobre prueba
Azar (1 de 6 valores)     : 16.67%
```

Hacen falta **dos** bloques porque la tarea es de dos saltos: uno marca cada
posición con «mi anterior es la marca» y el otro recoge esa posición desde el
`[CLS]`. El demo imprime los pesos de atención aprendidos, donde se ve una
cabeza concentrando ~0.9 sobre la posición que contiene la respuesta.

### Guardar y cargar un modelo

```cpp
#include "engine/serialize.hpp"

engine::save_parameters(model, "modelo.bin");
// ... en otra ejecución, sobre la misma arquitectura:
engine::load_parameters(model, "modelo.bin");
```

Los tensores se casan **por nombre**, no por posición, y la forma se comprueba
al cargar: un fichero de otro modelo se rechaza en vez de dejar la red
silenciosamente rota. `inspect_parameters(ruta)` enumera lo que hay dentro sin
tocar ningún modelo.

### Regularización y control del entrenamiento

```cpp
nn::Sequential model{
    nn::make<nn::Linear>(128, 256),
    nn::make<nn::GELU>(),        // también Sigmoid, Tanh y ReLU
    nn::make<nn::Dropout>(0.1f),
    nn::make<nn::Linear>(256, 10)
};

optim::Adam opt(model.parameters(), 0.001f);
optim::WarmupCosineLR scheduler(opt, /*calentamiento=*/5, /*total=*/50);

for (size_t epoch = 0; epoch < 50; ++epoch) {
    model.train();                                  // Dropout activo
    for (/* cada mini-lote */) {
        opt.zero_grad();
        loss.backward();
        optim::clip_grad_norm(model.parameters(), 1.0f);
        opt.step();
    }
    scheduler.step();

    model.eval();                                   // Dropout desactivado
    // ... evaluar
}
```

`clip_grad_norm` recorta por la norma **global** de todos los gradientes
juntos, así que limita la longitud del paso sin cambiar su dirección.

### Manipulación de tensores

```cpp
Tensor s = x.sum(1);           // reduce el eje 1
Tensor m = x.mean(0, true);    // keepdim deja el eje a 1
Tensor mx = x.max(2);          // el gradiente va solo al ganador

Tensor parte = x.slice(0, 2, 3);              // filas [2, 5)
Tensor junto = Tensor::concat({a, b}, 1);     // por columnas
Tensor pila  = Tensor::stack({a, b});         // eje nuevo al principio
```

Los cuatro operadores aritméticos admiten difusión por sufijo, y un tensor de
un elemento actúa como escalar sobre cualquier forma.

---

## ⚡ Notas de Rendimiento

Todas estas decisiones salieron de medir, no de suponer.

**El backward libera los gradientes intermedios según los consume.** En orden
topológico inverso, al llegar a un nodo su gradiente ya está completo; una vez
propagado a los padres nadie más lo necesita. Conservarlos hasta que muriese el
grafo multiplicaba por **24** la memoria de un backward (+27,6 MB frente a
+1,1 MB en un `TransformerBlock` de prueba).

**`Conv2d` guarda la entrada, no las columnas.** Las columnas de `im2col` ocupan
`kH*kW` veces la entrada — nueve veces con un kernel 3×3. Recalcularlas en el
backward cuesta un **+5 % de tiempo** y ahorra un **10×** de memoria (20 MB → 2 MB
en una convolución 16→16 sobre un lote de 32 imágenes de 32×32). Es el mismo
compromiso que hace PyTorch.

**Los pesos de atención no se guardan salvo que se pidan** con
`keep_attention(true)`: es una copia de `(B, H, S, S)` en cada paso que durante
el entrenamiento nadie mira. Quitarla además aceleró el demo un 5 %.

**El grafo cuesta unas 6 veces la inferencia.** El mismo forward ocupa 3,9 MB
bajo `NoGradGuard` y 25 MB construyendo grafo. Es inherente al diseño: hay que
retener las activaciones para derivar.

**El núcleo de `matmul` iza los punteros de fila y los marca `restrict`.** Sin
esa promesa de no solapamiento el compilador no vectoriza el acumulador. En
cambio la comprobación `a_ik == 0` **se conserva** aunque impida vectorizar esa
rama: las matrices que llegan a `matmul` suelen ser salidas de ReLU con la
mitad de los valores en cero exacto, y saltárselos gana más de lo que cuesta.
Medido de las dos formas sobre el ejemplo del Transformer: sin la rama 18,7 s,
con ella 15,9 s. Un microbenchmark con datos densos decía justo lo contrario —
por eso las cifras salen de los ejemplos reales.

**La difusión recorre por bloques, no con un módulo por elemento.** `rhs[i %
inner]` en el bucle interno impide vectorizar: la suma difundida costaba 3,44 ms
sobre un millón de valores frente a 0,56 ms de la suma normal. Recorriendo el
bloque repetido baja a 0,40 ms, **8,6× más rápido**.

`bench/bench.cpp` reproduce todas estas cifras (`cmake --build build --target
bench && ./build/bench`). No se ejecuta en CI: los tiempos de un runner
compartido no son comparables entre ejecuciones.

Sobre el tiempo de compilación: `engine/tensor.hpp` la incluye todo, así que
solo trae lo imprescindible (0,64 s → 0,34 s por unidad de traducción).
`<random>` vive en `engine/random.hpp` y `TensorImpl` en `engine/detail/`. Se
probaron también *unity build* y cabeceras precompiladas: **ambos empeoran** la
compilación en paralelo en un proyecto de este tamaño, y se descartaron.

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

**La convolución es un producto matricial.** `im2col` aplana cada ventana del
volumen de entrada en una fila, de modo que `Conv2d` se reduce a `(M, K) x (K,
outC)` en lugar de siete bucles anidados. Su derivada es `col2im`, que reparte
el gradiente de cada ventana a los píxeles que la formaron **sumando** allí
donde las ventanas se solapan; por eso col2im no es la inversa de im2col sino
su adjunto, y las pruebas lo comprueban con la identidad
`<im2col(x), y> == <x, col2im(y)>`.

**La atención se compone de operaciones existentes.** `scaled_dot_product_attention`
no necesita nodo propio en el grafo: es `matmul` por lotes, `transpose`,
`softmax` y una suma. Lo que hizo falta fue generalizar el tensor —`permute`
para reordenar ejes, `matmul` con lotes (y con un operando 2D compartido),
`softmax` sobre el último eje y difusión por sufijo— y la atención salió de ahí
sin escribir una sola derivada nueva. `LayerNorm`, en cambio, sí es un nodo
fusionado: su derivada arrastra dos términos de corrección porque la media y la
varianza dependen de todo el vector.

**Solo las hojas acumulan gradiente.** `add_grad` suma en lugar de sobrescribir,
así que hay que llamar a `zero_grad()` en cada iteración (igual que en PyTorch).
Los nodos intermedios, en cambio, se reinician al empezar cada `backward()`: su
gradiente es un valor temporal del recorrido, y conservarlo haría que un segundo
`backward()` sobre el mismo grafo propagase la suma de ambos recorridos.

---

## ✅ Pruebas

La suite cubre tensores, autograd, capas densas, convoluciones,
atención y optimizadores con **524 comprobaciones** (589 al compilar con
`-DENGINE_CUDA=ON` sobre una máquina con tarjeta, que añade las de paridad
CPU/GPU), y se ejecutan en CI sobre GCC, Clang, AppleClang y MSVC, más
AddressSanitizer, UBSan, ThreadSanitizer y una compilación en Debug. El grueso
de la verificación de autograd es una **comprobación
numérica de gradientes** por diferencias centradas, que compara cada derivada
analítica con `(f(x+h) - f(x-h)) / 2h`; también se verifica que los nodos
intermedios del grafo se liberen al salir de ámbito, que `col2im` sea el adjunto
exacto de `im2col`, y que un MLP resuelva el XOR donde un modelo lineal no puede.

Un detalle sobre la comprobación numérica: el tensor de ponderación de la
pérdida debe construirse **fuera** del closure. Si se genera dentro con
`randn`, cada evaluación usa pesos distintos y la comprobación deja de comparar
la misma función consigo misma —da errores enormes que parecen fallos del
motor.
