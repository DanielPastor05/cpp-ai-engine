# C++ AI Engine desde Cero

Un motor de Inteligencia Artificial y Aprendizaje Profundo (Deep Learning) avanzado construido desde cero en **C++17** con backend de aceleración GPU en **CUDA**.

## 🚀 Hoja de Ruta del Proyecto

- [x] **Fase 1: Librería de Tensores (CPU)**
  - Arreglo plano contiguo en memoria (`data`).
  - Cálculo de pasos/zancadas (*strides*) en orden C (*row-major*).
  - Indexación multidimensional a plano 1D.
  - Operaciones elemento a elemento (Suma, Resta, Multiplicación Hadamard, División, ReLU).
  - Multiplicación Matricial (`MatMul`) optimizada para la caché CPU ($i \to k \to j$).
- [ ] **Fase 2: Motor de Autograd**
  - Grafos dirigidos acíclicos (DAG) de cómputo.
  - Diferenciación automática en modo inverso (*Reverse-mode Automatic Differentiation*).
  - Algoritmo de ordenamiento topológico (*Topological Sort*) para `.backward()`.
- [ ] **Fase 3: Capas y Optimizadores**
  - Abstracción `nn::Module` (`Linear`, `ReLU`, `Softmax`, `CrossEntropyLoss`).
  - Optimizadores `SGD` y `Adam`.
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

# 3. Ejecutar el ejemplo de la Fase 1
./build/cpp_ai_engine
```
