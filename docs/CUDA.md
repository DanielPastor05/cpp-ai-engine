# Fase 6: el backend CUDA

Cómo está construido el camino de GPU, qué decisiones tiene detrás y qué
deliberadamente no hace.

```bash
cmake -B build-cuda -S . -DENGINE_CUDA=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure   # incluye la paridad CPU/GPU
./build-cuda/bench                                # tabla CPU vs GPU
```

El backend está **apagado por defecto**. El motor tiene que seguir compilando y
pasando las 509 comprobaciones en una máquina sin toolkit ni tarjeta, que es lo
que hay en CI; la GPU es una aceleración opcional, no un requisito.

---

## Lo primero fue separar el almacenamiento del tensor

Antes del primer kernel, `TensorImpl` guardaba un `std::vector<float>`. Eso es
host y punto. Con esa estructura, cada operación habría acabado con ramas
host/dispositivo repartidas por todo `src/tensor.cpp`, y —lo que es peor— cada
kernel habría pagado una ida y vuelta por PCIe, porque no había dónde anotar
que un tensor ya estaba arriba.

`include/engine/detail/storage.hpp` introduce `Storage`: el búfer de host, un
espejo opcional en el dispositivo, y dos banderas de validez.

```cpp
mutable float* device_ = nullptr;
mutable bool host_valid_ = true;
mutable bool device_valid_ = false;
```

**Invariante: al menos una de las dos copias es válida en todo momento.** A
partir de ahí, quien pide un lado obsoleto paga la copia y nadie más se entera:

| Se pide | Qué ocurre |
|---|---|
| `host()` | baja del dispositivo si el host está obsoleto |
| `host_mut()` | igual, y además marca el dispositivo como obsoleto |
| `device()` | sube si el dispositivo está obsoleto |
| `device_mut()` | igual, y marca el host como obsoleto |
| `device_write()` | reserva **sin subir nada**, y marca el host como obsoleto |

`device_write()` es el que más tráfico ahorra: la salida de un kernel se
escribe entera, así que subir su contenido previo sería tirar ancho de banda.

El resultado es que una cadena de operaciones en GPU se queda en la GPU. La
prueba de paridad lo comprueba, y no de oídas:

```cpp
cuda::reset_transfer_stats();
Tensor D = A.matmul(B).relu();
(void)D.data()[0];
check(cuda::transfer_stats().to_device_count == 0,
      "una segunda operacion no resube operandos ya residentes");
```

Sin `ENGINE_CUDA` los tres miembros del dispositivo **ni se declaran**, así que
la compilación de CPU no paga nada por que exista el backend. La macro es
`PUBLIC` en CMake precisamente por esto: cambia la disposición de `Storage`, y
si la librería y quien la usa no coincidieran, el desajuste daría corrupción de
memoria en vez de un error de compilación.

---

## El contrato de despacho

Cada operación de GPU devuelve un `bool`: **true si se hizo cargo del trabajo**.

```cpp
if (!cuda::ops::matmul(impl_->storage, B.impl_->storage, C.impl_->storage,
                       batch, M, K, N, a_batched, b_batched)) {
    // ... el camino de CPU de siempre, intacto ...
}
```

Devolver false significa «no hay dispositivo», «a este tamaño no compensa» o
«esta forma no cabe en la geometría de lanzamiento». Sin CUDA, esas funciones
las define `src/cuda_disabled.cpp` devolviendo false y el enlazador las elimina.

Esto es lo que mantiene `src/tensor.cpp` legible: una condición por operación,
no dos implementaciones enredadas ni un `#ifdef` por función. Y tiene una
consecuencia práctica: **un fallo al lanzar un kernel no tumba el programa**, se
calcula en CPU y se sigue. Un motor que se cae porque la GPU está ocupada es
peor que uno que va más lento.

Ese camino de recuperación tenía una trampa que costó encontrar razonando sobre
el invariante. `device_write()` marca el host como obsoleto *antes* de lanzar el
kernel. Si el lanzamiento falla y se cae al camino de CPU, `matmul` pide el
búfer de host, `Storage` se lo baja del dispositivo **sin inicializar**, y como
`matmul` acumula sobre una salida que da por puesta a cero, el resultado es
basura. Por eso existe `revert_device_write()`, y por eso se llama en el mismo
sitio donde se detecta el fallo:

```cpp
bool launched_ok(const char* what, Storage& out) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    out.revert_device_write();   // el host vuelve a ser la copia buena
    ...
    return false;
}
```

---

## El producto de matrices, con teselas en memoria compartida

Es la operación que domina el perfil (53% del ejemplo del Transformer), así que
es la que justifica el trabajo.

Cada bloque calcula una tesela de 32×32 de la salida y la recorre a lo largo de
K. En cada paso los 1024 hilos del bloque cargan una tesela de A y otra de B a
memoria compartida, y después cada hilo hace sus 32 productos leyendo de ahí.

El motivo es el tráfico a memoria global. Sin teselas, cada elemento de A se lee
N veces y cada uno de B se lee M veces. Con teselas de lado T se leen N/T y M/T
veces: **32 veces menos**. La memoria compartida tiene un orden de magnitud más
de ancho de banda y una latencia mucho menor que la global, y ese cambio de
proporción es todo el kernel.

Dos detalles que no se ven en el resultado pero sí en el rendimiento y en la
corrección:

- **Los bordes se rellenan con ceros en vez de acortar el bucle.** Todos los
  hilos del bloque tienen que llegar a los mismos `__syncthreads()`; si los del
  borde salieran antes, la barrera sería inválida y el comportamiento quedaría
  sin definir. Rellenar con cero es correcto *y* más simple.
- **La memoria compartida no necesita relleno.** `As[ty][k]` es una difusión
  dentro del warp y `Bs[k][tx]` recorre bancos consecutivos: ninguno de los dos
  accesos genera conflictos de banco. Añadir una columna de relleno, que es el
  reflejo habitual, aquí sólo gastaría memoria compartida y reduciría la
  ocupación.

El softmax usa un bloque por fila con dos reducciones sobre memoria compartida
—primero el máximo, para restarlo y que la exponencial no se desborde, y
después la suma—, con `expf` y no `__expf`: la versión rápida ahorra unos ciclos
a cambio de precisión, y estos valores se comparan contra PyTorch en la prueba
de referencia.

---

## Paridad: por qué la comparación es con tolerancia

`tests/test_cuda_parity.cpp` calcula la misma expresión dos veces sobre
exactamente los mismos datos, una con el backend apagado y otra encendido.

Es la única forma de comprobar un kernel que sirva de algo: **los kernels no
fallan devolviendo un error, fallan devolviendo números plausibles**.

La comparación es con tolerancia relativa (~1e-5), no exacta, y eso no es una
concesión. El compilador de dispositivo funde multiplicación y suma en una sola
instrucción FMA, que redondea una vez donde la CPU redondea dos; la diferencia
está en el último bit y se acumula con K. Exigir igualdad bit a bit entre CPU y
GPU sería exigir que la GPU calcule *peor*.

Lo que sí se mantiene es el determinismo dentro de cada lado: el orden de
acumulación del kernel es fijo (k ascendente, el mismo que en CPU), así que dos
ejecuciones en GPU dan exactamente lo mismo. La garantía de la Fase 5 —resultado
idéntico bit a bit sea cual sea el número de hilos— sigue valiendo para la CPU.

Durante la prueba los umbrales se ponen a cero. Con los umbrales normales todos
estos casos irían a la CPU, que es justo lo contrario de lo que quiere una
prueba de paridad: interesa ejercitar formas pequeñas y **con restos** —17×23×31,
33×65×129—, porque los bordes de las teselas son quienes fallan. El último caso
encadena un `TransformerBlock` entero con su paso hacia atrás: las pruebas por
operación pueden pasar todas y el modelo dar otra cosa si una operación deja un
tensor en el lado equivocado y otra lo lee sin sincronizar.

---

## Los umbrales, otra vez

La misma lección de la Fase 5, en otra escala. Lanzar un kernel cuesta unos
microsegundos, así que por debajo de cierto tamaño la GPU pierde contra **un
solo núcleo** de CPU. Los valores por defecto:

| Operación | Umbral | Equivale a |
|---|---|---|
| `matmul` | 2²² operaciones | ~128³ |
| elemento a elemento | 2²⁰ elementos | 4 MiB |

Se cambian sin recompilar con `ENGINE_CUDA_MIN_FLOPS` y
`ENGINE_CUDA_MIN_ELEMENTS`, que es lo que permite barrerlos y encontrar el cruce
en una máquina concreta en lugar de heredar el mío. `ENGINE_CUDA=0` apaga el
backend entero sobre el mismo binario, para comparar de las dos formas sin
recompilar nada.

---

## Lo que deliberadamente no está en la GPU

- **`Conv2d`.** Tiene su propio bucle escrito a mano y no pasa por
  `Tensor::matmul`, así que el backend no lo ve. Es exactamente el mismo motivo
  por el que el primer intento de paralelismo en CPU no aceleró MNIST en
  absoluto (589 s contra 587 s). En `mnist_demo` las capas densas sí se van a la
  GPU y las convoluciones no, y el ejemplo lo dice al arrancar en lugar de dejar
  que el lector lo suponga. Reescribirlo como `im2col` + `Tensor::matmul` es la
  continuación natural, y está sin hacer, no pasada por alto.
- **Los tensores guardados para el paso hacia atrás bajan a host.** El jacobiano
  del softmax depende de su salida, así que el forward hace `res.detach()`, y
  `detach()` pide el lado host: una bajada por cada softmax aunque el resultado
  fuera a quedarse arriba. La solución es una copia que se quede en el
  dispositivo; la contabilidad de transferencias la delata en cuanto se mira.
- **Fusión de kernels.** Cada operación es un kernel y un viaje a memoria
  global. Un `LayerNorm` fusionado o un `attention` fusionado ahorrarían la
  mayor parte de ese tráfico. Es la siguiente optimización de verdad.
- **Streams y solapamiento.** Todo va en el stream por defecto y las copias son
  sincronizantes. Solapar transferencia y cálculo con memoria anclada es lo que
  atacaría directamente el coste de PCIe.
- **Tensor cores.** El motor es fp32 en todas partes. Usarlos exige fp16 o tf32
  y una discusión de precisión que este proyecto no ha tenido.
- **cuBLAS.** Por el mismo motivo que no hay BLAS en el camino de CPU: el
  objetivo es implementarlo, no llamarlo. Un `matmul` con teselas se queda muy
  por debajo de cuBLAS, y eso es información, no un defecto.

---

## Reproducir las medidas

```bash
cmake -B build-cuda -S . -DENGINE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/bench
```

La sección «CPU frente a GPU» imprime tres cosas, y la tercera es la que
importa:

1. `matmul` de 64³ a 2048³, CPU contra GPU, **con los operandos ya residentes**.
   Mide el kernel.
2. El coste de las transferencias: la misma operación con los datos arriba
   frente a la misma con ida y vuelta completa por iteración, que es lo que pasa
   en un bucle de entrenamiento donde la pérdida y el optimizador siguen en CPU.
3. El ancho de banda H2D y D2H medido, y el número de subidas y bajadas.

Se informan **por separado a propósito**. En un motor real el enlace PCIe es el
cuello de botella mucho antes que el cálculo, y una tabla CPU/GPU que esconda
ese coste dentro del total no dice nada útil sobre el motor: dice cuánto ocupa
la matriz.

La bajada incluye la espera al kernel, porque `cudaMemcpy` sincroniza. Eso no es
un defecto de la medida, es la medida: el coste real de leer un resultado desde
el programa.
