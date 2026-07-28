# Perfilar los kernels con Nsight Compute

Medir el tiempo dice **cuánto** tarda un kernel. Perfilarlo dice **por qué**, que es
lo único que permite decidir qué tocar a continuación.

Este documento son los comandos exactos y, sobre todo, qué métricas mirar — la parte
que casi nunca está escrita en ningún sitio.

---

## Por qué hay un ejecutable aparte

`bench_matmul` existe para esto. Pasarle `bench` entero a `ncu` significa esperar a
que perfile decenas de lanzamientos distintos para leer uno; `bench_matmul` ejecuta
**un kernel sobre una forma** y nada más:

```bash
bench_matmul --kernel=register --size=2048 --iters=10
```

`--iters` fija el número de repeticiones en lugar de medir por tiempo, que es lo que
conviene bajo el perfilador: cada lanzamiento perfilado cuesta bastante más que uno
normal.

---

## Los comandos

```powershell
# Resumen rápido en la terminal
ncu --set default .\build-cuda\Release\bench_matmul.exe --kernel=register --size=2048 --iters=5

# Informe completo, con roofline, para abrir en la interfaz de Nsight Compute
ncu --set full -o perfil_register .\build-cuda\Release\bench_matmul.exe --kernel=register --size=2048 --iters=5

# Sólo las métricas que interesan, sin el resto del informe
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread .\build-cuda\Release\bench_matmul.exe --kernel=register --size=2048 --iters=5
```

En Linux es lo mismo cambiando la ruta por `./build-cuda/bench_matmul`.

Para comparar variantes, perfila las cuatro y abre los informes juntos: Nsight Compute
sabe poner dos perfiles lado a lado (*Add Baseline*), que es la forma más rápida de ver
qué cambió de verdad entre un kernel y el siguiente.

```powershell
foreach ($k in "naive","tiled","register","vectorized") {
    ncu --set full -o "perfil_$k" .\build-cuda\Release\bench_matmul.exe --kernel=$k --size=2048 --iters=5
}
```

> Nsight Compute necesita permisos para leer los contadores del hardware. En Windows
> hay que abrir la terminal como administrador; en Linux, o bien `sudo`, o poner
> `NVreg_RestrictProfilingToAdminUsers=0` en el módulo del driver.

---

## Qué métricas mirar, y qué significan

| Métrica | Qué dice |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Qué fracción del pico de **cálculo** se alcanza |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Qué fracción del pico de **memoria** |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Ocupación real, no la teórica |
| `launch__registers_per_thread` | Registros por hilo — lo que limita la ocupación |
| `l1tex__data_bank_conflicts_pipe_lsu_shared.sum` | Conflictos de banco en memoria compartida |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | Divergencia entre hilos del warp |

### Cómo se leen juntas

**Cálculo alto y memoria baja** → el kernel está donde debe estar para un matmul
grande. Lo que queda es afinar el bucle interno.

**Memoria alta y cálculo bajo** → está limitado por ancho de banda. En un matmul de
tamaño decente eso significa que las teselas no están reutilizando los datos, no que
la tarjeta se quede corta.

**Los dos bajos** → es latencia. O no hay bastantes warps para tapar las esperas, o
hay dependencias en cadena dentro de cada hilo. Aquí es donde miran los registros por
hilo y la ocupación.

**Conflictos de banco distintos de cero** → dos hilos del mismo warp están pidiendo
direcciones distintas del mismo banco de memoria compartida y la lectura se serializa.
Se arregla cambiando la disposición de la tesela: transponerla, o añadirle una columna
de relleno.

### Sobre la ocupación

Es la métrica más malinterpretada. **Ocupación baja no es un defecto por sí sola.**
El kernel `register` de este motor baja a propósito a la mitad de ocupación respecto
al de teselas, porque gasta unos 80-100 registros por hilo en los 64 acumuladores. A
cambio, cada hilo tiene mucho más trabajo independiente en vuelo.

La pregunta correcta no es «¿qué ocupación tengo?» sino «¿tengo bastante trabajo en
vuelo para tapar la latencia?». Si el rendimiento de cálculo es alto con ocupación
del 50%, la ocupación no es el problema.

---

## La sección de roofline

`--set full` genera el diagrama roofline, que sitúa el kernel en dos ejes: intensidad
aritmética (FLOP por byte movido) contra rendimiento alcanzado.

Para una RTX 3060 Ti el punto de inflexión está sobre los **36 FLOP/byte**
(≈16,2 TFLOP/s de pico fp32 contra ≈448 GB/s). Un matmul de N×N×N tiene una intensidad
de **N/6 FLOP/byte**, así que:

| Forma | Intensidad | Región |
|---|---|---|
| 512³ | ~85 FLOP/byte | limitado por cálculo |
| 2048³ | ~341 FLOP/byte | limitado por cálculo, con holgura |

Estar a la derecha del punto de inflexión y aun así lejos del techo horizontal es
exactamente el diagnóstico que motivó el teselado de registros: hay margen de cálculo
sin usar, y el problema está dentro del kernel.

`bench_matmul` imprime esos dos números al arrancar, así que el diagnóstico está a la
vista antes incluso de abrir el perfilador.

---

## Qué hacer con los resultados

Las cifras medidas van a la tabla de [CUDA.md](CUDA.md). Conviene apuntar, para cada
variante: tiempo, GFLOP/s, porcentaje del pico, registros por hilo y ocupación
alcanzada. Con esas cinco columnas, la progresión se explica sola — y si alguna
optimización no dio lo que prometía, también se ve, que es igual de útil.
