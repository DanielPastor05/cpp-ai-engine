# Phase 6: the CUDA backend

How the GPU path is built, what decisions sit behind it, and what it
deliberately does not do.

```bash
cmake -B build-cuda -S . -DENGINE_CUDA=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure   # includes CPU/GPU parity
./build-cuda/bench                                # CPU vs GPU table
```

The backend is **off by default**. The engine has to keep compiling and passing
its 524 checks on a machine with no toolkit and no card, which is what CI has;
the GPU is an optional acceleration, not a requirement.

---

## The first thing was separating storage from the tensor

Before the first kernel, `TensorImpl` held a `std::vector<float>`. That is host
and nothing else. With that structure every operation would have ended up with
host/device branches spread through `src/tensor.cpp` and — worse — every kernel
would have paid a round trip over PCIe, because there was nowhere to record that
a tensor was already up there.

`include/engine/detail/storage.hpp` introduces `Storage`: the host buffer, an
optional device mirror, and two validity flags.

```cpp
mutable float* device_ = nullptr;
mutable bool host_valid_ = true;
mutable bool device_valid_ = false;
```

**Invariant: at least one of the two copies is valid at all times.** From there,
whoever asks for a stale side pays for the copy and nobody else notices:

| Asked for | What happens |
|---|---|
| `host()` | downloads if the host copy is stale |
| `host_mut()` | the same, and marks the device stale |
| `device()` | uploads if the device copy is stale |
| `device_mut()` | the same, and marks the host stale |
| `device_write()` | allocates **without uploading anything**, and marks the host stale |

`device_write()` is the one that saves the most traffic: a kernel's output is
written in full, so uploading its previous contents would be bandwidth thrown
away.

The result is that a chain of GPU operations stays on the GPU. The parity test
checks it, and not by hearsay:

```cpp
cuda::reset_transfer_stats();
Tensor D = A.matmul(B).relu();
(void)D.data()[0];
check(cuda::transfer_stats().to_device_count == 0,
      "a second operation does not re-upload already resident operands");
```

Without `ENGINE_CUDA` the three device members are **not even declared**, so a
CPU build pays nothing for the backend existing. The macro is `PUBLIC` in CMake
precisely because of this: it changes `Storage`'s layout, and if the library and
its consumer disagreed, the mismatch would give memory corruption rather than a
compile error.

---

## The dispatch contract

Every GPU operation returns a `bool`: **true if it took the work**.

```cpp
if (!cuda::ops::matmul(impl_->storage, B.impl_->storage, C.impl_->storage,
                       batch, M, K, N, a_batched, b_batched)) {
    // ... the usual CPU path, untouched ...
}
```

Returning false means "no device", "not worth it at this size", or "this shape
does not fit the launch geometry". Without CUDA those functions are defined in
`src/cuda_disabled.cpp` returning false, and the linker removes them.

This is what keeps `src/tensor.cpp` readable: one condition per operation, not
two tangled implementations and not an `#ifdef` per function. And it has a
practical consequence: **a kernel failing to launch does not bring the program
down**, it computes on the CPU and carries on. An engine that crashes because
the GPU is busy is worse than one that runs slower.

That recovery path had a trap that took reasoning about the invariant to find.
`device_write()` marks the host stale *before* the kernel launches. If the
launch fails and it falls back to the CPU path, `matmul` asks for the host
buffer, `Storage` pulls it off the device **uninitialised**, and since `matmul`
accumulates into an output it assumes is zeroed, the result is garbage. That is
why `revert_device_write()` exists, and why it is called in the same place the
failure is detected:

```cpp
bool launched_ok(const char* what, Storage& out) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    out.revert_device_write();   // the host is the good copy again
    ...
    return false;
}
```

### The failure that check does not see

`cudaGetLastError()` reports **launch** errors: an invalid grid, a binary with
no code for the card, too much shared memory. A fault *inside* the kernel body —
an out-of-range access — does not appear there. The launch is asynchronous: by
the time the error exists, the call has already returned `cudaSuccess`. The
error sticks to the context and surfaces at the next synchronising operation,
which is usually a `cudaMemcpy` three operations later and is in no way to
blame. The symptom is a message pointing at the wrong place.

Catching it at the guilty launch means synchronising right after, and that is a
barrier per kernel when the engine launches hundreds per step: it would mean
paying in production for a diagnostic that only matters while chasing a fault.
So it sits behind an environment variable, off by default:

```bash
ENGINE_CUDA_SYNC=1 ./build-cuda/test_engine
```

With it, `launch_ok()` calls `cudaDeviceSynchronize()` before looking at the
status, and the error comes out naming the kernel that caused it. The recovery
path is the usual one: report once and compute on the CPU.

---

## The matrix product, with shared-memory tiles

It is the operation that dominates the profile (53% of the Transformer example),
so it is the one that justifies the work.

Each block computes a 32×32 tile of the output and walks it along K. At each
step the block's 1024 threads load one tile of A and one of B into shared
memory, and then each thread does its 32 products reading from there.

The reason is global memory traffic. Without tiles, each element of A is read N
times and each of B M times. With tiles of side T they are read N/T and M/T
times: **32 times fewer**. Shared memory has an order of magnitude more
bandwidth and far lower latency than global, and that change of ratio is the
whole kernel.

Two details that do not show in the result but do show in performance and in
correctness:

- **Edges are padded with zeros rather than shortening the loop.** Every thread
  in the block has to reach the same `__syncthreads()`; if the edge threads left
  early, the barrier would be invalid and the behaviour undefined. Padding with
  zero is correct *and* simpler.
- **Shared memory needs no padding.** `As[ty][k]` is a broadcast within the warp
  and `Bs[k][tx]` walks consecutive banks: neither access produces bank
  conflicts. Adding a padding column, which is the usual reflex, would only
  spend shared memory here and reduce occupancy.

Softmax uses one block per row with two reductions over shared memory — first
the maximum, to subtract it so the exponential does not overflow, and then the
sum — with `expf` rather than `__expf`: the fast version saves a few cycles in
exchange for precision, and these values are compared against PyTorch in the
reference test.

---

## Optimising the matmul: from tiling to the ceiling

The tiled kernel above is the textbook one, and it falls well short of the
card's ceiling. What is interesting is **why**, because both intuitive answers
are wrong.

It is not occupancy: with 1024 threads and 8 KB of shared memory per block there
are plenty of warps in flight. And it is not global memory traffic: the tiles
already fixed that, cutting it by a factor of 32.

It is **arithmetic intensity at the register level**. In `matmul_tiled`, each
thread, for each step of K, does:

> **1 FMA against 2 shared-memory reads.**

The load units saturate long before the arithmetic ones, and at that ratio it
makes no difference how many warps are waiting: the bottleneck is the inner loop
itself.

The fix is for each thread to compute a **block** of results instead of a single
one. With 8×8 outputs live in registers, each thread reads 8 values of A and 8
of B per step of K and does 64 products with them:

> **64 FMA against 16 shared-memory reads.** From 1:2 to 4:1 — eight times better.

### The four variants

| Variant | What changes | FMA : shared reads |
|---|---|---|
| `naive` | no shared memory | 1 : 2, and against **global** memory |
| `tiled` | 32×32 tiles, one result per thread | 1 : 2 |
| `register` | 128×128 blocks, 8×8 results per thread in registers | **4 : 1** |
| `vectorized` | the same, with `float4` loads from global to shared | 4 : 1, with 4× fewer load instructions |

All four stay alive in the binary and are selected with
`cuda::set_matmul_kernel`, or with `--kernel=` in the benchmark. This is not
indecision: **the progression is the result**, and it also makes it possible to
parity-check each one separately, which is what actually protects the work.

### The details that matter

**`As` is stored transposed** in shared memory (index `[k][m]`). Without that,
each thread's eight reads of A would go with stride K and each would land in a
different bank.

**Occupancy halves, and that is intentional.** 64 accumulators plus working
registers comes to some 80-100 registers per thread. Fewer resident warps, yes —
but the instruction-level parallelism inside each thread more than makes up for
it. It is this kernel's classic trade-off and it shows up immediately under the
profiler, so it is worth knowing it was put there on purpose.

**Vectorisation demands alignment.** `K` and `N` must be multiples of 4, or the
addresses do not land on multiples of 16 bytes. If that does not hold, the
dispatch degrades to `register` silently — a misaligned `float4` read does not
raise an error, **it returns a different value**, which is considerably worse. It
is the same kind of alignment-driven kernel selection cuBLAS does internally, and
there is a parity test dedicated to that path.

**Bounds are checked only where they are needed**: on the global-to-shared load
(padding with zeros) and on the final store. The inner loop runs with no checks
at all, which is what lets shapes with remainders be correct without penalising
the hot path.

### Measured, against cuBLAS

The progression is only worth something if there is a ruler beside it. Measured on
an RTX 3060 Ti (16 489 GFLOP/s fp32 peak), square products, operands already
resident on the device so no transfer is inside the timed region:

| Shape | `naive` | `tiled` | `register` | `vectorized` | **cuBLAS** |
|---|---|---|---|---|---|
| 512³ | 462 | 443 | 552 | 663 | **3 301** |
| 1024³ | 499 | 701 | 1 463 | 1 637 | **7 304** |
| 2048³ | 721 | 910 | 2 712 | 2 862 | **8 844** |
| 4096³ | — | 1 078 | 4 582 | **4 663** | **9 043** |

GFLOP/s. At 4096³ that is 28.3% of peak for the engine's best kernel against
54.8% for cuBLAS: **1.94x behind, or 52% of cuBLAS's throughput.**

Two things are worth reading off that table rather than the headline. The
register tiling is where the jump happens — 910 to 4 582 GFLOP/s at 4096³, a
factor of five for a change that touches no memory hierarchy, only how many
results each thread keeps in registers. And the gap to cuBLAS *narrows* with
size: 5x at 512³, 1.9x at 4096³, because the small shapes are dominated by launch
overhead and tile quantisation that cuBLAS's shape-specialised kernels handle and
a single 128x128 geometry does not.

`bench_matmul` prints this table, and it prints cuBLAS's agreement with the
engine before any of the timings, because a reference row that computed something
else would make the whole comparison worse than no comparison. The first version
of that check divided by `|expected|` and reported 7.2e-03, which looked like a
broken reference and was really catastrophic cancellation in the divisor; with
the parity test's `max(1, |expected|)` scale it reports 1.8e-05, which is FMA
rounding and nothing else.

cuBLAS is linked into `bench_matmul` and **nowhere else**. The engine never calls
it. Not measuring against it, which is what this project did until now, reads as
avoiding the comparison — and losing by 1.9x with the reason visible is a much
better answer than declining to look.

### The roofline says where to attack

Before optimising it is worth knowing which ceiling you are hitting. An N×N×N
product moves `3N²` values to do `2N³` operations: **N/6 FLOP/byte**. An RTX
3060 Ti's ridge point sits around 36 FLOP/byte (≈16.2 TFLOP/s against ≈448
GB/s), so any N above a few hundred is squarely in the **compute-bound** region.

That is what justifies everything above: at this shape the work is in the
kernel's arithmetic intensity, not in the transfers. `bench_matmul` prints the
ridge point and each shape's intensity so the decision is visible rather than
assumed.

### Checking the indices without a GPU

CI has no card, so the CUDA job only compiles. That catches syntax errors and
nothing else: **an indexing error compiles perfectly happily** and shows up weeks
later as wrong results.

`tests/test_cuda_indexing.cpp` covers that gap. It reproduces the kernel's
structure with loops — block grid, 256 threads, shared memory, barriers — using
the same index expressions, and compares against a reference product over eleven
shapes chosen for their remainders: 1×1×1, 127×128×129, 256×260×256, 130×4×130.
It runs on any machine and on every push.

The price is that the expressions are written twice and have to be maintained in
parallel. That is accepted knowingly: the alternative was having no check on the
indices at all until reaching a machine with a GPU, and by then the error is
already committed.

What it does not cover, and why device parity is still needed: races between
threads, real shared-memory coherence, `float4` load alignment and, obviously,
performance.

How to measure it and what to look at: **[docs/PROFILING.md](PROFILING.md)**.

---

## Parity: why the comparison is to a tolerance

`tests/test_cuda_parity.cpp` computes the same expression twice over exactly the
same data, once with the backend off and once with it on.

It is the only way to check a kernel that is worth anything: **kernels do not
fail by returning an error, they fail by returning plausible numbers**.

The comparison is to a relative tolerance (~1e-5) rather than exact, and that is
not a concession. The device compiler fuses multiply and add into a single FMA
instruction, which rounds once where the CPU rounds twice; the difference is in
the last bit and accumulates with K. Demanding bit-identical results between CPU
and GPU would be demanding that the GPU compute *worse*.

What does hold is determinism within each side: the kernel's accumulation order
is fixed (k ascending, the same as on the CPU), so two GPU runs give exactly the
same thing. The Phase 5 guarantee — identical bit for bit whatever the thread
count — still holds for the CPU.

During the test the thresholds are set to zero. At the normal thresholds all
these cases would go to the CPU, which is the opposite of what a parity test
wants: the point is to exercise small shapes **with remainders** — 17×23×31,
33×65×129 — because the tile edges are what fail. The last case chains a whole
`TransformerBlock` with its backward pass: the per-operation tests can all pass
and the model still give something else if one operation leaves a tensor on the
wrong side and another reads it without synchronising.

---

## The thresholds, again

The same lesson as Phase 5, at another scale. Launching a kernel costs a few
microseconds, so below a certain size the GPU loses to **a single CPU core**. The
defaults:

| Operation | Threshold | Equivalent to |
|---|---|---|
| `matmul` | 2²² operations | ~128³ |
| element-wise | 2²⁰ elements | 4 MiB |

They are changed without recompiling via `ENGINE_CUDA_MIN_FLOPS` and
`ENGINE_CUDA_MIN_ELEMENTS`, which is what allows sweeping them and finding the
crossover on a particular machine instead of inheriting mine. `ENGINE_CUDA=0`
turns the whole backend off on the same binary, to compare both ways without
recompiling anything.

---

## What is deliberately not on the GPU

- **The loss and the optimisers.** `cross_entropy` and the SGD and Adam steps
  are still on the CPU, and they are now the remaining break in MNIST: one
  download of every gradient and one upload of every parameter per step. It is
  the next thing to attack, because the chain ahead of it no longer breaks.
- **`LayerNorm`.** It is the one that is left, and it breaks the chain twice per
  block. The forward would have a kernel without difficulty; the one that
  decides is the backward, because `dgamma` and `dbeta` accumulate **across**
  rows and that cross-row reduction is a design problem of its own — the same
  one that keeps the CPU version serial. An accelerated forward with the
  backward on the host solves half of it, so it gets done whole or not at all.
- **`Tensor::sum()` to a scalar.** The accumulator is `double` on purpose, and a
  two-stage GPU reduction in `float` would lose exactly what that change fixed.
  Doing it properly means accumulating in `double` on the device too, which is
  not hard but is not free either.
- **Kernel fusion.** Every operation is one kernel and one trip to global
  memory. A fused `LayerNorm` or a fused `attention` would save most of that
  traffic. It is the next real optimisation.
- **Streams and overlap.** Everything runs on the default stream and the copies
  are synchronising. Overlapping transfer and compute with pinned memory is what
  would attack the PCIe cost directly.
- **Tensor cores.** The engine is fp32 everywhere. Using them requires fp16 or
  tf32 and a precision discussion this project has not had.
- **cuBLAS, as a backend.** For the same reason there is no BLAS on the CPU path:
  the goal is to implement it, not to call it. It *is* linked into
  `bench_matmul` as the reference row, and the engine's best kernel lands 1.94x
  behind it at 4096³ — measured above rather than asserted.

---

## Reproducing the measurements

```bash
cmake -B build-cuda -S . -DENGINE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/bench
```

The "CPU versus GPU" section prints three things, and the third is the one that
matters:

1. `matmul` from 64³ to 2048³, CPU against GPU, **with the operands already
   resident**. That measures the kernel.
2. The cost of the transfers: the same operation with the data up there against
   the same one with a full round trip per iteration, which is what happens in a
   training loop where the loss and the optimiser are still on the CPU.
3. The measured H2D and D2H bandwidth, and the number of uploads and downloads.

They are reported **separately on purpose**. In a real engine the PCIe link is
the bottleneck long before the arithmetic is, and a CPU/GPU table that hides
that cost inside the total says nothing useful about the engine: it says how big
the matrix is.

The download includes waiting for the kernel, because `cudaMemcpy` synchronises.
That is not a defect of the measurement, it is the measurement: the real cost of
reading a result from the program.
