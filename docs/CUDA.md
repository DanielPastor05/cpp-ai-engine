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

The progression is only worth something if there is a ruler beside it. RTX 3060
Ti, 16 489 GFLOP/s of fp32 peak, 4096³, operands already resident so no transfer
is inside the timed region:

| kernel | GFLOP/s | % of peak | vs cuBLAS |
|---|---|---|---|
| `naive` | 898 | 5.4% | 10.3× behind |
| `tiled` | 1 178 | 7.1% | 7.9× |
| `register` | 6 871 | 41.7% | 1.35× |
| **`vectorized`** | **7 660** | **46.5%** | **1.21×** |
| `tensorcore` (tf32) | 5 200 | 31.5% | 1.78× |
| **cuBLAS** | **9 258** | **56.1%** | — |

**46.5% of peak, 1.21× behind cuBLAS** for the best hand-written kernel. On a
single matrix product, which is the flattering comparison — end to end on MNIST
the engine is **1.90× behind PyTorch**, and
[docs/PERFORMANCE.md](PERFORMANCE.md#against-pytorch-on-the-same-card-which-is-the-number-that-counts)
takes that gap apart. A kernel close to cuBLAS does not make a framework close
to PyTorch, and saying only the first would be choosing the number. The
register tiling is where the jump lives: 1 178 → 6 871 GFLOP/s, a factor of
**5.8**, for a change that touches no memory hierarchy at all — only how many
results each thread keeps in registers.

cuBLAS is linked into `bench_matmul` and **nowhere else**; the engine never calls
it. Its row is checked against the engine before any timing is believed, because
a reference computing something else is worse than no reference: it agrees to
1.8e-05, which is FMA rounding. (The first version of that check divided by
`|expected|` and reported 7.2e-03, which looked like a broken reference and was
catastrophic cancellation in the divisor.)

### The measurement was lying, and the fix is the interesting part

The numbers above come from `tools/bench_matmul_isolated.sh`, **not** from
`bench_matmul`'s built-in sweep, and the difference is not cosmetic.

The sweep runs every kernel back to back in one process. On a consumer card that
measures temperature as much as code: by the fifth row the clocks have dropped,
so a kernel is slower partly *because of where it sits in the table*. Measured
here, `vectorized` at 4096³ came out at **4 888 GFLOP/s inside the sweep and
7 660 on its own** — a factor of 1.6, larger than most of the differences the
table exists to show. Taking the best of three windows inside the sweep did not
fix it; it made it worse, because three windows generate three times the heat.

An earlier draft of this document reported 4 663 GFLOP/s and "1.94× behind
cuBLAS" from exactly that broken sweep. Both numbers were wrong, and the
direction of the error was systematic rather than random: every row was
pessimistic, and the later ones more so.

So the honest measurement is one process per kernel with a pause between, keeping
the fastest observation — microbenchmark noise only ever adds time. The sweep is
still there as an overview and now prints a warning saying its rows are not
comparable to each other.

### tf32 on the tensor cores: implemented, measured, and it loses

`tensorcore` is a full WMMA implementation — 128×128 block tile, 8 warps, each
owning a 32×64 slab as a 2×4 grid of 16×16×8 tf32 fragments, staged through
shared memory. It reaches 31.5% of peak. `vectorized`, on the ordinary fp32
pipes, reaches 46.5%. **The tensor cores lose by 1.5×, and no amount of tuning
the kernel was going to change that.**

The reason is the hardware, and it is worth knowing before reaching for WMMA at
all: on GA10x — consumer Ampere — **dense tf32 tensor throughput is the same
16.2 TFLOP/s as fp32**. The 2× that tensor cores are famous for applies to fp16,
or to tf32 with structured sparsity, or to the data-centre A100 whose tensor
cores run at full rate. On a 3060 Ti there is simply no arithmetic headroom for
a tf32 kernel to exploit, so it pays WMMA's fragment-staging overhead for nothing.

Getting there took three wrong turns, each of which is a real lesson:

1. **A 64×64 block tile, 4 warps.** 4 227 GFLOP/s — slower than the fp32 kernel
   it was meant to beat. Each staged value fed 32 outputs where the fp32 kernel's
   fed 64: half the data reuse. A faster multiplier does not help a kernel that
   is not multiply-bound. Doubling the tile to 128×128 fixed the reuse.
2. **Rounding after every fragment load.** `wmma::precision::tf32` fragments hold
   fp32 bits and the caller must round them, and the obvious place is right after
   `load_matrix_sync`. That is 24 conversions per lane per K-step against 8
   `mma_sync` instructions — the conversion outnumbered the arithmetic three to
   one. Moving it to staging time, once per value instead of once per load, was
   worth 20%.
3. **A leading dimension of 67.** `store_matrix_sync` requires the leading
   dimension to be a multiple of 4 floats. Storing fragments straight to global
   memory with `ld = N` is undefined behaviour when `N % 4 != 0` — and undefined
   here means plausible numbers in the wrong places, not a crash. The
   exact-integer parity case caught it on 65×33×67 while every tolerance case
   around it still passed.

That third one is why the parity suite has a case that compares tf32 against fp32
with **no tolerance at all**. tf32 represents integers up to 2048 exactly, so on
small-integer inputs the tf32 product must equal the fp32 product bit for bit.
Any error in the fragment indices, the staging, the edge padding or the warp
mapping survives no tolerance — and the tolerance cases, whose bounds are
`4·sqrt(K)·2⁻¹¹`, were demonstrably hiding one.

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

**A threshold is only half the rule, and the half that stops applying.** It
answers "is this worth moving the data across PCIe?", which is the right question
only while the data is on the host. Once the chain stays resident, refusing a
small kernel is what costs the round trip — the CPU path has to pull the buffer
down to run. So every dispatch takes the size threshold **or** device residency,
whichever says yes:

```cpp
bool elementwise_worth_it(const Storage& input, size_t n) {
    if (!enabled() || n == 0) return false;
    return input.resident_on_device() || n >= min_elementwise_elements();
}
```

This was not a refinement, it was a bug the numbers were hiding. MNIST's largest
tensor is 802 816 values against a default of 2²⁰, so **no elementwise operation
dispatched at all** and the chain broke after every matmul: 15.8 s on stock
settings against 3.4 s with the threshold forced down by hand. Lowering the
default instead would have traded one demo for another — the same change made
`transformer_demo` go from 28 s to 39, because its matrices are small and
genuinely are not worth a launch. The residency rule improved both at once
(MNIST 15.8 → 4.3 s, transformer 28.0 → 26.0) and needs no tuning.

`matmul` asks for **both** operands to be resident rather than either: with one
side on the host, dispatching would trade a download for an upload instead of
avoiding one. In a training loop both is the normal case, because the weights
stay on the device between steps once the optimiser updates them there.

---

## What is deliberately not on the GPU

- **`cross_entropy`.** Still on the host, and now for a measured reason rather
  than an unexamined one. On an MNIST step it runs on `(64, 10)` and costs
  **0.02 ms of an 11.65 ms step, moving zero bytes** — the logits are already
  where it needs them. A kernel would replace two hundredths of a millisecond
  with a launch. It gets one when a model makes it matter, and the LayerNorm
  floor below is the shape that argument takes.
- **Dropout.** The mask comes from the host RNG, so the whole layer stays on the
  CPU. Moving it means a device RNG and a decision about whether the mask has to
  match the CPU's run for run, which is a reproducibility question rather than a
  performance one.
- **Kernel fusion.** Every operation is one kernel and one trip to global
  memory, and the profiler now says exactly where that costs most:
  `grad_accumulate` runs **twenty times per training step** and holds 13.5% of
  GPU time, because every gradient accumulation is its own launch. PyTorch folds
  it into the backward operation that produced the gradient.
  `transpose_tiled` is another twelve launches per step.

  Doing the same here is not a kernel, it is a change to the dispatch contract:
  every backward operation would need an "accumulate into this instead of
  returning a new tensor" variant, and `cuda::ops::*` currently returns a `bool`
  and writes one output. That is the next real piece of work and it is a
  refactor, not an optimisation, which is why it is named here rather than
  half-started.
- **Streams and overlap.** Everything runs on the default stream and the copies
  are synchronising. Overlapping transfer with compute is the textbook next step
  and it is deliberately not taken, because the measurement does not support it:
  one MNIST step moves under 5 MiB in total and spends **1.54 ms of 11.65 inside
  kernels**. What is left is host-side dispatch, and overlapping copies does not
  make dispatch cheaper. It needs a workload where the transfers are the cost,
  and this engine does not have one yet.
- **fp16 mixed precision.** tf32 is implemented and gave nothing, because
  consumer Ampere runs dense tf32 at the fp32 rate. fp16 really is 2×, but it is
  a precision project — master weights, loss scaling — not a kernel.
- **cuBLAS, as a backend.** For the same reason there is no BLAS on the CPU path:
  the goal is to implement it, not to call it. It *is* linked into
  `bench_matmul` as the reference row, and the engine's best kernel lands 1.21×
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
