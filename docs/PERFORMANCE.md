# Performance notes

Every number here was measured, on this machine, with `bench/bench.cpp` or by
profiling the examples. Where a measurement contradicted my expectation, the
contradiction is recorded rather than quietly dropped.

Reproduce with:

```bash
cmake --build build --target bench --parallel && ./build/bench
```

---

## Where the time actually goes

Profile of `transformer_demo`, built the way CMake builds it (`-O3 -DNDEBUG`):

| Function | Share |
|---|---|
| `matmul` | 53% |
| `permute` | 10% |
| `operator+` | 8% |
| `Adam::step` | 5% |
| `transpose` | 5% |
| Tensor allocation (zero fill) | 4% |

A first profile at `-O2` attributed 69% to `matmul`. That build was not the one
the project ships — see the note on baselines below.

---

## Vectorisation

Two things stopped the compiler from vectorising the hot loops.

**Aliasing.** `matmul`'s accumulator `c_row[j] += a_ik * b_row[j]` cannot be
vectorised unless the compiler knows the pointers do not overlap. Hoisting the
row pointers out of the inner loop and marking them `restrict` fixed it.

**Accessors inside the loop.** The element-wise operators called `data()` and
`size()` per element — three function calls per element — which also blocked
vectorisation. `Tensor::data()` appeared in the profile with **615 million
calls**. Hoisting the references out gave the rest.

Result: `transformer_demo` from 17.4 s to 15.9 s.

## The zero check in `matmul` stays, and it is not obvious why

```cpp
const float a_ik = a_row[k];
if (a_ik == 0.0f) continue;
```

On dense matrices this branch costs about 40%: it blocks vectorisation, and
multiplying by zero is cheaper than branching. A micro-benchmark says remove it.

The micro-benchmark is wrong for this workload. Matrices reaching `matmul` in a
real network are frequently **ReLU outputs**, with roughly half their entries at
exactly zero — the second dense layer of every Transformer block, for example.
There the branch skips half the work.

Measured on the real example rather than on synthetic data:

| Variant | `transformer_demo` |
|---|---|
| Baseline (branch, no hoisting) | 17.4 s |
| Hoisted + `restrict`, no branch | 18.7 s |
| Hoisted + `restrict`, with branch | **15.9 s** |

---

## Broadcasting: 8.6× from removing a modulo

The broadcast loop indexed the right operand with `rhs[i % inner]`. A modulo per
element prevents vectorisation:

| | 1M elements |
|---|---|
| Plain addition | 0.56 ms |
| Broadcast addition, with modulo | 3.44 ms |
| Broadcast addition, block by block | **0.40 ms** |

Because tensors are contiguous, broadcasting over leading axes is just repeating
the trailing block, so the loop can walk blocks and index the operand directly.
It ends up faster than the plain addition, which reads more memory.

---

## Memory

| | Before | After |
|---|---|---|
| Backward pass over a `TransformerBlock` | +27.6 MB | **+1.1 MB** |
| `Conv2d` 16→16 over 32 images of 32×32 | 20 MB | **2 MB** |
| Attention weights per step | one `(B,H,S,S)` copy | none unless requested |

**Intermediate gradients are released as they are consumed.** In reverse
topological order a node's gradient is complete when reached, and once pushed to
its parents nothing reads it again. Holding them until the graph died was
costing 24×.

**`Conv2d` stores its input, not its `im2col` columns.** The columns are
`kH·kW` times larger than the input — nine times for a 3×3 kernel. Recomputing
them in the backward pass costs **+5% time for a 10× memory saving**. This is
the same trade-off PyTorch makes.

**Attention weights are opt-in** via `keep_attention(true)`. Storing a
`(B, H, S, S)` copy on every step that nobody reads during training also made
the example 5% slower.

For reference, the graph itself costs about **6×** over inference: the same
`TransformerBlock` forward pass takes 3.9 MB under `NoGradGuard` and 25 MB while
building a graph. That is inherent — activations have to be retained to
differentiate.

There is no leak: RSS is flat across 60 training iterations.

---

## CPU parallelism

`matmul` and the element-wise operators are split across a persistent thread
pool. Threads are created once and reused: creating one costs tens of
microseconds, more than many of the engine's operations, so spawning per call
would be a regression rather than an improvement.

**The split is deterministic by construction.** Each row of the output is
computed start to finish by a single thread, so no reduction crosses a chunk
boundary and the accumulation order never changes. Results are **identical bit
for bit** with one thread or with eight — asserted in the test suite, not just
claimed.

Measured on 4 cores:

| Threads | matmul 512³ | Speedup | Element-wise, 4M | Speedup |
|---|---|---|---|---|
| 1 | 18.97 ms | 1.00× | 4.60 ms | 1.00× |
| 2 | 9.87 ms | 1.92× | 2.83 ms | 1.62× |
| 3 | 7.08 ms | 2.68× | 2.86 ms | 1.61× |
| 4 | 6.16 ms | **3.08×** | 2.60 ms | **1.77×** |

Element-wise operations scale much worse, and that is expected: they are
limited by memory bandwidth, not by arithmetic. Adding cores does not add
bandwidth.

End to end on the flagship example — full MNIST, 60 000 images, six epochs:

| | Time | Test accuracy |
|---|---|---|
| 1 thread | 587 s | 99.35% |
| 4 threads | **329 s** (1.78×) | 99.35% |

The per-epoch losses are identical to the last digit across both runs, which is
the determinism guarantee holding in practice rather than in a unit test.

`Conv2d` had to be parallelised separately, because at the time it had its own
hand-written loop and did not go through `Tensor::matmul`. The first threaded
build left it alone and MNIST gained **nothing** — 589 s against 587 s — because
the convolutions dominate that workload.

### Composing `Conv2d` out of tensor operations, and what it cost

That hand-written loop is gone. `Conv2d` is now `im2col` followed by
`Tensor::matmul`, a broadcast bias, a permute and two reshapes — every one of
which already carries its own derivative and its own kernel. The layer's
hand-written `backward_fn` disappeared with it: autograd derives all three
gradients by composition. Roughly ninety lines deleted.

The point was to let the CUDA backend see convolutions at all, and it does. On
the bench shape (`Conv2d(8→16, 3×3)`, batch 16 of 16×16, forward + backward):

| | CPU | GPU |
|---|---|---|
| hand-written loop | 5.12 ms | 5.14 ms — the backend never saw it |
| composed | 6.78 ms | **2.85 ms** |

**1.8× against the original**, and the CNN chain `conv → relu → pool → conv` now
stays resident on the device end to end, which needed `im2col`, `col2im` and
`MaxPool2d` to get kernels of their own.

The honest other half: composition does **five passes over the output buffer
where the hand-written version did one**, and that is not free. Measured on the
MNIST subset shipped in the repo (2 000 images, 12 epochs):

| | Training |
|---|---|
| hand-written, CPU | **18.4 s** |
| hand-written, CUDA | 19.8 s |
| composed, CPU | 25.8 s (**+40%**) |
| composed, CUDA, default thresholds | 24.3 s |
| composed, CUDA, `ENGINE_CUDA_MIN_ELEMENTS=65536` | **19.5 s** |

So the GPU build came out slightly ahead and the CPU-only build paid 40%. The
cost was concentrated in `reshape`, which copied the whole buffer to change
nothing but the shape: two of those five passes were pure metadata changes.

**That table is the state before "Making the GPU actually win" below, kept
because the retraction is the interesting part.** `reshape` is now a view, and
after four further fixes the same subset trains in **3.8 s on the GPU against
26.3 s on the CPU**. The composed convolution is no longer the slower one; what
was slow was never the composition.

Two things measured along the way that did **not** work, recorded so nobody
repeats them. Replacing that `permute({0,2,1})` with `transpose()` — identical
semantics on a 3D tensor, and a specialised loop instead of per-element index
arithmetic — came out **5% slower**: it writes with a stride where permute writes
contiguously, and at this size the memory access pattern outweighs the
arithmetic. And giving the matrix product a kernel while `im2col` still ran on
the host was a **loss**, because the columns are `kH*kW` times the input and
uploading them costs more than multiplying them: 24.6 s against 19.0 s.

### The thresholds matter more than the parallelism

The first attempt used thresholds ten times lower and made the examples
**slower** — `transformer_demo` went from 15.9 s to 22.6 s. That example chains
many small matrix products, and each one was paying for synchronisation without
gaining anything.

Dispatching a parallel region costs about **7.8 µs with four threads**,
measured directly. The thresholds are derived from that number: a matmul is only
split above ~1M multiply-adds (~130 µs of work), so the overhead stays under 6%.
Below that it runs inline, and nested parallel regions always run inline.

The lesson generalises: for a threaded fast path, the interesting engineering is
in deciding *when not to use it*.

| Configuration | `-j1` | `-j4` |
|---|---|---|
| Baseline | 23.4 s | 10.1 s |
| Unity build | 17.7 s | 11.1 s ✗ |
| Precompiled headers | 32.6 s ✗ | 14.2 s ✗ |
| Header pruning + split test suite | 28.2 s | **8.7 s** |

**Unity builds and precompiled headers both make the parallel build slower** at
this project's size — unity reduces available parallelism, and PCH is generated
once per target and never amortises across six of them. Both were measured and
discarded.

What worked: `engine/tensor.hpp` is included by everything, so it carries only
what the declaration of `Tensor` needs (0.64 s → 0.34 s per translation unit).
`<random>` lives in `engine/random.hpp` and `TensorImpl` in `engine/detail/`.
The test suite was one 4.78 s translation unit — 25% of all compilation work and
the critical path — and is now split by area.

The serial build got *slower*, because each new translation unit pays for the
headers again. The parallel build is the one that matters, and the README
recommends `--parallel` accordingly.

Incremental rebuild after touching one test file: **5.2 s → 2.6 s**. That is the
number that matters day to day.

---

## End-to-end: MNIST

The full 60 000-image training set, six epochs, single-threaded:

| Epoch | Loss | Test accuracy | Elapsed |
|---|---|---|---|
| 1 | 0.2078 | 98.17% | 107 s |
| 3 | 0.0449 | 98.97% | 299 s |
| 6 | 0.0205 | **99.35%** | 587 s |

Roughly 1.6 ms per image for a forward and backward pass through a 207 000-parameter
CNN, on one CPU core with no BLAS. That is the number Phase 6 has to beat.

---

## Making the GPU actually win

For a long stretch this backend was well built and pointless: 620 parity checks,
four matmul variants, gradients verified against the CPU bit for bit — and MNIST
trained **slower with a card than without one**, 19.5 s against 18.4.

It now trains in **3.8 s against 26.3 s: 6.9×**, on the same 2 000-image subset,
same twelve epochs, same final accuracy (94.4% against 94.6%, and the loss curves
agree to four decimals). One training step, batch 64, went from 40.2 ms to 9.99 ms
while the CPU stayed at 69.9.

What is worth reading here is not the number, it is that **the first two things I
was sure were the cause were not**.

### The 606 MiB that did not matter

The obvious culprit was PCIe traffic. `src/optim.cpp` dispatched to the backend
zero times, so every step read `p.data()` and `grad.data()` for all 206 922
parameters — a full download and upload each, **606 MiB per training run from the
optimiser alone**. Adding `sgd_step`/`adam_step` kernels, device-side reductions
for `clip_grad_norm`, and fixing a duplicated broadcast loop in `operator+` cut
transfers per step from 6.22 MiB to 1.62.

Training time fell **2%**.

Transfers were real and worth removing, but MNIST at this size was never
transfer-bound. Instrumenting the step per phase said so immediately: forward
9.2 ms, backward 30.1, and 11 transfers totalling under 2 MiB between them.

### The profiler disagreed with all of it

`nsys` on 100 steps, and three kernels held **93.5%** of GPU time:

| kernel | share | calls | mean |
|---|---|---|---|
| `sum_over_axis` | 42.4% | 200 | 2.67 ms |
| `matmul_tiled` | 38.7% | 900 | 542 µs |
| `permute_gather` | 12.4% | 800 | 195 µs |

All three had the **same bug**, and it is not a bug you find by reading the
kernels — each one is correct, and each looks reasonable in isolation. They all
draw their parallelism from the size of the **output** rather than the amount of
work:

- `sum_over_axis` gives one thread per output. A convolution's bias gradient has
  one output per channel: **16 threads** looping 50 176 times each. Fixed with a
  block per output and a shared-memory tree — 534 ms to 5.9 ms, **90×**.
- `matmul_tiled` gives one block per output tile. A convolution's weight gradient
  is `(9 × 50176) × (50176 × 16)`: **one block** walking a K of fifty thousand.
  Fixed with split-K, chunk count fixed on the host so the result does not shift
  between runs — 488 ms to 62 ms, **7.9×**.
- `permute_gather` reads at whatever stride the permutation implies. For the one
  permutation this engine leans on — Conv2d turning `(N·oH·oW, C)` into
  `(N, C, oH·oW)` — that stride is the channel count, so consecutive threads land
  64 bytes apart and every 32-byte sector fetched carries four useful bytes.
  33 GB/s on a card that does about 400. Fixed by detecting the last-two-axes
  swap and staging a tile through shared memory — 154 ms to 9.8 ms, **15.7×**.

Total GPU time across the run: 1 259 ms to 159 ms.

The step went from 40.2 ms to 32.2.

### Where it actually was

159 ms over 100 steps is **1.6 ms of kernel time in a 32 ms step**. Ninety-five
per cent of a "GPU-accelerated" step was the host, and not doing anything
interesting: `Storage(count, 0.0f)` allocated and zeroed a host mirror for every
tensor an operation produced, immediately before a kernel overwrote all of it.

One MNIST step creates **89 MiB** of those mirrors. Timed on its own: **20.4 ms**.

Making the host allocation lazy — `count` and `fill` describe the buffer,
`materialise()` builds it the first time anyone reads it — took the step to
**9.99 ms**. A tensor that is only ever written and read by kernels now allocates
no host memory at all.

| | step, batch 64 |
|---|---|
| CPU | 69.9 ms |
| GPU, before any of this | 40.2 ms |
| + three kernel parallelism fixes | 32.2 ms |
| + lazy host mirror | **9.99 ms** |

### What this cost to find

Three measurements, in this order, each of which contradicted the plan:

1. Transfer counters per phase — killed the PCIe hypothesis the plan was built on.
2. `nsys` per-kernel times — found three unrelated-looking kernels with one shared
   design mistake.
3. A twelve-line micro-benchmark of `std::vector::assign` — found the thing that
   was bigger than all three kernels put together.

None of them needed a tool that was not already installed. The first two
hypotheses were mine and both were wrong; what made the difference was measuring
before each change rather than after.

`ncu` wants administrator rights for its performance counters on Windows.
`nsys profile -t cuda` does not, and per-kernel totals were all this needed.

---

## A note on baselines

An early benchmark showed a 4.4× speedup from `-O2` to `-O3` on `matmul`. Before
recommending it, I checked what the project actually compiles with: CMake's
`Release` configuration already passes `-O3 -DNDEBUG`. The speedup was an
artefact of the benchmark command, not an improvement available to the project.

Verify the baseline is the real one before quoting an improvement against it.

---

## GPU

The CUDA backend, its design and how to reproduce its measurements are in
**[docs/CUDA.md](CUDA.md)**. Two things worth repeating here:

- The transfer cost is measured and reported **separately** from kernel time.
  On a real engine the PCIe link becomes the bottleneck long before arithmetic
  does, and a CPU/GPU table that folds that cost into the total says nothing
  about the engine — it says how big the matrix was.
- `Conv2d` **does** go through the GPU now. It used to have its own hand-written
  loop and never called `Tensor::matmul`; it is now composed out of `im2col`,
  `matmul` and `permute`, which is what let the whole CNN chain get kernels.

---

## What is deliberately not done

- **No BLAS.** The point is to implement it, not to call it. `matmul` reaches
  ~15 GFLOP/s single-threaded; a tuned BLAS does 5-10× better on this hardware.
- **No cache tiling in `matmul`.** It would pay off on large matrices; the ones
  this engine handles are small enough that loop order and vectorisation
  dominate.
- **No `-march=native`.** Worth another 1.4×, at the cost of a binary that will
  not run on a different CPU.
- **Views for `transpose` and `permute`.** `reshape` is a view now; those two
  are not, and stay that way deliberately. They genuinely move data, and making
  them views means non-contiguous strides through every kernel and every CPU
  loop — a different project, with no payoff measured here.
