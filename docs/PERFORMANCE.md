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

`Conv2d` had to be parallelised separately: it has its own hand-written loop and
does not go through `Tensor::matmul`. The first threaded build left it alone and
MNIST gained **nothing** — 589 s against 587 s — because the convolutions
dominate that workload. Its backward pass is now split into two passes over
disjoint axes (`m` for `dcols`, output channel for `dW` and `db`) rather than one
pass with shared accumulators, which keeps it both race-free and deterministic.

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
- `Conv2d` does **not** go through the GPU, for exactly the reason it did not
  benefit from CPU threading either: it has its own hand-written loop and never
  calls `Tensor::matmul`. In `mnist_demo` the dense layers dispatch to the
  device and the convolutions do not, and the example says so on startup.

---

## What is deliberately not done

- **No BLAS.** The point is to implement it, not to call it. `matmul` reaches
  ~15 GFLOP/s single-threaded; a tuned BLAS does 5-10× better on this hardware.
- **No cache tiling in `matmul`.** It would pay off on large matrices; the ones
  this engine handles are small enough that loop order and vectorisation
  dominate.
- **No `-march=native`.** Worth another 1.4×, at the cost of a binary that will
  not run on a different CPU.
- **Tensor views.** `reshape`, `transpose` and `permute` copy. Making them
  share storage is the largest remaining optimisation and the natural companion
  to a device abstraction — deferred deliberately, not overlooked.
