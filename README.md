# cpp-ai-engine

A deep learning engine written from scratch in C++17 — tensors, reverse-mode
autodiff, CNNs and Transformers — with **every gradient validated against
PyTorch**.

[![CI](https://github.com/DanielPastor05/cpp-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielPastor05/cpp-ai-engine/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![tests](https://img.shields.io/badge/tests-509%20checks-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

No dependencies. No BLAS. The point is to implement it, not to call it.

*[Versión en español](README.es.md)*

---

## Results

| Task | This engine | Baseline | Note |
|---|---|---|---|
| **MNIST** (CNN, 60k images) | **99.35%** | — | real data, 6 epochs, ~10 min on one core |
| Shapes at random positions | **100%** (CNN, 1 683 params) | 73.3% (MLP, 1 779 params) | translation invariance |
| Interleaved 3-class spiral | **99.3%** (MLP) | 52.7% (linear) | non-linear separability |
| "Token after the marker" | **99.3%** (Transformer) | 17.5% (mean pooling) | chance is 16.7% |
| **Gradient agreement with PyTorch** | **~1e-7** | — | 23 fixtures, single ops to full blocks |
| MNIST training on 4 cores | **1.78×** (329 s vs 587 s) | 1 core | identical loss to the last digit |
| matmul 512³ on 4 cores | **3.08×** | 1 core | bit-identical regardless of thread count |

Each baseline is a control that *provably cannot* solve its task. If one ever
starts succeeding, the experiment is broken — not the model.

---

## What makes this different

**Gradients are validated against PyTorch, not just against themselves.**
Numerical gradient checking proves internal consistency; it does not catch a
convention that is wrong in both the forward and the backward pass.
`tools/generate_reference.py` builds the same computation in PyTorch, dumps
inputs, weights, outputs and gradients into the engine's own binary format, and
`tests/test_reference.cpp` replays them. Coverage runs from `matmul` up to a
complete `TransformerBlock` with all 16 of its parameters, and a 10-step Adam
trajectory. Everything agrees to ~1e-7. The fixtures are committed, so CI checks
them without needing Python.

**Performance work is measured, and the measurements sometimes disagreed with
me.** A micro-benchmark said removing a branch from `matmul` would give 1.7×;
removing it made the Transformer 12% *slower*, because the matrices reaching it
are ReLU outputs that are half zeros. Both results are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md), including the optimisations I
measured and discarded.

**Multi-threading that does not change the answer.** `matmul` and the
element-wise operators split across a persistent thread pool, and the split is
by output row — so the accumulation order never changes and results are
**identical bit for bit** whatever the thread count. The first attempt at this
made the examples *slower*; the thresholds are derived from a measured 7.8 µs
dispatch cost. Set with `ENGINE_NUM_THREADS` or `parallel::set_num_threads`.

**Real bugs, found and fixed.** A `shared_ptr` cycle that leaked the entire
computation graph. A repeated `backward()` that multiplied gradients. A heap
overflow that AddressSanitizer caught and every test missed. Each is written up
with symptom, diagnosis and fix in [docs/ENGINEERING.md](docs/ENGINEERING.md),
and each has a regression test.

---

## Quick start

```bash
cmake -B build -S . && cmake --build build --parallel
ctest --test-dir build --output-on-failure     # 509 checks
./build/mnist_demo                             # trains a CNN on real data
```

The repository ships a 2 000-image MNIST subset so this works straight after
cloning. Run `tools/download_mnist.sh` for the full 60 000 — the example detects
which one is present.

---

## What's inside

```
include/engine/
  tensor.hpp       Tensor (handle) + N-dimensional operations
  autograd.hpp     backward(), NoGradGuard
  nn.hpp           Module, Linear, activations, Dropout, Sequential, losses
  conv.hpp         im2col/col2im, Conv2d, MaxPool2d, Flatten
  transformer.hpp  LayerNorm, Embedding, attention, MultiHeadAttention, blocks
  optim.hpp        SGD, Adam, gradient clipping, LR schedulers
  parallel.hpp     Deterministic multi-threading over the hot loops
  serialize.hpp    Save and load weights
  data.hpp         IDX/MNIST reader
examples/          Six runnable demos, one per phase plus MNIST
tests/             509 checks across six translation units + PyTorch fixtures
bench/             Reproducible performance benchmarks
tools/             PyTorch fixture generator, MNIST downloader
```

Design decisions and their trade-offs: **[docs/DESIGN.md](docs/DESIGN.md)**.

---

## Usage

### Tensors and autodiff

```cpp
#include "engine/tensor.hpp"
using engine::Tensor;

Tensor a({1}, {2.0f}, /*requires_grad=*/true);
Tensor b({1}, {3.0f}, true);

Tensor L = (a * b) + a.relu();   // L = a*b + relu(a)
L.backward();

a.grad().data()[0];   // 4.0  ->  dL/da = b + 1
b.grad().data()[0];   // 2.0  ->  dL/db = a
```

Implicit `backward()` is only allowed on a scalar. For any other root, supply
the seed: `y.backward(Tensor(y.shape(), 1.0f))`.

For inference, `NoGradGuard` skips graph construction entirely — about 6× less
memory:

```cpp
{
    engine::autograd::NoGradGuard no_grad;
    Tensor logits = model(X);
}
```

### Training a network

```cpp
#include "engine/nn.hpp"
#include "engine/optim.hpp"

engine::manual_seed(42);                       // reproducible

nn::Sequential model{
    nn::make<nn::Conv2d>(1, 16, nn::Window2d(3, 3, 1, 1)),
    nn::make<nn::ReLU>(),
    nn::make<nn::MaxPool2d>(2, 2),
    nn::make<nn::Flatten>(),
    nn::make<nn::Dropout>(0.25f),
    nn::make<nn::Linear>(16 * 14 * 14, 10)
};

optim::Adam opt(model.parameters(), 0.001f);
optim::CosineAnnealingLR scheduler(opt, /*epochs=*/6);

for (size_t epoch = 0; epoch < 6; ++epoch) {
    model.train();                             // Dropout active
    for (/* each mini-batch */) {
        opt.zero_grad();
        Tensor loss = nn::cross_entropy_loss(model(X.select_rows(idx)), y);
        loss.backward();
        optim::clip_grad_norm(model.parameters(), 5.0f);
        opt.step();
    }
    scheduler.step();

    model.eval();                              // Dropout off
    float acc = nn::accuracy(model(X_test), y_test);
}

engine::save_parameters(model, "model.bin");
```

Checkpoints are matched **by name**, and shapes are verified on load: a file
from a different model is rejected rather than silently producing a broken
network.

### Transformers

```cpp
#include "engine/transformer.hpp"

nn::Embedding embedding(vocab, 32);
nn::TransformerBlock block(/*d_model=*/32, /*heads=*/4, /*ff_hidden=*/64);
Tensor pe = nn::positional_encoding(seq_len, 32);

Tensor h = block(embedding(ids) + pe);         // the sum broadcasts (S, 32)

Tensor mask = nn::causal_mask(seq_len);        // no position sees the future
h = block.forward(h, &mask);
```

Attention needed **no new derivatives** — it composes batched `matmul`,
`transpose`, `softmax` and an addition. What it required was generalising the
tensor to N dimensions.

---

## Examples

```bash
./build/mnist_demo        # CNN on real MNIST digits + save/load round-trip
./build/cnn_demo          # CNN vs MLP on shapes drawn at random positions
./build/transformer_demo  # attention on a task that needs word order
./build/nn_demo           # MLP on an interleaved spiral
./build/autograd_demo     # backpropagation from first principles
./build/cpp_ai_engine     # tensors, strides and matmul
./build/bench             # performance benchmarks
```

`transformer_demo` prints the attention weights it learned. One head puts 0.94
on the position holding the answer:

```
Attention from [CLS] in the second block:
  Head 0: 0.00  0.00  0.00* 0.00  0.14  0.74  0.11  0.02
  Head 3: 0.00  0.06  0.94* 0.00  0.00  0.00  0.00  0.00
                     ^ the position containing the answer
```

---

## Roadmap

- [x] **Phase 1 — Tensor library** · strides, row-major indexing, cache-friendly matmul
- [x] **Phase 2 — Autograd** · DAG, reverse mode, iterative topological sort, `NoGradGuard`
- [x] **Phase 3 — Layers and optimisers** · `nn::Module`, SGD/Adam, schedulers, gradient clipping
- [x] **Phase 4 — CNNs** · im2col/col2im, `Conv2d`, `MaxPool2d`, `Flatten`
- [x] **Phase 5 — Transformers** · scaled dot-product attention, multi-head, `LayerNorm`
- [ ] **Phase 6 — CUDA backend** · host/device memory, custom kernels, shared-memory tiling

---

## Testing

509 checks over six translation units, run on every push against **GCC, Clang,
AppleClang and MSVC**, plus a job under **AddressSanitizer and
UndefinedBehaviorSanitizer** one under **ThreadSanitizer**, and one in
**Debug** with standard-library assertions enabled.

Three layers of verification:

1. **Against PyTorch** — 23 committed fixtures, agreement to ~1e-7.
2. **Numerical gradient checking** — every operation compared against
   `(f(x+h) - f(x-h)) / 2h`, weighted by a non-uniform upstream gradient so an
   indexing error cannot cancel itself out.
3. **Structural properties** — `col2im` is asserted to be the exact adjoint of
   `im2col` through `⟨im2col(x), y⟩ == ⟨x, col2im(y)⟩`; a `weak_ptr` proves the
   computation graph is released.

```bash
ctest --test-dir build --output-on-failure
python3 tools/generate_reference.py      # regenerate fixtures (needs PyTorch)
```

---

## Using it from another project

```bash
cmake --install build --prefix /where/you/want
```

```cmake
find_package(cpp_ai_engine REQUIRED)
target_link_libraries(my_app PRIVATE engine::engine)
```

---

## License

MIT — see [LICENSE](LICENSE).
