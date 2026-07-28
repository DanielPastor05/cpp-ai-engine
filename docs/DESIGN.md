# Design notes

Why the engine is built the way it is. Each section states a decision, the
alternative that was rejected, and the consequence.

---

## `Tensor` is a handle; `TensorImpl` is the node

`Tensor` holds nothing but a `shared_ptr<TensorImpl>`. Copying a `Tensor` shares
its data, its gradient and its history.

This is what makes `Module::parameters()` work. It returns tensors *by value*,
yet writing to them updates the layer's weights, because both refer to the same
implementation:

```cpp
optim::Adam opt(model.parameters(), 0.001f);   // copies, but shares storage
opt.step();                                     // updates the real weights
```

The cost is that assignment is aliasing, not copying. `Tensor::detach()` exists
for when a genuine copy is needed.

`TensorImpl` lives in `include/engine/detail/tensor_impl.hpp` rather than the
main header: it drags in `<functional>`, which every translation unit would
otherwise pay for.

---

## The buffer is a `Storage`, not a `std::vector<float>`

`TensorImpl` used to hold the vector directly. That is host memory and nothing
else, which is fine right up to the moment a second kind of memory exists.

`Storage` owns a host buffer plus an optional device mirror and two validity
flags, with one invariant: **at least one of the two copies is valid at all
times**. Asking for a stale side pays for the copy; asking for the fresh one
costs a flag check.

The reason to do this *before* writing any kernel is that the alternative is
worse in two ways. Without a place to record that a tensor is already on the
device, every kernel pays a round trip over PCIe. And without the buffer behind
a type, host/device branching spreads through every operation in
`src/tensor.cpp` instead of living in one file.

Without `ENGINE_CUDA` the device members are not even declared, so the CPU build
pays nothing for the backend existing. That makes the macro part of the ABI —
it changes the layout of `Storage` — which is why CMake exports it as `PUBLIC`.
A library and its consumer disagreeing here would corrupt memory rather than
fail to compile.

The same split is what would make `reshape`, `transpose` and `permute` into
views instead of copies: shape, strides and offset are already separate from
the buffer. That has not been done.

---

## The graph is acyclic in reference counting, not just in topology

A node holds `shared_ptr`s to its **parents** and never to its children. That
single rule is what allows the graph to be freed at all.

It has a direct consequence for the backward function's signature. The obvious
formulation captures the output tensor:

```cpp
// Wrong: node → lambda → node is a shared_ptr cycle, never collected
res.impl_->backward_fn = [self, res]() { self.add_grad(res.grad()); };
```

Instead the gradient arrives as an argument:

```cpp
std::function<void(const Tensor&)> backward_fn;
```

so the lambda captures only inputs, and ownership stays one-directional. This
was a real bug — see [ENGINEERING.md](ENGINEERING.md#1-a-reference-cycle-meant-the-computation-graph-was-never-freed).

---

## Only leaves accumulate gradient

A gradient means two different things depending on where it sits:

- On a **leaf** (a parameter) it accumulates, which is what makes gradient
  accumulation across mini-batches possible and why `zero_grad()` exists.
- On an **intermediate node** it is a temporary value belonging to one
  traversal of the graph.

`backward()` therefore clears intermediate gradients before propagating, and
releases each one as soon as it has been consumed. In reverse topological order
a node's gradient is complete by the time it is reached, and once pushed to its
parents nothing reads it again.

Same semantics as PyTorch, and a 24× reduction in the peak memory of a backward
pass.

---

## The backward pass must not build a graph

Computing `dA = dC · Bᵀ` uses the same `matmul` and `transpose` as the forward
pass, so without care each backward pass would register a second-order graph
that nobody ever uses.

`autograd::NoGradGuard` is a thread-local RAII switch that turns graph
construction off. `backward()` and every optimiser's `step()` enable it
internally. It is also the right tool for inference:

```cpp
{
    engine::autograd::NoGradGuard no_grad;
    Tensor logits = model(X);      // ~6× less memory than the training path
}
```

---

## Topological sort is iterative on purpose

A recursive depth-first traversal is shorter to write and blows the process
stack on deep graphs. `src/autograd.cpp` uses an explicit stack of
`(node, next-parent-index)` pairs.

---

## `col2im` is the adjoint of `im2col`, not its inverse

`im2col` flattens each sliding window into a row, which reduces convolution to
a single matrix product `(M, K) × (K, outC)` instead of seven nested loops.

Its derivative scatters each row back to the pixels that formed it, **summing**
wherever windows overlap. That sum is exactly what makes it the correct
derivative, and it is why `col2im(im2col(x)) != x`.

Because that is easy to get subtly wrong and hard to notice, the tests assert
the defining property directly over four window geometries:

```
⟨im2col(x), y⟩ == ⟨x, col2im(y)⟩
```

---

## Attention needed no new derivatives

`scaled_dot_product_attention` is batched `matmul`, `transpose`, `softmax` and
an addition. Not one new backward function was written for it.

What it *did* require was generalising the tensor:

| Operation | Generalisation |
|---|---|
| `matmul` | batching over leading axes, plus a shared 2-D operand |
| `transpose` | swaps the last two axes at any rank |
| `permute` | arbitrary axis reordering, inverse permutation as derivative |
| `softmax` | over the last axis at any rank |
| `operator+` | suffix broadcasting |

`LayerNorm` is the exception and *is* a fused node: mean and variance depend on
the whole vector, so its derivative carries two correction terms that do not
decompose into existing ops.

---

## Broadcasting is a suffix rule

The right-hand operand broadcasts if, after dropping leading ones, its shape is
a **suffix** of the left-hand shape. One rule covers every case the engine
needs:

| Case | Shapes |
|---|---|
| Dense layer bias | `(N,)` over `(M, N)` |
| Positional encoding | `(S, D)` over `(B, S, D)` |
| Attention mask | `(S, S)` over `(B, H, S, S)` |
| Scalar | `(1,)` over anything |

Because tensors are contiguous in C order, broadcasting over leading axes is
just repeating the trailing block — which also means the loop can run block by
block instead of computing a modulo per element. That detail is worth 8.6×; see
[PERFORMANCE.md](PERFORMANCE.md).

The gradient of a broadcast operand is the sum over its repetitions.

---

## Numerical stability

- `softmax` subtracts the row maximum. Without it, `exp(1000)` overflows.
- `cross_entropy_loss` fuses log-softmax and NLL into a single node. Chaining
  `Softmax` then `log` loses precision when a probability underflows; the fused
  version uses the shifted log-sum-exp identity and its gradient collapses to
  `(softmax(logits) - one_hot) / N`.
- `Sigmoid` switches formula by sign, so neither tail overflows.
- Attention scores are scaled by `1/√d_k` before the softmax, otherwise the dot
  product grows with dimension and saturates it.

---

## Serialisation matches by name, not by position

Tensors in a checkpoint are keyed by name, so adding a layer to the end of a
model does not invalidate an existing file. Shapes are always verified on load:
a checkpoint from a different model is **rejected** rather than silently
producing a broken network.

The format is deliberately plain — magic, version, then per tensor a name,
shape and `float32` payload — and little-endian is required rather than
assumed: a big-endian machine is rejected explicitly instead of reading numbers
backwards.

This same format is what `tools/generate_reference.py` writes from PyTorch, so
the reference fixtures are readable by the engine with no extra machinery.
