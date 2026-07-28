# Engineering log

Bugs that were actually found and fixed while building this engine, and two
cases where a measurement proved my reasoning wrong. Every one of them is
covered by a regression test today.

---

## 1. A reference cycle meant the computation graph was never freed

**Symptom.** Nothing visibly broke. Training produced correct results. But
memory grew on every epoch and never came back down.

**Diagnosis.** Every operation registered its backward function as a lambda
that captured the output tensor by value:

```cpp
Tensor res_copy = res;
res.impl_->backward_fn = [self_copy, other_copy, res_copy]() mutable {
    if (self_copy.requires_grad()) self_copy.add_grad(res_copy.grad());
    //                                                ^^^^^^^^^
};
```

`Tensor` is a handle holding a `shared_ptr<TensorImpl>`. So the node owned a
lambda, and that lambda owned the node: `node → backward_fn → node`. A cycle of
`shared_ptr` is never collected. Every intermediate tensor of every forward pass
leaked, for the lifetime of the process.

**Fix.** `backward_fn` now takes the output gradient as a parameter instead of
capturing it:

```cpp
std::function<void(const Tensor&)> backward_fn;
```

The lambda captures only the *inputs*. Since a node already holds `shared_ptr`s
to its parents and never to its children, ownership flows in exactly one
direction and the graph is acyclic in the reference-counting sense too.

**How it is prevented now.** `test_graph_is_released` takes a `weak_ptr` to an
intermediate node, lets the enclosing scope end, and asserts the pointer
expired. An RSS probe also confirms memory is flat across 60 training
iterations.

---

## 2. Calling `backward()` twice multiplied the gradients

**Symptom.** Running backward a second time over the same graph gave gradients
that were too large — and grew with each additional call.

```
first  backward: dL/dx = 6    correct
second backward: dL/dx = 18   should still be 6
```

**Diagnosis.** Intermediate nodes kept the gradient accumulated during the
previous traversal. On the second pass, each one propagated *last time's
gradient plus this time's*, and the error compounded down the graph.

The reason it went unnoticed: a training loop builds a fresh graph every
iteration, so the bug only shows when the *same* graph is traversed twice —
which the tests were not doing.

**Fix.** Draw the distinction the engine was missing. A node's gradient means
two different things depending on what it is:

- **Leaves** (parameters) *accumulate* — that is what makes gradient
  accumulation across mini-batches work, and why `zero_grad()` exists.
- **Intermediate nodes** hold a temporary value belonging to one traversal.

`backward()` now discards the gradient of every node that has a `backward_fn`
before propagating. This matches PyTorch's semantics.

**Bonus.** The same insight yielded a large memory win: an intermediate
gradient can be released *as soon as it has been consumed*. In reverse
topological order, by the time a node is reached its gradient is complete, and
once pushed to the parents nobody needs it again. Peak memory of a backward
pass on a `TransformerBlock` dropped from **+27.6 MB to +1.1 MB — 24×**.

---

## 3. A heap buffer overflow in the broadcast gradient

**Symptom.** None. Every test passed. The forward pass produced correct
numbers.

**Diagnosis.** Found by AddressSanitizer during a review pass. The broadcast
planner computed how many times the right-hand operand repeats using a helper
whose `to` parameter treats `0` as "until the end":

```cpp
plan.repeat = product(base, 0, offset);   // offset == 0 → full product, not 1
```

When the right operand had more axes than the base but only leading ones —
`(1,3,4)` broadcast over `(3,4)` — `offset` was `0` and the helper returned
`12` instead of `1`.

The forward pass was unaffected because it only depends on `inner`. The
backward pass walked `repeat * inner` positions of the output gradient and read
**12× past the end of the buffer**.

**Fix.** Derive the repeat count from the total instead of a partial product:

```cpp
plan.repeat = product(base) / plan.inner;
```

**How it is prevented now.** A regression test covers broadcasting with leading
unit axes and scalar broadcasting, and the CI runs the whole suite under
AddressSanitizer and UndefinedBehaviorSanitizer on every push. This bug is
precisely why that job exists.

---

## 4. `MaxPool2d` silently returned negative infinity

With padding greater than or equal to the kernel size, some windows fell
entirely inside the padded region — no real value to take a maximum over. The
layer returned `-inf` without complaint, which would poison every downstream
computation.

Not reachable through the public API (`MaxPool2d(k, s)` sets padding to zero),
but reachable through the `Window2d` constructor. The constructor now rejects
it.

---

# When measurement proved me wrong

## The `-O2` vs `-O3` result that was not real

A matmul benchmark showed a **4.4× speedup** going from `-O2` to `-O3`
(2.99 → 13.41 GFLOP/s) — auto-vectorisation of the inner loop.

Before recommending the change, I checked what the project actually compiles
with. CMake's `Release` configuration already passes `-O3 -DNDEBUG`. The
speedup was an artefact of my hand-written benchmark command, not an
improvement available to the project.

**Lesson.** Verify the baseline is the real one before quoting an improvement
against it.

## The micro-benchmark that recommended a 12% slowdown

`matmul` contained what looked like a premature optimisation:

```cpp
const float a_ik = a_row[k];
if (a_ik == 0.0f) continue;
```

A micro-benchmark on dense random matrices was unambiguous: removing the branch
gave **1.7×**, because the branch blocks vectorisation and multiplying by zero
is cheaper than branching.

Removing it made the Transformer example **12% slower**.

The benchmark used dense random data. Real matrices reaching `matmul` are often
*ReLU outputs*, with roughly half their entries at exactly zero — the second
dense layer of every Transformer block, for instance. There the branch skips
half the work, and that outweighs the lost vectorisation.

Measured three ways on the real example, not on synthetic data:

| Variant | Time |
|---|---|
| Baseline (branch, no hoisting) | 17.4 s |
| Hoisted row pointers + `restrict`, **no** branch | 18.7 s |
| Hoisted row pointers + `restrict`, **with** branch | **15.9 s** |

The winning combination keeps the branch *and* adds the vectorisation hints.
The reasoning is recorded in a comment in `src/tensor.cpp` so nobody
"optimises" it away again.

**Lesson.** A micro-benchmark measures the data you gave it. Benchmark the
workload you actually have.

---

## What this project taught me about testing

Numerical gradient checking against centred differences catches a wrong chain
rule. It does *not* catch a wrong convention shared by both the forward and
backward pass. That is why the engine also validates against PyTorch: 23
fixtures generated by `tools/generate_reference.py`, covering everything from
`matmul` to a full `TransformerBlock` and a 10-step Adam trajectory, all
matching to ~1e-7.

The other habit worth keeping is **control experiments that must fail**. Every
demo pairs the model under test with a baseline that provably cannot solve the
task:

- A linear classifier on the spiral dataset — not linearly separable.
- An MLP on shapes drawn at random positions — no translation invariance.
- Mean-pooled embeddings on the sequence task — every sequence contains exactly
  the same multiset of tokens, so pooling is at chance *by construction*.

If the baseline ever starts succeeding, the experiment is broken, not the
model.
