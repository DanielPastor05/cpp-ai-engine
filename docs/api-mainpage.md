# cpp-ai-engine API reference {#mainpage}

Generated from the public headers under `include/engine/`. This is the index:
every type, every signature, the class relationships, jump-to-definition without
a checkout.

**It is not where the reasoning lives.** The headers carry long prose explaining
why each decision is what it is, and it reads better in the file than extracted
into a signature table. If you want to know *why* `Storage` was split out before
the first kernel was written, or why every CUDA operation returns `bool`, or why
the tensor-core variant is real and never selected automatically, read the
header — or the documents below, which are on GitHub.

| | |
|---|---|
| [Repository and README](https://github.com/DanielPastor05/cpp-ai-engine) | what the project is, and the measured results |
| [Design notes](https://github.com/DanielPastor05/cpp-ai-engine/blob/main/docs/DESIGN.md) | each decision with the alternative it rejected |
| [Performance notes](https://github.com/DanielPastor05/cpp-ai-engine/blob/main/docs/PERFORMANCE.md) | every number, including the optimisations that were measured and discarded |
| [The CUDA backend](https://github.com/DanielPastor05/cpp-ai-engine/blob/main/docs/CUDA.md) | kernels, the dispatch contract, and what is deliberately not on the GPU |
| [Engineering log](https://github.com/DanielPastor05/cpp-ai-engine/blob/main/docs/ENGINEERING.md) | the bugs that were actually hit, with the regression test each left behind |

## Where to start

`engine::Tensor` is a handle over a shared `TensorImpl`, so copying one shares
its data, its gradient and its history. Everything else follows from that and
from one invariant: **at least one of the host and device copies of a buffer is
valid at all times**, and nothing crosses PCIe until somebody asks for the side
that has gone stale.

- `engine::Tensor` — the type nearly everything returns
- `engine::autograd` — `backward()` and `NoGradGuard`
- `engine::nn` — layers, losses, `Sequential`
- `engine::cuda` — device queries, the matmul variants, occupancy

`engine::detail` is excluded from this index on purpose. `Storage` and the
`cuda::ops` entry points are implementation that happens to live in a header,
and listing them here would invite code to depend on them.
