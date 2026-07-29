# Profiling the kernels with Nsight Compute

Timing a kernel tells you **how long** it takes. Profiling it tells you **why**,
which is the only thing that lets you decide what to touch next.

This document is the exact commands and, above all, which metrics to look at —
the part that is almost never written down anywhere.

---

## Why there is a separate executable

`bench_matmul` exists for this. Handing the whole of `bench` to `ncu` means
waiting for it to profile dozens of unrelated launches to read one;
`bench_matmul` runs **one kernel on one shape** and nothing else:

```bash
bench_matmul --kernel=register --size=2048 --iters=10
```

`--iters` fixes the repetition count rather than measuring against a wall clock,
which is what you want under the profiler: each profiled launch costs
considerably more than a normal one.

---

## The commands

```powershell
# Quick summary in the terminal
ncu --set default .\build-cuda\Release\bench_matmul.exe --kernel=register --size=2048 --iters=5

# Full report, with roofline, to open in the Nsight Compute UI
ncu --set full -o profile_register .\build-cuda\Release\bench_matmul.exe --kernel=register --size=2048 --iters=5

# Only the metrics that matter, without the rest of the report
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread .\build-cuda\Release\bench_matmul.exe --kernel=register --size=2048 --iters=5
```

On Linux it is the same with the path changed to `./build-cuda/bench_matmul`.

To compare variants, profile all four and open the reports together: Nsight
Compute can put two profiles side by side (*Add Baseline*), which is the fastest
way to see what actually changed between one kernel and the next.

```powershell
foreach ($k in "naive","tiled","register","vectorized") {
    ncu --set full -o "profile_$k" .\build-cuda\Release\bench_matmul.exe --kernel=$k --size=2048 --iters=5
}
```

> Nsight Compute needs permission to read the hardware counters. On Windows the
> terminal has to be opened as administrator; on Linux, either `sudo` or set
> `NVreg_RestrictProfilingToAdminUsers=0` on the driver module.

---

## Which metrics to look at, and what they mean

| Metric | What it says |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | What fraction of **compute** peak is reached |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | What fraction of **memory** peak |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy, not the theoretical one |
| `launch__registers_per_thread` | Registers per thread — what caps occupancy |
| `l1tex__data_bank_conflicts_pipe_lsu_shared.sum` | Shared-memory bank conflicts |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | Divergence within a warp |

### How to read them together

**Compute high, memory low** → the kernel is where it should be for a large
matmul. What is left is tuning the inner loop.

**Memory high, compute low** → it is bandwidth bound. On a matmul of any decent
size that means the tiles are not reusing their data, not that the card is
falling short.

**Both low** → it is latency. Either there are not enough warps to hide the
waits, or there are dependency chains inside each thread. This is where
registers per thread and occupancy are worth reading.

**Bank conflicts above zero** → two threads in the same warp are asking for
different addresses in the same shared-memory bank and the read serialises. It
is fixed by changing the tile's layout: transposing it, or adding a padding
column.

### On occupancy

It is the most misread metric of the set. **Low occupancy is not a defect on its
own.** This engine's `register` kernel deliberately drops to half the occupancy
of the tiled one, because it spends some 80-100 registers per thread on its 64
accumulators. In exchange, each thread has far more independent work in flight.

The right question is not "what occupancy do I have?" but "do I have enough work
in flight to hide the latency?". If compute throughput is high at 50% occupancy,
occupancy is not the problem.

---

## The roofline section

`--set full` generates the roofline chart, which places the kernel on two axes:
arithmetic intensity (FLOP per byte moved) against achieved throughput.

For an RTX 3060 Ti the ridge point sits around **36 FLOP/byte** (≈16.2 TFLOP/s
of fp32 peak against ≈448 GB/s). An N×N×N matmul has an intensity of **N/6
FLOP/byte**, so:

| Shape | Intensity | Region |
|---|---|---|
| 512³ | ~85 FLOP/byte | compute bound |
| 2048³ | ~341 FLOP/byte | compute bound, with room to spare |

Sitting to the right of the ridge point and still far below the horizontal
ceiling is exactly the diagnosis that motivated the register tiling: there is
unused compute headroom, and the problem is inside the kernel.

`bench_matmul` prints both numbers on start-up, so the diagnosis is visible
before the profiler is even opened.

---

## What to do with the results

The measured figures go into the table in [CUDA.md](CUDA.md). It is worth
recording, for each variant: time, GFLOP/s, percentage of peak, registers per
thread and achieved occupancy. With those five columns the progression explains
itself — and if an optimisation did not deliver what it promised, that shows too,
which is just as useful.
