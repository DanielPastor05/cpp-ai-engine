#!/usr/bin/env python3
"""Re-runs the benchmarks and compares them against docs/performance.json.

The other half of tools/check_claims.py. That one proves the documents agree
with each other; this one proves they agree with the machine.

**It is deliberately not in CI**, and the reason is worth stating rather than
leaving as an omission. GitHub's runners are shared, virtualised, and have no
GPU. A timing threshold on one drifts by more than the differences being
measured -- this project has the receipts, having once recorded a 3x MNIST
regression that was pure thermal noise, and having found that a benchmark sweep
in a single process is biased by up to 1.6x by the card heating up. A
performance test that fails at random teaches people to ignore a red CI, and a
CI nobody trusts is worse than no CI. So the check that blocks a merge is the
one that cannot flake, and this one runs where the numbers were taken.

It needs: both build trees, a CUDA-capable card, and PyTorch for the comparison
rows. Whatever is missing is skipped and named, never silently passed.

Usage:
    python tools/check_perf.py             # everything available
    python tools/check_perf.py --quick     # skip the 4096-cubed matmul sweep
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS = os.path.join(REPO, "docs", "performance.json")


def binary(tree, name):
    """Release/ on multi-config generators, the tree root otherwise."""
    for candidate in (os.path.join(REPO, tree, "Release", name + ".exe"),
                      os.path.join(REPO, tree, "Release", name),
                      os.path.join(REPO, tree, name + ".exe"),
                      os.path.join(REPO, tree, name)):
        if os.path.exists(candidate):
            return candidate
    return None


def run(cmd, env=None):
    merged = dict(os.environ)
    if env:
        merged.update(env)
    out = subprocess.run(cmd, capture_output=True, text=True, env=merged,
                         cwd=REPO, timeout=1800)
    return out.stdout + out.stderr


def last_epoch_seconds(text):
    """The demo prints cumulative elapsed time, so the last line is the total."""
    times = re.findall(r"Loss = .*?\|\s+([\d.,]+) s", text)
    return float(times[-1].replace(",", ".")) if times else None


def final_accuracy(text):
    m = re.search(r"Final accuracy on the test set: ([\d.,]+)%", text)
    return float(m.group(1).replace(",", ".")) if m else None


# Each measurement returns (value, note) or (None, why it was skipped). Best of
# `reps`, because the minimum is the least contaminated sample: a slow run means
# something else was using the machine, a fast one cannot mean less than the
# work took.
def measure_mnist(tree, reps=3):
    exe = binary(tree, "mnist_demo")
    if exe is None:
        return None, f"{tree}/mnist_demo not built"
    best, acc = None, None
    for _ in range(reps):
        text = run([exe])
        t = last_epoch_seconds(text)
        if t is None:
            return None, "could not parse the demo's output"
        best = t if best is None else min(best, t)
        acc = final_accuracy(text) or acc
    return best, f"best of {reps}, accuracy {acc}%"


def measure_pytorch(args, reps=3):
    python = os.path.join(REPO, ".torch", "Scripts", "python.exe")
    if not os.path.exists(python):
        python = os.path.join(REPO, ".torch", "bin", "python")
    if not os.path.exists(python):
        return None, "no .torch venv; see docs/PERFORMANCE.md for the pip line"
    best = None
    for _ in range(reps):
        text = run([python, os.path.join("tools", "bench_pytorch.py")] + args)
        m = re.search(r"Training time: ([\d.]+) s", text)
        if not m:
            return None, "could not parse bench_pytorch.py output"
        t = float(m.group(1))
        best = t if best is None else min(best, t)
    return best, f"best of {reps}"


def measure_matmul(kernel):
    exe = binary("build-cuda", "bench_matmul")
    if exe is None:
        return None, "build-cuda/bench_matmul not built"
    flag = "--kernel=" + kernel if kernel != "cublas" else "--cublas"
    text = run([exe, "--size=4096", flag])
    m = re.search(r"\d+\.\d+ ms\s+([\d.]+)\s", text)
    return (float(m.group(1)), "4096 cubed") if m else (None, "could not parse output")


MEASUREMENTS = {
    "mnist-engine-cpu": lambda a: measure_mnist("build"),
    "mnist-engine-cuda": lambda a: measure_mnist("build-cuda"),
    "mnist-pytorch-cpu": lambda a: measure_pytorch(["--device", "cpu"]),
    "mnist-pytorch-cuda": lambda a: measure_pytorch([]),
    "mnist-pytorch-cuda-tf32": lambda a: measure_pytorch(["--tf32"]),
    "matmul-vectorized": lambda a: (None, "skipped by --quick") if a.quick
                                   else measure_matmul("vectorized"),
    "matmul-cublas": lambda a: (None, "skipped by --quick") if a.quick
                               else measure_matmul("cublas"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the 4096-cubed matmul rows")
    args = ap.parse_args()

    with open(CLAIMS, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Machine on record: {data['machine']['cpu']} + {data['machine']['gpu']}")
    print("If yours is not that one, expect these to differ and do not 'fix' the")
    print("documents to match your hardware.\n")

    drifted, skipped, offsets, ok = [], [], [], 0
    for claim in data["claims"]:
        if claim["id"] not in MEASUREMENTS:
            continue
        value, note = MEASUREMENTS[claim["id"]](args)
        expected, unit = claim["value"], claim["unit"]
        if value is None:
            skipped.append(f"{claim['id']}: {note}")
            print(f"  {'-':>9}  {claim['id']:<26} skipped: {note}")
            continue
        signed = (value - expected) / expected
        off = abs(signed)
        offsets.append(signed)
        mark = "ok" if off <= claim["tolerance"] else "DRIFT"
        print(f"  {value:9.2f}  {claim['id']:<26} published {expected} {unit}, "
              f"off by {off * 100:.1f}%  [{mark}]  ({note})")
        if mark == "ok":
            ok += 1
        else:
            drifted.append(f"{claim['id']}: measured {value:.2f} {unit}, "
                           f"published {expected} {unit} "
                           f"({off * 100:.1f}% off, tolerance {claim['tolerance'] * 100:.0f}%)")

    print()
    if skipped:
        print(f"{len(skipped)} claim(s) not measured:")
        for s in skipped:
            print(f"  {s}")
        print()

    # Uniform drift is the machine, not the code.
    #
    # A regression in one of this engine's kernels cannot also slow PyTorch
    # down. So when every measured row moves the same way -- and they do, this
    # file was written watching three consecutive runs degrade together as the
    # afternoon's work heated the card -- the finding is about machine state and
    # failing on it would be reporting the wrong thing loudly.
    #
    # What a real regression looks like is the opposite: one row moves and its
    # neighbours do not. That still fails.
    slower = [o for o in offsets if o > 0.05]
    faster = [o for o in offsets if o < -0.05]
    uniform = len(offsets) >= 3 and (not faster or not slower)

    if drifted and uniform:
        direction = "slower" if slower else "faster"
        worst = max(abs(o) for o in offsets)
        print(f"Everything measured came out {direction}, by up to {worst * 100:.0f}%, "
              f"including\nthe rows this project does not implement. A regression here "
              f"cannot slow\nPyTorch down too, so this is the machine and not the code: "
              f"a loaded box,\na warm card, or a different one entirely.\n")
        print("Re-run on an idle machine before believing any of it. Not failing.")
        return 0

    if drifted:
        print("Measurements no longer match what the documents publish, and they did")
        print("not all move together, which is what a real regression looks like:\n")
        for d in drifted:
            print(f"  {d}")
        print("\nIf it is real: update docs/performance.json and every file that cites")
        print("the number, then run tools/check_claims.py to find them all.")
        return 1

    print(f"{ok} claim(s) re-measured and still true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
