#!/usr/bin/env python3
"""Trains the same network in PyTorch, on the same card, for a number that means
something.

The engine's headline was "5.5x faster on the GPU than on the CPU", and that
number is weaker than it looks: it does not say the GPU path is fast, it says
the CPU path is slow. The only comparison a reader can calibrate against is a
production framework running the same model on the same hardware.

Fairness is the whole point, so everything below is matched to
examples/mnist_demo.cpp deliberately:

  same subset          data/mnist/subset-*, read from the same IDX files,
                       normalised to [0, 1] the same way (src/data.cpp:79)
  same architecture    two 3x3 convolutions with padding 1, 16 then 32 channels,
                       2x2 max pooling after each, dropout 0.25, 1568 -> 128 -> 10
  same optimiser       Adam at 1e-3, cosine annealing to 1e-4 over the run
  same everything else batch 64, 12 epochs, global-norm gradient clipping at 5.0

And two settings that matter more than they look:

  TF32 off by default. Ampere runs fp32 matmuls through the tensor cores at
  reduced precision unless told otherwise, and the engine is fp32 everywhere.
  Leaving it on would be comparing a 10-bit mantissa against a 23-bit one and
  calling the difference a speedup. --tf32 measures it anyway, because it is
  what a PyTorch user actually gets, and the gap between the two rows is itself
  worth knowing.

  cuDNN benchmark off. It autotunes the convolution algorithm on first sight of
  a shape, which is a real advantage of the framework, but it makes the first
  epoch unrepresentative and the rest depend on a cache. --cudnn-benchmark turns
  it on.

Usage:
    .torch/Scripts/python tools/bench_pytorch.py            # matched, fp32
    .torch/Scripts/python tools/bench_pytorch.py --tf32     # what PyTorch gives you
    .torch/Scripts/python tools/bench_pytorch.py --device cpu
"""

import argparse
import gzip
import os
import struct
import sys
import time

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch is required: pip install torch --index-url "
             "https://download.pytorch.org/whl/cu128")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST = os.path.join(REPO, "data", "mnist")


# ---------------------------------------------------------
# The same IDX files the engine reads, parsed the same way.
# ---------------------------------------------------------

def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def read_images(path):
    with _open(path) as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: not an IDX image file (magic {magic})")
        raw = f.read(count * rows * cols)
    # Normalised to [0, 1], as src/data.cpp does: with values 0-255 the first
    # layer's gradients are two orders of magnitude larger than they should be.
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).float().div_(255.0)
    return t.view(count, 1, rows, cols)


def read_labels(path):
    with _open(path) as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: not an IDX label file (magic {magic})")
        raw = f.read(count)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()


def find_mnist():
    """Prefers the full set if it has been downloaded, exactly as the demo does."""
    full = [("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")]
    subset = [("subset-train-images-idx3-ubyte", "subset-train-labels-idx1-ubyte",
               "subset-test-images-idx3-ubyte", "subset-test-labels-idx1-ubyte")]
    for names, is_full in ((full[0], True), (subset[0], False)):
        paths = []
        for n in names:
            p = os.path.join(MNIST, n)
            paths.append(p if os.path.exists(p) else p + ".gz"
                         if os.path.exists(p + ".gz") else None)
        if all(paths):
            return paths, is_full
    sys.exit(f"No MNIST found under {MNIST}. Run tools/download_mnist.sh, or use "
             "the subset the repository ships.")


# ---------------------------------------------------------
# The same architecture as examples/mnist_demo.cpp:155.
# ---------------------------------------------------------

def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=1, padding=1),   # (N,1,28,28) -> (N,16,28,28)
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                         # -> (N,16,14,14)
        nn.Conv2d(16, 32, 3, stride=1, padding=1),  # -> (N,32,14,14)
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                         # -> (N,32,7,7)
        nn.Flatten(),                               # -> (N,1568)
        nn.Dropout(0.25),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


@torch.no_grad()
def accuracy(model, images, labels, batch, device):
    model.eval()
    hits = 0
    for start in range(0, len(labels), batch):
        x = images[start:start + batch].to(device, non_blocking=True)
        pred = model(x).argmax(dim=1).cpu()
        hits += int((pred == labels[start:start + batch]).sum())
    model.train()
    return hits / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tf32", action="store_true",
                    help="allow tensor cores for fp32 matmuls and convolutions")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="let cuDNN autotune the convolution algorithm")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available to this PyTorch build.")
    device = torch.device(args.device)

    # See the module docstring: off unless asked for, so the comparison is
    # fp32 against fp32.
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    torch.manual_seed(args.seed)

    paths, is_full = find_mnist()
    train_x = read_images(paths[0])
    train_y = read_labels(paths[1])
    test_x = read_images(paths[2])
    test_y = read_labels(paths[3])
    epochs = args.epochs if args.epochs is not None else (6 if is_full else 12)

    print("====================================================")
    print("  MNIST in PyTorch, for comparison with the engine  ")
    print("====================================================\n")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"Backend: PyTorch {torch.__version__} on {name}")
        print(f"         cuDNN {torch.backends.cudnn.version()}, "
              f"TF32 {'on' if args.tf32 else 'off'}, "
              f"cuDNN benchmark {'on' if args.cudnn_benchmark else 'off'}")
    else:
        print(f"Backend: PyTorch {torch.__version__} on CPU, "
              f"{torch.get_num_threads()} thread(s)")
    print(f"Data:    {'full MNIST' if is_full else 'repository subset'}, "
          f"{len(train_y)} training images, {len(test_y)} test\n")

    model = build_model().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model:   {params:,} parameters\n")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Both sets live on the device for the whole run, which is what the engine
    # does too once the residency chain holds. Uploading per batch would measure
    # the PCIe link rather than the model.
    train_x = train_x.to(device)
    train_y_dev = train_y.to(device)

    print(f"--- Training ({epochs} epochs, batches of {args.batch}) ---")
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_y), device=device)
        model.train()
        total, batches = 0.0, 0

        for start in range(0, len(train_y), args.batch):
            idx = order[start:start + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[idx]), train_y_dev[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss)
            batches += 1

        scheduler.step()
        acc = accuracy(model, test_x, test_y, 500, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(f"  Epoch {epoch:2d} | Loss = {total / batches:.4f} | "
              f"Test = {acc * 100:.2f}% | lr = {opt.param_groups[0]['lr']:.1e} | "
              f"{elapsed:.1f} s")

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - started
    final = accuracy(model, test_x, test_y, 500, device)

    print(f"\nFinal accuracy on the test set: {final * 100:.2f}%")
    print(f"Training time: {total_time:.1f} s")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak device memory: {peak:.1f} MiB")


if __name__ == "__main__":
    main()
