#!/usr/bin/env python3
"""Generates the fixtures that validate the engine against PyTorch.

Checking gradients numerically by central differences proves the engine is
consistent with itself. This proves something stronger: that it agrees with the
industry's reference implementation.

For each case the same computation is built in PyTorch, run forward and
backward, and the inputs, weights, output and gradients are dumped in the
engine's own binary format (engine/serialize.hpp). The resulting files are
committed under tests/fixtures/, so CI checks them without needing Python or
PyTorch: this script is only needed to regenerate them.

Usage:
    pip install torch
    python3 tools/generate_reference.py

The seeds are fixed, so regenerating produces exactly the same files unless the
PyTorch version changes.
"""

import math
import os
import struct
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("Hace falta PyTorch: pip install torch")

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures")
MAGIC = b"CPPAIENG"
VERSION = 1


# ---------------------------------------------------------------------------
# Writer for the engine/serialize.hpp format
#
#   "CPPAIENG" | version u32 | n_tensores u32
#   per tensor: name_len u32 | name | ndim u32 | dims u64[] | float32[]
#
# All little-endian, as the engine requires.
# ---------------------------------------------------------------------------
def write_fixture(name, tensors):
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, name + ".bin")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(tensors)))

        for key, tensor in tensors.items():
            t = tensor.detach().contiguous().to(torch.float32)
            encoded = key.encode("utf-8")
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<I", t.dim() if t.dim() > 0 else 1))
            dims = list(t.shape) if t.dim() > 0 else [1]
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(t.reshape(-1).numpy().astype("<f4").tobytes())

    print(f"  {name+'.bin':38s} {len(tensors):2d} tensores")


def randn(*shape, seed=None, requires_grad=False):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(*shape, requires_grad=requires_grad)


def backward_with(output, grad_output):
    """Backpropagates a specific gradient rather than a tensor of ones.

    An output gradient of all ones can hide indexing errors:
    varias posiciones equivocadas dan la misma suma.
    """
    output.backward(grad_output)


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------
def case_matmul():
    torch.manual_seed(1)
    a = randn(4, 5, requires_grad=True)
    b = randn(5, 3, requires_grad=True)
    out = a @ b
    g = randn(4, 3)
    backward_with(out, g)
    write_fixture("matmul", {"a": a, "b": b, "grad_output": g,
                             "output": out, "grad.a": a.grad, "grad.b": b.grad})


def case_matmul_batched():
    torch.manual_seed(2)
    a = randn(2, 3, 4, requires_grad=True)
    b = randn(2, 4, 5, requires_grad=True)
    out = a @ b
    g = randn(2, 3, 5)
    backward_with(out, g)
    write_fixture("matmul_batched", {"a": a, "b": b, "grad_output": g,
                                     "output": out, "grad.a": a.grad, "grad.b": b.grad})


def case_matmul_shared():
    """(B, M, K) x (K, N): the same matrix applied to the whole batch."""
    torch.manual_seed(3)
    a = randn(3, 4, 5, requires_grad=True)
    b = randn(5, 2, requires_grad=True)
    out = a @ b
    g = randn(3, 4, 2)
    backward_with(out, g)
    write_fixture("matmul_shared", {"a": a, "b": b, "grad_output": g,
                                    "output": out, "grad.a": a.grad, "grad.b": b.grad})


def case_softmax():
    torch.manual_seed(4)
    x = randn(3, 6, requires_grad=True)
    out = F.softmax(x, dim=-1)
    g = randn(3, 6)
    backward_with(out, g)
    write_fixture("softmax", {"input": x, "grad_output": g, "output": out, "grad.input": x.grad})


def case_softmax_3d():
    torch.manual_seed(5)
    x = randn(2, 3, 4, requires_grad=True)
    out = F.softmax(x, dim=-1)
    g = randn(2, 3, 4)
    backward_with(out, g)
    write_fixture("softmax_3d", {"input": x, "grad_output": g,
                                 "output": out, "grad.input": x.grad})


def case_activations():
    for name, fn in [("relu", F.relu), ("sigmoid", torch.sigmoid),
                     ("tanh", torch.tanh), ("gelu", lambda t: F.gelu(t, approximate="tanh"))]:
        torch.manual_seed(6)
        # Exact 0 is avoided: ReLU is not differentiable there and the fixture
        # would be ambiguous
        x = (randn(4, 5) + 0.37).requires_grad_(True)
        out = fn(x)
        g = randn(4, 5)
        backward_with(out, g)
        write_fixture("act_" + name, {"input": x, "grad_output": g,
                                      "output": out, "grad.input": x.grad})


def case_cross_entropy():
    torch.manual_seed(7)
    logits = randn(8, 5, requires_grad=True)
    targets = torch.tensor([0, 4, 2, 1, 3, 0, 2, 4])
    loss = F.cross_entropy(logits, targets, reduction="mean")
    loss.backward()
    write_fixture("cross_entropy", {
        "logits": logits,
        "targets": targets.to(torch.float32),
        "output": loss.reshape(1),
        "grad.logits": logits.grad,
    })


def case_linear():
    torch.manual_seed(8)
    x = randn(4, 6, requires_grad=True)
    layer = torch.nn.Linear(6, 3)
    out = layer(x)
    g = randn(4, 3)
    backward_with(out, g)
    # PyTorch stores the weight as (out, in) and computes x @ W.T; the engine
    # stores it as (in, out) and computes x @ W. It is transposed on dump.
    write_fixture("linear", {
        "input": x, "weight": layer.weight.T, "bias": layer.bias.reshape(1, 3),
        "grad_output": g, "output": out,
        "grad.input": x.grad, "grad.weight": layer.weight.grad.T,
        "grad.bias": layer.bias.grad.reshape(1, 3),
    })


def case_linear_3d():
    torch.manual_seed(9)
    x = randn(2, 5, 6, requires_grad=True)
    layer = torch.nn.Linear(6, 4)
    out = layer(x)
    g = randn(2, 5, 4)
    backward_with(out, g)
    write_fixture("linear_3d", {
        "input": x, "weight": layer.weight.T, "bias": layer.bias.reshape(1, 4),
        "grad_output": g, "output": out,
        "grad.input": x.grad, "grad.weight": layer.weight.grad.T,
        "grad.bias": layer.bias.grad.reshape(1, 4),
    })


def case_conv2d(name, in_c, out_c, kernel, stride, padding, shape, seed):
    torch.manual_seed(seed)
    x = randn(*shape, requires_grad=True)
    conv = torch.nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding)
    out = conv(x)
    g = randn(*out.shape)
    backward_with(out, g)
    write_fixture(name, {
        "input": x, "weight": conv.weight, "bias": conv.bias,
        "grad_output": g, "output": out,
        "grad.input": x.grad, "grad.weight": conv.weight.grad, "grad.bias": conv.bias.grad,
    })


def case_maxpool():
    torch.manual_seed(12)
    # Well-separated values: on a tie the argmax is ambiguous and PyTorch and
    # the engine could pick different positions, both correct.
    x = (torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) * 0.37)
    x = (x % 19.0).requires_grad_(True)
    out = F.max_pool2d(x, kernel_size=2, stride=2)
    g = randn(*out.shape)
    backward_with(out, g)
    write_fixture("maxpool2d", {"input": x, "grad_output": g,
                                "output": out, "grad.input": x.grad})


def case_layernorm():
    torch.manual_seed(13)
    x = randn(4, 6, 8, requires_grad=True)
    norm = torch.nn.LayerNorm(8, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(randn(8, seed=14))
        norm.bias.copy_(randn(8, seed=15))
    out = norm(x)
    g = randn(4, 6, 8)
    backward_with(out, g)
    write_fixture("layernorm", {
        "input": x, "gamma": norm.weight, "beta": norm.bias,
        "grad_output": g, "output": out,
        "grad.input": x.grad, "grad.gamma": norm.weight.grad, "grad.beta": norm.bias.grad,
    })


def case_attention(name, causal, seed):
    torch.manual_seed(seed)
    q = randn(2, 3, 5, 4, requires_grad=True)
    k = randn(2, 3, 5, 4, requires_grad=True)
    v = randn(2, 3, 5, 4, requires_grad=True)

    scale = 1.0 / math.sqrt(4)
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.full((5, 5), -1e9), diagonal=1)
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    out = weights @ v

    g = randn(2, 3, 5, 4)
    backward_with(out, g)

    tensors = {"q": q, "k": k, "v": v, "grad_output": g, "output": out,
               "weights": weights, "grad.q": q.grad, "grad.k": k.grad, "grad.v": v.grad}
    if causal:
        tensors["mask"] = torch.triu(torch.full((5, 5), -1e9), diagonal=1)
    write_fixture(name, tensors)


def case_multihead_attention():
    """Composed from the same pieces the engine uses.

    nn.MultiheadAttention is not used because it packs q/k/v into one matrix
    and the weight correspondence would be different. What is validated is the
    computation, and every primitive is still PyTorch's.
    """
    torch.manual_seed(20)
    d_model, heads, seq, batch = 8, 2, 5, 2
    head_dim = d_model // heads

    x = randn(batch, seq, d_model, requires_grad=True)
    wq = torch.nn.Linear(d_model, d_model)
    wk = torch.nn.Linear(d_model, d_model)
    wv = torch.nn.Linear(d_model, d_model)
    wo = torch.nn.Linear(d_model, d_model)

    def split(t):
        return t.reshape(batch, seq, heads, head_dim).permute(0, 2, 1, 3)

    q, k, v = split(wq(x)), split(wk(x)), split(wv(x))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    merged = (attn @ v).permute(0, 2, 1, 3).reshape(batch, seq, d_model)
    out = wo(merged)

    g = randn(batch, seq, d_model)
    backward_with(out, g)

    tensors = {"input": x, "grad_output": g, "output": out, "grad.input": x.grad}
    for tag, layer in [("query", wq), ("key", wk), ("value", wv), ("out", wo)]:
        tensors[f"w.{tag}.weight"] = layer.weight.T
        tensors[f"w.{tag}.bias"] = layer.bias.reshape(1, d_model)
        tensors[f"grad.{tag}.weight"] = layer.weight.grad.T
        tensors[f"grad.{tag}.bias"] = layer.bias.grad.reshape(1, d_model)
    write_fixture("multihead_attention", tensors)


def case_transformer_block():
    """Pre-norm block, the same as the engine's nn::TransformerBlock:

        h = x + Attention(LayerNorm(x))
        y = h + FF2(ReLU(FF1(LayerNorm(h))))
    """
    torch.manual_seed(30)
    d_model, heads, ff, seq, batch = 8, 2, 16, 4, 2
    head_dim = d_model // heads

    x = randn(batch, seq, d_model, requires_grad=True)
    norm1 = torch.nn.LayerNorm(d_model, eps=1e-5)
    norm2 = torch.nn.LayerNorm(d_model, eps=1e-5)
    wq, wk = torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, d_model)
    wv, wo = torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, d_model)
    ff1, ff2 = torch.nn.Linear(d_model, ff), torch.nn.Linear(ff, d_model)

    # gamma and beta away from the identity, so that ignoring them shows up
    with torch.no_grad():
        norm1.weight.copy_(randn(d_model, seed=31))
        norm1.bias.copy_(randn(d_model, seed=32))
        norm2.weight.copy_(randn(d_model, seed=33))
        norm2.bias.copy_(randn(d_model, seed=34))

    def split(t):
        return t.reshape(batch, seq, heads, head_dim).permute(0, 2, 1, 3)

    normed = norm1(x)
    q, k, v = split(wq(normed)), split(wk(normed)), split(wv(normed))
    attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
    merged = (attn @ v).permute(0, 2, 1, 3).reshape(batch, seq, d_model)
    h = x + wo(merged)
    out = h + ff2(F.relu(ff1(norm2(h))))

    g = randn(batch, seq, d_model)
    backward_with(out, g)

    tensors = {"input": x, "grad_output": g, "output": out, "grad.input": x.grad}
    for tag, layer in [("norm1", norm1), ("norm2", norm2)]:
        tensors[f"w.{tag}.gamma"] = layer.weight
        tensors[f"w.{tag}.beta"] = layer.bias
        tensors[f"grad.{tag}.gamma"] = layer.weight.grad
        tensors[f"grad.{tag}.beta"] = layer.bias.grad
    for tag, layer in [("query", wq), ("key", wk), ("value", wv), ("out", wo),
                       ("ff1", ff1), ("ff2", ff2)]:
        cols = layer.out_features
        tensors[f"w.{tag}.weight"] = layer.weight.T
        tensors[f"w.{tag}.bias"] = layer.bias.reshape(1, cols)
        tensors[f"grad.{tag}.weight"] = layer.weight.grad.T
        tensors[f"grad.{tag}.bias"] = layer.bias.grad.reshape(1, cols)
    write_fixture("transformer_block", tensors)


def case_reductions():
    torch.manual_seed(40)
    x = randn(2, 3, 4, requires_grad=True)
    tensors = {"input": x}
    for tag, fn, shape in [("sum0", lambda t: t.sum(0), (3, 4)),
                           ("sum1", lambda t: t.sum(1), (2, 4)),
                           ("mean1", lambda t: t.mean(1), (2, 4))]:
        xi = x.detach().clone().requires_grad_(True)
        out = fn(xi)
        g = randn(*shape)
        backward_with(out, g)
        tensors[f"grad_output.{tag}"] = g
        tensors[f"output.{tag}"] = out
        tensors[f"grad.{tag}"] = xi.grad
    # The maximum over well-separated values, so the argmax is not ambiguous
    xm = ((torch.arange(24, dtype=torch.float32) * 7.0) % 23.0).reshape(2, 3, 4)
    xm = xm.requires_grad_(True)
    out = xm.max(dim=1).values
    g = randn(2, 4)
    backward_with(out, g)
    tensors["input.max"] = xm
    tensors["grad_output.max1"] = g
    tensors["output.max1"] = out
    tensors["grad.max1"] = xm.grad
    write_fixture("reductions", tensors)


def case_adam():
    """A 10-step trajectory, to validate the whole optimiser and not one step.

    An error in the bias correction only shows up by comparing the trajectory:
    the first step can match by chance.
    """
    torch.manual_seed(50)
    w = randn(3, 4, requires_grad=True)
    target = randn(3, 4)
    initial = w.detach().clone()

    opt = torch.optim.Adam([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    tensors = {"initial": initial, "target": target}
    for step in range(10):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
        tensors[f"step.{step}"] = w.detach().clone()
    write_fixture("adam", tensors)


def case_sgd_momentum():
    torch.manual_seed(51)
    w = randn(3, 4, requires_grad=True)
    target = randn(3, 4)
    initial = w.detach().clone()

    opt = torch.optim.SGD([w], lr=0.05, momentum=0.9)
    tensors = {"initial": initial, "target": target}
    for step in range(10):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
        tensors[f"step.{step}"] = w.detach().clone()
    write_fixture("sgd_momentum", tensors)


def main():
    print(f"PyTorch {torch.__version__} -> {os.path.normpath(FIXTURES)}\n")
    torch.set_default_dtype(torch.float32)

    case_matmul()
    case_matmul_batched()
    case_matmul_shared()
    case_softmax()
    case_softmax_3d()
    case_activations()
    case_cross_entropy()
    case_linear()
    case_linear_3d()
    case_conv2d("conv2d_pad1", 3, 4, 3, 1, 1, (2, 3, 8, 8), 10)
    case_conv2d("conv2d_stride2", 1, 2, 3, 2, 0, (2, 1, 8, 8), 11)
    case_maxpool()
    case_layernorm()
    case_attention("attention", causal=False, seed=16)
    case_attention("attention_causal", causal=True, seed=17)
    case_multihead_attention()
    case_transformer_block()
    case_reductions()
    case_adam()
    case_sgd_momentum()

    print("\nDone. The files are committed: CI checks them without PyTorch.")


if __name__ == "__main__":
    main()
