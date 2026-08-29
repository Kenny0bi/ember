"""Honest throughput comparison: ember vs PyTorch, forward + backward.

Same shapes a small transformer actually uses. PyTorch should win (it has
fused kernels and a C++ autograd); the point is to measure the price of the
pure-NumPy tape, not to pretend there isn't one.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ember import Tensor
from ember import functional as F

REPEATS = 30


def bench(fn):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        fn()
    return (time.perf_counter() - t0) / REPEATS * 1000  # ms


def main():
    rng = np.random.default_rng(0)
    cases = []

    # batched matmul at attention shapes
    a = rng.standard_normal((24, 4, 96, 32)).astype(np.float32)
    b = rng.standard_normal((24, 4, 32, 96)).astype(np.float32)

    def e_matmul():
        x, y = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
        (x @ y).sum().backward()

    def t_matmul():
        x = torch.tensor(a, requires_grad=True)
        y = torch.tensor(b, requires_grad=True)
        (x @ y).sum().backward()

    cases.append(("batched matmul (24,4,96,32)@(24,4,32,96)", e_matmul, t_matmul))

    # softmax over attention logits
    logits = rng.standard_normal((24, 4, 96, 96)).astype(np.float32)

    def e_softmax():
        x = Tensor(logits, requires_grad=True)
        F.softmax(x, axis=-1).sum().backward()

    def t_softmax():
        x = torch.tensor(logits, requires_grad=True)
        torch.softmax(x, dim=-1).sum().backward()

    cases.append(("softmax (24,4,96,96)", e_softmax, t_softmax))

    # layernorm-shaped reduction chain
    h = rng.standard_normal((24, 96, 128)).astype(np.float32)

    def e_ln():
        x = Tensor(h, requires_grad=True)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2.0).mean(axis=-1, keepdims=True)
        ((x - mu) / (var + 1e-5).sqrt()).sum().backward()

    def t_ln():
        x = torch.tensor(h, requires_grad=True)
        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2.0).mean(dim=-1, keepdim=True)
        ((x - mu) / (var + 1e-5).sqrt()).sum().backward()

    cases.append(("layernorm chain (24,96,128)", e_ln, t_ln))

    print(f"{'op':<45} {'ember ms':>10} {'torch ms':>10} {'ratio':>7}")
    for name, ef, tf in cases:
        em, tm = bench(ef), bench(tf)
        print(f"{name:<45} {em:>10.2f} {tm:>10.2f} {em / tm:>6.1f}x")


if __name__ == "__main__":
    main()
