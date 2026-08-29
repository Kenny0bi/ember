"""Gradient checks: every ember op against PyTorch's autograd.

Each test builds the same computation in ember (float64) and torch (float64),
backprops both, and asserts values and gradients agree to tight tolerance.
Run directly (python tests/test_grads.py) or via pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ember import Tensor
from ember import functional as F

RNG = np.random.default_rng(7)
ATOL = 1e-9


def pair(*shape):
    """Same random float64 data as an ember Tensor and a torch tensor."""
    data = RNG.standard_normal(shape)
    e = Tensor(data.copy(), requires_grad=True)
    t = torch.tensor(data.copy(), dtype=torch.float64, requires_grad=True)
    return e, t


def check(e_out, t_out, pairs, name):
    assert np.allclose(e_out.data, t_out.detach().numpy(), atol=ATOL), f"{name}: forward mismatch"
    e_out.sum().backward()
    t_out.sum().backward()
    for e, t in pairs:
        assert np.allclose(e.grad, t.grad.numpy(), atol=ATOL), f"{name}: grad mismatch"
    print(f"  ok  {name}")


def test_add_broadcast():
    (a, ta), (b, tb) = pair(4, 5), pair(5)
    check(a + b, ta + tb, [(a, ta), (b, tb)], "add broadcast")


def test_mul_broadcast():
    (a, ta), (b, tb) = pair(3, 4, 5), pair(4, 1)
    check(a * b, ta * tb, [(a, ta), (b, tb)], "mul broadcast")


def test_div_pow():
    (a, ta), (b, tb) = pair(4, 5), pair(4, 5)
    check(a / (b * b + 2.0), ta / (tb * tb + 2.0), [(a, ta), (b, tb)], "div/pow")


def test_matmul():
    (a, ta), (b, tb) = pair(4, 5), pair(5, 3)
    check(a @ b, ta @ tb, [(a, ta), (b, tb)], "matmul 2d")


def test_matmul_batched_broadcast():
    (a, ta), (b, tb) = pair(2, 3, 4, 5), pair(3, 5, 6)
    check(a @ b, ta @ tb, [(a, ta), (b, tb)], "matmul batched+broadcast")


def test_elementwise_unaries():
    for name in ["exp", "tanh", "relu", "sigmoid"]:
        a, ta = pair(4, 5)
        check(getattr(a, name)(), getattr(torch, name)(ta), [(a, ta)], name)
    a, ta = pair(4, 5)
    check((a * a + 1.0).log(), (ta * ta + 1.0).log(), [(a, ta)], "log")


def test_reductions():
    a, ta = pair(3, 4, 5)
    check(a.sum(axis=1), ta.sum(dim=1), [(a, ta)], "sum axis")
    a, ta = pair(3, 4, 5)
    check(a.mean(axis=2, keepdims=True), ta.mean(dim=2, keepdim=True), [(a, ta)], "mean keepdims")
    a, ta = pair(3, 4)
    check(a.max(axis=1), ta.max(dim=1).values, [(a, ta)], "max axis")


def test_shape_ops():
    a, ta = pair(2, 3, 4)
    check(a.reshape(6, 4), ta.reshape(6, 4), [(a, ta)], "reshape")
    a, ta = pair(2, 3, 4)
    check(a.transpose(2, 0, 1), ta.permute(2, 0, 1), [(a, ta)], "transpose")


def test_getitem_advanced():
    a, ta = pair(6, 5)
    idx = np.array([0, 2, 2, 4])  # repeated row: backward must accumulate
    cols = np.array([1, 3, 3, 0])
    check(a[idx, cols], ta[torch.tensor(idx), torch.tensor(cols)], [(a, ta)], "advanced indexing")


def test_softmax_logsoftmax():
    a, ta = pair(4, 7)
    check(F.softmax(a, axis=-1), torch.softmax(ta, dim=-1), [(a, ta)], "softmax")
    a, ta = pair(4, 7)
    check(F.log_softmax(a, axis=-1), torch.log_softmax(ta, dim=-1), [(a, ta)], "log_softmax")


def test_cross_entropy():
    a, ta = pair(8, 10)
    targets = RNG.integers(0, 10, 8)
    e_loss = F.cross_entropy(a, targets)
    t_loss = torch.nn.functional.cross_entropy(ta, torch.tensor(targets))
    check(e_loss, t_loss, [(a, ta)], "cross_entropy")


def test_gelu():
    a, ta = pair(4, 5)
    check(F.gelu(a), torch.nn.functional.gelu(ta, approximate="tanh"), [(a, ta)], "gelu(tanh)")


def test_layernorm_composition():
    a, ta = pair(3, 4, 8)
    g, tg = pair(8)
    b, tb = pair(8)
    mu = a.mean(axis=-1, keepdims=True)
    var = ((a - mu) ** 2.0).mean(axis=-1, keepdims=True)
    e_out = (a - mu) / (var + 1e-5).sqrt() * g + b
    t_out = torch.nn.functional.layer_norm(ta, (8,), tg, tb, eps=1e-5)
    check(e_out, t_out, [(a, ta), (g, tg), (b, tb)], "layernorm")


def test_attention_against_torch():
    """Full causal multi-head attention with identical weights in both."""
    B, T, C, H = 2, 6, 16, 4
    x, tx = pair(B, T, C)
    wqkv, twqkv = pair(C, 3 * C)
    wproj, twproj = pair(C, C)

    hd = C // H
    qkv = x @ wqkv
    q = qkv[:, :, :C].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
    k = qkv[:, :, C:2 * C].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
    v = qkv[:, :, 2 * C:].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
    att = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(hd))
    att = att + Tensor(np.triu(np.full((T, T), -1e9), k=1))
    att = F.softmax(att, axis=-1)
    e_out = ((att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)) @ wproj

    tqkv = tx @ twqkv
    tq = tqkv[:, :, :C].reshape(B, T, H, hd).permute(0, 2, 1, 3)
    tk = tqkv[:, :, C:2 * C].reshape(B, T, H, hd).permute(0, 2, 1, 3)
    tv = tqkv[:, :, 2 * C:].reshape(B, T, H, hd).permute(0, 2, 1, 3)
    tatt = (tq @ tk.transpose(-1, -2)) * (1.0 / np.sqrt(hd))
    tatt = tatt + torch.tensor(np.triu(np.full((T, T), -1e9), k=1))
    tatt = torch.softmax(tatt, dim=-1)
    t_out = ((tatt @ tv).permute(0, 2, 1, 3).reshape(B, T, C)) @ twproj

    check(e_out, t_out, [(x, tx), (wqkv, twqkv), (wproj, twproj)], "causal attention")


def test_gpt_finite_difference():
    """Numerical gradient check on the full GPT training loss.

    Perturb random parameters of a real (tiny) GPT and compare the analytic
    gradient against a central finite difference of the loss.
    """
    from ember import nn

    model = nn.GPT(vocab_size=11, block_size=8, dim=16, num_heads=2,
                   num_layers=2, dropout=0.0, seed=3)
    # float64 for finite-difference precision
    for p in model.parameters():
        p.data = p.data.astype(np.float64)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, 11, (2, 8))
    targets = rng.integers(0, 11, (2, 8))

    def loss_value():
        logits = model(idx)
        flat = logits.reshape(16, 11)
        return F.cross_entropy(flat, targets.reshape(-1))

    loss = loss_value()
    model.zero_grad()
    loss.backward()

    eps = 1e-6
    checked = 0
    for p in model.parameters():
        flat_idx = rng.integers(0, p.data.size, 3)
        for fi in flat_idx:
            orig = p.data.flat[fi]
            p.data.flat[fi] = orig + eps
            up = loss_value().item()
            p.data.flat[fi] = orig - eps
            down = loss_value().item()
            p.data.flat[fi] = orig
            numeric = (up - down) / (2 * eps)
            analytic = p.grad.flat[fi]
            assert abs(numeric - analytic) < 1e-6, (
                f"finite diff mismatch: numeric={numeric}, analytic={analytic}"
            )
            checked += 1
    print(f"  ok  GPT finite-difference ({checked} parameters checked)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all gradient checks passed")
