"""Composite differentiable functions built from Tensor primitives.

Nothing here defines its own backward pass: everything is a composition of
tensor.py ops, so correctness reduces to the primitives being correct.
"""

import numpy as np

from .tensor import Tensor


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    # Subtracting the (detached) max is the standard overflow guard; it cancels
    # in the ratio so it does not change the value or the gradient.
    shifted = x - x.max(axis=axis, keepdims=True).detach()
    e = shifted.exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.max(axis=axis, keepdims=True).detach()
    shifted = x - m
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean negative log-likelihood. logits: (N, C), targets: int array (N,)."""
    n = logits.shape[0]
    logp = log_softmax(logits, axis=-1)
    picked = logp[np.arange(n), np.asarray(targets)]
    return -picked.mean()


def gelu(x: Tensor) -> Tensor:
    # tanh approximation, same one GPT-2 uses
    c = float(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + (c * (x + 0.044715 * x**3.0)).tanh())
