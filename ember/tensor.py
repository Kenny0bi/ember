"""The core of ember: a NumPy-backed Tensor with reverse-mode autodiff.

Every differentiable op builds a node in an implicit computation graph by
recording its parents and a closure that knows how to push gradients backward.
Calling .backward() on a scalar output topologically sorts the graph and runs
those closures in reverse.
"""

from __future__ import annotations

import contextlib

import numpy as np

# Global autograd switch, flipped by no_grad() during evaluation/generation.
_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` back to `shape` by summing the axes NumPy broadcast over.

    Broadcasting copies values forward, so the gradient of each original
    element is the sum over all its broadcast copies.
    """
    if grad.shape == shape:
        return grad
    # Extra leading axes were created by broadcasting: sum them away.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Axes of size 1 were stretched: sum back with keepdims.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float32)
        self.data = arr
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = None
        self._prev = ()

    # ------------------------------------------------------------------ setup

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})\n{self.data}"

    def _accumulate(self, grad: np.ndarray):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    @staticmethod
    def _lift(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def _make(self, data: np.ndarray, parents: tuple, backward) -> "Tensor":
        """Create a result tensor, wiring it into the graph if grad is on."""
        out = Tensor(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._prev = parents
            out._backward = backward
        return out

    # ------------------------------------------------------------- arithmetic

    def __add__(self, other):
        other = self._lift(other)
        out_data = self.data + other.data

        def backward(g):
            return _unbroadcast(g, self.shape), _unbroadcast(g, other.shape)

        return self._make(out_data, (self, other), backward)

    def __mul__(self, other):
        other = self._lift(other)
        out_data = self.data * other.data

        def backward(g):
            return (
                _unbroadcast(g * other.data, self.shape),
                _unbroadcast(g * self.data, other.shape),
            )

        return self._make(out_data, (self, other), backward)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __rsub__(self, other):
        return self._lift(other) + (-self)

    def __truediv__(self, other):
        return self * self._lift(other) ** -1.0

    def __rtruediv__(self, other):
        return self._lift(other) * self**-1.0

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, exponent: float):
        assert isinstance(exponent, (int, float)), "only scalar powers"
        out_data = self.data**exponent

        def backward(g):
            return (g * exponent * self.data ** (exponent - 1),)

        return self._make(out_data, (self,), backward)

    def __matmul__(self, other):
        other = self._lift(other)
        assert self.ndim >= 2 and other.ndim >= 2, "matmul needs ndim >= 2"
        out_data = self.data @ other.data

        def backward(g):
            ga = _unbroadcast(g @ other.data.swapaxes(-1, -2), self.shape)
            gb = _unbroadcast(self.data.swapaxes(-1, -2) @ g, other.shape)
            return ga, gb

        return self._make(out_data, (self, other), backward)

    # ------------------------------------------------------------ elementwise

    def exp(self):
        out_data = np.exp(self.data)

        def backward(g):
            return (g * out_data,)

        return self._make(out_data, (self,), backward)

    def log(self):
        out_data = np.log(self.data)

        def backward(g):
            return (g / self.data,)

        return self._make(out_data, (self,), backward)

    def tanh(self):
        out_data = np.tanh(self.data)

        def backward(g):
            return (g * (1.0 - out_data**2),)

        return self._make(out_data, (self,), backward)

    def relu(self):
        out_data = np.maximum(self.data, 0.0)

        def backward(g):
            return (g * (self.data > 0),)

        return self._make(out_data, (self,), backward)

    def sigmoid(self):
        out_data = 1.0 / (1.0 + np.exp(-self.data))

        def backward(g):
            return (g * out_data * (1.0 - out_data),)

        return self._make(out_data, (self,), backward)

    def sqrt(self):
        return self**0.5

    # -------------------------------------------------------------- reductions

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)

        def backward(g):
            g = np.asarray(g)
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, self.shape).copy(),)

        return self._make(out_data, (self,), backward)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))]
        )
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)

        def backward(g):
            g = np.asarray(g)
            expanded = out_data
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)
                expanded = np.expand_dims(out_data, axis)
            mask = (self.data == expanded).astype(self.data.dtype)
            # Split the gradient evenly among tied maxima.
            mask /= mask.sum(axis=axis, keepdims=True)
            return (mask * g,)

        return self._make(out_data, (self,), backward)

    # ------------------------------------------------------------ shape / index

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out_data = self.data.reshape(shape)
        original = self.shape

        def backward(g):
            return (g.reshape(original),)

        return self._make(out_data, (self,), backward)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out_data = self.data.transpose(axes)
        inverse = np.argsort(axes)

        def backward(g):
            return (g.transpose(inverse),)

        return self._make(out_data, (self,), backward)

    def swapaxes(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        # Advanced integer indexing can select the same element twice, so the
        # backward pass must accumulate with np.add.at, not plain assignment.

        def backward(g):
            buf = np.zeros_like(self.data)
            np.add.at(buf, idx, g)
            return (buf,)

        return self._make(out_data, (self,), backward)

    # ---------------------------------------------------------------- backward

    def backward(self, grad=None):
        assert self.requires_grad, "called backward on a tensor without grad"
        if grad is None:
            assert self.data.size == 1, "backward() without grad needs a scalar"
            grad = np.ones_like(self.data)

        topo, visited = [], set()
        stack = [(self, False)]
        while stack:  # iterative DFS: graphs of deep models overflow recursion
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))

        self._accumulate(grad)
        for node in reversed(topo):
            if node._backward is None:
                continue
            grads = node._backward(node.grad)
            for parent, parent_grad in zip(node._prev, grads):
                if parent.requires_grad and parent_grad is not None:
                    parent._accumulate(parent_grad)
