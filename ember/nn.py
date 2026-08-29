"""Neural network modules on top of the Tensor engine."""

import numpy as np

from . import functional as F
from .tensor import Tensor


class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def num_parameters(self):
        return sum(p.data.size for p in self.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self):
        self._set_mode(True)

    def eval(self):
        self._set_mode(False)

    def _set_mode(self, training):
        self.training = training
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value._set_mode(training)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item._set_mode(training)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def state_dict(self, prefix=""):
        state = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                state[prefix + name] = value.data
            elif isinstance(value, Module):
                state.update(value.state_dict(prefix + name + "."))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        state.update(item.state_dict(f"{prefix}{name}.{i}."))
        return state

    def load_state_dict(self, state, prefix=""):
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                value.data = np.asarray(state[prefix + name], dtype=value.data.dtype)
            elif isinstance(value, Module):
                value.load_state_dict(state, prefix + name + ".")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        item.load_state_dict(state, f"{prefix}{name}.{i}.")


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng()
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(
            rng.uniform(-bound, bound, (in_features, out_features)).astype(np.float32),
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
            if bias
            else None
        )

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, dim, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.weight = Tensor(
            (rng.standard_normal((num_embeddings, dim)) * 0.02).astype(np.float32),
            requires_grad=True,
        )

    def forward(self, idx):
        return self.weight[np.asarray(idx)]


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2.0).mean(axis=-1, keepdims=True)
        xhat = (x - mu) / (var + self.eps).sqrt()
        return xhat * self.gamma + self.beta


class Dropout(Module):
    def __init__(self, p=0.1, rng=None):
        super().__init__()
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = (self.rng.random(x.shape) >= self.p).astype(np.float32)
        return x * Tensor(keep / (1.0 - self.p))


class CausalSelfAttention(Module):
    def __init__(self, dim, num_heads, block_size, dropout=0.0, rng=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = Linear(dim, 3 * dim, rng=rng)
        self.proj = Linear(dim, dim, rng=rng)
        self.drop = Dropout(dropout, rng=rng)
        # Additive mask: 0 on and below the diagonal, -1e9 above. Adding a huge
        # negative before softmax zeroes attention to future positions.
        mask = np.triu(np.full((block_size, block_size), -1e9, dtype=np.float32), k=1)
        self.mask = mask
        # introspection: set store_att=True to keep the last softmax(QK^T)
        self.store_att = False
        self.last_att = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q = qkv[:, :, :C].reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, C : 2 * C].reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * C :].reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        att = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(self.head_dim))
        att = att + Tensor(self.mask[:T, :T])
        att = F.softmax(att, axis=-1)
        if self.store_att:
            self.last_att = att.data.copy()
        att = self.drop(att)
        out = att @ v  # (B, H, T, hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(Module):
    def __init__(self, dim, num_heads, block_size, dropout=0.0, rng=None):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, block_size, dropout, rng=rng)
        self.ln2 = LayerNorm(dim)
        self.fc1 = Linear(dim, 4 * dim, rng=rng)
        self.fc2 = Linear(4 * dim, dim, rng=rng)
        self.drop = Dropout(dropout, rng=rng)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = F.gelu(self.fc1(self.ln2(x)))
        x = x + self.drop(self.fc2(h))
        return x


class GPT(Module):
    """A small pre-norm GPT: learned positional embeddings, causal attention."""

    def __init__(self, vocab_size, block_size, dim, num_heads, num_layers,
                 dropout=0.0, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.tok_emb = Embedding(vocab_size, dim, rng=rng)
        self.pos_emb = Embedding(block_size, dim, rng=rng)
        self.drop = Dropout(dropout, rng=rng)
        self.blocks = [
            TransformerBlock(dim, num_heads, block_size, dropout, rng=rng)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab_size, bias=False, rng=rng)

    def forward(self, idx):
        idx = np.asarray(idx)
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(np.arange(T))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))  # (B, T, vocab)
