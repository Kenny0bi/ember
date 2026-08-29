"""Optimizers and schedules. All state lives in plain NumPy arrays."""

import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v *= self.momentum
                v += g
                g = v
            p.data -= self.lr * g


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self.weight_decay:
                update = update + self.weight_decay * p.data
            p.data -= self.lr * update


def clip_grad_norm(params, max_norm):
    """Global-norm clipping, same semantics as torch.nn.utils.clip_grad_norm_."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float((p.grad**2).sum())
    norm = np.sqrt(total)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return norm


class CosineWithWarmup:
    """Linear warmup to peak_lr, then cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr=0.0):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.peak = peak_lr
        self.min = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup:
            lr = self.peak * self.step_num / self.warmup
        else:
            progress = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            progress = min(1.0, progress)
            lr = self.min + 0.5 * (self.peak - self.min) * (1 + np.cos(np.pi * progress))
        self.opt.lr = lr
        return lr
