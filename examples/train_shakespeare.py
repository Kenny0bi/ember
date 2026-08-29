"""Train a character-level GPT on Tiny Shakespeare, entirely in ember.

Everything in the training loop (forward, backward, AdamW, gradient clipping,
LR schedule) runs on the from-scratch engine. PyTorch is not imported.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ember import Tensor, nn, no_grad
from ember import functional as F
from ember import optim

ROOT = Path(__file__).resolve().parents[1]

# laptop-scale but real: pre-norm transformer, learned positions
BLOCK = 96
BATCH = 24
DIM = 128
HEADS = 4
LAYERS = 3
STEPS = 1500
EVAL_EVERY = 100


def get_batch(data, rng):
    ix = rng.integers(0, len(data) - BLOCK - 1, BATCH)
    x = np.stack([data[i : i + BLOCK] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + BLOCK] for i in ix])
    return x, y


def estimate_loss(model, data, rng, iters=8):
    with no_grad():
        losses = []
        for _ in range(iters):
            x, y = get_batch(data, rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(BATCH * BLOCK, -1), y.reshape(-1))
            losses.append(loss.item())
    return float(np.mean(losses))


def generate(model, stoi, itos, prompt, n_tokens, rng, temperature=0.8):
    idx = np.array([[stoi[c] for c in prompt]])
    with no_grad():
        for _ in range(n_tokens):
            window = idx[:, -BLOCK:]
            logits = model(window).data[0, -1] / temperature
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = rng.choice(len(p), p=p)
            idx = np.concatenate([idx, [[nxt]]], axis=1)
    return "".join(itos[i] for i in idx[0])


def main():
    text = (ROOT / "data" / "input.txt").read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"corpus: {len(text):,} chars, vocab {len(chars)}", flush=True)

    model = nn.GPT(vocab_size=len(chars), block_size=BLOCK, dim=DIM,
                   num_heads=HEADS, num_layers=LAYERS, dropout=0.1, seed=1)
    print(f"model: {model.num_parameters():,} parameters", flush=True)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    sched = optim.CosineWithWarmup(opt, warmup_steps=100, total_steps=STEPS,
                                   peak_lr=1e-3, min_lr=1e-4)
    rng = np.random.default_rng(0)
    eval_rng = np.random.default_rng(123)

    log = {"step": [], "train_loss": [], "val_loss": [], "lr": [],
           "tokens_per_sec": []}
    t0 = time.time()
    tokens_seen = 0
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(BATCH * BLOCK, -1), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm(model.parameters(), 1.0)
        lr = sched.step()
        opt.step()
        tokens_seen += BATCH * BLOCK

        if step % EVAL_EVERY == 0 or step == 1:
            model.eval()
            val = estimate_loss(model, val_data, eval_rng)
            train = estimate_loss(model, train_data, eval_rng)
            model.train()
            tps = tokens_seen / (time.time() - t0)
            log["step"].append(step)
            log["train_loss"].append(train)
            log["val_loss"].append(val)
            log["lr"].append(lr)
            log["tokens_per_sec"].append(tps)
            print(f"step {step:5d} | train {train:.4f} | val {val:.4f} | "
                  f"lr {lr:.2e} | {tps:,.0f} tok/s | {time.time() - t0:,.0f}s",
                  flush=True)
            (ROOT / "assets" / "shakespeare_log.json").write_text(json.dumps(log))

    model.eval()
    np.savez_compressed(ROOT / "assets" / "shakespeare_model.npz",
                        **model.state_dict())
    sample = generate(model, stoi, itos, "ROMEO:", 400, np.random.default_rng(5))
    (ROOT / "assets" / "shakespeare_sample.txt").write_text(sample)
    print("\n--- sample ---\n" + sample, flush=True)


if __name__ == "__main__":
    main()
