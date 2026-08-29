"""The parity experiment: is ember's autograd actually equivalent to PyTorch's?

Same MLP, same weights (copied bit for bit), same data order, same SGD
hyperparameters, trained side by side for 300 steps on MNIST. If the engine is
correct, the two loss curves should be numerically indistinguishable, with only
float32 accumulation-order noise between them. This is a much stronger claim
than "my loss goes down."
"""

import gzip
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ember import Tensor, nn
from ember import functional as F
from ember import optim

DATA = Path(__file__).resolve().parents[1] / "data"


def read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        dims = struct.unpack(">" + "I" * (magic & 0xFF), f.read(4 * (magic & 0xFF)))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def main():
    x = (read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784) / 255.0).astype(np.float32)
    y = read_idx(DATA / "train-labels-idx1-ubyte.gz")

    # ember model
    rng = np.random.default_rng(42)
    e_fc1 = nn.Linear(784, 128, rng=rng)
    e_fc2 = nn.Linear(128, 10, rng=rng)

    # torch model with the exact same weights
    t_fc1 = torch.nn.Linear(784, 128)
    t_fc2 = torch.nn.Linear(128, 10)
    with torch.no_grad():
        t_fc1.weight.copy_(torch.tensor(e_fc1.weight.data.T))
        t_fc1.bias.copy_(torch.tensor(e_fc1.bias.data))
        t_fc2.weight.copy_(torch.tensor(e_fc2.weight.data.T))
        t_fc2.bias.copy_(torch.tensor(e_fc2.bias.data))

    e_opt = optim.SGD([e_fc1.weight, e_fc1.bias, e_fc2.weight, e_fc2.bias],
                      lr=0.1, momentum=0.9)
    t_opt = torch.optim.SGD(list(t_fc1.parameters()) + list(t_fc2.parameters()),
                            lr=0.1, momentum=0.9)

    order_rng = np.random.default_rng(7)
    steps, batch = 300, 64
    e_losses, t_losses = [], []
    for step in range(steps):
        idx = order_rng.integers(0, len(x), batch)
        xb, yb = x[idx], y[idx]

        e_logits = e_fc2(e_fc1(Tensor(xb)).relu())
        e_loss = F.cross_entropy(e_logits, yb)
        e_opt.zero_grad()
        e_loss.backward()
        e_opt.step()

        t_logits = t_fc2(torch.relu(t_fc1(torch.tensor(xb))))
        t_loss = torch.nn.functional.cross_entropy(t_logits, torch.tensor(yb, dtype=torch.long))
        t_opt.zero_grad()
        t_loss.backward()
        t_opt.step()

        e_losses.append(e_loss.item())
        t_losses.append(t_loss.item())

    diffs = np.abs(np.array(e_losses) - np.array(t_losses))
    print(f"steps: {steps}")
    print(f"final loss   ember={e_losses[-1]:.6f}  torch={t_losses[-1]:.6f}")
    print(f"max  |ember - torch| over all steps: {diffs.max():.2e}")
    print(f"mean |ember - torch| over all steps: {diffs.mean():.2e}")

    out = Path(__file__).resolve().parents[1] / "assets" / "parity_log.json"
    out.write_text(json.dumps({
        "ember": e_losses, "torch": t_losses,
        "max_abs_diff": float(diffs.max()), "mean_abs_diff": float(diffs.mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
