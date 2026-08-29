"""Train an MLP classifier on MNIST with ember. No PyTorch anywhere.

The idx files are parsed by hand (magic number, dims, raw bytes), because if
the point of the project is doing things from scratch, the data loader should
not be the exception.
"""

import gzip
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ember import Tensor, nn, no_grad
from ember import functional as F
from ember import optim

DATA = Path(__file__).resolve().parents[1] / "data"


def read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


class MLP(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.fc1 = nn.Linear(784, 256, rng=rng)
        self.fc2 = nn.Linear(256, 128, rng=rng)
        self.fc3 = nn.Linear(128, 10, rng=rng)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


def main():
    x_train = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784) / 255.0
    y_train = read_idx(DATA / "train-labels-idx1-ubyte.gz")
    x_test = read_idx(DATA / "t10k-images-idx3-ubyte.gz").reshape(-1, 784) / 255.0
    y_test = read_idx(DATA / "t10k-labels-idx1-ubyte.gz")
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    model = MLP(seed=0)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(0)
    batch_size, epochs = 128, 5
    n = len(x_train)
    log = {"train_loss": [], "test_acc": []}

    t0 = time.time()
    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n - batch_size + 1, batch_size):
            idx = order[i : i + batch_size]
            logits = model(Tensor(x_train[idx]))
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        with no_grad():
            preds = []
            for i in range(0, len(x_test), 1000):
                preds.append(model(Tensor(x_test[i : i + 1000])).data.argmax(-1))
            acc = float((np.concatenate(preds) == y_test).mean())
        avg = epoch_loss / (n // batch_size * batch_size)
        log["train_loss"].append(avg)
        log["test_acc"].append(acc)
        print(f"epoch {epoch + 1}: train loss {avg:.4f}, test acc {acc:.4%}, "
              f"{time.time() - t0:.0f}s elapsed", flush=True)

    out = Path(__file__).resolve().parents[1] / "assets" / "mnist_log.json"
    out.write_text(json.dumps(log, indent=2))
    print(f"final test accuracy: {log['test_acc'][-1]:.4%}")


if __name__ == "__main__":
    main()
