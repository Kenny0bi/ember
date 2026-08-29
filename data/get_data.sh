#!/bin/sh
# Fetch the two training corpora: Tiny Shakespeare and MNIST (idx format).
cd "$(dirname "$0")"
curl -sL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sL -o "$f.gz" "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz"
done
ls -la
