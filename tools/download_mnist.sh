#!/usr/bin/env bash
# Downloads the full MNIST set (60,000 + 10,000 images) into data/mnist/.
#
# The repository already ships a 2,000 + 1,000 subset so the examples work on a
# fresh clone; this is for the real result. The examples
# detect by themselves which one is present.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/mnist"
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
mkdir -p "$DIR"

for name in train-images-idx3 train-labels-idx1 t10k-images-idx3 t10k-labels-idx1; do
    file="${name}-ubyte"
    if [ -f "$DIR/$file" ]; then
        echo "  already there: $file"
        continue
    fi
    echo "  descargando $file..."
    curl -sSL --fail -o "$DIR/$file.gz" "$BASE/$file.gz"
    gunzip -f "$DIR/$file.gz"
done

echo
echo "MNIST completo en $DIR"
echo "Run ./build/mnist_demo to train on it."
