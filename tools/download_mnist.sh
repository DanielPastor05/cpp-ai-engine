#!/usr/bin/env bash
# Descarga el conjunto MNIST completo (60.000 + 10.000 imágenes) en data/mnist/.
#
# El repositorio ya trae un subconjunto de 2.000 + 1.000 para que los ejemplos
# funcionen recién clonado; esto es para obtener el resultado real. Los ejemplos
# detectan automáticamente cuál está presente.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/mnist"
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
mkdir -p "$DIR"

for name in train-images-idx3 train-labels-idx1 t10k-images-idx3 t10k-labels-idx1; do
    file="${name}-ubyte"
    if [ -f "$DIR/$file" ]; then
        echo "  ya está: $file"
        continue
    fi
    echo "  descargando $file..."
    curl -sSL --fail -o "$DIR/$file.gz" "$BASE/$file.gz"
    gunzip -f "$DIR/$file.gz"
done

echo
echo "MNIST completo en $DIR"
echo "Ejecuta ./build/mnist_demo para entrenar con él."
