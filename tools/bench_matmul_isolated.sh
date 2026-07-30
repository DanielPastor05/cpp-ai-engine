#!/usr/bin/env bash
# One process per kernel, because a single sweep cannot compare them.
#
# bench_matmul's built-in sweep runs every kernel back to back in one process. On
# a consumer card that measures temperature as much as code: by the time the
# fifth kernel runs, the clocks have dropped, and the table's own row ordering
# biases its conclusion. Measured on an RTX 3060 Ti, the same kernel came out at
# 4888 GFLOP/s inside a sweep and 7903 GFLOP/s on its own -- a factor of 1.6,
# which is larger than most of the differences the table is trying to show.
#
# This runs each kernel in a fresh process with a pause between, and keeps the
# best of several attempts. Noise in a microbenchmark only ever adds time, so the
# fastest observation is the honest estimate.
#
#   tools/bench_matmul_isolated.sh [size] [iters] [repeats]
#
# The numbers in docs/CUDA.md come from this script, not from the sweep.
set -euo pipefail

SIZE="${1:-4096}"
ITERS="${2:-12}"
REPEATS="${3:-3}"

BIN=build-cuda/bench_matmul
[ -x "build-cuda/Release/bench_matmul.exe" ] && BIN=build-cuda/Release/bench_matmul.exe
[ -x "$BIN" ] || { echo "not built: configure with -DENGINE_CUDA=ON"; exit 1; }

echo "Isolated measurement: ${SIZE}^3, ${ITERS} iterations, best of ${REPEATS}"
echo "One process per kernel, 2 s between them."
echo
printf '  %-14s %12s %11s\n' kernel GFLOP/s '% of peak'

for kernel in naive tiled register vectorized tensorcore cublas; do
    best=""
    for _ in $(seq "$REPEATS"); do
        line=$("$BIN" "--kernel=$kernel" "--size=$SIZE" "--iters=$ITERS" 2>/dev/null |
               grep -E "^\s+(${kernel}|cuBLAS)\s" | tail -1 || true)
        [ -n "$line" ] || break
        # time  GFLOP/s  %peak -> keep the row with the largest GFLOP/s
        g=$(echo "$line" | awk '{print $3}')
        if [ -z "$best" ] || awk "BEGIN{exit !($g > $bestg)}"; then best="$line"; bestg="$g"; fi
        sleep 2
    done
    if [ -n "$best" ]; then
        echo "$best" | awk '{printf "  %-14s %12s %11s\n", $1, $3, $4}'
    else
        printf '  %-14s %12s\n' "$kernel" "unavailable"
    fi
    unset bestg
done
