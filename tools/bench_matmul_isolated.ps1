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
#   tools\bench_matmul_isolated.ps1 [-Size 4096] [-Iters 12] [-Repeats 3]
#
# The numbers in docs/CUDA.md come from this script, not from the sweep.
param(
    [int]$Size = 4096,
    [int]$Iters = 12,
    [int]$Repeats = 3
)

$bin = "build-cuda\Release\bench_matmul.exe"
if (-not (Test-Path $bin)) { $bin = "build-cuda\bench_matmul.exe" }
if (-not (Test-Path $bin)) {
    Write-Host "not built: configure with -DENGINE_CUDA=ON"
    exit 1
}

Write-Host "Isolated measurement: $($Size)^3, $Iters iterations, best of $Repeats"
Write-Host "One process per kernel, 2 s between them."
Write-Host ""
Write-Host ("  {0,-14} {1,12} {2,11}" -f "kernel", "GFLOP/s", "% of peak")

foreach ($kernel in @("naive", "tiled", "register", "vectorized", "tensorcore", "cublas")) {
    $bestG = 0.0
    $bestPct = ""
    for ($i = 0; $i -lt $Repeats; $i++) {
        $out = & $bin "--kernel=$kernel" "--size=$Size" "--iters=$Iters" 2>&1 |
               Select-String -Pattern "^\s+($kernel|cuBLAS)\s" | Select-Object -Last 1
        if (-not $out) { break }
        # Parse with the invariant culture so a locale that uses a decimal comma
        # does not silently read 4888.4 as 48884.
        if ($out -match '\s([\d]+\.[\d]+)\s+ms\s+([\d]+\.[\d]+)\s+([\d]+\.[\d]+)%') {
            $g = [double]::Parse($matches[2], [Globalization.CultureInfo]::InvariantCulture)
            if ($g -gt $bestG) { $bestG = $g; $bestPct = $matches[3] }
        }
        Start-Sleep -Seconds 2
    }
    if ($bestG -gt 0.0) {
        Write-Host ("  {0,-14} {1,12:N1} {2,10}%" -f $kernel, $bestG, $bestPct)
    } else {
        Write-Host ("  {0,-14} {1,12}" -f $kernel, "unavailable")
    }
}
