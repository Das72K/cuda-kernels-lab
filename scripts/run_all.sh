#!/usr/bin/env bash
# Regenerates every result in results/ from scratch.
# Run from anywhere, laptop plugged in, Windows power mode "Best performance", other apps closed.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=build
RAW=results/raw
COOLDOWN=10   # seconds of idle between programs, so heat from one doesn't slow the next
mkdir -p "$RAW"

echo "== Build (Release) =="
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" -j

echo "== Environment =="
{
    echo "date: $(date -Iseconds)"
    dirty=""
    git diff --quiet || dirty="-dirty"
    echo "git_commit: $(git rev-parse --short HEAD)${dirty}"
    echo "nvcc: $(nvcc --version | tail -n 1)"
    echo "cmake: $(cmake --version | head -n 1)"
    echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
    # BatteryStatus 2 = on AC power (Windows, queried through WSL interop; "unknown" if unavailable)
    ac=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d '\r' || true)
    echo "battery_status: ${ac:-unknown} (2 = on AC power)"
    nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.max_limit,temperature.gpu --format=csv
} | tee results/env.txt

# name of the binary -> name of its results files
run() {
    local bin=$1 out=$2
    echo
    echo "== $bin =="
    "./$BUILD/bin/$bin" --csv "$RAW/$out.csv" | tee "results/$out.txt"
    sleep "$COOLDOWN"
}

run peak_bench        peaks
run level1_vector_add level1_vector_add
run level2_reduction  level2_reduction
run level3_matmul     level3_matmul

echo
echo "Done. CSVs in $RAW/, console outputs in results/, environment in results/env.txt"
