#!/usr/bin/env bash
# Nsight Compute profile of every version: --profile mode runs one NVTX-labelled launch per version.
# Writes results/profile/<program>.csv (one row per kernel launch).
# Bytes and instruction counts do not depend on the clock; durations and percentages are measured at the
# clock the GPU reaches in profile mode (~2.06 GHz on this laptop, since there is no warm-up).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/profile
mkdir -p "$OUT"

METRICS=$(paste -sd, <<'EOF'
gpu__time_duration.sum
sm__cycles_elapsed.avg.per_second
dram__bytes_read.sum
dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
smsp__inst_executed_op_global_ld.sum
smsp__inst_executed_op_shared_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
sm__warps_active.avg.pct_of_peak_sustained_active
EOF
)

profile() {
    local bin=$1 out=$2
    echo "== ncu: $bin =="
    ncu --nvtx --clock-control none --metrics "$METRICS" --csv --page raw \
        --log-file "$OUT/$out.csv" "./build/bin/$bin" --profile > /dev/null
}

profile peak_bench          peaks
profile level1_vector_add   level1_vector_add
profile level2_reduction    level2_reduction
profile level3_matmul       level3_matmul
profile level4_image_filter level4_image_filter
echo "Done: $OUT/*.csv"
