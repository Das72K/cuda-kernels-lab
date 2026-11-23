# Profiling notes

## Level 3 (SGEMM): matmulV4 vs cuBLAS, Nsight Compute, n = 2048

| Metric | matmulV4 | cuBLAS `ampere_sgemm_64x64_nn` | matmulV5 | matmulV6 |
| --- | --- | --- | --- | --- |
| Threads per block | 256 | 64 | 256 | 256 |
| Outputs per thread | 4x4 | 8x8 | 8x8 | 8x8 |
| Registers | 56 | 126 | 127 | 127 |
| Occupancy | 65% | 33% | 32.2% | 32.2% |
| L1/TEX (Speed of Light) | 85.6% | 62.4% | 82.9% | 59.5% |
| Compute (SM) throughput | 64.5% | 73.1% | 60.7% | 65.9% |
| Cycles per issued instruction | 13.9 | 5.2 | 6.3 | 5.8 |
| Top stalls | MIO throttle 3.34, barrier 2.40, short scoreboard 2.11, long scoreboard 1.99 | almost no memory or barrier stalls | long scoreboard 0.94, barrier 0.93, short scoreboard 0.72, MIO throttle 0.50, dispatch stall 0.49, wait 0.10 | long scoreboard 0.95, barrier 0.84, dispatch stall 0.50, short scoreboard 0.32, MIO throttle 0.13, wait 0.11 |
| Shared load bank conflicts | 0 | 31,879 | 33,554,432 | 0 |
| Shared store bank conflicts | 0 | 908,309 | 0 | 0 |

Stalls are `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` from the raw page, leaving
out `selected` and `not_selected`, which are scheduling states rather than stalls.

### Conclusion

matmulV4 is shared-memory bound, not occupancy bound. ncu's "raise occupancy" hint is misleading
here: fewer registers would force a smaller register tile, which means more shared-memory traffic
per FMA, and shared memory is the actual bottleneck.

### Pitfall: ncu distorts timings

At n = 512, cuBLAS dropped from ~5.5 to ~2.1 TFLOPS under ncu. Timings come only from normal runs,
metrics come only from ncu runs.
