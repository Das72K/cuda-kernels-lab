| Level | Workload | Bound by | Best own kernel | vs library | vs measured ceiling |
|---|---|---|---|---|---|
| 1 Vector add | 100M floats | memory | 180 GB/s | — | 96% of 188 GB/s |
| 2 Reduction | 33.6M ints | memory | v5: 184 GB/s | 99% of CUB | 98% of 188 GB/s |
| 3 SGEMM | n = 2048, FP32 | compute | v4: 5.82 TFLOPS | 71% of cuBLAS | 44% of 13.1 TFLOPS |
| 4 Gaussian blur 7×7 | 8.3 MP image | memory | v4: 0.46 ms (145 GB/s eff.) | 2.3× NPP speed | 77% of 188 GB/s |

Ceilings measured on this GPU by `bench/peak.cu`: DRAM 187.6 GB/s (best read test), FP32 13.14 TFLOPS (FMA test). All values: median across 3 full runs.
