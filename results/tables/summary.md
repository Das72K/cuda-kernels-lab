| Level | Workload | Bound by | Best own kernel | vs library | vs measured ceiling |
|---|---|---|---|---|---|
| 1 Vector add | 100M floats | memory | 180 GB/s | — | 96% of 188 GB/s |
| 2 Reduction | 33.6M ints | memory | v5: 185 GB/s | 99% of CUB | 98% of 188 GB/s |
| 3 SGEMM | n = 2048, FP32 | compute | v4: 5.81 TFLOPS | 71% of cuBLAS | 44% of 13.2 TFLOPS |

Ceilings measured on this GPU by `bench/peak.cu`: DRAM 187.6 GB/s (best read test), FP32 13.16 TFLOPS (FMA test). All values: median across 3 full runs.
