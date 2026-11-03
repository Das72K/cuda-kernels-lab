| version | ms (median) | range across runs | effective GB/s | % of DRAM ceiling | % of NPP speed | max err | check |
|---|---|---|---|---|---|---|---|
| CPU (1 core) | 194.462 | 194.404–195.365 | 0.3 | — | — | 0.0e+00 | PASS |
| v0 naive | 0.997 | 0.997–1.000 | 66.6 | 35.5% | 106.9% | 3.0e-07 | PASS |
| v1 constant-memory weights | 0.546 | 0.545–0.547 | 121.4 | 64.7% | 195.4% | 3.0e-07 | PASS |
| v2 shared tile + halo | 0.692 | 0.688–0.693 | 95.8 | 51.1% | 154.3% | 3.0e-07 | PASS |
| v3 separable, 2 passes | 0.729 | 0.729–0.731 | 91.0 | 48.5% | 146.2% | 4.8e-07 | PASS |
| v4 separable, fused smem | 0.458 | 0.455–0.461 | 144.9 | 77.2% | 232.7% | 4.8e-07 | PASS |
| NPP FilterBorder 7x7 | 1.066 | 1.066–1.068 | 62.2 | 33.2% | 100.0% | 3.0e-07 | PASS |
