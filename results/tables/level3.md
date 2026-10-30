| version | ms (median) | range across runs | GFLOPS | % of FP32 ceiling | % of cuBLAS | max err | check |
|---|---|---|---|---|---|---|---|
| CPU (1 core, ikj) | 2561.923 | 2218.346–2574.875 | 7 | — | — | 3.8e-05 | PASS |
| v0 naive (uncoalesced) | 85.556 | 85.548–85.570 | 201 | 1.5% | 2.4% | 0.0e+00 | PASS |
| v1 naive (coalesced) | 20.180 | 20.171–20.194 | 851 | 6.5% | 10.3% | 0.0e+00 | PASS |
| v2 tiled (shared mem) | 15.497 | 15.485–15.530 | 1109 | 8.4% | 13.5% | 0.0e+00 | PASS |
| v3 register tiling 4x4 | 3.103 | 3.098–3.105 | 5536 | 42.1% | 67.2% | 0.0e+00 | PASS |
| v4 float4 + transposed A | 2.955 | 2.944–2.957 | 5814 | 44.2% | 70.6% | 0.0e+00 | PASS |
| cuBLAS sgemm (reference) | 2.087 | 2.078–2.090 | 8232 | 62.6% | 100.0% | 0.0e+00 | PASS |
| cuBLAS TF32 Tensor Cores | 1.346 | 1.342–1.347 | 12767 | — | 155.1% | 2.0e-02 | PASS |
