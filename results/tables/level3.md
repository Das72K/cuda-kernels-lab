| version | ms (median) | range across runs | GFLOPS | % of FP32 ceiling | % of cuBLAS | max err | check |
|---|---|---|---|---|---|---|---|
| CPU (1 core, ikj) | 2338.222 | 2236.766–2392.701 | 7 | — | — | 3.8e-05 | PASS |
| v0 naive (uncoalesced) | 85.549 | 85.089–85.564 | 201 | 1.5% | 2.4% | 0.0e+00 | PASS |
| v1 naive (coalesced) | 20.168 | 20.105–20.182 | 852 | 6.5% | 10.4% | 0.0e+00 | PASS |
| v2 tiled (shared mem) | 15.489 | 15.489–15.500 | 1109 | 8.4% | 13.5% | 0.0e+00 | PASS |
| v3 register tiling 4x4 | 3.106 | 3.089–3.107 | 5532 | 42.1% | 67.3% | 0.0e+00 | PASS |
| v4 float4 + transposed A | 2.950 | 2.939–2.951 | 5824 | 44.3% | 70.9% | 0.0e+00 | PASS |
| cuBLAS sgemm (reference) | 2.091 | 2.089–2.100 | 8217 | 62.5% | 100.0% | 0.0e+00 | PASS |
| cuBLAS TF32 Tensor Cores | 1.344 | 1.342–1.353 | 12783 | — | 155.4% | 2.0e-02 | PASS |
