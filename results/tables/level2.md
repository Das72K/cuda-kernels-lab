| version | ms (median) | range across runs | GB/s | % of measured ceiling | % of CUB | check |
|---|---|---|---|---|---|---|
| CPU (1 core) | 10.6980 | 10.6271–12.6679 | 12.5 | — | — | PASS |
| v0 atomicAdd | 0.7227 | 0.7212–0.7229 | 185.7 | 99.0% | 99.9% | PASS |
| v1 tree interleaved | 1.7985 | 1.7932–1.8026 | 74.6 | 39.8% | 40.1% | PASS |
| v2 tree strided | 1.2577 | 1.2548–1.2621 | 106.7 | 56.9% | 57.4% | PASS |
| v3 tree sequential | 1.2186 | 1.2134–1.2197 | 110.1 | 58.7% | 59.2% | PASS |
| v4 first add on load | 0.7275 | 0.7274–0.7306 | 184.5 | 98.3% | 99.2% | PASS |
| v5 warp shuffle | 0.7271 | 0.7268–0.7298 | 184.6 | 98.4% | 99.3% | PASS |
| CUB DeviceReduce::Sum | 0.7220 | 0.7218–0.7237 | 185.9 | 99.1% | 100.0% | PASS |
