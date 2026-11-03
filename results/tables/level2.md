| version | ms (median) | range across runs | GB/s | % of measured ceiling | % of CUB | check |
|---|---|---|---|---|---|---|
| CPU (1 core) | 11.8881 | 10.7131–12.7918 | 11.3 | — | — | PASS |
| v0 atomicAdd | 0.7186 | 0.7186–0.7237 | 186.8 | 99.5% | 100.5% | PASS |
| v1 tree interleaved | 1.7963 | 1.7929–1.8003 | 74.7 | 39.8% | 40.2% | PASS |
| v2 tree strided | 1.2576 | 1.2544–1.2579 | 106.7 | 56.9% | 57.4% | PASS |
| v3 tree sequential | 1.2160 | 1.2159–1.2194 | 110.4 | 58.8% | 59.3% | PASS |
| v4 first add on load | 0.7303 | 0.7286–0.7305 | 183.8 | 98.0% | 98.9% | PASS |
| v5 warp shuffle | 0.7289 | 0.7265–0.7304 | 184.1 | 98.1% | 99.0% | PASS |
| CUB DeviceReduce::Sum | 0.7221 | 0.7201–0.7228 | 185.9 | 99.1% | 100.0% | PASS |
