| test | ms (median) | range across runs | result | % of 192 GB/s spec |
|---|---|---|---|---|
| copy (read + write) | 6.126 | 6.125–6.129 | 175.3 GB/s | 91.3% |
| read-only | 2.862 | 2.861–2.872 | 187.6 GB/s | 97.7% |
| copy unrolled x4 | 6.167 | 6.149–6.170 | 174.1 GB/s | 90.7% |
| read-only unrolled x4 | 2.862 | 2.861–2.862 | 187.6 GB/s | 97.7% |
| FMA (FP32) | 2.448 | 2.437–2.449 | 13.16 TFLOPS | — |
