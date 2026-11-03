| test | ms (median) | range across runs | result | % of 192 GB/s spec |
|---|---|---|---|---|
| copy (read + write) | 6.130 | 6.124–6.141 | 175.2 GB/s | 91.2% |
| read-only | 2.862 | 2.861–2.863 | 187.6 GB/s | 97.7% |
| copy unrolled x4 | 6.168 | 6.159–6.174 | 174.1 GB/s | 90.7% |
| read-only unrolled x4 | 2.861 | 2.860–2.869 | 187.6 GB/s | 97.7% |
| FMA (FP32) | 2.451 | 2.442–2.453 | 13.14 TFLOPS | — |
