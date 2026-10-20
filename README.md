# cuda-kernels-lab

Hand-written CUDA kernels, built up level by level and benchmarked.

**Target:** NVIDIA GeForce RTX 4050 Laptop GPU (Ada, `sm_89`)

## Layout

```
.
├── CMakeLists.txt
├── common/                  # shared helpers (CUDA_CHECK, GpuTimer)
└── src/
    └── level1_vector_add/
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Binaries land in `build/bin/`. Kernels are compiled with `-lineinfo` so Nsight Compute can map metrics back to source.

## Results

| Level | Kernel | Problem size | Time (ms) | Throughput | % of peak | Notes |
|-------|--------|--------------|-----------|------------|-----------|-------|
|       |        |              |           |            |           |       |

## Level 1: Vector Add

The kernel computes `c[i] = a[i] + b[i]`, with one thread per element.

**Test setup:** RTX 4050 Laptop GPU (`sm_89`, 6 GB), CUDA 13.3, WSL Ubuntu 24.04, Release build,
laptop plugged in. CPU baseline = single-threaded loop.

|           n | MB/array | CPU ms | kern ms | kern GB/s | kern x | total ms | total x | pinned ms | pinned x | check |
|------------:|---------:|-------:|--------:|----------:|-------:|---------:|--------:|----------:|---------:|:-----:|
|       1,000 |      0.0 | 0.0003 |  0.0212 |       0.6 |    0.0 |   0.1685 |    0.00 |    0.2406 |     0.00 | PASS  |
|      10,000 |      0.0 | 0.0042 |  0.0231 |       5.2 |    0.2 |   0.1548 |    0.03 |    0.2095 |     0.02 | PASS  |
|     100,000 |      0.4 | 0.0415 |  0.0106 |     112.9 |    3.9 |   0.3496 |    0.12 |    0.3476 |     0.12 | PASS  |
|   1,000,000 |      4.0 | 0.5282 |  0.0243 |     493.8 |   21.7 |   2.1604 |    0.24 |    1.6040 |     0.33 | PASS  |
|  10,000,000 |     40.0 | 9.4933 |  0.7223 |     166.1 |   13.1 |  27.4110 |    0.35 |   11.2420 |     0.84 | PASS  |
|  50,000,000 |    200.0 | 48.5073 | 3.5075 |     171.1 |   13.8 | 129.8434 |    0.37 |   53.0997 |     0.91 | PASS  |
| 100,000,000 |    400.0 | 86.1964 | 8.9903 |     133.5 |    9.6 | 200.7804 |    0.43 |  120.4299 |     0.72 | PASS  |

### Findings

- **Kernel bandwidth and the L2 cache effect:** For large arrays (10M–50M elements) the kernel
  reaches ~170 GB/s, roughly 90% of the RTX 4050 Laptop's ~190 GB/s VRAM bandwidth. At 1M elements
  the measured "bandwidth" (~500–750 GB/s) exceeds the VRAM limit: the 12 MB working set fits in
  the L2 cache, so repeated runs read from cache, not VRAM. Benchmarks must use data larger than
  L2 to measure real memory bandwidth.

- **Kernel vs CPU speed-up:** Vector add is memory-bound (12 bytes moved per addition), so the
  speed-up is set by the ratio of memory bandwidths: ~170 GB/s on the GPU vs ~14 GB/s for one CPU
  core gives the measured 10–14×. Below ~100K elements the fixed launch overhead (~10–20 µs, varies between runs) dominates and the
  CPU is faster.

- **Transfers dominate (pageable vs pinned):** Including host↔device copies, the GPU never beats
  the CPU. Pageable copies run at ~4.5–5 GB/s because the driver stages them through a page-locked
  buffer; pinned memory (`cudaMallocHost`) lets the copy engine read directly and raises copy throughput
  ~2.5× to ~11–12 GB/s (2.4× faster total at 10M–50M). Even then the best result is 0.91× the CPU: copying
  over PCIe costs more than the single addition it enables. GPUs pay off when data stays on the
  device or the work per byte is high (the goal of Level 3, matrix multiplication).

- **Laptop noise and throttling:** Short runs (<100 µs) vary between runs due to GPU clock changes.
  At 100M the kernel dropped to ~134 GB/s, consistent with the laptop GPU lowering clocks under
  sustained load. A CPU-side gotcha: in Release builds the compiler removed repeated identical CPU
  runs, making the CPU look ~100× faster; an `asm volatile("" ::: "memory")` barrier forces every
  repeat to execute.

### How to run

```bash
cmake -S . -B build && cmake --build build && ./build/bin/level1_vector_add
```
