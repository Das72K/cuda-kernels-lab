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
