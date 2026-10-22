#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_utils.h"

constexpr int BLOCK = 256;  // threads per block (power of 2, needed by the tree)

// ============================================================
// v0: every thread atomically adds its element to ONE global counter
// ============================================================
__global__ void reduceV0Atomic(const int* in, int* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out, in[i]);
    }
}

// ============================================================
// v1: tree reduction in shared memory, interleaved addressing
//     (simple but slow: divergent branches + slow modulo)
// ============================================================
__global__ void reduceV1(const int* in, int* out, int n)
{
    __shared__ int sdata[BLOCK];

    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. each thread copies one element from global memory to shared memory
    sdata[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    // 2. tournament: in round s, thread tid adds the value of its partner tid + s
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. thread 0 writes this block's partial sum
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// v2: tree reduction, strided index -> active threads are contiguous (no divergence)
//     (new problem: shared memory bank conflicts)
// ============================================================
__global__ void reduceV2(const int* in, int* out, int n)
{
    __shared__ int sdata[BLOCK];

    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int index = 2 * s * tid;      // thread tid works on slot 2*s*tid
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// v3: tree reduction, sequential addressing ("fold in half")
//     -> contiguous active threads AND no shared memory bank conflicts
// ============================================================
__global__ void reduceV3(const int* in, int* out, int n)
{
    __shared__ int sdata[BLOCK];

    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {   // s = 128, 64, 32, ..., 1
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// v4: first add during load -> each block reduces 2 * BLOCK elements
//     (no idle threads in the first round, half as many blocks)
// ============================================================
__global__ void reduceV4(const int* in, int* out, int n)
{
    __shared__ int sdata[BLOCK];

    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // each thread loads two elements, blockDim.x apart (both loads stay coalesced), and adds them
    int sum = (i < n) ? in[i] : 0;
    if (i + blockDim.x < n) {
        sum += in[i + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// v5: like v4, but the last 32 values are reduced with warp shuffles
//     (no shared memory and no __syncthreads for the final rounds)
//     requires BLOCK >= 64
// ============================================================
__global__ void reduceV5(const int* in, int* out, int n)
{
    __shared__ int sdata[BLOCK];

    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    int sum = (i < n) ? in[i] : 0;
    if (i + blockDim.x < n) {
        sum += in[i + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    // tree in shared memory only while more than one warp is needed
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // last 64 values: warp 0 finishes in registers
    if (tid < 32) {
        int v = sdata[tid] + sdata[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (tid == 0) {
            out[blockIdx.x] = v;
        }
    }
}

// ============================================================
// Host helpers
// ============================================================
long long sumCPU(const int* in, int n)
{
    long long s = 0;
    for (int i = 0; i < n; i++) {
        s += in[i];
    }
    return s;
}

// Stops the compiler from deleting "unused" benchmark work (same idea as Level 1's barrier)
template <typename T>
inline void doNotOptimize(const T& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

using ReduceKernel = void (*)(const int*, int*, int);

// Launches `kernel` repeatedly: n values -> one partial sum per block -> ... -> 1 value.
// Returns a device pointer to the final result.
const int* reduceMultiPass(ReduceKernel kernel, int elemsPerBlock,
                           const int* d_in, int* d_bufA, int* d_bufB, int n)
{
    const int* src = d_in;
    int*       dst = d_bufA;
    while (n > 1) {
        int blocks = (n + elemsPerBlock - 1) / elemsPerBlock;
        kernel<<<blocks, BLOCK>>>(src, dst, n);
        n   = blocks;
        src = dst;
        dst = (dst == d_bufA) ? d_bufB : d_bufA;
    }
    return src;
}

// Warm-up once, then average `reps` timed runs of `launch` with CUDA events
template <typename F>
float timeGpu(F launch, int reps)
{
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int k = 0; k < reps; k++) {
        launch();
    }
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    return timer.elapsed_ms() / reps;
}

int main()
{
    const int    n     = 1 << 25;                    // 33.5M ints = 128 MB (much bigger than L2)
    const size_t bytes = size_t(n) * sizeof(int);
    const int    reps  = 20;

    // ---------- input: random digits 0..9, fixed seed so every run is identical ----------
    std::vector<int> h_in(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 9);
    for (int& x : h_in) {
        x = dist(rng);
    }

    // ---------- CPU baseline (also the reference result) ----------
    const long long cpu_sum = sumCPU(h_in.data(), n);   // warm-up + reference
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < reps; k++) {
        long long s = sumCPU(h_in.data(), n);
        doNotOptimize(s);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    // ---------- GPU buffers ----------
    const int maxBlocks = (n + BLOCK - 1) / BLOCK;
    int *d_in = nullptr, *d_bufA = nullptr, *d_bufB = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_bufA, maxBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bufB, maxBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // ---------- results table ----------
    std::printf("n = %d (%.1f MB), block = %d threads, CPU sum = %lld\n\n",
                n, bytes / 1e6, BLOCK, cpu_sum);
    std::printf("%-22s %10s %10s %9s %12s %6s\n", "version", "ms", "GB/s", "vs CPU", "sum", "check");
    std::printf("%-22s %10.4f %10.1f %9s %12lld %6s\n", "CPU (1 core)", cpu_ms,
                bytes / (cpu_ms / 1000.0) / 1e9, "1.0", cpu_sum, "ref");

    auto printRow = [&](const char* name, float ms, long long result) {
        double gbps = bytes / (ms / 1000.0) / 1e9;
        std::printf("%-22s %10.4f %10.1f %9.1f %12lld %6s\n", name, ms, gbps, cpu_ms / ms,
                    result, result == cpu_sum ? "PASS" : "FAIL");
    };

    int result = 0;

    // ---------- v0: atomics ----------
    float ms = timeGpu([&] {
        CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(int)));   // reset the counter each run
        reduceV0Atomic<<<maxBlocks, BLOCK>>>(d_in, d_out, n);
    }, reps);
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    printRow("v0 atomicAdd", ms, result);

    // ---------- v1: shared-memory tree, interleaved ----------
    const int* d_res = nullptr;
    ms = timeGpu([&] {
        d_res = reduceMultiPass(reduceV1, BLOCK, d_in, d_bufA, d_bufB, n);
    }, reps);
    CUDA_CHECK(cudaMemcpy(&result, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    printRow("v1 tree interleaved", ms, result);

    // ---------- v2: shared-memory tree, strided index ----------
    ms = timeGpu([&] {
        d_res = reduceMultiPass(reduceV2, BLOCK, d_in, d_bufA, d_bufB, n);
    }, reps);
    CUDA_CHECK(cudaMemcpy(&result, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    printRow("v2 tree strided", ms, result);

    // ---------- v3: shared-memory tree, sequential addressing ----------
    ms = timeGpu([&] {
        d_res = reduceMultiPass(reduceV3, BLOCK, d_in, d_bufA, d_bufB, n);
    }, reps);
    CUDA_CHECK(cudaMemcpy(&result, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    printRow("v3 tree sequential", ms, result);

    // ---------- v4: first add during load (2 * BLOCK elements per block) ----------
    ms = timeGpu([&] {
        d_res = reduceMultiPass(reduceV4, 2 * BLOCK, d_in, d_bufA, d_bufB, n);
    }, reps);
    CUDA_CHECK(cudaMemcpy(&result, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    printRow("v4 first add on load", ms, result);

    // ---------- v5: v4 + warp shuffle for the last warp ----------
    ms = timeGpu([&] {
        d_res = reduceMultiPass(reduceV5, 2 * BLOCK, d_in, d_bufA, d_bufB, n);
    }, reps);
    CUDA_CHECK(cudaMemcpy(&result, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    printRow("v5 warp shuffle", ms, result);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_bufA));
    CUDA_CHECK(cudaFree(d_bufB));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
