#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cub/cub.cuh>

#include "bench.h"
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

int main(int argc, char** argv)
{
    const std::string       csvPath = bench::csvPathFromArgs(argc, argv);
    const bench::DeviceInfo dev     = bench::queryDevice();
    bench::printDevice(dev);

    const int    n     = 1 << 25;                    // 33.5M ints = 134 MB (much bigger than L2)
    const size_t bytes = size_t(n) * sizeof(int);

    // ---------- input: random digits 0..9, fixed seed ----------
    std::vector<int> h_in(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 9);
    for (int& x : h_in) {
        x = dist(rng);
    }

    // ---------- CPU baseline (also the reference result) ----------
    const long long    cpu_sum = sumCPU(h_in.data(), n);
    const bench::Stats cpu     = bench::timeCpu([&] {
        long long s = sumCPU(h_in.data(), n);
        bench::doNotOptimize(s);
    });

    // ---------- GPU buffers ----------
    const int maxBlocks = (n + BLOCK - 1) / BLOCK;
    int *d_in = nullptr, *d_bufA = nullptr, *d_bufB = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_bufA, maxBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bufB, maxBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int        result = 0;
    const int* d_res  = nullptr;

    // ---------- library reference first (CUB), so every row can report % of library ----------
    void*  d_temp     = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    const bench::Stats cub_t = bench::timeGpu([&] {
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n));
    });
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    const int cub_result = result;

    // ---------- results table + CSV ----------
    bench::CsvWriter csv(csvPath);
    std::printf("\nn = %d (%.1f MB), block = %d threads, CPU sum = %lld\n", n, bytes / 1e6, BLOCK, cpu_sum);
    std::printf("Times: median of %d trials (p10/p90 spread), after a time-based GPU warm-up\n\n",
                bench::TimingConfig{}.trials);
    std::printf("%-22s %9s %9s %9s %8s %7s %8s %12s %6s\n",
                "version", "ms", "p10", "p90", "GB/s", "%peak", "vs CPU", "sum", "check");

    auto report = [&](const char* name, const bench::Stats& t, long long sum, bool onGpu) {
        const double gbps = bytes / (t.median / 1e3) / 1e9;
        const bool   pass = (sum == cpu_sum);
        char pct[16];
        if (onGpu) {
            std::snprintf(pct, sizeof pct, "%.1f%%", 100.0 * gbps / dev.peak_gbps);
        } else {
            std::snprintf(pct, sizeof pct, "-");
        }
        std::printf("%-22s %9.4f %9.4f %9.4f %8.1f %7s %8.1f %12lld %6s\n",
                    name, t.median, t.p10, t.p90, gbps, pct, cpu.median / t.median, sum,
                    pass ? "PASS" : "FAIL");

        bench::Row row;
        row.level     = "2_reduction";
        row.kernel    = name;
        row.n         = n;
        row.t         = t;
        row.bytes     = static_cast<double>(bytes);
        row.peak_gbps = onGpu ? dev.peak_gbps : 0.0;
        row.lib_ms    = onGpu ? cub_t.median : 0.0;
        row.max_err   = static_cast<double>(std::llabs(sum - cpu_sum));
        row.pass      = pass;
        csv.write(row);
    };

    report("CPU (1 core)", cpu, cpu_sum, false);
    report("CUB DeviceReduce::Sum", cub_t, cub_result, true);

    // ---------- v0: atomics ----------
    const bench::Stats t0 = bench::timeGpu([&] {
        CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(int)));   // reset the counter each run
        reduceV0Atomic<<<maxBlocks, BLOCK>>>(d_in, d_out, n);
    });
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    report("v0 atomicAdd", t0, result, true);

    // ---------- v1..v5: multi-pass tree reductions ----------
    auto runTree = [&](const char* name, ReduceKernel kernel, int elemsPerBlock) {
        const bench::Stats t = bench::timeGpu([&] {
            d_res = reduceMultiPass(kernel, elemsPerBlock, d_in, d_bufA, d_bufB, n);
        });
        CUDA_CHECK(cudaMemcpy(&result, d_res, sizeof(int), cudaMemcpyDeviceToHost));
        report(name, t, result, true);
    };
    runTree("v1 tree interleaved",  reduceV1, BLOCK);
    runTree("v2 tree strided",      reduceV2, BLOCK);
    runTree("v3 tree sequential",   reduceV3, BLOCK);
    runTree("v4 first add on load", reduceV4, 2 * BLOCK);
    runTree("v5 warp shuffle",      reduceV5, 2 * BLOCK);

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_bufA));
    CUDA_CHECK(cudaFree(d_bufB));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
