#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>

#include "cuda_utils.h"

// GPU: one thread per element
__global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;  // global index of this thread

    if (i < n) {                                    // skip extra threads in the last block
        c[i] = a[i] + b[i];
    }
}

// CPU baseline: one core, one loop
void vectorAddCPU(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// All measurements for one problem size
struct Result {
    double cpu_ms;
    float  kernel_ms;
    float  total_ms;
    float  total_pinned_ms;
    double gbps;        // kernel bandwidth
    int    errors;
};

Result runBenchmark(int n, int reps)
{
    const size_t bytes = n * sizeof(float);
    Result r{};

    // ---------- Host data ----------
    std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // ---------- CPU: warm-up, then timed runs ----------
    vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < reps; k++) {
        vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);
        asm volatile("" : : : "memory");   // compiler barrier: forces every repeat to really happen
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    // ---------- GPU memory ----------
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // ---------- GPU warm-up ----------
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;

    // ---------- GPU: kernel only ----------
    timer.start();
    for (int k = 0; k < reps; k++) {
        vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    }
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = timer.elapsed_ms() / reps;

    // ---------- GPU: total trip ----------
    timer.start();
    for (int k = 0; k < reps; k++) {
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
        vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    }
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    r.total_ms = timer.elapsed_ms() / reps;

    // ---------- GPU: total trip, pinned host memory ----------
    float *p_a = nullptr, *p_b = nullptr, *p_c = nullptr;
    CUDA_CHECK(cudaMallocHost(&p_a, bytes));
    CUDA_CHECK(cudaMallocHost(&p_b, bytes));
    CUDA_CHECK(cudaMallocHost(&p_c, bytes));
    std::memcpy(p_a, h_a.data(), bytes);
    std::memcpy(p_b, h_b.data(), bytes);

    timer.start();
    for (int k = 0; k < reps; k++) {
        CUDA_CHECK(cudaMemcpy(d_a, p_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, p_b, bytes, cudaMemcpyHostToDevice));
        vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaMemcpy(p_c, d_c, bytes, cudaMemcpyDeviceToHost));
    }
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    r.total_pinned_ms = timer.elapsed_ms() / reps;

    for (int i = 0; i < n; i++) {
        if (p_c[i] != h_c_cpu[i]) {
            r.errors++;
        }
    }

    CUDA_CHECK(cudaFreeHost(p_a));
    CUDA_CHECK(cudaFreeHost(p_b));
    CUDA_CHECK(cudaFreeHost(p_c));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // ---------- Correctness + bandwidth ----------
    for (int i = 0; i < n; i++) {
        if (h_c_gpu[i] != h_c_cpu[i]) {
            r.errors++;
        }
    }
    r.gbps = (3.0 * bytes) / (r.kernel_ms / 1000.0) / 1e9;

    return r;
}

int main()
{
    const int sizes[] = { 1'000, 10'000, 100'000, 1'000'000,
                          10'000'000, 50'000'000, 100'000'000 };

    std::printf("%12s %9s %10s %10s %9s %9s %10s %9s %10s %9s %6s\n",
                "n", "MB/array", "CPU ms", "kern ms", "kern GB/s",
                "kern x", "total ms", "total x", "pinned ms", "pinned x", "check");

    for (int n : sizes) {
        int reps = (n <= 1'000'000) ? 100 : 10;   // more repeats for tiny, noisy sizes
        Result r = runBenchmark(n, reps);

        std::printf("%12d %9.1f %10.4f %10.4f %9.1f %9.1f %10.4f %9.2f %10.4f %9.2f %6s\n",
                    n, n * sizeof(float) / 1e6,
                    r.cpu_ms, r.kernel_ms, r.gbps, r.cpu_ms / r.kernel_ms,
                    r.total_ms, r.cpu_ms / r.total_ms,
                    r.total_pinned_ms, r.cpu_ms / r.total_pinned_ms,
                    r.errors == 0 ? "PASS" : "FAIL");
    }
    return 0;
}