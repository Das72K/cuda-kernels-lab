#include <cstdio>
#include <vector>
#include <chrono>

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

int main()
{
    // ---------- 0. Problem size ----------
    const int    n     = 1'000'000;
    const size_t bytes = n * sizeof(float);
    const int    reps  = 20;                      // repeat timed runs, then average

    // ---------- 1. Host data ----------
    std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // ---------- 2. CPU timing ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++) {
        vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    // ---------- 3. GPU memory ----------
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // ---------- 4. Warm-up (not timed) ----------
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;

    // ---------- 5. GPU timing: kernel only ----------
    timer.start();
    for (int r = 0; r < reps; r++) {
        vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    }
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    float kernel_ms = timer.elapsed_ms() / reps;

    // ---------- 6. GPU timing: total trip (copy in + kernel + copy out) ----------
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    timer.stop();
    float total_ms = timer.elapsed_ms();

    // ---------- 7. Free GPU memory ----------
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // ---------- 8. Correctness ----------
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c_gpu[i] != h_c_cpu[i]) {
            errors++;
        }
    }

    // ---------- 9. Results ----------
    // Kernel moves 3 arrays: reads a and b, writes c
    double gbps = (3.0 * bytes) / (kernel_ms / 1000.0) / 1e9;

    std::printf("n = %d (%.1f MB per array)\n", n, bytes / 1e6);
    std::printf("CPU               : %8.3f ms\n", cpu_ms);
    std::printf("GPU kernel only   : %8.3f ms  (%.1f GB/s, %.1fx vs CPU)\n",
                kernel_ms, gbps, cpu_ms / kernel_ms);
    std::printf("GPU total (+copy) : %8.3f ms  (%.2fx vs CPU)\n",
                total_ms, cpu_ms / total_ms);
    std::printf("%s (%d mismatches)\n", errors == 0 ? "PASS" : "FAIL", errors);

    return errors == 0 ? 0 : 1;
}