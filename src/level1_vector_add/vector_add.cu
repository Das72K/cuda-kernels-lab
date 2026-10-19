#include <cstdio>
#include <vector>

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
    const size_t bytes = n * sizeof(float);   // 4 MB per array

    // ---------- 1. Input data on the CPU (host) ----------
    std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);

    // ---------- 2. Rent GPU workspace (device) ----------
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ---------- 3. Bring the ingredients: RAM -> VRAM ----------
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // ---------- 4. Send the cooks: launch the kernel ----------
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;   // 3907
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());        // was the launch itself valid?
    CUDA_CHECK(cudaDeviceSynchronize());   // wait for the kernel, catch errors inside it

    // ---------- 5. Bring the food out: VRAM -> RAM ----------
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // ---------- 6. Return the workspace ----------
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // ---------- 7. Correctness: GPU vs CPU ----------
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c_gpu[i] != h_c_cpu[i]) {
            errors++;
        }
    }
    std::printf("n = %d, blocks = %d, threads/block = %d\n", n, blocks, threadsPerBlock);
    std::printf("%s (%d mismatches)\n", errors == 0 ? "PASS" : "FAIL", errors);

    return errors == 0 ? 0 : 1;
}