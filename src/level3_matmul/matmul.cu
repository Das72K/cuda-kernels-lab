#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cublas_v2.h>

#include "bench_utils.h"
#include "cuda_utils.h"

#define CUBLAS_CHECK(call)                                                          \
    do {                                                                            \
        cublasStatus_t st_ = (call);                                                \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(st_), \
                         __FILE__, __LINE__);                                       \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

constexpr int TILE = 16;  // each block is TILE x TILE threads

// ============================================================
// v0: naive, one thread per C element, BAD mapping:
//     threadIdx.x -> row, so neighbouring threads read A and write C n floats apart
// ============================================================
__global__ void matmulV0(const float* A, const float* B, float* C, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ============================================================
// v1: naive, one thread per C element, GOOD mapping:
//     threadIdx.x -> col, so neighbouring threads read neighbouring B and C elements
// ============================================================
__global__ void matmulV1(const float* A, const float* B, float* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ============================================================
// CPU baseline: i-k-j loop order (walks rows of B and C, which is cache-friendly)
// ============================================================
void matmulCPU(const float* A, const float* B, float* C, int n)
{
    for (size_t i = 0; i < size_t(n) * n; i++) {
        C[i] = 0.0f;
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            const float a = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a * B[k * n + j];
            }
        }
    }
}

float maxAbsDiff(const std::vector<float>& x, const std::vector<float>& y)
{
    float m = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        m = std::max(m, std::fabs(x[i] - y[i]));
    }
    return m;
}

int main()
{
    const int    n     = 2048;
    const size_t elems = size_t(n) * n;
    const size_t bytes = elems * sizeof(float);
    const double flops = 2.0 * n * n * n;   // one multiply + one add per (i, j, k)
    const int    reps  = 3;
    const float  tol   = 1e-2f;             // floats: different summation order -> tiny differences

    // ---------- input: random floats in [-1, 1], fixed seed ----------
    std::vector<float> h_A(elems), h_B(elems), h_C(elems), h_ref(elems), h_cpu(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : h_A) x = dist(rng);
    for (float& x : h_B) x = dist(rng);

    // ---------- CPU baseline: one run only (it takes seconds) ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(h_A.data(), h_B.data(), h_cpu.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- GPU buffers ----------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // ---------- cuBLAS reference (also the correctness reference) ----------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major. Row-major C = A*B has the same memory layout as
    // column-major C^T = B^T * A^T, so we pass B first and A second.
    float cublas_ms = timeGpu([&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, d_B, n, d_A, n, &beta, d_C, n));
    }, reps);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    const float cpu_err = maxAbsDiff(h_cpu, h_ref);

    // ---------- results table ----------
    std::printf("n = %d (%.1f MB per matrix), %.2f GFLOP per multiply, block = %dx%d\n\n",
                n, bytes / 1e6, flops / 1e9, TILE, TILE);
    std::printf("%-26s %10s %10s %10s %10s %6s\n",
                "version", "ms", "GFLOPS", "% cuBLAS", "max err", "check");
    std::printf("%-26s %10.1f %10.1f %9.2f%% %10.2e %6s\n", "CPU (1 core, ikj)",
                cpu_ms, flops / (cpu_ms / 1000.0) / 1e9, 100.0 * cublas_ms / cpu_ms,
                cpu_err, cpu_err < tol ? "PASS" : "FAIL");

    auto printRow = [&](const char* name, float ms) {
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        const float  err    = maxAbsDiff(h_C, h_ref);
        const double gflops = flops / (ms / 1000.0) / 1e9;
        std::printf("%-26s %10.3f %10.1f %9.1f%% %10.2e %6s\n", name, ms, gflops,
                    100.0 * cublas_ms / ms, err, err < tol ? "PASS" : "FAIL");
    };

    printRow("cuBLAS sgemm (reference)", cublas_ms);

    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // ---------- v0: naive, uncoalesced ----------
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));   // a broken kernel must not "pass" with old results
    float ms = timeGpu([&] { matmulV0<<<grid, block>>>(d_A, d_B, d_C, n); }, reps);
    printRow("v0 naive (uncoalesced)", ms);

    // ---------- v1: naive, coalesced ----------
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    ms = timeGpu([&] { matmulV1<<<grid, block>>>(d_A, d_B, d_C, n); }, reps);
    printRow("v1 naive (coalesced)", ms);

    std::printf("\nGPU times are the median of 7 trials; CPU is a single run.\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
