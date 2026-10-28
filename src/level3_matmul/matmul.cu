#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "bench.h"
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
constexpr float kTol = 1e-2f;   // max allowed abs error vs cuBLAS (FP32)

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
// v2: tiled with shared memory
//     each block loads TILE x TILE tiles of A and B into shared memory once,
//     then every thread reuses them TILE times -> TILE x less global memory traffic
// ============================================================
__global__ void matmulV2(const float* A, const float* B, float* C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        // 1. each thread loads ONE element of the A tile and ONE of the B tile (coalesced)
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;
        As[ty][tx] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();   // the whole tile must be loaded before anyone uses it

        // 2. partial dot product using only shared memory
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();   // everyone must be done with this tile before it is overwritten
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// ============================================================
// v3: 2D register tiling
//     block computes a BM x BN tile of C with (BM/TM) x (BN/TN) = 16 x 16 threads,
//     each thread computes TM x TN = 4 x 4 outputs held in registers.
//     Per k step: TM + TN = 8 shared loads for TM * TN = 16 FMAs (v2: 2 loads per FMA)
//     Requires n % BM == 0 and n % BN == 0 and n % BK == 0.
// ============================================================
constexpr int BM = 64, BN = 64, BK = 8;   // block tile sizes
constexpr int TM = 4,  TN = 4;            // per-thread tile sizes
constexpr int V3_THREADS = (BM / TM) * (BN / TN);   // 256

__global__ void matmulV3(const float* A, const float* B, float* C, int n)
{
    __shared__ float As[BM][BK];   // 64 x 8 tile of A
    __shared__ float Bs[BK][BN];   // 8 x 64 tile of B

    const int tx  = threadIdx.x;              // 0..15 -> which group of TN columns
    const int ty  = threadIdx.y;              // 0..15 -> which group of TM rows
    const int tid = ty * blockDim.x + tx;     // 0..255, linear id for loading

    const int rowBase = blockIdx.y * BM;      // top-left corner of this block's C tile
    const int colBase = blockIdx.x * BN;

    float acc[TM][TN] = {};                   // 16 running sums, all in registers

    for (int t = 0; t < n; t += BK) {
        // 1. cooperative load: 512 floats of A and 512 of B, 2 of each per thread
        #pragma unroll
        for (int l = 0; l < (BM * BK) / V3_THREADS; l++) {
            const int idx = tid + l * V3_THREADS;
            const int r = idx / BK, c = idx % BK;
            As[r][c] = A[(rowBase + r) * n + (t + c)];
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / V3_THREADS; l++) {
            const int idx = tid + l * V3_THREADS;
            const int r = idx / BN, c = idx % BN;
            Bs[r][c] = B[(t + r) * n + (colBase + c)];
        }
        __syncthreads();

        // 2. for each k: 4 values of A and 4 of B into registers, then 16 FMAs
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // 3. write the 4 x 4 results
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            C[(rowBase + ty * TM + i) * n + (colBase + tx * TN + j)] = acc[i][j];
}

// ============================================================
// v4: v3 + vectorized float4 shared-memory reads
//     - A tile stored TRANSPOSED (AsT[k][row]) so a thread's 4 A values are contiguous
//     - +4 padding per row removes the bank conflicts of the transposed stores
//     - per k step: 2 float4 loads instead of 8 scalar loads; float4 stores of C
// ============================================================
constexpr int PAD = 4;
static_assert(TM == 4 && TN == 4, "v4 uses float4, so TM and TN must be 4");

__global__ void matmulV4(const float* A, const float* B, float* C, int n)
{
    __shared__ __align__(16) float AsT[BK][BM + PAD];   // transposed + padded A tile
    __shared__ __align__(16) float Bs[BK][BN];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    float acc[TM][TN] = {};

    for (int t = 0; t < n; t += BK) {
        // 1. load A tile (stored transposed) and B tile, 2 elements of each per thread
        #pragma unroll
        for (int l = 0; l < (BM * BK) / V3_THREADS; l++) {
            const int idx = tid + l * V3_THREADS;
            const int r = idx / BK, c = idx % BK;
            AsT[c][r] = A[(rowBase + r) * n + (t + c)];
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / V3_THREADS; l++) {
            const int idx = tid + l * V3_THREADS;
            const int r = idx / BN, c = idx % BN;
            Bs[r][c] = B[(t + r) * n + (colBase + c)];
        }
        __syncthreads();

        // 2. one float4 load for A's 4 values, one for B's 4 values, then 16 FMAs
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            const float4 a4 = *reinterpret_cast<const float4*>(&AsT[k][ty * TM]);
            const float4 b4 = *reinterpret_cast<const float4*>(&Bs[k][tx * TN]);
            const float a[TM] = {a4.x, a4.y, a4.z, a4.w};
            const float b[TN] = {b4.x, b4.y, b4.z, b4.w};
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // 3. write each row of the 4 x 4 result with one float4 store
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        *reinterpret_cast<float4*>(&C[(rowBase + ty * TM + i) * n + (colBase + tx * TN)]) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
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

// ============================================================
// Size sweep: cuBLAS (FP32) vs v3 vs v4, correctness checked against cuBLAS
// ============================================================
void sizeSweep(cublasHandle_t handle, bench::CsvWriter& csv, const bench::DeviceInfo& dev)
{
    const int   sizes[] = {512, 1024, 2048, 4096};
    const float alpha = 1.0f, beta = 0.0f;

    std::printf("\nSize sweep: GFLOPS (%% of cuBLAS)\n");
    std::printf("%8s %12s %22s %22s %6s\n", "n", "cuBLAS", "v3", "v4", "check");

    for (int n : sizes) {
        const size_t elems = size_t(n) * n;
        const size_t bytes = elems * sizeof(float);
        const double flops = 2.0 * n * n * n;

        std::vector<float> h_A(elems), h_B(elems), h_ref(elems), h_C(elems);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& x : h_A) x = dist(rng);
        for (float& x : h_B) x = dist(rng);

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        const bench::Stats ref = bench::timeGpu([&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                     &alpha, d_B, n, d_A, n, &beta, d_C, n));
        });
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        const dim3 block(TILE, TILE);
        const dim3 grid(n / BN, n / BM);

        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        const bench::Stats s3 = bench::timeGpu([&] { matmulV3<<<grid, block>>>(d_A, d_B, d_C, n); });
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        const float err3 = maxAbsDiff(h_C, h_ref);

        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        const bench::Stats s4 = bench::timeGpu([&] { matmulV4<<<grid, block>>>(d_A, d_B, d_C, n); });
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        const float err4 = maxAbsDiff(h_C, h_ref);

        auto gf = [&](const bench::Stats& t) { return flops / (t.median / 1e3) / 1e9; };
        std::printf("%8d %12.1f %13.1f (%5.1f%%) %13.1f (%5.1f%%) %6s\n",
                    n, gf(ref),
                    gf(s3), 100.0 * ref.median / s3.median,
                    gf(s4), 100.0 * ref.median / s4.median,
                    (err3 < kTol && err4 < kTol) ? "PASS" : "FAIL");

        auto row = [&](const char* name, const bench::Stats& t, float err) {
            bench::Row r;
            r.level       = "3_matmul_sweep";
            r.kernel      = name;
            r.n           = n;
            r.t           = t;
            r.flops       = flops;
            r.peak_gflops = dev.peak_gflops;
            r.lib_ms      = ref.median;
            r.max_err     = err;
            r.pass        = err < kTol;
            csv.write(r);
        };
        row("cuBLAS sgemm", ref, 0.0f);
        row("v3 register tiling 4x4", s3, err3);
        row("v4 float4 + transposed A", s4, err4);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
}

int main(int argc, char** argv)
{
    const std::string       csvPath = bench::csvPathFromArgs(argc, argv);
    const bench::DeviceInfo dev     = bench::queryDevice();
    bench::printDevice(dev);

    const int    n     = 2048;
    const size_t elems = size_t(n) * n;
    const size_t bytes = elems * sizeof(float);
    const double flops = 2.0 * n * n * n;   // one multiply + one add per (i, j, k)

    // ---------- input: random floats in [-1, 1], fixed seed ----------
    std::vector<float> h_A(elems), h_B(elems), h_C(elems), h_ref(elems), h_cpu(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : h_A) x = dist(rng);
    for (float& x : h_B) x = dist(rng);

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
    auto cublasRun = [&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, d_B, n, d_A, n, &beta, d_C, n));
    };
    const bench::Stats ref = bench::timeGpu(cublasRun);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // ---------- results table + CSV ----------
    bench::CsvWriter csv(csvPath);
    std::printf("\nn = %d (%.1f MB per matrix), %.2f GFLOP per multiply, block = %dx%d\n",
                n, bytes / 1e6, flops / 1e9, TILE, TILE);
    std::printf("Times: median of %d trials (p10/p90 spread); %%peak uses the driver-reported clock\n\n",
                bench::TimingConfig{}.trials);
    std::printf("%-26s %10s %10s %10s %10s %7s %9s %10s %6s\n",
                "version", "ms", "p10", "p90", "GFLOPS", "%peak", "% cuBLAS", "max err", "check");

    auto report = [&](const char* name, const bench::Stats& t, float err, float rowTol, bool onGpu,
                      bool fp32Peak = true) {
        const double gflops = flops / (t.median / 1e3) / 1e9;
        char pk[16];
        if (onGpu && fp32Peak) {
            std::snprintf(pk, sizeof pk, "%.1f%%", 100.0 * gflops / dev.peak_gflops);
        } else {
            std::snprintf(pk, sizeof pk, "-");
        }
        std::printf("%-26s %10.3f %10.3f %10.3f %10.1f %7s %8.1f%% %10.2e %6s\n",
                    name, t.median, t.p10, t.p90, gflops, pk, 100.0 * ref.median / t.median,
                    err, err < rowTol ? "PASS" : "FAIL");

        bench::Row row;
        row.level       = "3_matmul";
        row.kernel      = name;
        row.n           = n;
        row.t           = t;
        row.flops       = flops;
        row.peak_gflops = (onGpu && fp32Peak) ? dev.peak_gflops : 0.0;
        row.lib_ms      = onGpu ? ref.median : 0.0;
        row.max_err     = err;
        row.pass        = err < rowTol;
        csv.write(row);
    };

    // error of whatever is in d_C right now, against the cuBLAS reference
    auto errOfDC = [&] {
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        return maxAbsDiff(h_C, h_ref);
    };

    // reset d_C so a kernel that writes nothing can't "pass" with old results, time it, check it
    auto runKernel = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        const bench::Stats t = bench::timeGpu(launch);
        report(name, t, errOfDC(), kTol, true);
    };

    report("cuBLAS sgemm (reference)", ref, 0.0f, kTol, true);

    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    runKernel("v0 naive (uncoalesced)", [&] { matmulV0<<<grid, block>>>(d_A, d_B, d_C, n); });
    runKernel("v1 naive (coalesced)",   [&] { matmulV1<<<grid, block>>>(d_A, d_B, d_C, n); });
    runKernel("v2 tiled (shared mem)",  [&] { matmulV2<<<grid, block>>>(d_A, d_B, d_C, n); });

    if (n % BM == 0 && n % BN == 0 && n % BK == 0) {
        const dim3 gridT(n / BN, n / BM);
        runKernel("v3 register tiling 4x4",   [&] { matmulV3<<<gridT, block>>>(d_A, d_B, d_C, n); });
        runKernel("v4 float4 + transposed A", [&] { matmulV4<<<gridT, block>>>(d_A, d_B, d_C, n); });
    } else {
        std::printf("v3/v4 skipped: n must be a multiple of %d\n", BM);
    }

    // ---------- cuBLAS with TF32 Tensor Cores (lower precision: 10-bit mantissa) ----------
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    const bench::Stats tf32 = bench::timeGpu(cublasRun);
    report("cuBLAS TF32 Tensor Cores", tf32, errOfDC(), 1e-1f, true, false);   // looser tolerance: TF32 rounds inputs
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    sizeSweep(handle, csv, dev);

    // ---------- CPU baseline: one run takes seconds, so 5 trials of one run each ----------
    bench::TimingConfig cpuCfg = bench::cpuDefaults();
    cpuCfg.trials = 5;
    const bench::Stats cpu = bench::timeCpu([&] {
        matmulCPU(h_A.data(), h_B.data(), h_cpu.data(), n);
        bench::doNotOptimize(h_cpu[0]);
    }, cpuCfg);
    const float cpu_err = maxAbsDiff(h_cpu, h_ref);

    std::printf("\nCPU baseline (timed after all GPU measurements):\n");
    report("CPU (1 core, ikj)", cpu, cpu_err, kTol, false);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
