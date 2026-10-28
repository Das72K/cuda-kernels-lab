// Peak microbenchmarks: achievable DRAM bandwidth (copy, read-only) and FP32 FMA throughput.
// These are the empirical ceilings for the roofline plot.

#include <cstdio>
#include <string>

#include "bench.h"
#include "cuda_utils.h"

// ============================================================
// Bandwidth: grid-stride loops over float4 (16-byte loads/stores)
// ============================================================
__global__ void copyKernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n4)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        out[i] = in[i];
    }
}

__global__ void readKernel(const float4* __restrict__ in, float* sink, size_t n4)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    float acc = 0.0f;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) {   // practically never true, but the compiler can't remove the loads
        sink[0] = acc;
    }
}

// Unrolled variants: 4 independent float4 requests per thread per iteration
// -> more memory requests in flight (memory-level parallelism)
__global__ void copyKernelUnrolled(const float4* __restrict__ in, float4* __restrict__ out, size_t n4)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n4; i += 4 * stride) {
        const float4 a = in[i];
        const float4 b = in[i + stride];
        const float4 c = in[i + 2 * stride];
        const float4 d = in[i + 3 * stride];
        out[i]              = a;
        out[i + stride]     = b;
        out[i + 2 * stride] = c;
        out[i + 3 * stride] = d;
    }
    for (; i < n4; i += stride) {
        out[i] = in[i];
    }
}

__global__ void readKernelUnrolled(const float4* __restrict__ in, float* sink, size_t n4)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (; i + 3 * stride < n4; i += 4 * stride) {
        const float4 a = in[i];
        const float4 b = in[i + stride];
        const float4 c = in[i + 2 * stride];
        const float4 d = in[i + 3 * stride];
        acc += (a.x + a.y + a.z + a.w) + (b.x + b.y + b.z + b.w)
             + (c.x + c.y + c.z + c.w) + (d.x + d.y + d.z + d.w);
    }
    for (; i < n4; i += stride) {
        const float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) {
        sink[0] = acc;
    }
}

// ============================================================
// Compute: CHAINS independent FMA chains per thread, all in registers.
// Independent chains let the SM overlap FMAs instead of waiting for each result.
// ============================================================
template <int CHAINS>
__global__ void fmaKernel(float* sink, int iters, float a, float b)
{
    float x[CHAINS];
    #pragma unroll
    for (int c = 0; c < CHAINS; c++) {
        x[c] = threadIdx.x * 0.001f + c;
    }
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            x[c] = fmaf(x[c], a, b);
        }
    }
    float s = 0.0f;
    #pragma unroll
    for (int c = 0; c < CHAINS; c++) {
        s += x[c];
    }
    if (s == 12345.678f) {   // keeps the results "used" so the FMAs can't be removed
        sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
    }
}

int main(int argc, char** argv)
{
    const std::string       csvPath = bench::csvPathFromArgs(argc, argv);
    const bench::DeviceInfo dev     = bench::queryDevice();
    bench::printDevice(dev);
    bench::CsvWriter csv(csvPath);

    const int threads = 256;
    const int blocks  = dev.sm_count * 32;   // enough blocks to keep every SM busy

    // ---------- buffers ----------
    const size_t n4    = size_t(1) << 25;            // 32M float4 = 512 MB per buffer (far bigger than L2)
    const size_t bytes = n4 * sizeof(float4);
    float4* d_in   = nullptr;
    float4* d_out  = nullptr;
    float*  d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_sink, size_t(blocks) * threads * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, bytes));

    // ---------- measurements ----------
    const bench::Stats copy = bench::timeGpu([&] { copyKernel<<<blocks, threads>>>(d_in, d_out, n4); });
    const bench::Stats read = bench::timeGpu([&] { readKernel<<<blocks, threads>>>(d_in, d_sink, n4); });
    const bench::Stats copyU = bench::timeGpu([&] { copyKernelUnrolled<<<blocks, threads>>>(d_in, d_out, n4); });
    const bench::Stats readU = bench::timeGpu([&] { readKernelUnrolled<<<blocks, threads>>>(d_in, d_sink, n4); });

    constexpr int CHAINS = 8;
    const int     iters  = 4096;
    const double  fmaFlops = double(blocks) * threads * iters * CHAINS * 2.0;   // one FMA = 2 FLOP
    const bench::Stats fma = bench::timeGpu([&] {
        fmaKernel<CHAINS><<<blocks, threads>>>(d_sink, iters, 0.999999f, 1e-6f);
    });

    // ---------- results ----------
    const double copyBytes = 2.0 * bytes;   // read + write
    const double readBytes = double(bytes);
    const double copyGBs = copyBytes / (copy.median / 1e3) / 1e9;
    const double readGBs = readBytes / (read.median / 1e3) / 1e9;
    const double copyUGBs = copyBytes / (copyU.median / 1e3) / 1e9;
    const double readUGBs = readBytes / (readU.median / 1e3) / 1e9;
    const double fmaGF   = fmaFlops  / (fma.median  / 1e3) / 1e9;

    std::printf("\nTimes: median of %d trials (p10/p90 spread)\n\n", bench::TimingConfig{}.trials);
    std::printf("%-22s %9s %9s %9s %12s %14s\n", "test", "ms", "p10", "p90", "result", "% theoretical");
    std::printf("%-22s %9.3f %9.3f %9.3f %7.1f GB/s %13.1f%%\n", "copy (read + write)",
                copy.median, copy.p10, copy.p90, copyGBs, 100.0 * copyGBs / dev.peak_gbps);
    std::printf("%-22s %9.3f %9.3f %9.3f %7.1f GB/s %13.1f%%\n", "read-only",
                read.median, read.p10, read.p90, readGBs, 100.0 * readGBs / dev.peak_gbps);
    std::printf("%-22s %9.3f %9.3f %9.3f %7.1f GB/s %13.1f%%\n", "copy unrolled x4",
                copyU.median, copyU.p10, copyU.p90, copyUGBs, 100.0 * copyUGBs / dev.peak_gbps);
    std::printf("%-22s %9.3f %9.3f %9.3f %7.1f GB/s %13.1f%%\n", "read-only unrolled x4",
                readU.median, readU.p10, readU.p90, readUGBs, 100.0 * readUGBs / dev.peak_gbps);
    std::printf("%-22s %9.3f %9.3f %9.3f %6.0f GFLOPS %12.1f%%\n", "FMA (FP32)",
                fma.median, fma.p10, fma.p90, fmaGF, 100.0 * fmaGF / dev.peak_gflops);
    if (dev.fp32_lanes_per_sm > 0) {
        const double impliedGHz = fmaGF / (2.0 * dev.sm_count * dev.fp32_lanes_per_sm);
        std::printf("\nImplied sustained SM clock during the FMA test: %.3f GHz "
                    "(driver-reported: %.3f GHz)\n", impliedGHz, dev.sm_clock_ghz);
    }

    auto row = [&](const char* name, const bench::Stats& t, long long n, double b, double f) {
        bench::Row r;
        r.level       = "0_peak";
        r.kernel      = name;
        r.n           = n;
        r.t           = t;
        r.bytes       = b;
        r.flops       = f;
        r.peak_gbps   = b > 0 ? dev.peak_gbps : 0.0;
        r.peak_gflops = f > 0 ? dev.peak_gflops : 0.0;
        csv.write(r);
    };
    row("copy (read + write)", copy, (long long)(n4 * 4), copyBytes, 0.0);
    row("read-only",           read, (long long)(n4 * 4), readBytes, 0.0);
    row("copy unrolled x4",      copyU, (long long)(n4 * 4), copyBytes, 0.0);
    row("read-only unrolled x4", readU, (long long)(n4 * 4), readBytes, 0.0);
    row("FMA (FP32)",          fma,  (long long)blocks * threads, 0.0, fmaFlops);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
