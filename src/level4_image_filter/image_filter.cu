// Level 4: 2D Gaussian blur (radius R) on a real grayscale image, clamp-to-edge borders.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <npp.h>

#include "stb_image.h"
#include "stb_image_write.h"

#include "bench.h"
#include "cuda_utils.h"

constexpr int R  = 3;            // filter radius
constexpr int K  = 2 * R + 1;    // 7 x 7 = 49 taps
constexpr int BX = 32;           // block width: one warp per row segment -> coalesced loads
constexpr int BY = 8;            // block height
constexpr float kTol = 1e-4f;    // max abs error vs the CPU reference (pixel values are in [0, 1])

// clamp-to-edge border: out-of-range coordinates read the nearest edge pixel
__host__ __device__ inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ============================================================
// v0: naive. One thread per output pixel, 49 reads from global memory, weights in global memory.
// ============================================================
__global__ void blurV0(const float* in, float* out, const float* w, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -R; dy <= R; dy++) {
        const int sy = clampi(y + dy, 0, height - 1);
        for (int dx = -R; dx <= R; dx++) {
            const int sx = clampi(x + dx, 0, width - 1);
            sum += w[(dy + R) * K + (dx + R)] * in[sy * width + sx];
        }
    }
    out[y * width + x] = sum;
}

// 49 Gaussian weights in constant memory: cached, and BROADCAST when all threads of a warp
// read the same address at the same time (which they do: every thread uses w[dy][dx] together)
__constant__ float c_w[K * K];

// ============================================================
// v1: v0 with the weights in constant memory
// ============================================================
__global__ void blurV1(const float* in, float* out, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -R; dy <= R; dy++) {
        const int sy = clampi(y + dy, 0, height - 1);
        for (int dx = -R; dx <= R; dx++) {
            const int sx = clampi(x + dx, 0, width - 1);
            sum += c_w[(dy + R) * K + (dx + R)] * in[sy * width + sx];
        }
    }
    out[y * width + x] = sum;
}

// Tile in shared memory: the block's 32 x 8 outputs plus a halo of R pixels on every side
constexpr int TW = BX + 2 * R;   // 38
constexpr int TH = BY + 2 * R;   // 14

// ============================================================
// v2: shared-memory tile with halo. Each input pixel is loaded from global memory once per block
//     (~2 loads per output instead of 49); borders are clamped once, during the load.
// ============================================================
__global__ void blurV2(const float* in, float* out, int width, int height)
{
    __shared__ float tile[TH][TW];

    const int x0  = blockIdx.x * BX - R;          // image coordinates of the tile's top-left (incl. halo)
    const int y0  = blockIdx.y * BY - R;
    const int tid = threadIdx.y * BX + threadIdx.x;

    // 1. cooperative load: 256 threads fill the 38 x 14 = 532-element tile (clamp-to-edge)
    for (int i = tid; i < TH * TW; i += BX * BY) {
        const int ty = i / TW;
        const int tx = i % TW;
        const int sx = clampi(x0 + tx, 0, width - 1);
        const int sy = clampi(y0 + ty, 0, height - 1);
        tile[ty][tx] = in[sy * width + sx];
    }
    __syncthreads();   // the whole tile must be loaded before anyone reads it

    // 2. compute from shared memory (the guard comes AFTER the barrier: every thread must reach it)
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < K; dy++) {
        #pragma unroll
        for (int dx = 0; dx < K; dx++) {
            sum += c_w[dy * K + dx] * tile[threadIdx.y + dy][threadIdx.x + dx];
        }
    }
    out[y * width + x] = sum;
}

// 7 weights of the 1D Gaussian, in constant memory
__constant__ float c_g[K];

// ============================================================
// v3: separable blur in two passes (7 + 7 = 14 taps per pixel instead of 49).
//     The intermediate image goes through DRAM: 16 bytes/pixel of traffic instead of 8.
// ============================================================
__global__ void blurRowsV3(const float* in, float* tmp, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    #pragma unroll
    for (int d = -R; d <= R; d++) {
        sum += c_g[d + R] * in[y * width + clampi(x + d, 0, width - 1)];
    }
    tmp[y * width + x] = sum;
}

__global__ void blurColsV3(const float* tmp, float* out, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    #pragma unroll
    for (int d = -R; d <= R; d++) {
        sum += c_g[d + R] * tmp[clampi(y + d, 0, height - 1) * width + x];
    }
    out[y * width + x] = sum;
}

// ============================================================
// v4: separable blur fused into ONE kernel through shared memory:
//     load tile + halo once (8 bytes/pixel of DRAM traffic) -> horizontal pass into shared memory
//     -> vertical pass from shared memory. About 19 taps per output instead of 49.
// ============================================================
__global__ void blurV4(const float* in, float* out, int width, int height)
{
    __shared__ float tile[TH][TW];   // input tile + halo: 14 x 38
    __shared__ float rows[TH][BX];   // after the horizontal pass: 14 x 32

    const int x0  = blockIdx.x * BX - R;
    const int y0  = blockIdx.y * BY - R;
    const int tid = threadIdx.y * BX + threadIdx.x;

    // 1. load tile + halo once (clamp-to-edge), as in v2
    for (int i = tid; i < TH * TW; i += BX * BY) {
        const int ty = i / TW;
        const int tx = i % TW;
        tile[ty][tx] = in[clampi(y0 + ty, 0, height - 1) * width + clampi(x0 + tx, 0, width - 1)];
    }
    __syncthreads();

    // 2. horizontal pass: all TH rows (the vertical halo rows are needed by step 3), BX output columns
    for (int i = tid; i < TH * BX; i += BX * BY) {
        const int ty = i / BX;   // BX = 32: the compiler turns / and % into shifts
        const int tx = i % BX;
        float s = 0.0f;
        #pragma unroll
        for (int d = 0; d < K; d++) {
            s += c_g[d] * tile[ty][tx + d];
        }
        rows[ty][tx] = s;
    }
    __syncthreads();

    // 3. vertical pass (bounds check after the last barrier)
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < K; d++) {
        sum += c_g[d] * rows[threadIdx.y + d][threadIdx.x];
    }
    out[y * width + x] = sum;
}

// ============================================================
// CPU reference (same loops, same border rule)
// ============================================================
void blurCPU(const float* in, float* out, const float* w, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int dy = -R; dy <= R; dy++) {
                const int sy = clampi(y + dy, 0, height - 1);
                for (int dx = -R; dx <= R; dx++) {
                    const int sx = clampi(x + dx, 0, width - 1);
                    sum += w[(dy + R) * K + (dx + R)] * in[sy * width + sx];
                }
            }
            out[y * width + x] = sum;
        }
    }
}

// 1D Gaussian (K weights), normalized to sum to 1
std::vector<float> gaussian1D(float sigma)
{
    std::vector<float> g(K);
    float s = 0.0f;
    for (int i = 0; i < K; i++) {
        const float d = float(i - R);
        g[i] = std::exp(-d * d / (2.0f * sigma * sigma));
        s += g[i];
    }
    for (float& v : g) v /= s;
    return g;
}

// 2D Gaussian = outer product of the 1D Gaussian with itself (this is what makes it separable)
std::vector<float> gaussianWeights(float sigma)
{
    const std::vector<float> g = gaussian1D(sigma);
    std::vector<float> w(K * K);
    for (int j = 0; j < K; j++)
        for (int i = 0; i < K; i++)
            w[j * K + i] = g[j] * g[i];
    return w;
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

bool savePng(const std::string& path, const std::vector<float>& img, int width, int height)
{
    std::vector<unsigned char> px(img.size());
    for (size_t i = 0; i < img.size(); i++) {
        px[i] = static_cast<unsigned char>(std::lround(std::clamp(img[i], 0.0f, 1.0f) * 255.0f));
    }
    if (!stbi_write_png(path.c_str(), width, height, 1, px.data(), width)) {
        std::fprintf(stderr, "could not write %s\n", path.c_str());
        return false;
    }
    return true;
}

std::string argValue(int argc, char** argv, const char* key, const char* fallback)
{
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return fallback;
}

int main(int argc, char** argv)
{
    const std::string       csvPath = bench::csvPathFromArgs(argc, argv);
    const std::string       imgPath = argValue(argc, argv, "--image", "data/test_image.png");
    const std::string       outDir  = argValue(argc, argv, "--outdir", "results/images");
    std::filesystem::create_directories(outDir);
    const bench::DeviceInfo dev     = bench::queryDevice();
    bench::printDevice(dev);

    // ---------- load the image as grayscale floats in [0, 1] ----------
    int width = 0, height = 0, channels = 0;
    unsigned char* raw = stbi_load(imgPath.c_str(), &width, &height, &channels, 1);
    if (!raw) {
        std::fprintf(stderr, "could not load %s\n", imgPath.c_str());
        return 1;
    }
    const size_t pixels = size_t(width) * height;
    std::vector<float> h_in(pixels), h_ref(pixels), h_out(pixels);
    for (size_t i = 0; i < pixels; i++) h_in[i] = raw[i] / 255.0f;
    stbi_image_free(raw);

    const std::vector<float> h_w = gaussianWeights(1.5f);
    const std::vector<float> h_g = gaussian1D(1.5f);
    const double bytes = 2.0 * pixels * sizeof(float);         // ideal DRAM traffic: read once, write once
    const double flops = 2.0 * K * K * double(pixels);         // one multiply-add per tap

    // ---------- CPU reference result (untimed; the CPU is timed after all GPU work) ----------
    blurCPU(h_in.data(), h_ref.data(), h_w.data(), width, height);

    // ---------- GPU buffers ----------
    float *d_in = nullptr, *d_out = nullptr, *d_w = nullptr, *d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, h_w.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, pixels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_w, h_w.data(), K * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_g, h_g.data(), K * sizeof(float)));

    const dim3 block(BX, BY);
    const dim3 grid((width + BX - 1) / BX, (height + BY - 1) / BY);

    struct GpuRow { const char* name; bench::Stats t; float err; };
    std::vector<GpuRow> gpuRows;

    // reset the output, time the launch, check against the CPU reference
    auto runKernel = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(d_out, 0, pixels * sizeof(float)));
        const bench::Stats t = bench::timeGpu(launch);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, pixels * sizeof(float), cudaMemcpyDeviceToHost));
        gpuRows.push_back({name, t, maxAbsDiff(h_out, h_ref)});
    };

    runKernel("v0 naive", [&] { blurV0<<<grid, block>>>(d_in, d_out, d_w, width, height); });
    runKernel("v1 constant-memory weights", [&] { blurV1<<<grid, block>>>(d_in, d_out, width, height); });
    runKernel("v2 shared tile + halo", [&] { blurV2<<<grid, block>>>(d_in, d_out, width, height); });
    runKernel("v3 separable, 2 passes", [&] {
        blurRowsV3<<<grid, block>>>(d_in, d_tmp, width, height);
        blurColsV3<<<grid, block>>>(d_tmp, d_out, width, height);
    });
    runKernel("v4 separable, fused smem", [&] { blurV4<<<grid, block>>>(d_in, d_out, width, height); });
    if (!savePng(outDir + "/level4_blur.png", h_out, width, height)) {
        std::fprintf(stderr, "error: failed to save output image\n");
        return 1;
    }

    // ---------- NPP library reference ----------
    // NPP stream context, filled in by hand (CUDA 13 has no nppGetStreamContext); default stream, like our kernels
    NppStreamContext nppCtx = {};
    nppCtx.hStream = 0;
    CUDA_CHECK(cudaGetDevice(&nppCtx.nCudaDeviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, nppCtx.nCudaDeviceId));
    nppCtx.nMultiProcessorCount         = prop.multiProcessorCount;
    nppCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppCtx.nMaxThreadsPerBlock          = prop.maxThreadsPerBlock;
    nppCtx.nSharedMemPerBlock           = prop.sharedMemPerBlock;
    nppCtx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nppCtx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    CUDA_CHECK(cudaStreamGetFlags(nppCtx.hStream, &nppCtx.nStreamFlags));

    const int      step = width * int(sizeof(float));
    const NppiSize size = {width, height};
    auto nppFilter = [&] {
        return nppiFilterBorder_32f_C1R_Ctx(d_in, step, size, NppiPoint{0, 0},
                                            d_out, step, size,
                                            d_w, NppiSize{K, K}, NppiPoint{R, R},
                                            NPP_BORDER_REPLICATE, nppCtx);
    };

    // fail fast: one checked call before any timing
    NppStatus nppStatus = nppFilter();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (nppStatus != NPP_SUCCESS) {
        std::fprintf(stderr, "NPP error: nppiFilterBorder_32f_C1R_Ctx returned %d\n", int(nppStatus));
        return 1;
    }
    runKernel("NPP FilterBorder 7x7", [&] { nppStatus = nppFilter(); });
    if (nppStatus != NPP_SUCCESS) {   // safety net for the timed calls
        std::fprintf(stderr, "NPP error during timing: %d\n", int(nppStatus));
        return 1;
    }
    const auto nppIt = std::find_if(gpuRows.begin(), gpuRows.end(),
        [](const GpuRow& r) { return std::strcmp(r.name, "NPP FilterBorder 7x7") == 0; });
    const double nppMs = nppIt->t.median;

    // ---------- CPU timing LAST (one run takes ~1 s, so 5 trials) ----------
    bench::TimingConfig cpuCfg = bench::cpuDefaults();
    cpuCfg.trials = 5;
    const bench::Stats cpu = bench::timeCpu([&] {
        blurCPU(h_in.data(), h_out.data(), h_w.data(), width, height);
        bench::doNotOptimize(h_out[0]);
    }, cpuCfg);

    // ---------- results table + CSV ----------
    bench::CsvWriter csv(csvPath);
    std::printf("\nimage %dx%d (%.1f MP), Gaussian %dx%d (sigma 1.5), clamp-to-edge borders, block %dx%d\n",
                width, height, pixels / 1e6, K, K, BX, BY);
    std::printf("Times: median of %d trials (p10/p90 spread); GB/s assumes ideal traffic (read + write once)\n\n",
                bench::TimingConfig{}.trials);
    std::printf("%-26s %10s %10s %10s %9s %9s %8s %7s %10s %6s\n",
                "version", "ms", "p10", "p90", "GB/s", "GFLOPS", "vs CPU", "% NPP", "max err", "check");

    auto report = [&](const char* name, const bench::Stats& t, float err, bool onGpu) {
        const double sec = t.median / 1e3;
        char pct[16];
        if (onGpu) {
            std::snprintf(pct, sizeof pct, "%.1f", 100.0 * nppMs / t.median);
        } else {
            std::snprintf(pct, sizeof pct, "-");
        }
        std::printf("%-26s %10.3f %10.3f %10.3f %9.1f %9.1f %8.1f %7s %10.2e %6s\n",
                    name, t.median, t.p10, t.p90, bytes / sec / 1e9, flops / sec / 1e9,
                    cpu.median / t.median, pct, err, err < kTol ? "PASS" : "FAIL");
        bench::Row row;
        row.level       = "4_blur";
        row.kernel      = name;
        row.n           = static_cast<long long>(pixels);
        row.t           = t;
        row.bytes       = bytes;
        row.flops       = flops;
        row.peak_gbps   = onGpu ? dev.peak_gbps : 0.0;
        row.peak_gflops = onGpu ? dev.peak_gflops : 0.0;
        row.lib_ms      = onGpu ? nppMs : 0.0;
        row.max_err     = err;
        row.pass        = err < kTol;
        csv.write(row);
    };

    report("CPU (1 core)", cpu, 0.0f, false);
    for (const GpuRow& r : gpuRows) report(r.name, r.t, r.err, true);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_tmp));
    return 0;
}
