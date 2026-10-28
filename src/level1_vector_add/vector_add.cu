#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>

#include "bench.h"
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

// GPU results for one problem size (the CPU is timed in a separate pass, after all GPU work)
struct GpuResult {
    int          n = 0;
    bench::Stats kern;
    bench::Stats total;
    bench::Stats pinned;
    int          errors = 0;
};

// Pass 1 (GPU): kernel only, total trip with pageable and with pinned host memory
GpuResult measureGpu(int n)
{
    const size_t bytes = size_t(n) * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // ---------- CPU reference for the correctness check (one untimed run) ----------
    vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);

    // ---------- GPU buffers ----------
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // ---------- GPU: kernel only ----------
    const bench::Stats kern = bench::timeGpu([&] {
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    });

    // ---------- GPU: total trip, pageable host memory ----------
    const bench::Stats total = bench::timeGpu([&] {
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    });
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c_gpu[i] != h_c_cpu[i]) errors++;
    }

    // ---------- GPU: total trip, pinned host memory ----------
    float *p_a = nullptr, *p_b = nullptr, *p_c = nullptr;
    CUDA_CHECK(cudaMallocHost(&p_a, bytes));
    CUDA_CHECK(cudaMallocHost(&p_b, bytes));
    CUDA_CHECK(cudaMallocHost(&p_c, bytes));
    std::memcpy(p_a, h_a.data(), bytes);
    std::memcpy(p_b, h_b.data(), bytes);
    const bench::Stats pinned = bench::timeGpu([&] {
        CUDA_CHECK(cudaMemcpy(d_a, p_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, p_b, bytes, cudaMemcpyHostToDevice));
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaMemcpy(p_c, d_c, bytes, cudaMemcpyDeviceToHost));
    });
    for (int i = 0; i < n; i++) {
        if (p_c[i] != h_c_cpu[i]) errors++;
    }

    CUDA_CHECK(cudaFreeHost(p_a));
    CUDA_CHECK(cudaFreeHost(p_b));
    CUDA_CHECK(cudaFreeHost(p_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    GpuResult g;
    g.n      = n;
    g.kern   = kern;
    g.total  = total;
    g.pinned = pinned;
    g.errors = errors;
    return g;
}

// Pass 2 (CPU): timed after all GPU measurements
bench::Stats measureCpu(int n)
{
    std::vector<float> h_a(n), h_b(n), h_c_cpu(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // ---------- CPU ----------
    const bench::Stats cpu = bench::timeCpu([&] {
        vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);
        bench::doNotOptimize(h_c_cpu[0]);
    });
    return cpu;
}

// One console row and the four CSV rows of one problem size
void reportSize(const GpuResult& g, const bench::Stats& cpu, const bench::DeviceInfo& dev,
                bench::CsvWriter& csv)
{
    const int           n      = g.n;
    const size_t        bytes  = size_t(n) * sizeof(float);
    const double        moved  = 3.0 * bytes;   // the kernel reads a and b and writes c
    const bench::Stats& kern   = g.kern;
    const bench::Stats& total  = g.total;
    const bench::Stats& pinned = g.pinned;
    const int           errors = g.errors;

    // ---------- console row ----------
    std::printf("%12d %9.1f %10.4f %10.4f %9.1f %9.1f %10.4f %9.2f %10.4f %9.2f %6s\n",
                n, bytes / 1e6,
                cpu.median, kern.median, moved / (kern.median / 1e3) / 1e9, cpu.median / kern.median,
                total.median, cpu.median / total.median,
                pinned.median, cpu.median / pinned.median,
                errors == 0 ? "PASS" : "FAIL");

    // ---------- CSV rows ----------
    auto row = [&](const char* name, const bench::Stats& t, bool isKernel) {
        bench::Row r;
        r.level     = "1_vector_add";
        r.kernel    = name;
        r.n         = n;
        r.t         = t;
        r.bytes     = moved;
        r.peak_gbps = isKernel ? dev.peak_gbps : 0.0;
        r.max_err   = errors;
        r.pass      = errors == 0;
        csv.write(r);
    };
    row("CPU (1 core)", cpu, false);
    row("GPU kernel", kern, true);
    row("GPU total pageable", total, false);
    row("GPU total pinned", pinned, false);
}

int main(int argc, char** argv)
{
    const std::string       csvPath = bench::csvPathFromArgs(argc, argv);
    const bench::DeviceInfo dev     = bench::queryDevice();
    bench::printDevice(dev);
    bench::CsvWriter csv(csvPath);

    const int sizes[] = { 1'000, 10'000, 100'000, 1'000'000,
                          10'000'000, 50'000'000, 100'000'000 };

    // all GPU measurements first, then all CPU timing
    std::vector<GpuResult> gpu;
    for (int n : sizes) {
        gpu.push_back(measureGpu(n));
    }
    std::vector<bench::Stats> cpu;
    for (int n : sizes) {
        cpu.push_back(measureCpu(n));
    }

    std::printf("\nTimes: median of %d trials; GB/s = 3 arrays moved by the kernel; x = speed-up vs CPU (1 core)\n\n",
                bench::TimingConfig{}.trials);
    std::printf("%12s %9s %10s %10s %9s %9s %10s %9s %10s %9s %6s\n",
                "n", "MB/array", "CPU ms", "kern ms", "kern GB/s", "kern x",
                "total ms", "total x", "pinned ms", "pinned x", "check");

    for (size_t i = 0; i < gpu.size(); i++) {
        reportSize(gpu[i], cpu[i], dev, csv);
    }
    return 0;
}
