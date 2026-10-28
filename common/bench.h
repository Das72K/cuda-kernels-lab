#pragma once

// Shared benchmark framework for all levels:
// time-based warm-up, auto-calibrated trials, median + p10/p90, device peaks, CSV output.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda_utils.h"

namespace bench {

// ============================================================
// Statistics
// ============================================================
struct Stats {
    double median = 0.0;  // ms per launch
    double p10    = 0.0;
    double p90    = 0.0;
    double min    = 0.0;
    int    trials = 0;
    int    reps   = 0;    // launches per trial (auto-calibrated)
};

inline Stats summarize(std::vector<double> v, int reps)
{
    std::sort(v.begin(), v.end());
    auto at = [&](double q) {
        return v[static_cast<size_t>(std::lround(q * (v.size() - 1)))];
    };
    Stats s;
    s.trials = static_cast<int>(v.size());
    s.reps   = reps;
    s.min    = v.front();
    s.median = at(0.5);
    s.p10    = at(0.1);
    s.p90    = at(0.9);
    return s;
}

// Stops the compiler from deleting "unused" benchmark work
template <typename T>
inline void doNotOptimize(const T& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

struct TimingConfig {
    int    trials       = 15;     // timed batches
    double warmup_ms    = 500.0;  // GPU busy time before measuring (laptop clocks ramp up)
    double min_trial_ms = 10.0;   // each batch lasts at least this long
};

// CPU runs are noisier (OS scheduling, shared laptop power budget), so CPU trials last longer
inline TimingConfig cpuDefaults()
{
    TimingConfig c;
    c.min_trial_ms = 100.0;
    return c;
}

// ============================================================
// GPU timing: first launch (error check) -> time-based warm-up -> calibrate -> trials
// ============================================================
template <typename F>
Stats timeGpu(F launch, const TimingConfig& cfg = TimingConfig{})
{
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;

    double warmed = 0.0;
    for (int batch = 1; warmed < cfg.warmup_ms; batch *= 2) {
        timer.start();
        for (int k = 0; k < batch; k++) launch();
        timer.stop();
        warmed += timer.elapsed_ms();
    }
    CUDA_CHECK(cudaGetLastError());

    timer.start();
    launch();
    timer.stop();
    const double one  = std::max(static_cast<double>(timer.elapsed_ms()), 1e-4);
    const int    reps = std::max(1, static_cast<int>(std::ceil(cfg.min_trial_ms / one)));

    std::vector<double> per_launch;
    for (int t = 0; t < cfg.trials; t++) {
        timer.start();
        for (int k = 0; k < reps; k++) launch();
        timer.stop();
        per_launch.push_back(timer.elapsed_ms() / reps);
    }
    CUDA_CHECK(cudaGetLastError());
    return summarize(per_launch, reps);
}

// ============================================================
// CPU timing: warm-up run -> calibrate -> trials. `run` must call doNotOptimize on its result.
// ============================================================
template <typename F>
Stats timeCpu(F run, const TimingConfig& cfg = cpuDefaults())
{
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    run();  // warm-up: page faults, caches
    auto t0 = clock::now();
    run();
    const double one  = std::max(ms_since(t0), 1e-4);
    const int    reps = std::max(1, static_cast<int>(std::ceil(cfg.min_trial_ms / one)));

    std::vector<double> per_run;
    for (int t = 0; t < cfg.trials; t++) {
        t0 = clock::now();
        for (int k = 0; k < reps; k++) run();
        per_run.push_back(ms_since(t0) / reps);
    }
    return summarize(per_run, reps);
}

// ============================================================
// Device info and theoretical peaks (from the driver)
// ============================================================
struct DeviceInfo {
    std::string name;
    int    cc_major = 0, cc_minor = 0;
    int    sm_count = 0;
    int    fp32_lanes_per_sm = 0;
    double sm_clock_ghz   = 0.0;  // max SM clock reported by the driver
    double mem_clock_ghz  = 0.0;
    int    bus_width_bits = 0;
    double l2_mb          = 0.0;
    double peak_gbps      = 0.0;  // theoretical DRAM bandwidth
    double peak_gflops    = 0.0;  // theoretical FP32 (one FMA = 2 FLOP)
};

// FP32 lanes per SM for known architectures (0 = unknown)
inline int fp32LanesPerSm(int major, int minor)
{
    if (major == 8 && minor == 0) return 64;                                   // A100
    if ((major == 8 && (minor == 6 || minor == 7 || minor == 9)) || major == 9) return 128;  // Ampere consumer, Ada, Hopper
    return 0;
}

inline DeviceInfo queryDevice(int dev = 0)
{
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int sm_khz = 0, mem_khz = 0, bus = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_khz, cudaDevAttrClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));

    DeviceInfo d;
    d.name              = p.name;
    d.cc_major          = p.major;
    d.cc_minor          = p.minor;
    d.sm_count          = p.multiProcessorCount;
    d.fp32_lanes_per_sm = fp32LanesPerSm(p.major, p.minor);
    d.sm_clock_ghz      = sm_khz / 1e6;
    d.mem_clock_ghz     = mem_khz / 1e6;
    d.bus_width_bits    = bus;
    d.l2_mb             = p.l2CacheSize / (1024.0 * 1024.0);
    d.peak_gbps         = 2.0 * mem_khz * 1e3 * (bus / 8.0) / 1e9;  // double data rate
    d.peak_gflops       = 2.0 * d.sm_count * d.fp32_lanes_per_sm * d.sm_clock_ghz;
    return d;
}

inline void printDevice(const DeviceInfo& d)
{
    std::printf("GPU: %s (sm_%d%d), %d SMs, L2 %.0f MB\n",
                d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count, d.l2_mb);
    std::printf("Theoretical peaks: DRAM %.1f GB/s (%d-bit, %.3f GHz x 2), "
                "FP32 %.1f GFLOPS at driver-reported clock (%d SMs x %d lanes x 2 x %.3f GHz)\n",
                d.peak_gbps, d.bus_width_bits, d.mem_clock_ghz,
                d.peak_gflops, d.sm_count, d.fp32_lanes_per_sm, d.sm_clock_ghz);
}

// ============================================================
// CSV output: one row per (level, kernel, size)
// ============================================================
struct Row {
    std::string level;
    std::string kernel;
    long long   n           = 0;
    Stats       t;
    double      bytes       = 0.0;  // DRAM bytes per launch (0 = not reported)
    double      flops       = 0.0;  // operations per launch (0 = not reported)
    double      peak_gbps   = 0.0;
    double      peak_gflops = 0.0;
    double      lib_ms      = 0.0;  // library reference median (0 = none)
    double      max_err     = 0.0;
    bool        pass        = true;
};

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path)
    {
        if (path.empty()) return;
        f_ = std::fopen(path.c_str(), "w");
        if (!f_) {
            std::fprintf(stderr, "cannot open CSV file %s\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        std::fprintf(f_, "level,kernel,n,ms_median,ms_p10,ms_p90,ms_min,trials,reps,"
                         "gbps,gflops,pct_peak_bw,pct_peak_flops,pct_library,max_err,pass\n");
    }
    ~CsvWriter() { if (f_) std::fclose(f_); }
    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;

    void write(const Row& r)
    {
        if (!f_) return;
        const double sec    = r.t.median / 1e3;
        const double gbps   = r.bytes > 0 ? r.bytes / sec / 1e9 : 0.0;
        const double gflops = r.flops > 0 ? r.flops / sec / 1e9 : 0.0;
        std::fprintf(f_, "%s,%s,%lld,%.6f,%.6f,%.6f,%.6f,%d,%d,",
                     r.level.c_str(), r.kernel.c_str(), r.n,
                     r.t.median, r.t.p10, r.t.p90, r.t.min, r.t.trials, r.t.reps);
        opt(gbps);
        opt(gflops);
        opt(r.peak_gbps > 0 && gbps > 0 ? 100.0 * gbps / r.peak_gbps : 0.0);
        opt(r.peak_gflops > 0 && gflops > 0 ? 100.0 * gflops / r.peak_gflops : 0.0);
        opt(r.lib_ms > 0 ? 100.0 * r.lib_ms / r.t.median : 0.0);
        std::fprintf(f_, "%.6g,%s\n", r.max_err, r.pass ? "PASS" : "FAIL");
    }

private:
    void opt(double v) { v > 0 ? std::fprintf(f_, "%.4f,", v) : std::fprintf(f_, ","); }
    FILE* f_ = nullptr;
};

// `--csv <path>` on the command line -> path, otherwise "" (no CSV written)
inline std::string csvPathFromArgs(int argc, char** argv)
{
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], "--csv") == 0) return argv[i + 1];
    }
    return "";
}

}  // namespace bench
