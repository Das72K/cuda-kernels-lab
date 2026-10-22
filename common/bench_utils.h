#pragma once

#include <algorithm>
#include <vector>

#include "cuda_utils.h"

// Stops the compiler from deleting "unused" benchmark work
template <typename T>
inline void doNotOptimize(const T& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

// Warm-up once, then run `trials` timed batches of `reps` launches and return the
// MEDIAN per-launch time in ms (robust against laptop clock changes)
template <typename F>
float timeGpu(F launch, int reps, int trials = 7)
{
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    GpuTimer timer;
    for (int t = 0; t < trials; t++) {
        timer.start();
        for (int k = 0; k < reps; k++) {
            launch();
        }
        timer.stop();
        CUDA_CHECK(cudaGetLastError());
        times.push_back(timer.elapsed_ms() / reps);
    }
    std::sort(times.begin(), times.end());
    return times[trials / 2];
}
