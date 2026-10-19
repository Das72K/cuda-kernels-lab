#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Abort with file/line on any CUDA runtime error.
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                   cudaGetErrorName(err_), __FILE__, __LINE__,                \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// Check for errors from the most recent kernel launch.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Times GPU work on a stream with CUDA events.
class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
  void stop(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(stop_, stream)); }

  // Blocks until the stop event completes; returns elapsed milliseconds.
  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
