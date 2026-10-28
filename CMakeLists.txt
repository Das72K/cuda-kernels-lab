cmake_minimum_required(VERSION 3.24)

project(cuda_kernels_lab LANGUAGES CXX CUDA)

# Target Ada Lovelace (RTX 40-series, compute capability 8.9).
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

# Shared helpers (header-only).
add_library(lab_common INTERFACE)
target_include_directories(lab_common INTERFACE ${CMAKE_SOURCE_DIR}/common)
target_link_libraries(lab_common INTERFACE CUDA::cudart)

# add_lab_executable(<name> <sources...>)
# Builds one exercise binary with the common helpers and profiling-friendly flags.
function(add_lab_executable name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE lab_common)
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
endfunction()

add_subdirectory(src/level1_vector_add)
add_subdirectory(src/level2_reduction)
add_subdirectory(src/level3_matmul)
add_subdirectory(bench)
