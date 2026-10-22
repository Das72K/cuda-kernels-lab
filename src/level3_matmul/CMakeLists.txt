add_lab_executable(level3_matmul matmul.cu)
target_link_libraries(level3_matmul PRIVATE CUDA::cublas)
