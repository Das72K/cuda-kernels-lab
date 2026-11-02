add_lab_executable(level4_image_filter image_filter.cu stb_impl.cpp)
target_include_directories(level4_image_filter PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(level4_image_filter PRIVATE CUDA::nppif CUDA::nppc)
