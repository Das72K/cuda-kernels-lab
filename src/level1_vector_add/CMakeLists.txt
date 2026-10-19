# Level 1: vector add.
# The target is created once vector_add.cu exists (re-run cmake after adding it).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.cu)
  add_lab_executable(level1_vector_add vector_add.cu)
endif()
