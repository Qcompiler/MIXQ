



cmake_minimum_required(VERSION 3.8)
set(TARGET_NAME kernel)
project(${TARGET_NAME})

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++17 -O3 -Xcompiler "-fPIC" -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 )
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(./)
include_directories(cutlass/include)
cuda_add_library(
  ${TARGET_NAME} SHARED
  i8gemm.cu
 )


