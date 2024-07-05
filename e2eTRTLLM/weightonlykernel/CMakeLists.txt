



cmake_minimum_required(VERSION 3.8)
set(TARGET_NAME weightonlykernel)
project(${TARGET_NAME})

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++17 -O3 -Xcompiler "-fPIC" 
      -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 )
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(cutlass_kernels)
include_directories(utils)
include_directories(cutlass_extensions/include)
include_directories(cutlass/include)
include_directories(./)



cuda_add_library(
  ${TARGET_NAME} SHARED
   fpA_intB_gemm_wrapper.cu
      weightOnlyBatchedGemv/kernelLauncher.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs1Int4b.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs1Int8b.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs2Int4b.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs2Int8b.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs3Int4b.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs3Int8b.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs4Int4b.cu
     weightOnlyBatchedGemv/weightOnlyBatchedGemvBs4Int8b.cu
      cutlass_kernels/fpA_intB_gemm.cu
      cutlass_kernels/cutlass_heuristic.cc
      cutlass_kernels/cutlass_preprocessors.cc
      utils/logger.cc
      utils/cuda_utils.cc
)


