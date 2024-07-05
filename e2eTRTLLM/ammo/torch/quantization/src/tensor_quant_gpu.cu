/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

#define BLOCK_SIZE 128
#define EPSILON (1. / (1 << 24)) // Minimum representable of fp16

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                        \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                            \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

__host__ __device__ float bits_to_bound(int num_bits, int is_unsigned) {
  float bound = (1 << (num_bits - 1 + int(is_unsigned))) - 1;
  return bound;
}

__device__ float fake_tensor_quant_device(float input, float amax,
                                          int min_bound, int max_bound) {
  CUDA_KERNEL_ASSERT(amax >= 0);

  if (amax < EPSILON) {
    return 0.f;
  }

  float scale = max_bound / amax;
  float output = rint(input * scale);
  output = output > max_bound ? max_bound : output;
  output = output < min_bound ? min_bound : output;

  return output / scale;
}

template <typename T>
__global__ void fake_tensor_quant_kernel(const T *inputs, size_t n, T *outputs,
                                         const float *amax, int num_bits = 8,
                                         bool is_unsigned = false,
                                         bool narrow_range = true) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    if (is_unsigned) {
      CUDA_KERNEL_ASSERT(inputs[tid] >= 0);
    }
    float bound = bits_to_bound(num_bits, is_unsigned);
    float max_bound = bound;
    float min_bound = -(bound + !narrow_range);
    outputs[tid] = fake_tensor_quant_device((float)inputs[tid], amax[0],
                                            min_bound, max_bound);
  }
}

void fake_tensor_quant_cuda_inplace(at::Tensor inputs, at::Tensor amax,
                                    int num_bits = 8, bool is_unsigned = false,
                                    bool narrow_range = true) {
  size_t numel = inputs.numel();
  AT_DISPATCH_FLOATING_TYPES(
      inputs.type().scalarType(), "fake_tensor_quant_cuda_inplace", [&] {
        fake_tensor_quant_kernel<<<numel / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
            inputs.data_ptr<scalar_t>(), numel, inputs.data_ptr<scalar_t>(),
            amax.to(at::ScalarType::Float).data_ptr<float>(), num_bits,
            is_unsigned, narrow_range);
      });
}

at::Tensor fake_tensor_quant_cuda(at::Tensor inputs, at::Tensor amax,
                                  int num_bits = 8, bool is_unsigned = false,
                                  bool narrow_range = true) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs);
  AT_DISPATCH_FLOATING_TYPES(
      inputs.type().scalarType(), "fake_tensor_quant_cuda", [&] {
        fake_tensor_quant_kernel<<<numel / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
            inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>(),
            amax.to(at::ScalarType::Float).data_ptr<float>(), num_bits,
            is_unsigned, narrow_range);
      });

  return outputs;
}

template <typename T>
__global__ void fake_tensor_quant_with_axis_cuda_kernel(
    const T *inputs, size_t n, T *outputs, const float *amax, int axis_size,
    int outer_size, int num_bits = 8, bool is_unsigned = false,
    bool narrow_range = true) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float bound = bits_to_bound(num_bits, is_unsigned);
  float max_bound = bound;
  float min_bound = -(bound + !narrow_range);

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    if (is_unsigned) {
      CUDA_KERNEL_ASSERT(inputs[idx] >= 0);
    }
    int axis_idx = (idx / outer_size) % axis_size;

    outputs[idx] = fake_tensor_quant_device((float)inputs[idx], amax[axis_idx],
                                            min_bound, max_bound);
  }
}

at::Tensor fake_tensor_quant_with_axis_cuda(at::Tensor inputs, at::Tensor amax,
                                            int axis, int num_bits = 8,
                                            bool is_unsigned = false,
                                            bool narrow_range = true) {
  auto outputs = torch::empty_like(inputs);
  size_t numel = inputs.numel();
  int axis_size = inputs.size(axis);

  int outer_size = inputs.stride(axis);

  AT_DISPATCH_FLOATING_TYPES(
      inputs.type().scalarType(), "fake_tensor_quant_cuda_with_axis", [&] {
        fake_tensor_quant_with_axis_cuda_kernel<<<numel / (BLOCK_SIZE * 4) + 1,
                                                  BLOCK_SIZE>>>(
            inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>(),
            amax.to(at::ScalarType::Float).data_ptr<float>(), axis_size,
            outer_size, num_bits, is_unsigned, narrow_range);
      });
  return outputs;
}
