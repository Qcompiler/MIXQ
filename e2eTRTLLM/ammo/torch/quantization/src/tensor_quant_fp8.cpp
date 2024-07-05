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
#include <cuda_fp8.h>
#include <torch/extension.h>

at::Tensor fake_e4m3fy_cuda(at::Tensor inputs);

at::Tensor fake_e4m3fy(at::Tensor inputs) {
  if (inputs.is_cuda()) {
    return fake_e4m3fy_cuda(inputs.contiguous());
  } else {
    TORCH_CHECK(inputs.dtype() == at::ScalarType::Float);
    TORCH_CHECK(inputs.is_contiguous());
    auto out = at::zeros_like(inputs);
    for (int i = 0; i < inputs.numel(); ++i) {
      out.data_ptr<float>()[i] = static_cast<float>(
          static_cast<__nv_fp8_e4m3>(inputs.data_ptr<float>()[i]));
    }
    return out;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_e4m3fy", &fake_e4m3fy, "Reduce precision to E4M3",
        py::arg("inputs"));
}
