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

#include <torch/extension.h>

void fake_tensor_quant_cuda_inplace(at::Tensor, at::Tensor, int, bool, bool);
at::Tensor fake_tensor_quant_cuda(at::Tensor, at::Tensor, int, bool, bool);
at::Tensor fake_tensor_quant_with_axis_cuda(at::Tensor, at::Tensor, int, int,
                                            bool, bool);
float bits_to_bound(int, int);
at::Tensor fake_e4m3fy_cuda(at::Tensor inputs);

void fake_tensor_quant_(at::Tensor inputs, at::Tensor amax, int num_bits = 8,
                        bool is_unsigned = false, bool narrow_range = true) {
  TORCH_CHECK(inputs.is_cuda());
  TORCH_CHECK(inputs.is_contiguous()) // in-place on non-contiguous tensor is
                                      // more difficult
  TORCH_CHECK(amax.numel(), 1);
  fake_tensor_quant_cuda_inplace(inputs, amax, num_bits, is_unsigned,
                                 narrow_range);
}
// TODO: Can we add support for CPU tensors here?
at::Tensor fake_tensor_quant(at::Tensor inputs, at::Tensor amax,
                             int num_bits = 8, bool is_unsigned = false,
                             bool narrow_range = true) {
  TORCH_CHECK(inputs.is_cuda());
  TORCH_CHECK(amax.numel(), 1);
  return fake_tensor_quant_cuda(inputs.contiguous(), amax.contiguous(),
                                num_bits, is_unsigned, narrow_range);
}

at::Tensor fake_tensor_quant_with_axis(at::Tensor inputs, at::Tensor amax,
                                       int axis, int num_bits = 8,
                                       bool is_unsigned = false,
                                       bool narrow_range = true) {
  TORCH_CHECK(inputs.is_cuda());
  TORCH_CHECK(amax.numel(), inputs.size(axis));
  return fake_tensor_quant_with_axis_cuda(inputs.contiguous(),
                                          amax.contiguous(), axis, num_bits,
                                          is_unsigned, narrow_range);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_tensor_quant_", &fake_tensor_quant_, "Fake Tensor Quant Inplace",
        py::arg("inputs"), py::arg("amax"), py::arg("num_bits") = 8,
        py::arg("unsigned") = false, py::arg("narrow_range") = true);
  m.def("fake_tensor_quant", &fake_tensor_quant, "Fake Tensor Quant",
        py::arg("inputs"), py::arg("amax"), py::arg("num_bits") = 8,
        py::arg("unsigned") = false, py::arg("narrow_range") = true);
  m.def("fake_tensor_quant_with_axis", &fake_tensor_quant_with_axis,
        "Fake Tensor Quant with axis", py::arg("inputs"), py::arg("amax"),
        py::arg("axis"), py::arg("num_bits") = 8, py::arg("unsigned") = false,
        py::arg("narrow_range") = true);
}
