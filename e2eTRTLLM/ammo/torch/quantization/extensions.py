# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module to load C++ / CUDA extensions."""

from pathlib import Path

from ammo.torch.utils import load_cpp_extension

path = Path(__file__).parent


cuda_ext = load_cpp_extension(
    name="ammo_cuda_ext",
    sources=[path / "src/tensor_quant.cpp", path / "src/tensor_quant_gpu.cu"],
    cuda_version_specifiers=">=11",
)


cuda_ext_fp8 = load_cpp_extension(
    name="ammo_cuda_ext_fp8",
    sources=[path / "src/tensor_quant_fp8.cpp", path / "src/tensor_quant_gpu_fp8.cu"],
    cuda_version_specifiers=">=11.8",
)
