# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions for PyTorch tensors."""
from collections import abc
from typing import List

import numpy as np
import torch

__all__ = ["torch_to", "torch_detach", "torch_to_numpy", "numpy_to_torch"]


def torch_to(data, *args, **kwargs):
    """Try to recursively move the data to the specified args/kwargs."""
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_to(val, *args, **kwargs) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_to(val, *args, **kwargs) for k, val in data.items()}
    return data


def torch_detach(data):
    """Try to recursively detach the data from the computation graph."""
    if isinstance(data, torch.Tensor):
        return torch.detach(data)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_detach(val) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_detach(val) for k, val in data.items()}
    return data


def torch_to_numpy(inputs: List[torch.Tensor]) -> List[np.ndarray]:
    """Convert torch tensors to numpy arrays."""
    return [t.detach().cpu().numpy() for t in inputs]


def numpy_to_torch(np_outputs: List[np.ndarray]) -> List[torch.Tensor]:
    """Convert numpy arrays to torch tensors."""
    return [torch.from_numpy(arr) for arr in np_outputs]
