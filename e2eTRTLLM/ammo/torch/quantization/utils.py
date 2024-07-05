# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantization utilities."""

from contextlib import contextmanager

import torch

__all__ = ["reduce_amax", "is_quantized", "is_quantized_linear", "replace_function"]


def reduce_amax(input, axis=None, keepdims=True):
    """Compute the absolute maximum value of a tensor.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computation is disabled as this function is never meant learning reduces amax

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True
        granularity: DEPRECTED. specifies if the statistic has to be calculated at tensor or channel granularity

    Returns:
        The reduced tensor.

    Raises:
        ValueError: Any axis which doesn't make sense or is not supported
        ValueError: If unknown granularity is passed in.
    """
    with torch.no_grad():
        # A memory-efficient implementation that avoids copying input tensor
        if axis is None:
            max_val = torch.max(input)
            min_val = torch.min(input)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            max_val = torch.amax(input, dim=axis, keepdim=keepdims)
            min_val = torch.amin(input, dim=axis, keepdim=keepdims)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
            if output.numel() == 1:
                output.squeeze_()
        return output


def is_quantized(module):
    """Check if a module is quantized."""
    from ammo.torch.quantization.nn import TensorQuantizer

    for _module in module.modules():
        if isinstance(_module, TensorQuantizer):
            return True
    return False


def is_quantized_linear(module):
    """Check if a module is a quantized linear module."""
    is_quant = hasattr(module, "input_quantizer") and hasattr(module, "weight_quantizer")
    is_linear = hasattr(module, "weight") and module.weight.dim() == 2
    return is_quant and is_linear


@contextmanager
def replace_function(package, name, new_func):
    """Replace a function with a new one within a context."""
    old_func = getattr(package, name)
    setattr(package, name, new_func)
    setattr(package, "_" + name, old_func)
    yield
    setattr(package, name, old_func)
    delattr(package, "_" + name)
