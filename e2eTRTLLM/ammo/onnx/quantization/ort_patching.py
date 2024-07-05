# Adapted from https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/quant_utils.py
# and https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/calibrate.py
#
# MIT License
#
# Copyright (c) Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""This module contains all the patched functions from ORT."""

import numpy as np
from onnxruntime.quantization import quant_utils
from onnxruntime.quantization.calibrate import HistogramCollector
from tqdm import tqdm


def _compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=False):
    """Calculates the scale and zero point.

    Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    Args:
        rmin: minimum value of r
        rmax: maximum value of r
        qmin: minimum value representable by the target quantization data type
        qmax: maximum value representable by the target quantization data type

    Returns:
        A tuple zero and scale (z, s)
    """
    if qmin > 0 or qmax < 0:
        raise ValueError(
            f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin},"
            f" qmmax:{qmax}"
        )

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = -absmax
        rmax = +absmax

    scale = (rmax - rmin) / float(qmax - qmin)
    if scale < 1e-9 or np.isinf(scale):
        scale = 1.0
        zero_point = 0
    else:
        zero_point = round(qmin - rmin / scale)

    return zero_point, scale


def _collect_value(self, name_to_arr):
    """Collect histogram on real value."""
    for tensor, data_arr in tqdm(name_to_arr.items()):
        data_arr = np.asarray(data_arr)  # noqa: PLW2901
        data_arr = data_arr.flatten()  # noqa: PLW2901

        if data_arr.size > 0:
            min_value = np.min(data_arr)
            max_value = np.max(data_arr)
        else:
            min_value = 0
            max_value = 0

        # Change the inf and nan values to meaningful min/max
        min_value = (
            np.finfo(np.float32).tiny if np.isinf(min_value) or np.isnan(min_value) else min_value
        )
        max_value = (
            np.finfo(np.float32).max if np.isinf(max_value) or np.isnan(max_value) else max_value
        )

        threshold = max(abs(min_value), abs(max_value))

        if tensor in self.histogram_dict:
            old_histogram = self.histogram_dict[tensor]
            self.histogram_dict[tensor] = self.merge_histogram(
                old_histogram, data_arr, min_value, max_value, threshold
            )
        else:
            hist, hist_edges = np.histogram(data_arr, self.num_bins, range=(-threshold, threshold))
            self.histogram_dict[tensor] = (
                hist,
                hist_edges,
                min_value,
                max_value,
                threshold,
            )


def patch_ort_modules():
    """Patches the ORT modules."""
    HistogramCollector.collect_value = _collect_value
    quant_utils.compute_scale_zp = _compute_scale_zp
