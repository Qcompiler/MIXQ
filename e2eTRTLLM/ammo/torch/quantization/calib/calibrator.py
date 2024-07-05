# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Abstract base class for calibrators."""


class _Calibrator:
    """Abstract base class of calibrators.

    Args:
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.

    Readonly Properties:
        axis:
    """

    def __init__(self, num_bits, axis, unsigned):
        """Initialize."""
        self._num_bits = num_bits
        self._axis = axis
        self._unsigned = unsigned

    def collect(self, x):
        """Abstract method: collect tensor statistics used to compute amax.

        Args:
            x: A tensor
        """
        raise NotImplementedError

    def reset(self):
        """Abstract method: reset calibrator to initial state."""
        raise NotImplementedError

    def compute_amax(self, *args, **kwargs):
        """Abstract method: compute the amax from the collected data.

        Returns:
            amax: a tensor
        """
        raise NotImplementedError

    def __repr__(self):
        s = "num_bits={_num_bits}"
        s += " axis={_axis}"
        s += " unsigned={_unsigned}"
        return s.format(**self.__dict__)
