# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Calibrator that returns the absolute max of all collected tensors."""
import warnings

import torch

from ammo.torch.utils import distributed as dist

from .. import utils as quant_utils
from .calibrator import _Calibrator

__all__ = ["MaxCalibrator"]


class MaxCalibrator(_Calibrator):
    """Max calibrator, tracks the maximum value globally.

    Args:
        calib_desc: A MaxCalibDescriptor.
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.

    Readonly Properties:
        amaxs: A list of amax. Numpy array is saved as it is likely to be used for some plot.
    """

    def __init__(self, num_bits, axis, unsigned, track_amax=False):
        """Initialize."""
        super(MaxCalibrator, self).__init__(num_bits, axis, unsigned)
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []  # shall we have a better name?
        self._calib_amax = None

    @property
    def amaxs(self):
        """Returns the list of amax`s collected so far."""
        return self._amaxs

    @torch.no_grad()
    def collect(self, x):
        """Tracks the absolute max of all tensors.

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        # Swap axis to reduce.
        axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
        # Handle negative axis.
        axis = [x.dim() + i if isinstance(i, int) and i < 0 else i for i in axis]
        reduce_axis = []
        for i in range(x.dim()):
            if i not in axis:
                reduce_axis.append(i)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax = torch.max(self._calib_amax, local_amax)

        if self._track_amax:
            self._amaxs.append(local_amax.cpu().numpy())

    def reset(self):
        """Reset the collected absolute max."""
        self._calib_amax = None

    def compute_amax(self):
        """Return the absolute max of all tensors collected."""
        if dist.size() > 1 and dist.is_master():
            warnings.warn(
                "This method does not perform any synchronization across DistributedDataParallel"
                " (DDP) https://pytorch.org/docs/stable/notes/ddp.html modules. The recommended"
                " method is to use the same calibration dataset across all distributed data"
                " parallel groups so that `amax` is the same for all DDP modules."
            )
        return self._calib_amax

    def __str__(self):
        s = "MaxCalibrator("
        s += "track_amax={_track_amax}"
        s += ")"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "MaxCalibrator("
        s += super(MaxCalibrator, self).__repr__()
        s += " calib_amax={_calib_amax}"
        s += " track_amax={_track_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)
