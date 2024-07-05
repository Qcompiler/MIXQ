# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Handles quantization plugins to correctly quantize third-party modules.

Please check out the source code of this module for examples of how plugins work and how you can
write your own one. Currently, we support plugins for

- :meth:`apex<ammo.torch.quantization.plugins.apex>`
- :meth:`huggingface<ammo.torch.quantization.plugins.huggingface>`
- :meth:`megatron<ammo.torch.quantization.plugins.megatron>`
- :meth:`nemo<ammo.torch.quantization.plugins.nemo>`
"""
from .huggingface import *

try:
    from .apex import *
except ImportError:
    pass
try:
    from .nemo import *
except ImportError:
    pass
try:
    from .megatron import *
except ImportError:
    pass
