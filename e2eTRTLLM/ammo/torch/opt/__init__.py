# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module for general-purpose model optimization infrastructure.

``ammo.torch.opt`` contains tooling to:

* ingest a user-provided model and set it up for optimization;
* define and implement search and optimization procedures;
* export a model back to a regular pytorch model after optimization;
* save, restore, and manage checkpoints from which the model modifications can be resumed.

Please refer to each individual sub-module to learn more about the various concepts wihin
``ammo.torch.opt`` and how to use them to implement a model optimization algorithm.
"""

from .conversion import *
from .mode import *
from .searcher import *
