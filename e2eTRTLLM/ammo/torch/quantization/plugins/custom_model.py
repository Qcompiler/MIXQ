# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom on the fly model plugins for quantization."""

from .huggingface import register_falcon_linears_on_the_fly


# TODO: This is a temporary solution
# In future implement a decorator to register methods updating QUANT_MODULE on the fly
def register_custom_model_plugins_on_the_fly(model):
    """Registers custom modules as QUANT_MODULE on the fly."""
    register_falcon_linears_on_the_fly(model)
