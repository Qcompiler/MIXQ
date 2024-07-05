# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This module contains the mode descriptor for the quantization mode."""

from typing import Optional, Set

from ammo.torch.opt.mode import (
    ConfigDict,
    ConvertEntrypoint,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeDescriptor,
    _ModeRegistryCls,
)

from .conversion import (
    convert_to_quantized_model,
    restore_quantized_model,
    update_quantize_metadata,
)

QuantizeModeRegistry = _ModeRegistryCls()


# TODO: OMNIML-717 Reuse search infra for quantization calibration algorithms
@QuantizeModeRegistry.register_mode
class QuantizeModeDescriptor(_ModeDescriptor):
    """Class to describe the ``"quant"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "quantize"

    @property
    def config(self) -> ConfigDict:
        """Specifies the default config for the mode.

        Please refer to :mod:`ammo.torch.quantization.config` for more quantization configurations.
        """
        return {"quant_cfg": {"default": {"num_bits": 8, "axis": None}}, "algorithm": "max"}

    # TODO: [OMNIML-716] Enables chaining distillation mode after quantization mode.
    @property
    def next_modes(self) -> Optional[Set[str]]:
        """Modes that must immediately follow this mode."""
        return set()

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_to_quantized_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_quantized_model

    @property
    def update(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating a model."""
        return update_quantize_metadata
