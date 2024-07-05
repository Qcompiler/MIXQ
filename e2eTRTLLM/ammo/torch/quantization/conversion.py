# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Quantization conversion/restore utilities."""

import fnmatch
import warnings
from typing import Any, Callable, Dict, Union

import torch.nn as nn

from ..opt.conversion import ApplyModeError
from ..opt.mode import (
    ConfigDict,
    MetadataDict,
)
from .module_mapping import QUANT_MODULE_MAPPING
from .nn import SequentialQuantizer, TensorQuantizer
from .plugins.custom_model import register_custom_model_plugins_on_the_fly
from .tensor_quant import QuantDescriptor
from .utils import is_quantized

__all__ = [
    "replace_quant_module",
    "set_quantizer_by_cfg",
    "set_quantizer_attribute",
]


def convert_to_quantized_model(model: nn.Module, config: ConfigDict):
    """Convert the model to a quantized one as per `config`."""
    replace_quant_module(model)
    set_quantizer_by_cfg(model, config["quant_cfg"])
    metadata = {"quantizer_state": quantizer_state(model)}
    return model, metadata


def restore_quantized_model(
    model: nn.Module, config: ConfigDict, metadata: MetadataDict
) -> nn.Module:
    """Restores the quantizer states from the given state dict."""

    def _get_parent_device(child_name):
        parent_module = model.get_submodule(child_name.rpartition(".")[0])
        # If the parent module is a sequential quantizer, get the device of the parent of the parent
        if isinstance(parent_module, SequentialQuantizer):
            return _get_parent_device(child_name.rpartition(".")[0].rpartition(".")[0])

        try:
            return next(parent_module.parameters()).device
        except StopIteration:
            # For modules without parameters
            return None

    assert not is_quantized(model), "Model must not be quantized!"

    quantizer_state_dict = metadata["quantizer_state"]

    replace_quant_module(model)
    SequentialQuantizer.restore_sequential_quantizers(model, quantizer_state_dict)

    unmatched_keys = quantizer_state_dict.keys() - quantizer_state(model).keys()
    extra_keys = quantizer_state(model).keys() - quantizer_state_dict.keys()

    if unmatched_keys:
        raise ApplyModeError(f"Unmatched keys in quantizer state_dict: {unmatched_keys}")
    if extra_keys:
        raise ApplyModeError(f"Extra keys in quantizer state_dict: {extra_keys}")

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            device = _get_parent_device(name)
            module.set_from_ammo_state(quantizer_state_dict[name], name, device)
            if device is None:
                warnings.warn(
                    f"Restoring quantizer {name} from state dict. Could not look up parent"
                    " module device. Please move the model to the correct device after this. Model"
                    " forward might throw error otherwise."
                )

    return model


def update_quantize_metadata(model: nn.Module, metadata: MetadataDict) -> None:
    """Update the quantizer state in the metadata dict."""
    metadata["quantizer_state"] = quantizer_state(model)


def quantizer_state(model: nn.Module) -> Dict[str, Any]:
    """Returns the quantizer state dict describing the quantizer states in the model."""
    return {
        n: m.get_ammo_state()
        for n, m in model.named_modules()
        if isinstance(m, (TensorQuantizer, SequentialQuantizer))
    }


def replace_quant_module(model: nn.Module):
    """Recursively replace the module with quantized module."""
    assert not is_quantized(model), "Model must not be quantized!"

    register_custom_model_plugins_on_the_fly(model)

    def _replace_quant_module(model):
        for name, module in model.named_children():
            if type(module) in QUANT_MODULE_MAPPING:
                setattr(model, name, QUANT_MODULE_MAPPING[type(module)](module))
            # Continue replacing in case of nested quantization as well
            _replace_quant_module(getattr(model, name))

    _replace_quant_module(model)

    replaced_modules = sum(isinstance(m, TensorQuantizer) for _, m in model.named_modules())
    print(f"Replaced {replaced_modules} modules to quantized modules")


def set_quantizer_by_cfg(quant_model: nn.Module, quant_cfg):
    """Change the configuration of all quantizers."""
    quant_cfg = quant_cfg.copy()
    if "default" in quant_cfg:
        set_quantizer_attribute(quant_model, "*", quant_cfg["default"])
        quant_cfg.pop("default")
    for pattern, cfg in quant_cfg.items():
        set_quantizer_attribute(quant_model, pattern, cfg)


def set_quantizer_attribute(
    quant_model: nn.Module,
    wildcard_or_filter_func: Union[str, Callable],
    attribute,
):
    """Finegrained adjustment of quantizer attribute by wildcard or filter function.

    Args:
        quant_model: A pytorch model
        wildcard_or_filter_func: a wildcard string or a filter function. The wildcard string is matched
            against the quantizer module names. The quantizer modules are
            instances of :class:`TensorQuantizer <ammo.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
            The filter function takes a quantized module name as input and returns ``True`` if the
            quantizer should be adjusted and ``False`` otherwise.
        attribute: a dict of quantizer attributes or a list of quantizer attribute dicts.
            An example attribute dict is: ``{"num_bits": 8, "axis": 0, "enable": True}``.
            If ``attribute`` is a list of dicts, the matched
            :class:`TensorQuantizer <nn.modules.tensor_quantizer.TensorQuantizer>` modules will be replaced with
            :class:`SequentialQuantizer <nn.modules.tensor_quantizer.SequentialQuantizer>` modules having one quantizer
            for each attribute dict from the list.

    """
    for name, module in quant_model.named_modules():
        if isinstance(module, TensorQuantizer):
            if isinstance(wildcard_or_filter_func, str):
                if not fnmatch.fnmatch(name, wildcard_or_filter_func):
                    continue
            elif callable(wildcard_or_filter_func):
                if not wildcard_or_filter_func(name):
                    continue
            else:
                raise NotImplementedError(f"Unsupported type {type(wildcard_or_filter_func)}")

            if isinstance(attribute, list):
                parent_module = quant_model.get_submodule(name.rpartition(".")[0])
                module = SequentialQuantizer(
                    *(TensorQuantizer(QuantDescriptor()) for _ in range(len(attribute)))
                )
                setattr(parent_module, name.split(".")[-1], module)

            module.set_from_attribute_dict(attribute)
