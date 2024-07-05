# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""TensorQuantizer Module."""
import contextlib
import math
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx._globals import GLOBALS

from ... import calib
from ... import utils as quant_utils
from ...tensor_quant import (
    QuantDescriptor,
    fake_tensor_quant,
    scaled_e4m3,
    tensor_quant,
)
from .clip import Clip

__all__ = ["TensorQuantizer", "SequentialQuantizer"]


class TensorQuantizer(nn.Module):
    """Tensor quantizer module.

    This module uses tensor_quant or fake_tensor_quant function to quantize a tensor. And wrappers
    variable, moving statistics we'd want when training a quantized network.

    Experimental features:
        * ``clip`` stage learns range before enabling quantization.
        * ``calib`` stage runs calibration

    Args:
        quant_desc: An instance of :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`.
        disabled: A boolean. If True, by pass the whole module returns input. Default False.
        if_quant: A boolean. If True, run main quantization body. Default True.
        if_clip: A boolean. If True, clip before quantization and learn amax. Default False.
        if_calib: A boolean. If True, run calibration. Not implemented yet. Settings of calibration
            will probably go to :class:`QuantDescriptor <ammo.torch.quantization.QuantDescriptor>`.

    Readonly Properties:
        - axis:
        - fake_quant:
        - scale:
        - step_size:

    Mutable Properties:
        - num_bits:
        - unsigned:
        - amax:
    """

    def __init__(
        self,
        quant_desc=QuantDescriptor(),
        disabled=False,
        if_quant=True,
        if_clip=False,
        if_calib=False,
    ):
        """Initialize quantizer and set up required variables."""
        super(TensorQuantizer, self).__init__()
        # Expand quant_desc. Use quant_desc.dict would be easier, but adding one-by-one explicitly gives more control
        self._num_bits = quant_desc.num_bits
        self._fake_quant = quant_desc.fake_quant
        self._axis = quant_desc.axis
        self._block_sizes = quant_desc.block_sizes
        self._scale_amax = quant_desc.scale_amax
        self._learn_amax = quant_desc.learn_amax
        self._unsigned = quant_desc.unsigned
        self._narrow_range = quant_desc.narrow_range

        self._scale = None if not quant_desc.fake_quant else 1.0
        self._disabled = disabled
        self._if_quant = if_quant
        self._if_clip = False
        self._if_calib = if_calib

        if quant_desc.amax is not None:
            self.register_buffer("_amax", torch.tensor(quant_desc.amax))

        # Clip module consumes a lot of memory, so only create it if learn_amax is True
        if self._learn_amax:
            init_amax = quant_desc.amax if quant_desc.amax is not None else 1.0
            self.clip = Clip(-init_amax, init_amax, learn_min=True, learn_max=True)
            # It makes more sense to enable clip stage (which learns amax) if learn_amax is true
            self.enable_clip()
        if if_clip:
            self.enable_clip()

        if quant_desc.calib_method == "histogram":
            self._calibrator = calib.HistogramCalibrator(
                num_bits=self._num_bits, axis=self._axis, unsigned=self._unsigned
            )
        elif quant_desc.calib_method == "max":
            self._calibrator = calib.MaxCalibrator(
                num_bits=self._num_bits, axis=self._axis, unsigned=self._unsigned
            )

    @property
    def num_bits(self):
        """Return num_bits for quantization."""
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        self._num_bits = value

    @property
    def maxbound(self):
        """Return maxbound for quantization."""
        if self._num_bits == (4, 3):
            return 448.0
        return (1 << (self._num_bits - 1 + int(self._unsigned))) - 1

    @property
    def unsigned(self):
        """Return True if unsigned quantization is used."""
        return self._unsigned

    @unsigned.setter
    def unsigned(self, value):
        self._unsigned = value

    @property
    def scale(self):
        """Return scale used for quantization."""
        return self._scale

    @property
    def pre_quant_scale(self):
        """Return pre_quant_scale used for smoothquant."""
        if not hasattr(self, "_pre_quant_scale"):
            return None
        return self._pre_quant_scale

    @pre_quant_scale.setter
    def pre_quant_scale(self, value):
        assert value is not None, "pre_quant_scale cannot be set to None."
        if not hasattr(self, "_pre_quant_scale"):
            self.register_buffer("_pre_quant_scale", torch.tensor(value))
        else:
            value = torch.tensor(value, device=self._pre_quant_scale.device)
            if self._pre_quant_scale.shape != value.shape:
                raise RuntimeError("Changing shape when setting pre_quant_scale is not allowed.")
            self._pre_quant_scale.data.copy_(value.data)

    @property
    def amax(self):
        """Return amax for quantization."""
        if not hasattr(self, "_amax"):
            return None
        return self._amax

    @amax.setter
    def amax(self, value):
        assert value is not None, "amax cannot be set to None."

        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", torch.tensor(value))
        else:
            value = torch.tensor(value, device=self._amax.device)
            if self._amax.shape != value.shape:
                raise RuntimeError("Changing shape when setting amax is not allowed.")
            self._amax.data.copy_(value.data)

    @property
    def step_size(self):
        """Return step size for integer quantization."""
        if not hasattr(self, "_amax"):
            warnings.warn("step_size is undefined under dynamic amax mode!")
            return None
        assert isinstance(
            self._num_bits, int
        ), "Step size is not defined for non-integer quantization."
        return self._amax / (2.0 ** (self._num_bits - 1 + int(self._unsigned)) - 1.0)

    @property
    def axis(self):
        """Return axis for quantization."""
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = value

    @property
    def block_sizes(self):
        """Return block_sizes for quantization."""
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, value):
        self._axis = None
        self._block_sizes = value

    @property
    def fake_quant(self):
        """Return True if fake quantization is used."""
        return self._fake_quant

    @property
    def narrow_range(self):
        """Return True if symmetric integer range for signed quantization is used."""
        return self._narrow_range

    @narrow_range.setter
    def narrow_range(self, value):
        self._narrow_range = value

    @property
    def is_enabled(self):
        """Return true if the modules is not disabled."""
        return not self._disabled

    def disable(self):
        """Bypass the module.

        Neither of calibration, clipping and quantization will be performed if the module is disabled.
        """
        self._disabled = True

    def enable(self):
        """Enable the module."""
        self._disabled = False

    def disable_clip(self):
        """Disable clip stage."""
        self._if_clip = False
        self.clip.clip_value_min.requires_grad = False
        self.clip.clip_value_max.requires_grad = False

    def enable_clip(self):
        """Enable clip stage."""
        if not self._learn_amax:
            raise ValueError("learn_amax is False. Cannot enable clip.")
        self.clip.clip_value_min.requires_grad = True
        self.clip.clip_value_max.requires_grad = True
        self._if_clip = True

    def disable_calib(self):
        """Disable calibration."""
        self._if_calib = False

    def enable_calib(self):
        """Enable calibration."""
        if self._calibrator is None:
            raise ValueError("Calibrator was not created, cannot enable calibration.")
        self._if_calib = True

    def disable_quant(self):
        """Disable quantization."""
        self._if_quant = False

    def enable_quant(self):
        """Enable quantization."""
        self._if_quant = True

    def load_calib_amax(self, *args, **kwargs):
        """Load amax from calibrator.

        Updates the amax buffer with value computed by the calibrator, creating it if necessary.
        ``*args`` and ``**kwargs`` are directly passed to ``compute_amax``, except ``"strict"`` in
        ``kwargs``. Refer to ``compute_amax`` for more details.
        """
        strict = kwargs.pop("strict", True)
        if getattr(self, "_calibrator", None) is None:
            raise RuntimeError("Calibrator not created.")
        calib_amax = self._calibrator.compute_amax(*args, **kwargs)
        if calib_amax is None:
            err_msg = (
                "Calibrator returned None. This usually happens when calibrator hasn't seen any"
                " tensor."
            )
            if not strict:
                warnings.warn(err_msg)
                warnings.warn("Set amax to NaN!")
                calib_amax = torch.tensor(math.nan)
            else:
                raise RuntimeError(
                    err_msg
                    + " Passing 'strict=False' to `load_calib_amax()` will ignore the error."
                )
        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", calib_amax.data)
        else:
            self._amax.copy_(calib_amax)

    def init_learn_amax(self):
        """Initialize learned amax from fixed amax."""
        if self._learn_amax is False:
            raise RuntimeError("Called init_learn_amax with learn_amax=False.")

        if self._amax.numel() != 1:
            warnings.warn("Per channel learned amax not supported. Initializing with max(amax).")
            init_amax = torch.max(self._amax)
        else:
            init_amax = self._amax
        self.clip.clip_value_min.data.copy_(-init_amax.data)
        self.clip.clip_value_max.data.copy_(init_amax.data)

    def _get_amax(self, inputs):
        """Get amax from buffer or compute it dynamically."""
        if hasattr(self, "_amax"):
            amax = self._amax
        else:
            if self._axis is None:
                reduce_axis = None
            else:
                reduce_axis = []
                # Swap axis to reduce
                axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
                for i in range(inputs.dim()):
                    if not (i in axis or (i - inputs.dim()) in axis):
                        reduce_axis.append(i)
            amax = quant_utils.reduce_amax(inputs, axis=reduce_axis, keepdims=True).detach()
        if self._scale_amax is not None:
            amax = amax.detach() * self._scale_amax

        amax = amax.data
        return amax

    def _quant_forward(self, inputs):
        """Quantized forward pass."""
        if self._learn_amax:
            inputs = self.clip(inputs)
            amax = torch.max(-self.clip.clip_value_min, self.clip.clip_value_max).detach()
        else:
            amax = self._get_amax(inputs)

        if self._fake_quant:
            if isinstance(self._num_bits, tuple):
                E, M = self._num_bits  # noqa: N806
                outputs = scaled_e4m3(inputs, self._get_amax(inputs), E, M)
            else:
                outputs = fake_tensor_quant(
                    inputs, amax, self._num_bits, self._unsigned, self._narrow_range
                )
        else:
            assert not isinstance(
                self._num_bits, tuple
            ), "only fake quantization supports ExMy type quantization."
            outputs, self._scale = tensor_quant(inputs, amax, self._num_bits, self._unsigned)

        return outputs

    def _check_onnx_readiness(self, inputs):
        """Check if quantizer is ready for ONNX export."""
        assert hasattr(self, "_amax"), (
            "Quantizer has not been calibrated. ONNX export requires the quantizer to be"
            " calibrated.Calibrate and load amax before exporting to ONNX."
        )

        if self._if_calib:
            warnings.warn(
                "Quantizer is in calibration mode. "
                "Please complete calibration before exporting to ONNX for correct results."
            )

        amax = self._get_amax(inputs)

        # We only support scalar amax for E4M3 ONNX export
        if isinstance(self.num_bits, tuple):
            assert amax.numel() == 1, (
                "E4M3 supports ONNX export only for per-tensor quantization."
                " Per-tensor quantization requires scalar amax. "
                f"Received non-scalar amax of shape: {amax.shape}"
            )

        if self.block_sizes is not None:
            raise Exception("Blockquant does not support ONNX export.")

    def _setup_for_blockquant(self, inputs: torch.Tensor):
        # Get reshape sizes and paddings for block-quantization
        def get_axis_quant_params(ax):
            ax = ax if ax in self.block_sizes else ax - inputs.dim()
            bsize = self.block_sizes.get(ax, None)
            padding, ax_slice = None, None
            if bsize is not None and inputs.shape[ax] % bsize != 0:
                padding = (bsize - (inputs.shape[ax] % bsize), 0)
                ax_slice = slice(inputs.shape[ax])
            return bsize, padding, ax_slice

        def set_quant_params(axis, block_reshape_size, padding, slices, amax_shape=None):
            self._axis = tuple(axis)
            if hasattr(self, "_calibrator"):
                self._calibrator._axis = self._axis
            self._original_shape = inputs.shape
            self._block_reshape_size = torch.Size(block_reshape_size)
            if padding is not None:
                self._padding = tuple(padding)
                self._original_shape = F.pad(inputs, self._padding, "constant", 0).shape
            if slices is not None:
                self._slices = slices
            if amax_shape:
                self._amax_shape_for_export = amax_shape

        # Reshape size have already been set
        if hasattr(self, "_block_reshape_size"):
            return

        reshape_size, quantize_axis, paddings, slices = [], [], [], []

        # special handling for block-quantization along the last axis:
        # flatten the input for faster execution
        if (self.block_sizes.get(inputs.dim() - 1, None) or self.block_sizes.get(-1, None)) and len(
            self._block_sizes
        ) == 1:
            bsize, padding, ax_slice = get_axis_quant_params(inputs.dim() - 1)
            slices = None if ax_slice is None else (*(slice(None),) * (inputs.dim() - 1), ax_slice)
            padding = padding if not padding else tuple(reversed(padding))
            amax_shape_for_export = (*(inputs.shape[:-1]), -1)
            set_quant_params((0,), (-1, bsize), padding, slices, amax_shape_for_export)
            return

        for ax in range(inputs.dim()):
            bsize, padding, ax_slice = get_axis_quant_params(ax)
            paddings.append(padding)
            slices.append(ax_slice)
            if bsize is not None:
                reshape_size.extend([math.ceil(inputs.shape[ax] / bsize), bsize])
                quantize_axis.extend([True, False])
            else:
                reshape_size.append(inputs.shape[ax])
                quantize_axis.append(True)

        quant_axis = [i for i in range(len(quantize_axis)) if quantize_axis[i]]

        if all(s is None for s in slices):
            slices = None
        else:
            slices = [s if s else slice(None) for s in slices]

        if all(p is None for p in paddings):
            paddings = None
        else:
            new_paddings = []
            for padding in paddings:
                if not (new_paddings or padding):
                    continue
                new_paddings.extend(padding if padding else (0, 0))
            paddings = tuple(reversed(new_paddings))

        set_quant_params(quant_axis, reshape_size, paddings, slices)

    def _process_for_blockquant(self, inputs: torch.Tensor):
        if hasattr(self, "_padding"):
            inputs = F.pad(inputs, self._padding, "constant", 0)
        assert inputs.shape == self._original_shape, (
            f"Input shape has changed from {self._original_shape} to {inputs.shape}."
            " Block-quantization requires a fixed input shape."
        )
        inputs = inputs.reshape(self._block_reshape_size)
        return inputs

    def _reset_to_original_shape(self, outputs: torch.Tensor):
        outputs = outputs.reshape(self._original_shape)
        if hasattr(self, "_slices"):
            outputs = outputs[self._slices]
        return outputs

    def export_amax(self) -> Optional[torch.Tensor]:
        """Export correctly formatted/shaped amax."""
        if self.amax is None:
            return None

        if not hasattr(self, "_amax_shape_for_export"):
            amax = self.amax
        else:
            amax = self.amax.reshape(self._amax_shape_for_export)
        amax[amax == 0] = self.maxbound
        return amax

    def forward(self, inputs):
        """Apply tensor_quant function to inputs.

        Args:
            inputs: A Tensor of type float32.

        Returns:
            outputs: A Tensor of type output_dtype
        """
        # Activation scaling for smoothquant
        if self.pre_quant_scale is not None:
            inputs = inputs * self.pre_quant_scale

        if self._disabled:
            return inputs

        if GLOBALS.in_onnx_export:
            self._check_onnx_readiness(inputs)

        if self.block_sizes is not None:
            self._setup_for_blockquant(inputs)
            inputs = self._process_for_blockquant(inputs)

        outputs = inputs

        if self._if_calib:
            if self._calibrator is None:
                raise RuntimeError("Calibrator was not created.")
            # Shape is only known when it sees the first tensor
            self._calibrator.collect(inputs)

        if self._if_clip:
            if not self._learn_amax:
                raise RuntimeError("Clip without learning amax is not implemented.")
            outputs = self.clip(inputs)

        if self._if_quant:
            outputs = self._quant_forward(inputs)

        if self.block_sizes is not None:
            outputs = self._reset_to_original_shape(outputs)

        return outputs

    def _short_amax(self, fmt=".4f"):
        """Short description of amax.

        Returns:
            'dynamic': if _amax is not registered
            'amax': if _amax is per-tensor
            '[min, max](size)': if _amax is per-channel
        """
        if not hasattr(self, "_amax"):
            return "dynamic"
        if self._amax is None:
            return "None"
        if self._amax.numel() == 1:
            return f"{self._amax.item():{fmt}}"
        return (
            f"[{self._amax.min().item():{fmt}},"
            f" {self._amax.max().item():{fmt}}]({self._amax.numel()})"
        )

    def extra_repr(self):
        """Set the extra information about this module."""
        if self._disabled:
            return "disabled"
        s = f"{'unsigned ' if self._unsigned else ''}{self._num_bits} bit"
        s += " narrow" if (self._narrow_range) else ""
        s += " fake" if (self._fake_quant) else ""
        if self.block_sizes is not None:
            s += f" block_sizes={self._block_sizes},"
        else:
            s += f" axis={self._axis}" if self._axis is not None else " per-tensor"
        s += f" amax={self._short_amax()}"
        s += f" *{self._scale_amax}" if self._scale_amax else ""
        s += " pre_quant_scale" if self.pre_quant_scale is not None else ""
        s += " learned" if (self._learn_amax) else ""
        s += (
            f" calibrator={self._calibrator.__class__.__name__}"
            if (self._calibrator is not None)
            else ""
        )
        s += f" scale={self._scale}" if self._scale is not None else ""
        s += " quant" if (self._if_quant) else ""
        s += " clip" if (self._if_clip) else ""
        s += " calib" if (self._if_calib) else ""
        return s

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Overloaded module function.

        Adds warnings during state_dict loading.
        A workaround is implemented for loading amax from checkpoint and only supports CUDA.

        Args:
            state_dict: A dict containing the state of the top level module
            prefix: A string that prefixes all of this modules state in state_dict, e.g. 'model.conv1.'
        """
        dst_has_amax = "_amax" in self._buffers
        src_has_amax = prefix + "_amax" in state_dict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not src_has_amax and dst_has_amax:
            warnings.warn(f"{prefix[:-1]}: No amax in state_dict.")
        elif src_has_amax and not dst_has_amax:
            warnings.warn(
                f"{prefix[:-1]}: No '_amax' buffer to load amax into."
                " '_amax` will be created as WAR for now. "
                "This behavior will change in future."
            )
            self.register_buffer("_amax", state_dict[prefix + "_amax"].data.to(device))

        dst_has_pre_quant_scale = "_pre_quant_scale" in self._buffers
        src_has_pre_quant_scale = prefix + "_pre_quant_scale" in state_dict

        if not src_has_pre_quant_scale and dst_has_pre_quant_scale:
            warnings.warn(f"{prefix[:-1]}: No pre_quant_scale in state_dict.")
        elif src_has_pre_quant_scale and not dst_has_pre_quant_scale:
            warnings.warn(
                f"{prefix[:-1]}: No '_pre_quant_scale' buffer to load pre_quant_scale into."
                " '_pre_quant_scale` will be created as WAR for now. "
                "This behavior will change in future."
            )
            self.register_buffer(
                "_pre_quant_scale", state_dict[prefix + "_pre_quant_scale"].data.to(device)
            )

        super(TensorQuantizer, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _get_properties_for_ammo_state(self):
        return self.__dict__.keys() - nn.Module().__dict__.keys() - {"clip", "_calibrator"}

    def get_ammo_state(self) -> Dict[str, Any]:
        """Get meta state to be saved in checkpoint."""
        ammo_state = {}
        for k in self._get_properties_for_ammo_state():
            ammo_state[k] = getattr(self, k)

        if self.amax is not None:
            ammo_state["_amax"] = self.amax

        if self.pre_quant_scale is not None:
            ammo_state["_pre_quant_scale"] = self.pre_quant_scale

        if hasattr(self, "clip"):
            ammo_state["_init_clip"] = True

        if hasattr(self, "_calibrator"):
            ammo_state["_calibrator_type"] = str(type(self._calibrator))

        return ammo_state

    def set_from_ammo_state(self, ammo_state, prefix="", device=None):
        """Set meta state from checkpoint."""
        for key in self._get_properties_for_ammo_state():
            setattr(self, key, ammo_state[key])

        if "_init_clip" in ammo_state:
            # clip min and max parameters will be loaded from checkpoint
            self.clip = Clip(-1.0, 1.0, learn_min=True, learn_max=True)

        if "_amax" in ammo_state:
            self.amax = ammo_state["_amax"]

        if "_pre_quant_scale" in ammo_state:
            self.pre_quant_scale = ammo_state["_pre_quant_scale"]

        if "_calibrator_type" in ammo_state:
            if "MaxCalibrator" in ammo_state["_calibrator_type"]:
                calib_cls = calib.MaxCalibrator
            elif "HistogramCalibrator" in ammo_state["_calibrator_type"]:
                calib_cls = calib.HistogramCalibrator
            else:
                raise RuntimeError(
                    f"{prefix[:-1]}: Unknown calibrator type: {ammo_state['_calibrator_type']}"
                )

            self._calibrator = calib_cls(
                num_bits=self._num_bits, axis=self._axis, unsigned=self._unsigned
            )

        if device is not None:
            self.to(device)

    def set_from_attribute_dict(self, attribute_dict: Dict[str, Any]):
        """Set quantizer attributes from attribute_dict."""
        if "num_bits" in attribute_dict:
            self.num_bits = attribute_dict["num_bits"]
        if "axis" in attribute_dict:
            self.axis = attribute_dict["axis"]
            if hasattr(self, "_calibrator"):
                self._calibrator._axis = attribute_dict["axis"]
        if "block_sizes" in attribute_dict:
            assert (
                not attribute_dict["block_sizes"] or attribute_dict.get("axis", None) is None
            ), "axis must be None when block_sizes is not None."
            self.block_sizes = attribute_dict["block_sizes"]
        if "enable" in attribute_dict:
            if attribute_dict["enable"]:
                self.enable()
            else:
                self.disable()


class SequentialQuantizer(nn.Sequential):
    """A sequential container for  :class:`TensorQuantizer` modules.

    This modules is used to quantize a tensor in multiple formats sequentially. It takes as input
    :class:`TensorQuantizer` modules and containerize them similar to :class:`torch.nn.Sequential`.

    Args:
        quantizers (TensorQuantizer): :class:`TensorQuantizer` modules to be added to the container.

    """

    def __init__(self, *quantizers: TensorQuantizer):  # noqa: N803
        """Initialize SequentialQuantizer module."""
        assert not any(
            not isinstance(q, TensorQuantizer) for q in quantizers
        ), "All quantizers must be a TensorQuantizer."
        super().__init__(*quantizers)

    def get_ammo_state(self) -> Dict[str, Any]:
        """Get meta state to be saved in checkpoint."""
        return {"num_quantizers": len(self), "is_sequential_quantizer": True}

    def disable(self):
        """Disable the quantizer modules."""
        for quantizer in self:
            quantizer.disable()

    def set_from_attribute_dict(self, attributes: List[Dict[str, Any]]):
        """Set the attributes of contained quantizers from a list of attribute_dicts."""
        for attribute, quantizer in zip(attributes, self):
            quantizer.set_from_attribute_dict(attribute)

    @staticmethod
    def restore_sequential_quantizers(model, ammo_state: Dict[str, Any]):
        """Restore sequential quantizers from checkpoint."""
        for name, ammo_state in ammo_state.items():
            if "is_sequential_quantizer" in ammo_state and ammo_state["is_sequential_quantizer"]:
                sequential_quantizer = SequentialQuantizer(
                    *(
                        TensorQuantizer(QuantDescriptor())
                        for _ in range(ammo_state["num_quantizers"])
                    )
                )
                parent_module = model.get_submodule(name.rpartition(".")[0])
                setattr(parent_module, name.rpartition(".")[-1], sequential_quantizer)

    @staticmethod
    @contextlib.contextmanager
    def replace_sequential_quantizer_with_single_quantizer(model, indx: int = 0):
        """Replace instances of :class:`SequentialQuantizer` in the model with single quantizers.

        The quantizer indexed by ``indx`` from the sequential quantizer is used to replace it.
        This method is useful for individually calibrating the quantizers in a sequential quantizer.
        """
        for name, module in list(model.named_modules()):
            if isinstance(module, SequentialQuantizer):
                assert len(module) > indx
                parent_module = model.get_submodule(name.rpartition(".")[0])
                setattr(parent_module, "_original_" + name.rpartition(".")[-1], module)
                setattr(parent_module, name.rpartition(".")[-1], module[indx])

        yield

        for name, module in list(model.named_modules()):
            if isinstance(module, SequentialQuantizer) and "_original_" in name.rpartition(".")[-1]:
                parent_module = model.get_submodule(name.rpartition(".")[0])
                original_name = name.rpartition(".")[-1].replace("_original_", "")
                setattr(parent_module, original_name, module)
                delattr(parent_module, name.rpartition(".")[-1])
