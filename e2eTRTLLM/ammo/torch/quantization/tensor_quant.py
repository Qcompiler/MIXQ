# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Basic tensor quantization functions."""

import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch.autograd import Function
from torch.onnx import symbolic_helper


class ScaledQuantDescriptor:
    """Supportive descriptor of quantization.

    Describe how a tensor should be quantized. A QuantDescriptor and a tensor defines a quantized
    tensor.

    Args:
        num_bits: An integer or a tuple of two integers.
            Specifically, `num_bits` can be:

            #. A positive integer argument for integer quantization. `num_bits` specify
                the number of bits used for integer quantization.

            #. Constant integer tuple (4,3) for E4M3 floating point quantization emulating
                Nvidia's FP8 quantization. E4M3 quantization only supports per-tensor quantization.

            Default: 8.
        name: Seems a nice thing to have
        fake_quant: A boolean. If True, use fake quantization mode. Default True.
        axis: None, int or tuple of int. The specified axis/axes will have its own amax for
            computing scaling factor. If None (the default), use per tensor scale. Must be in the
            range [-rank(input_tensor), rank(input_tensor)). E.g. For a KCRS weight tensor,
            ``quant_axis=(0)`` will yield per channel scaling.
        block_sizes: None or a dictionary with integer keys and values. The dictionary specifies
            block quantization parameters. The keys are the axes for block quantization and the
            values are block sizes for quantization along the respective axes. Keys must be in the
            range ``[-rank(input_tensor), rank(input_tensor)]``. Values, which are the block sizes
            for quantization must be positive integers. For example, ``block_sizes = {-1: 32}`` will
            quantize the last axis of the input tensor in blocks of size 32. If None, block
            quantization is not performed. ``axis`` must be None when ``block_sizes`` is not None.
        amax: A float or list/ndarray of floats of user specified absolute max range. If supplied,
            ignore quant_axis and use this to quantize. If learn_amax is True, will be used to
            initialize learnable amax.
        learn_amax: A boolean. If True, learn amax.
        scale_amax: A float. If supplied, multiply amax by scale_amax. Default None. It is useful
            for some quick experiment.
        calib_method: A string. One of ``["max", "histogram"]`` indicates which calibration to use.
            Except the simple max calibration, other methods are all histogram based.
        unsigned: A boolean. If True, use unsigned.
        narrow_range: A boolean. if True, symmetric integer range for signed quantization is used.

    Read-only properties:
        - fake_quant:
        - name:
        - learn_amax:
        - scale_amax:
        - axis:
        - calib_method:
        - num_bits:
        - amax:
        - unsigned:
    """

    def __init__(
        self,
        num_bits=8,
        name=None,
        fake_quant=True,
        axis=None,
        block_sizes=None,
        amax=None,
        learn_amax=False,
        scale_amax=None,
        calib_method="max",
        unsigned=False,
        narrow_range=False,
    ):
        """Initialize QuantDescriptor."""
        if isinstance(num_bits, int):
            if num_bits < 0:
                raise ValueError(f"num_bits must be > 0, not {num_bits}.")
        elif num_bits != (4, 3):
            raise TypeError(
                f"num_bits must be a postive integer or tuple (4,3), not {type(num_bits)}."
            )

        self._num_bits = num_bits
        if not isinstance(name, str) and name is not None:
            raise TypeError(f"name must be a string or None, not {type(name)}.")
        self._name = name

        self._fake_quant = fake_quant
        self._axis = axis
        self._block_sizes = None
        if block_sizes is not None:
            assert axis is None, "axis must be None when block_sizes is not None."
            assert isinstance(block_sizes, dict), "block_sizes must be a dictionary."
            assert all(
                isinstance(key, int) for key in block_sizes.keys()
            ), "Keys of block_sizes must be integers."
            assert all(
                isinstance(value, int) and value > 0 for value in block_sizes.values()
            ), "Values of block_sizes must be positive integers."
            self._block_sizes = block_sizes

        self._learn_amax = learn_amax
        if self._learn_amax and self._axis is not None:
            raise TypeError(
                f"axis is ignored and must be None when learn_amax is true, got {type(self._axis)}."
            )
        if amax is not None:
            if (
                not isinstance(amax, float)
                and not isinstance(amax, list)
                and not isinstance(amax, np.ndarray)
            ):
                raise TypeError(f"amax must be float, list or ndarray, not {type(amax)}")
            # Make it single precision array
            self._amax = np.array(amax, dtype=np.float32)
        else:
            self._amax = amax

        self._scale_amax = scale_amax
        self._calib_method = calib_method
        self._unsigned = unsigned
        self._narrow_range = narrow_range

    @property
    def num_bits(self):
        """Return num_bits."""
        return self._num_bits

    @property
    def fake_quant(self):
        """Return True if fake quantization is used."""
        return self._fake_quant

    @property
    def axis(self):
        """Return axis for quantization."""
        return self._axis

    @property
    def block_sizes(self):
        """Return block_sizes for quantization."""
        return self._block_sizes

    @property
    def amax(self):
        """Return amax."""
        return self._amax

    @property
    def learn_amax(self):
        """Return True if amax is learnable."""
        return self._learn_amax

    @property
    def scale_amax(self):
        """Return scale_amax."""
        return self._scale_amax

    @property
    def name(self):
        """Return name."""
        return self._name

    @property
    def calib_method(self):
        """Return calibration method."""
        return self._calib_method

    @property
    def unsigned(self):
        """Return True if unsigned integer range is used."""
        return self._unsigned

    @property
    def narrow_range(self):
        """Return True if symmetric integer range for signed quantization is used."""
        return self._narrow_range

    def __str__(self):
        s = (self._name + ": ") if self._name is not None else "QuantDescriptor"
        s += f"({'unsigned ' if self._unsigned else ''}{self._num_bits}bit"
        s += " fake" if self._fake_quant else " real"
        if self._block_sizes is not None:
            s += f" block_sizes={self._block_sizes}"
        else:
            s += f" axis={self._axis if self._axis is not None else ' per-tensor'}"
        if isinstance(self._amax, torch.Tensor):
            amax_str = np.array2string(
                self._amax.cpu().numpy().flatten(), edgeitems=1, formatter={"all": "{:.2e}".format}
            )
            s += f" amax={amax_str}"
        elif self._amax is not None:
            s += f" amax={self._amax}"
            s += " full_range"
        if self._learn_amax:
            s += " learn_amax"
        if self._scale_amax:
            s += f" scale_amax={self._scale_amax}"
        s += ")"
        return s

    def __eq__(self, rhs):
        """Compare 2 descriptors."""
        return self.__dict__ == rhs.__dict__

    def dict(self):
        """Serialize to dict.

        The build-in __dict__ method returns all the attributes, which includes those have default value and have
        protected prefix "_". This method only returns those have values other than the default one and don't have _ in
        key. Construct a instance by dict returned by this method should get exactly the same instance.
        """
        obj_dict = {}
        obj_dict["num_bits"] = self._num_bits
        obj_dict["name"] = self._name

        if not self._fake_quant:
            obj_dict["fake_quant"] = self._fake_quant
        if self._axis is not None:
            obj_dict["axis"] = self._axis
        if self._amax is not None:
            obj_dict["amax"] = self._amax.tolist()
        if self._scale_amax is not None:
            obj_dict["scale_amax"] = self._scale_amax
        if self._learn_amax:
            obj_dict["learn_amax"] = self._learn_amax
        if self._unsigned:
            obj_dict["unsigned"] = self._unsigned

        return obj_dict


QuantDescriptor = ScaledQuantDescriptor

# Predefined descriptors
QUANT_DESC_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))


@torch.jit.script
def _fake_tensor_quant_backward(inputs, amax, grad_outputs):
    zero = grad_outputs.new_zeros(1)
    grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
    return grad_inputs


def _onnx_int8_helper(g, inputs, amax, num_bits, unsigned, narrow_range):
    assert num_bits == 8, "Only INT8 ONNX export is supported for now."
    maxbound = (1 << (num_bits - 1 + int(unsigned))) - 1

    if amax.numel() == 1:
        zero_point, axis = torch.tensor(0.0, device=amax.device), None
    else:
        amax_init_shape = amax.shape
        amax = amax.squeeze().data
        assert len(amax.shape) == 1, "ONNX does not support multi-axis quantization."
        zero_point = torch.zeros_like(amax, dtype=torch.int32).data
        axis = list(amax_init_shape).index(list(amax.shape)[0])

    zero_point = g.op("Constant", value_t=zero_point)

    if not unsigned:
        assert not narrow_range, "ONNX does not support unsigned narrow range INT8."
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)

    amax = amax.to(torch.float32)
    scale = amax / maxbound
    scale.masked_fill_(scale == 0, 1.0)
    scale = g.op("Constant", value_t=scale)

    input_type = inputs.type().scalarType()

    # Q inputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the input if needed.
    if input_type == "Half" or input_type == "BFloat16":
        inputs = g.op("Cast", inputs, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
    out = g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)

    # DQ outputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the output if needed.
    if input_type == "Half":
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
    elif input_type == "BFloat16":
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)

    return out


class FakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction use CUDA extension."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "i", "b", "b")
    def symbolic(g, inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        """ONNX symbolic function."""
        return _onnx_int8_helper(g, inputs, amax, num_bits, unsigned, narrow_range)

    @staticmethod
    def forward(ctx, inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        """Forward method."""
        ctx.save_for_backward(inputs, amax)

        def legacy_quant_func():
            # The LegacyFakeTensorQuantFunction support cpu and amax with any shape that can be broadcasted to inputs.
            outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
            return outputs / scale.to(inputs.dtype)

        if not inputs.is_cuda:
            outputs = legacy_quant_func()
        else:
            try:
                from .extensions import cuda_ext

                with torch.cuda.device(
                    None
                    if inputs.device.index == torch.cuda.current_device()
                    else inputs.device.index
                ):
                    if amax.numel() == 1:
                        outputs = cuda_ext.fake_tensor_quant(  # type: ignore[union-attr]
                            inputs, amax, num_bits, unsigned, narrow_range
                        )
                    else:
                        axis = amax.shape.index(amax.numel())
                        outputs = cuda_ext.fake_tensor_quant_with_axis(  # type: ignore[union-attr]
                            inputs, amax.squeeze(), axis, num_bits, unsigned, narrow_range
                        )
            except (AttributeError, ValueError):
                # AttributeError: cuda_ext is not imported, possibly due to CPU only installation
                # ValueError: cuda_ext is installed, but trying to perform multidimensional quantization (amax dim > 1)
                outputs = legacy_quant_func()

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        inputs, amax = ctx.saved_tensors
        return _fake_tensor_quant_backward(inputs, amax, grad_outputs), None, None, None, None


def _onnx_fp8_quantize(g, inputs, scale_inv):
    """Helper Function for Quantization."""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    # Q inputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the input if needed.
    if inputs.type().scalarType() == "Half" or inputs.type().scalarType() == "BFloat16":
        inputs = g.op("Cast", inputs, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    scale = g.op("Constant", value_t=torch.tensor(scale_inv))
    q_op = g.op("trt::TRT_FP8QuantizeLinear", inputs, scale).setType(
        inputs.type().with_dtype(torch.uint8).with_sizes(output_shape)
    )
    return q_op


def _onnx_fp8_dequantize(g, inputs, scale_inv, otype=None):
    """Helper Function for Dequantization."""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    scale = g.op("Constant", value_t=torch.tensor(scale_inv))
    out = g.op("trt::TRT_FP8DequantizeLinear", inputs, scale).setType(
        inputs.type().with_dtype(torch.float32).with_sizes(output_shape)
    )

    # DQ outputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the output if needed.
    if otype == "Half":
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
    elif otype == "BFloat16":
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return out


class ScaledE4M3Function(Function):
    """E4M3fy input with scale."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "i", "b", "b")
    def symbolic(g, inputs, amax=None, E=4, M=3):  # noqa: N803
        """ONNX symbolic function."""
        if amax is None:
            scale = 1.0
        else:
            scale = 448.0 / float(amax)
        otype = inputs.type().scalarType()
        q_tensor = _onnx_fp8_quantize(g, inputs, 1.0 / scale)
        return _onnx_fp8_dequantize(g, q_tensor, 1.0 / scale, otype)

    @staticmethod
    def forward(ctx, inputs, amax=None, E=4, M=3):  # noqa: N803
        """Forward method."""
        if E != 4 or M != 3:
            raise NotImplementedError("Only support E=4 & M=3 for now.")

        ctx.save_for_backward(inputs)
        ctx.amax = amax
        zero_mask = inputs.abs() < 1.0 / (1 << 24)

        from .extensions import cuda_ext_fp8

        assert (
            cuda_ext_fp8 is not None
        ), "cuda_ext_fp8 could not be imported. E4M3 quantization requires CUDA and cuda_ext_fp8."

        with torch.cuda.device(
            None if inputs.device.index == torch.cuda.current_device() else inputs.device.index
        ):
            if amax is None:
                outputs = cuda_ext_fp8.fake_e4m3fy(inputs)
            else:
                scale = 448.0 / amax
                outputs = cuda_ext_fp8.fake_e4m3fy(inputs * scale) / scale

        # Zero out values that are tiny.
        # Tiny values could lead to tiny amax and then large scale which cause overflow/saturation
        # and won't go back to normal value after dividing by scale. The right behavior is to mark them
        # as zero which also get rid of inf/nan
        outputs[zero_mask] = 0.0

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        (inputs,) = ctx.saved_tensors
        amax = torch.tensor(
            ctx.amax if ctx.amax is not None else 448.0, dtype=torch.float32, device=inputs.device
        )
        grad_inputs = _fake_tensor_quant_backward(inputs, amax, grad_outputs)
        return grad_inputs, None, None, None


class TensorQuantFunction(Function):
    """A universal tensor quantization function.

    Take an input tensor, output an quantized tensor. The granularity of scale can be interpreted from the
    shape of amax.
    output_dtype indicates whether the quantized value will be stored in integer or float. The reason we want to store
    it in float is the pytorch function takes the quantized value may not accept integer input, e.g. Conv2D.

    It uses 2^num_bits -1 values instead of 2^num_bits. e.g., for num_bits=8, it uses [-127, 127] instead of [-128, 127]
    """

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "i", "b", "b")
    def symbolic(g, inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        """ONNX symbolic function."""
        return _onnx_int8_helper(g, inputs, amax, num_bits, unsigned, narrow_range)

    @staticmethod
    def forward(ctx, inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        """Forward method.

        Follow tensorflow convention, max value is passed in and used to decide scale, instead of inputing scale
        directly. Though inputing scale directly may be more natural to use.

        Args:
            ctx: A Context object to store tensors for backward.
            inputs: A Tensor of type float32.
            amax: A Tensor of type float32. Inputs will be quantized within range [-amax, amax]
                amax will be broadcasted to inputs tensor.
            num_bits: A integer used to calculate scaling factor, scale = (2^(num_bits-1) - 1) / max
                Effectively, it indicates how many integer bits is used to represent the value. Default 8.
            output_dtype: A type of Tensor. torch.int32 or torch.float32.
            unsigned: A boolean. Use unsigned integer range. E.g. [0, 255] for num_bits=8. Default False.
            narrow_range: A boolean. Use symmetric integer range for signed quantization
                E.g. [-127,127] instead of [-128,127] for num_bits=8. Default True.

        Returns:
            outputs: A Tensor of type output_dtype.
            scale: A Tensor of type float32. outputs / scale will dequantize outputs tensor.

        Raises:
            ValueError:
        """
        ctx.save_for_backward(inputs, amax)
        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        # Check if scale overflows FP16
        if outputs.dtype == torch.half and scale.max() > 65504:
            raise ValueError(f"scale is too large for FP16 with amax={amax}")
        return outputs, scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs, grad_scale):
        """Implements straight through estimation with clipping.

        For -amax <= input <= amax the gradient passes straight through, otherwise the gradient is zero.

        Args:
            ctx: A Context object with saved tensors from forward.
            grad_outputs: A tensor of gradient of outputs.
            grad_scale: A tensor of gradient of scale.

        Returns:
            grad_inputs: A tensor of gradient.
        """
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)  # create a zero tensor with the same type and device
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None


class LegacyFakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction.

    See comments of TensorQuantFunction, arguments are the same.
    """

    @staticmethod
    def forward(ctx, inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        """Forward method."""
        ctx.save_for_backward(inputs, amax)
        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        return outputs / scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation."""
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None


def _tensor_quant(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
    """Shared function body between TensorQuantFunction and FakeTensorQuantFunction."""
    # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if unsigned:
        if inputs.min() < 0.0:
            raise TypeError("Negative values encountered in unsigned quantization.")

    # Computation must be in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()

    min_amax = amax.min()
    if min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor((2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1
    scale = max_bound / amax

    epsilon = 1.0 / (1 << 24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = amax <= epsilon
        scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    if min_amax <= epsilon:
        scale[zero_amax_mask] = (
            1.0  # Return 1 makes more sense for values quantized to 0 with amax=0
        )

    if input_dtype == torch.half:
        outputs = outputs.half()

    return outputs, scale


class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization.

    gemmlowp style scale+shift quantization. See more details in
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md.

    We DO NOT recommend affine quantization on weights for performance reason. There might be value to affine quantize
    activation as it can be cancelled by bias and comes with no performance penalty. This functionality is only added
    for experimental purpose.
    """

    @staticmethod
    def forward(ctx, inputs, min_range, max_range, num_bits=8):
        """As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        ctx.save_for_backward(inputs, min_range, max_range)

        step_size = (max_range - min_range) / (2.0**num_bits - 1)

        min_bound = -(2.0 ** (num_bits - 1))
        max_bound = 2.0 ** (num_bits - 1) - 1

        quant_zero = torch.round(min_range / step_size) - min_bound
        quantized = torch.round(inputs / step_size) - quant_zero
        quantized = torch.clamp(quantized, min_bound, max_bound)

        outputs = (quantized + quant_zero) * step_size

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping.

        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs.

        Returns:
            grad_inputs: A tensor of gradient
        """
        inputs, min_range, max_range = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where((inputs <= max_range) * (inputs >= min_range), grad_outputs, zero)
        return grad_inputs, None, None, None


tensor_quant = TensorQuantFunction.apply
legacy_fake_tensor_quant = LegacyFakeTensorQuantFunction.apply
fake_tensor_quant = FakeTensorQuantFunction.apply
fake_affine_tensor_quant = FakeAffineTensorQuantFunction.apply
scaled_e4m3 = ScaledE4M3Function.apply
