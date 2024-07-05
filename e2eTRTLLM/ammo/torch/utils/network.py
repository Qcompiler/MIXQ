# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions for PyTorch models."""
import inspect
import warnings
from collections import abc, deque
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from .tensor import torch_to

__all__ = [
    "ModelLike",
    "compare_dict",
    "get_model_attributes",
    "get_module_device",
    "get_same_padding",
    "init_model_from_model_like",
    "is_channels_last",
    "is_parallel",
    "make_divisible",
    "model_to",
    "param_num",
    "param_num_from_forward",
    "remove_bn",
    "set_submodule",
    "standardize_model_args",
    "standardize_model_like_tuple",
    "standardize_named_model_args",
    "unwrap_model",
    "zero_grad",
]

ModelLike = Union[nn.Module, Type[nn.Module], Tuple]


def is_parallel(model: nn.Module) -> bool:
    """Check if a PyTorch model is parallelized."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def get_module_device(module: nn.Module) -> torch.device:
    """Get the device of a PyTorch module."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        # For modules without parameters
        return torch.device("cpu")


def param_num(network: nn.Module, trainable_only: bool = False, unit=1e6) -> float:
    """Get the number of parameters of a PyTorch model.

    Args:
        network: The PyTorch model.
        trainable_only: Whether to only count trainable parameters. Default is False.
        unit: The unit to return the number of parameters in. Default is 1e6 (million).

    Returns:
        The number of parameters in the model in the given unit.
    """
    return (
        sum(
            p.numel() if not trainable_only or p.requires_grad else 0
            for mod in network.modules()
            for p in mod.parameters(recurse=False)
            if not isinstance(mod, _BatchNorm)
        )
        / unit
    )


# TODO: we could also use the same approach as in inference_flops to get the number of params,
# which might be more accurate. Another approach could be to run a backwards pass and use a hook
# on the tensor directly.
def param_num_from_forward(
    model: nn.Module,
    trainable_only: bool = False,
    args: Union[torch.Tensor, Tuple, None] = None,
    unit: float = 1e6,
):
    """Get the number of parameters of a PyTorch model from a forward pass.

    Args:
        network: The PyTorch model.
        trainable_only: Whether to only count trainable parameters. Default is False.
        unit: The unit to return the number of parameters in. Default is 1e6 (million).

    Returns:
        The number of parameters from the model's forward pass in the given unit.

    This can helpful for dynamic modules, where the state dict might contain extra parameters that
    is not actively used in the model, e.g., because of a DynamicModule that is deactivated for the
    forward pass. We circumvent this issue by just counting parameters of modules that appear in a
    forward pass.
    """
    params = {}

    def count_hook(m: nn.Module, *_):
        if m not in params:  # don't double-count parameters
            params[m] = sum(
                getattr(m, n).numel()  # use getattr to retrieve param since it might be dynamic
                for n, p in m.named_parameters(recurse=False)  # don't recurse!
                if not trainable_only or p.requires_grad
            )

    # add hook to count parameters to all modules except _BatchNorm
    hooks = [
        m.register_forward_hook(count_hook)
        for m in model.modules()
        if not isinstance(m, _BatchNorm)
    ]

    # run forward pass
    args = standardize_model_args(model, args, use_kwargs=True)
    args = torch_to(args, get_module_device(model))
    model(*args[:-1], **args[-1])

    # remove hooks
    for h in hooks:
        h.remove()

    # count parameters and return
    return sum(params.values()) / unit


def get_same_padding(kernel_size: Union[int, Tuple[int, int]]) -> Union[int, tuple]:
    """Get the same padding for a given kernel size."""
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, f"invalid kernel size: {kernel_size}"
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    else:
        assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
        assert kernel_size % 2 == 1, "kernel size should be odd number"
        return kernel_size // 2


def make_divisible(v: Union[int, float], divisor: Optional[int], min_val=None) -> Union[int, float]:
    """Function taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if divisor is None:
        return v

    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def is_channels_last(model: nn.Module):
    """Check if the model is using channels last memory format."""
    # Infer target_model's memory_format
    # from https://github.com/pytorch/tutorials/blob/444fbd16f2ddf9967baf8b06e83867a141b071c2/
    # intermediate_source/memory_format_tutorial.py#L283
    has_channels_last = any(
        p.is_contiguous(memory_format=torch.channels_last) and not p.is_contiguous()
        for p in model.parameters()
    )
    return has_channels_last


def model_to(model: nn.Module, target_model: nn.Module):
    """Convert model to the same device, dtype and memory layout as the target_model."""
    has_channels_last = is_channels_last(target_model)
    # return model with same device, dtype, memory_format as self
    return model.to(
        tensor=next(target_model.parameters()),
        memory_format=torch.channels_last if has_channels_last else torch.contiguous_format,
    )


def set_submodule(model: nn.Module, target: str, target_submodule: nn.Module):
    """The set function that complements nn.Module.get_submodule()."""
    assert target != "", "Cannot set root module"

    # Verify the original submodule exists
    model.get_submodule(target)
    parent_module = model.get_submodule(target.rpartition(".")[0])
    child_name = target.split(".")[-1]
    parent_module.add_module(child_name, target_submodule)


def remove_bn(model: nn.Module):
    """Remove all batch normalization layers in the network."""
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.weight = m.bias = None
            m.forward = lambda x: x


def _preprocess_args(args: Union[Any, Tuple]) -> Tuple:
    """Return args in standardized format as tuple with last entry as kwargs."""
    # Re: torch.onnx.utils._decide_input_format, which is used in torch.onnx.export:
    # Starting in pytorch >= 1.12 tuplification is done in the beginning instead of at the end
    # for the args. We want to be consistent with that behavior here.
    # Specifically, this affects the case of just passing in one dict as args:
    #   * In torch < 1.12, this will be treated as a single positional argument; see here:
    #     https://github.com/pytorch/pytorch/blob/bc2c6edaf163b1a1330e37a6e34caf8c553e4755/torch/onnx/utils.py#L336
    #   * In torch >= 1.12, this will be treated as variable keyword argument (**kwargs), see here:
    #     https://github.com/pytorch/pytorch/blob/91e754b268c1869df5b2836f15c73e6ec1e265f1/torch/onnx/utils.py#L774
    if torch.__version__ < "1.12" and isinstance(args, abc.Mapping):
        args = (args, {})

    # now we can safely tuplify the args and add kwargs if necessary
    if not isinstance(args, tuple):
        args = (args,)
    if not isinstance(args[-1], abc.Mapping):
        args = args + ({},)

    return args


def standardize_named_model_args(
    model_or_fw_or_sig: Union[nn.Module, Callable, inspect.Signature], args: Union[Any, Tuple]
) -> Tuple[Dict[str, Any], Set]:
    """Standardize model arguments according to torch.onnx.export and give them a name.

    Args:
        model_or_fw_or_sig: A nn.Module, its forward method, or its forward method's signature.
        args: A tuple of args/kwargs or torch.Tensor feed into the model's ``forward()`` method.

    Returns: A tuple (args_normalized, args_default_added) where
        args_normalized is a dictionary of ordered model args where the key represents a unique
            serialized string based on the the argument's name in the function signature and the
            value contains the actual argument,
        args_default_added is a set indicating whether the argument was retrieved from the default
            value in the function signature of the model's ``forward()`` method.

    .. note::

        See :meth:`standardize_model_args() <ammo.torch.utils.network.standardize_model_args>` for
        more info as well.
    """
    # pre-process args
    args = _preprocess_args(args)

    # extract parameters from model signature
    if isinstance(model_or_fw_or_sig, nn.Module):
        model_or_fw_or_sig = inspect.signature(model_or_fw_or_sig.forward)
    elif callable(model_or_fw_or_sig):
        model_or_fw_or_sig = inspect.signature(model_or_fw_or_sig)
    params = model_or_fw_or_sig.parameters

    # we now continue to process the parameters in the function signature and classify them according
    # to their kind (see https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind for
    # an overview of the different kinds of parameters in a function signature)

    # sanity-check: kw-only must have default value and cannot be provided by user
    kw_only = [
        n
        for n, p in params.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY and (p.default == p.empty or n in args[-1])
    ]
    if kw_only:
        raise AssertionError(f"Keyword-only args ({kw_only}) can only be used w/ default values.")

    # sanity-check: kwargs in signature are okay but cannot be used by user!
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs_unexpected = any(kw not in params for kw in args[-1])
    if has_kwargs and kwargs_unexpected:
        raise AssertionError("Variable kwargs (**kwargs) are not supported.")

    # sanity-check: no unexpected kwargs provided by user
    assert not kwargs_unexpected, "Cannot provide unexpected keyword args!"

    # now sort in args_dict and default values
    args_queue = deque(args[:-1])
    args_dict = args[-1]
    args_normalized = {}
    args_default_added = set()
    for pname, param in params.items():
        # we peel off all positional/keyword arguments and fill them accordingly
        if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]:
            if args_queue:
                args_normalized[pname] = args_queue.popleft()
            elif pname in args_dict:
                args_normalized[pname] = args_dict[pname]
            elif param.default != param.empty:
                args_normalized[pname] = param.default
                args_default_added.add(pname)
            else:
                # sanity check: any args without default value must be provided by the user
                raise AssertionError(f"Argument {pname} must be provided by the user.")
        # when we have a var-positional arg (*args) we fill in the rest of the args
        elif param.kind == param.VAR_POSITIONAL:
            idx = 0
            while args_queue:
                args_normalized[f"{pname}.{idx}"] = args_queue.popleft()
                idx += 1
            # we also do not need to process further since everything following a var-positional
            # argument is keyword-only, which we don't allow!
            break

    # sanity-check: no positional arguments left
    assert not args_queue, "Positional arguments left unprocessed; too many provided!"

    # return the args (without kw-only args and kwargs) and set to indicate which args were
    # retrieved from default value in the function signature
    return args_normalized, args_default_added


def standardize_model_args(
    model_or_fw_or_sig: Union[nn.Module, Callable, inspect.Signature],
    args: Union[Any, Tuple],
    use_kwargs=False,
) -> Tuple:
    """Standardize model arguments according to torch.onnx.export.

    Args:
        model_or_fw_or_sig: A nn.Module, its forward method, or its forward method's signature.
        args: Arguments of ``model.forward()``. The format of ``args`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
        use_kwargs: Affects the return value, see below. For ``use_kwargs==False``, the returned
            args are also compatible with ``torch.onnx.export``.

    Returns:
        Standardized model args that can be used in ``model.forward()`` in the same standardized
        way no matter how they were provided, see below for more info.

    * If ``use_kwargs == False``, the returned args can be used as

      .. code-block:: python

            args = standardize_model_args(model, args, use_kwargs=False)
            model(*args)

    * If ``use_kwargs == True``, the returned args can be used as

      .. code-block:: python

            args = standardize_model_args(model, args, use_kwargs=True)
            model.forward(*args[:-1], **args[-1])

    .. warning::

        If ``use_kwargs == False`` the model's ``forward()`` method **cannot** contain keyword-only
        arguments (e.g. ``forward(..., *, kw_only_args)``) without default values and you must not
        provide them in ``args``.

    .. warning::

        If ``use_kwargs == False`` you must not provide variable keyword arguments in ``args`` that
        are processed via variable keyword arguments in the model's ``forward()`` method
        (e.g. ``forward(..., **kwargs)``).

    """
    # preprocess args
    args = _preprocess_args(args)

    # simply return as args/kwargs in this case
    if use_kwargs:
        return args

    # return sorted args without names in this case
    return tuple(standardize_named_model_args(model_or_fw_or_sig, args)[0].values())


def get_model_attributes(model: nn.Module) -> Dict[str, Any]:
    """Get the key attributes of a PyTorch model."""
    attrs = {}
    attrs["type(model)"] = type(model).__name__
    attrs["model.forward"] = getattr(model.forward, "__name__", None)
    keys = ["training"]
    for key in keys:
        attrs[key] = getattr(model, key)
    return attrs


def compare_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Tuple[str, ...]:
    """Compare two dictionaries and return keys with unmatched values."""
    keys_unmatched = tuple(k for k in dict1.keys() & dict2.keys() if dict1[k] != dict2[k])
    keys_unmatched += tuple(dict1.keys() ^ dict2.keys())
    return keys_unmatched


def unwrap_model(
    model: nn.Module, warn: bool = False, raise_error: bool = False, msg: str = ""
) -> nn.Module:
    """Unwrap a model that is wrapped by supported wrapper module or return original model."""
    # NOTE: can be extended in the future for other frameworks and wrappers
    supported_wrappers = {
        nn.parallel.DataParallel: "module",  # indicating attribute key to unwrap
        nn.parallel.DistributedDataParallel: "module",
    }
    if isinstance(model, tuple(supported_wrappers)):
        if raise_error:
            raise ValueError(msg or f"Model {model} is wrapped by {type(model)}!")
        elif warn:
            warnings.warn(msg or f"Model {model} is wrapped by {type(model)}; unwrapping...")
        return getattr(model, supported_wrappers[type(model)])
    return model


def zero_grad(model: nn.Module) -> None:
    """Set any gradients in the model's parameters to None."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def standardize_model_like_tuple(model: ModelLike) -> Tuple[Type[nn.Module], Tuple, Dict]:
    """Standardize a model-like tuple."""
    if not isinstance(model, (type, tuple)):
        raise ValueError(f"Expected type or tuple but got {model}")

    if isinstance(model, type):
        model = (model,)

    if len(model) == 1:
        model = (*model, (), {})
    elif len(model) == 2:
        model = (*model, {})

    model_cls, args, kwargs = model
    assert isinstance(model_cls, type) and issubclass(
        model_cls, nn.Module
    ), f"Invalid model cls: {model_cls}"
    assert isinstance(args, (tuple, list)), f"Invalid model args: {args}"
    assert isinstance(kwargs, dict), f"Invalid model kwargs: {kwargs}"
    return model_cls, tuple(args), kwargs


def init_model_from_model_like(model: ModelLike) -> nn.Module:
    """Initialize a model from a model-like object.

    Args:
        model: A model-like object. Can be a nn.Module (returned as it is), a model class type, or a tuple.
            If a tuple, it must be of the form (model_cls,) or (model_cls, args) or (model_cls, args, kwargs).
            Model will be initialized as ``model_cls(*args, **kwargs)``.
    """
    if isinstance(model, nn.Module):
        return model

    model_cls, args, kwargs = standardize_model_like_tuple(model)
    return model_cls(*args, **kwargs)
