# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module to handle model converting and restoring for optimization methods.

When applying a model optimization algorithm, we usually need to modify the model in each step
(mode) of the algorithm. This module provides the state manager, which is a standardized interface
(class) to record and store state information in the model.

Op top of the state manager, this module provides utilities to save a history of these modifications
("ammo state dict") and restoring a unmodified model to the state indicated in the state dict.
"""

import copy
import os
import warnings
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ammo import __version__
from ammo.torch.utils import ModelLike, init_model_from_model_like, unwrap_model

from .mode import (
    ConfigDict,
    MetadataDict,
    ModeLike,
    ModeState,
    ModeType,
    _ModeDescriptor,
    _ModeRegistryCls,
    get_mode_config,
)

__all__ = [
    "AmmoStateManager",
    "apply_mode",
    "ammo_state",
    "save",
    "restore_from_ammo_state",
    "restore",
]

AmmoStateList = List[Tuple[str, ModeState]]  # state data structure for multiple modes


class AmmoStateManager:
    """A class to handle the ammo state stored for each mode correspondig to an AMMO task/mode."""

    _state_key = "_ammo_state"

    def __init__(self, model: Optional[nn.Module] = None, init_state: bool = False) -> None:
        """Initialize state manager.

        Args:
            model: Module that has ammo_state stored. If None, a fake module is created to store
                any state that might be added with the manager.
            init_state: Whether to initialize the ammo state for the model if it does not exist.
        """
        # just assume fake module for easy implementation below
        if not model:
            model = nn.Module()
            init_state = True  # always initialize fake module

        # initialize ammo state if desired. Note that by default we don't do that to avoid
        # accidentally modifying a user-provided model.
        if init_state:
            assert not hasattr(model, self._state_key), "Model already has ammo state!"
            setattr(model, self._state_key, [])

        # sanity check that root module has ammo state now
        assert self.is_converted(model, is_root=True), "Model must have ammo state!"

        # store reference to state
        self._state: AmmoStateList = getattr(model, self._state_key)

    @property
    def has_state(self) -> bool:
        """Return whether the model has a non-trivial ammo state."""
        return bool(self._state)

    @classmethod
    def is_converted(cls, model: nn.Module, is_root: bool = False) -> bool:
        """Check if model is converted.

        Args:
            model: A model to be checked for state/metadata from the convert process.
            is_root: Additionally check whether the module with state is the root module.

        Returns:
            True if the model contains convert Ammo state indicating that it has been converted.

        This method raises an assertion when multiple ammo_states are detected or when is_root is
        set to True but the module with state is not the root module.
        """
        # check for submodules with state
        mods_with_state = [name for name, m in model.named_modules() if hasattr(m, cls._state_key)]
        # check if there is multiple submodules with state
        assert len(mods_with_state) <= 1, "Model has multiple ammo states!"
        is_converted = bool(mods_with_state)

        # check if mod with state is root module if desired
        if is_converted:
            assert not is_root or mods_with_state[0] == "", "Model has ammo state but not the root!"

        return is_converted

    # TODO: consider renaming state_dict???
    def state_dict(self) -> AmmoStateList:
        """Return the metadata of the model."""
        return self._state

    def load_state_dict(self, state_dict: AmmoStateList) -> None:
        """Load the provided ``state_dict`` to the ammo_state."""
        assert not self.has_state, "Cannot load state_dict if there is already one."

        # make sure we operate on deepcopy
        state_dict = copy.deepcopy(state_dict)
        # add modes one-by-one
        for m_str, m_state in state_dict:
            # adds config and metadata with sanity checks
            self.add_mode(m_str, m_state["config"], m_state["metadata"])

        # overwrite state manually afterwards to ensure exact consistency with provided state_dict
        self._state.clear()
        self._state.extend(state_dict)

    @classmethod
    def transfer_state_dict(cls, model_from: nn.Module, model_to: nn.Module) -> None:
        """Transfer the state (same instance) from one model to another."""
        manager_from = AmmoStateManager(model_from, init_state=False)  # state must exist
        manager_to = AmmoStateManager(model_to, init_state=True)  # state must NOT exist

        # transfer state_dict (this uses sanity checks + deepcopy)
        manager_to.load_state_dict(manager_from.state_dict())

        # manually set the state dict to be the exact same instance
        setattr(model_to, cls._state_key, manager_from.state_dict())
        manager_to = AmmoStateManager(model_to, init_state=False)  # state must exist now

        # remove state from model_from
        delattr(model_from, cls._state_key)

    def modes_with_states(self) -> Iterator[Tuple[_ModeDescriptor, ConfigDict, MetadataDict]]:
        """Yield the mode together with the full config and metadata from the state."""
        for m_str, m_state in self._state:
            yield _ModeRegistryCls.get_from_any(m_str), m_state["config_full"], m_state["metadata"]

    @property
    def last_mode(self) -> Optional[_ModeDescriptor]:
        """Return the last mode applied to the model (last stored mode)."""
        return _ModeRegistryCls.get_from_any(self._state[-1][0]) if self._state else None

    @property
    def last_metadata(self) -> MetadataDict:
        """Return the metadata of the last mode applied to the model (must exist!)."""
        return self._state[-1][1]["metadata"]

    def get_full_config(self, mode: ModeType, config: ConfigDict) -> ConfigDict:
        """Standardize the provided config and return partial+full config."""
        # standardize mode to descriptor
        mode_d = _ModeRegistryCls.get_from_any(mode)

        # validate config
        config_default = mode_d.config
        unexpected_keys = set(config.keys()) - set(config_default.keys())
        if unexpected_keys:
            raise KeyError(f"Unexpected keys in config: {unexpected_keys}")

        # process config as follows:
        # 1. use default config
        # 2. overwrite with config provided
        # 3. delete config keys whose values are set to None
        config_full = {**config_default, **config}

        return config_full

    def check_mode(self, mode: ModeType) -> None:
        """Check if the proposed mode is compatible with the current state."""
        # standardize mode to descriptor
        mode_d = _ModeRegistryCls.get_from_any(mode)

        # sanity checks for mode incompatibilities
        last_mode = self.last_mode
        assert mode_d.prior_modes is None or str(last_mode) in mode_d.prior_modes, (
            f"Cannot add {mode_d} after {last_mode}! Prior modes of {mode_d} are"
            f" {mode_d.prior_modes}."
        )
        if last_mode:
            assert last_mode.next_modes is None or str(mode_d) in last_mode.next_modes, (
                f"Cannot add {mode_d} after {last_mode}! Next modes of {last_mode} are"
                f" {last_mode.next_modes}."
            )

    def add_mode(self, mode: ModeType, config: ConfigDict, metadata: MetadataDict) -> None:
        """Add mode and update state in-place.

        Note that self._state is a list (preserves insertion order of keys) and we can therefore
        recall the order of modes!
        """
        # standardize mode to descriptor
        mode_d = _ModeRegistryCls.get_from_any(mode)

        # sanity checks for mode incompatibilities
        self.check_mode(mode_d)

        # store mode information
        m_state = {
            "config": config,
            "config_default": mode_d.config,
            "config_full": self.get_full_config(mode, config),
            "metadata": metadata,
        }
        self._state.append((str(mode_d), m_state))


class ApplyModeError(RuntimeError):
    """Error raised when applying a mode to a model fails."""


class ModelLikeModule(nn.Module):
    """Just a temp module type to store the initialization recipe for the actual model."""

    def __init__(self, modellike: ModelLike) -> None:
        super().__init__()
        assert not isinstance(modellike, nn.Module), "modellike should not be a nn.Module!"
        self.modellike = modellike

    def init_modellike(self) -> nn.Module:
        """Initialize the modellike to be an actual model."""
        model = init_model_from_model_like(self.modellike)
        AmmoStateManager.transfer_state_dict(self, model)
        return model


def apply_mode(
    model: ModelLike,
    mode: ModeLike,
    registry: Optional[_ModeRegistryCls] = None,
    init_state: Optional[bool] = None,
) -> nn.Module:
    """Apply the provided modes the model, record the changes, and return the model.

    Args:
        model: A model-like object. Can be an nn.Module, a model class type, or a tuple.
            Tuple must be of the form ``(model_cls,)`` or ``(model_cls, args)`` or
            ``(model_cls, args, kwargs)``. Model will be initialized as
            ``model_cls(*args, **kwargs)``.
        mode: A mode, a list of modes or a list of tuples containing the mode and its config. The
            mode may be specified as a string or as the actual
            :mod:`_ModeDescriptor<ammo.torch.opt.mode._ModeDescriptor>` class.
        registry: An optional mode registry from which to retrieve the mode. If not provided, all
            registries will be searched.
        init_state: Flag indicating whether we should initialize the state manager for the model. If
            not provided, it will be inferred from the model. This flag can be used to enforce a
            certain behavior. For example, for ``init_state=True`` the state manager will raise an
            error if the model already contains state.

    Returns:
        The converted model after applying the desired modes.
    """
    # initialize ModelLikeModule if needed.
    model = model if isinstance(model, nn.Module) else ModelLikeModule(model)

    # vanilla case
    if not mode:
        return model

    # check if the model is in a wrapper
    model = unwrap_model(model, raise_error=True)

    # standardize mode to ModeConfigDict
    mode_and_config = get_mode_config(mode)

    # get or initialize the state manager for the model
    manager = AmmoStateManager(
        model,
        init_state=not AmmoStateManager.is_converted(model) if init_state is None else init_state,
    )

    # get mode function based on registry argument
    get_mode = registry.__getitem__ if registry else _ModeRegistryCls.get_from_any

    # update metadata of currently last mode before adding new modes
    last_mode = manager.last_mode
    if last_mode is not None:
        last_mode.update(model, manager.last_metadata)

    # loop through modes and call convert entrypoint for each mode and record data in manager.
    for m, config in mode_and_config:
        manager.check_mode(m)
        model, metadata = get_mode(m).convert(model, manager.get_full_config(m, config))
        manager.add_mode(m, config, metadata)

    # it cannot be a ModelLikeModule anymore at the end
    assert not isinstance(model, ModelLikeModule), "Model must be a regular Module now!"

    # return model with state recorded
    return model


def get_mode(model: nn.Module) -> Optional[_ModeDescriptor]:
    """Get mode of converted network.

    model: A model that contains ammo_state

    The mode of the model is defined as the last mode activated during the convert process.
    """
    if AmmoStateManager.is_converted(model):
        return AmmoStateManager(model).last_mode
    return None


def ammo_state(model: nn.Module) -> Dict[str, Any]:
    """Return the AMMO state dict describing the modifications to the model.

    Note that the returned ``ammo_state`` does not contain the model parameters such as weights and biases.
    ``ammo_state`` is useful for saving and loading various ammo optimization states separately from the
    model parameters. For example:

    .. code-block::

        import ammo.torch.opt as ato

        # Save the ammo state and model weights separately
        torch.save(ato.ammo_state(model), "ammo_state.pt") # Save the ammo state
        torch.save(model.state_dict(), "model_weights.pt") # Save the model weights

    If you want to save the model weights and the ammo state together, please use
    :meth:`ato.save()<ammo.torch.opt.conversion.save>`.

    Args:
        model: the AMMO-modified model.

    Returns:
        An AMMO state dictionary describing the modifications to the model.
    """
    # unwrap model
    model = unwrap_model(model, warn=True)

    # retrieve state manager
    manager = AmmoStateManager(model=model if AmmoStateManager.is_converted(model) else None)

    # update metadata of current mode as needed
    last_mode = manager.last_mode
    if last_mode is not None:
        last_mode.update(model, manager.last_metadata)

    # construct state dict and return it
    objs = {
        "ammo_state_dict": manager.state_dict(),  # empty state_dict is okay (saving regular models)
        "ammo_version": __version__,
    }
    return objs


def save(model: nn.Module, f: Union[str, os.PathLike, BinaryIO], **kwargs) -> None:
    """Save a model's state dict together with the AMMO state dict to restore its architecture.

    Args:
        model: Any model.
        f: Target file location.
        **kwargs: additional args for ``torch.save()``.

    .. note::

        If model is a wrapper such as DistributedDataParallel, it will be unwrapped for saving.
    """
    # unwrap model
    model = unwrap_model(model, warn=True)

    # store ckpt
    ckpt_dict = {
        "ammo_state": ammo_state(model),
        "model_state_dict": model.state_dict(),
    }

    # store object
    torch.save(ckpt_dict, f, **kwargs)


def restore_from_ammo_state(model: ModelLike, ammo_state: Dict[str, Any]) -> nn.Module:
    """Restore the model architecture from the AMMO state dictionary based on the user-provided model.

    This method does not restore the model parameters such as weights and biases.
    Please load the weights and biases with the original checkpoint loading method after restoring
    ammo states with `restore_from_ammo_state`. For example:

    .. code-block:: python

        import ammo.torch.opt as ato

        model = ...  # Create the model-like object

        # Restore the previously saved ammo state followed by model weights
        ato.restore_from_ammo_state(model, torch.load("ammo_state.pt"))  # Restore ammo state
        model.load_state_dict(torch.load("model_weights.pt"), ...)  # Load the model weights

    If you want to restore the model weights and the ammo state together, please use
    :meth:`ato.restore()<ammo.torch.opt.conversion.restore>`.

    Args:
        model: A model-like object. Can be an nn.Module, a model class type, or a tuple.
            Tuple must be of the form ``(model_cls,)`` or ``(model_cls, args)`` or
            ``(model_cls, args, kwargs)``. Model will be initialized as
            ``model_cls(*args, **kwargs)``.
        ammo_state: The AMMO state dict describing the AMMO modifications to the model. The
            ``ammo_state`` can be generated via
            :meth:`ato.ammo_state()<ammo.torch.opt.conversion.ammo_state>`.

    Returns:
        A modified model architecture based on the restored modifications with the unmodified
        weights as stored in the provided ``model`` argument.

    .. note::

        Note that wrappers such as DistributedDataParallel are `not` supported during the restore
        process. Please wrap the model after the restore process.
    """
    # initialize ModelLikeModule if needed.
    model = model if isinstance(model, nn.Module) else ModelLikeModule(model)

    # Alert if the first 2 version numbers do not match, e.g., 0.3.2 vs 0.4.0.
    version = ammo_state["ammo_version"]
    if tuple(version.split(".")[:2]) != tuple(__version__.split(".")[:2]):
        warnings.warn(
            f"The checkpoint is stored with version {version}, but current version is"
            f" {__version__}. Compatibility of checkpoint with current version is not guaranteed!"
        )

    # initialize state manager and load state
    manager = AmmoStateManager(model=model, init_state=True)
    manager.load_state_dict(ammo_state["ammo_state_dict"])

    # apply restore entrypoints for each of the modes
    for m, config, metadata in manager.modes_with_states():
        model = m.restore(model, config, metadata)

    # it cannot be a ModelLikeModule anymore at the end
    assert not isinstance(model, ModelLikeModule), "Model must be a regular Module now!"

    return model


def restore(model: ModelLike, f: Union[str, os.PathLike, BinaryIO], **kwargs) -> nn.Module:
    """Load the checkpoint, restore the AMMO model modifications, and load the model's weights.

    Args:
        model: A model-like object. Can be an nn.Module, a model class type, or a tuple.
            Tuple must be of the form ``(model_cls,)`` or ``(model_cls, args)`` or ``(model_cls, args, kwargs)``.
            Model will be initialized as ``model_cls(*args, **kwargs)``.
        f: Target file location generated by :meth:`ato.save()<ammo.torch.opt.conversion.save>`.
        **kwargs: additional args for ``torch.load()``.

    Returns:
        The model with original weights and stored architecture.

    .. note::

        Note that wrappers such as DistributedDataParallel are `not` supported during the restore
        process. Please wrap the model after the restore process.
    """
    # initialize ModelLikeModule if needed.
    model = model if isinstance(model, nn.Module) else ModelLikeModule(model)

    # load checkpoint
    kwargs.setdefault("map_location", "cpu")
    objs = torch.load(f, **kwargs)

    # restore model architecture
    model_restored = restore_from_ammo_state(model, objs["ammo_state"])

    # load weights from checkpoint
    model_restored.load_state_dict(objs["model_state_dict"])

    # it cannot be a ModelLikeModule anymore at the end
    assert not isinstance(model_restored, ModelLikeModule), "Model must be a regular Module now!"

    return model_restored
