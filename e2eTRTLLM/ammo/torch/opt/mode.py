# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Interface and utilities for optimization modes/algorithms.

A mode is a specific type or algorithms for model optimization, e.g., some type of algorithm for
pruning or quantization. It can also specify a single step within an optimization algorithm instead
of the whole algorithm. For example, a mode can prepare a model for pruning or export (i.e. fix the
optimal model configuration) after pruning.

Within Ammo, a ``mode`` constitutes the unit for model optimization. We can define arbitrary modes,
each mode gets recorded in the model's ammo state dict, and we can define workflows as a sequence of
modes.
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch.nn as nn

from ammo.torch.utils import val2list

from .searcher import BaseSearcher

ConfigDict = Dict[str, Any]  # config dict for one mode
MetadataDict = Dict[str, Any]  # metadata dict for one mode
ModeConfigList = List[Tuple[str, ConfigDict]]  # config list for multiple modes
ModeState = Dict[str, Union[ConfigDict, MetadataDict]]  # state dict for one mode

ModeEntrypoint = Callable[[nn.Module, ConfigDict, MetadataDict], Tuple[nn.Module, MetadataDict]]
ConvertReturnType = Tuple[nn.Module, MetadataDict]
ConvertEntrypoint = Callable[[nn.Module, ConfigDict], ConvertReturnType]
RestoreEntrypoint = Callable[[nn.Module, ConfigDict, MetadataDict], nn.Module]
UpdateEntrypoint = Callable[[nn.Module, MetadataDict], None]

__all__ = []


# TODO: add more unit tests to test prior_modes and next_modes


class _ModeDescriptor(ABC):
    """Abstract class to describe a mode."""

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: '{self.name}'>"

    def __hash__(self):
        return hash(self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the string name of the mode."""

    @property
    @abstractmethod
    def config(self) -> ConfigDict:
        """Specifies the default config for the mode (see source code for actual configs)."""

    @property
    def prior_modes(self) -> Optional[Set[str]]:
        """Modes that must immediately precede this mode.

        Certain modes only makes sense if they are preceded by certain other modes.

        An empty set indicates that _no_ mode can precede this mode. A None value indicates that
        there are no restrictions on the preceding mode.

        Returns:
            A set of mode names that must immediately precede this mode. Defaults to None.
        """
        return None

    @property
    def next_modes(self) -> Optional[Set[str]]:
        """Modes that must immediately follow this mode.

        Certain modes only makes sense if they are followed by certain other modes.

        An empty set indicates that _no_ mode can follow this mode. A None value indicates that
        there are no restrictions on the following mode.

        Returns:
            A set of mode names that must immediately follow this mode. Defaults to None.
        """
        return None

    @property
    def search_algorithm(self) -> Type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode (if any)."""
        raise RuntimeError("Search is not supported for this mode.")

    @property
    @abstractmethod
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model.

        The function signature of the convert entrypoint is described below:

        Args:
            model: Model to be restored.
            config: Config used for the model that was also used during convert.

        Returns:
            A tuple consisting of
                1.  the in-place modified model. If the modification failed, the entrypoint can
                    return None instead
                2.  The metatdata that can be used to call the restore entrypoint to instantly
                    *restore* the modified model from the provided initial state, see below's
                    description for the restore entrypoint to get more info about ``metadata``.

        Raises:
            :meth:`ApplyModeError<ammo.torch.opt._conversion.ApplyModeError>` to indicate that the
            conversion process failed. This error can be caught by user-facing APIs if they want to
            enable a fall-back behavior.
        """

    @property
    @abstractmethod
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model.

        The function signature of the restore entrypoint is described below:

        Args:
            model: Model to be restored.
            config: Config used for the model that was also used during convert.
            metadata: The metadata is used during restoration of the model architecture to instantly
                restore the modified model. The metadata is used on top of the config to ensure that
                the model can be instantly restored/modified from the provided state. This is
                helpful when the ``convert`` entrypoint contains non-deterministic operations whose
                outcome can be stored in the metadata to ensure that the model can be restored
                reliably. A few examples of potential non-deterministic operations are provided
                below:
                    * Latency measurements: if the conversion leverages latency measurements during
                      conversion the conversion process may become non-deterministic.
                    * Random operations: if the conversion leverages random operations during
                      conversion, we should store the samples or random seed.
                    * Module's train flag: the conversion process might be affected by the module's
                      train flag (e.g. tracing is indirectly affected by train flag since the
                      forward may be affected by the train flag). If so, we should store the train
                      flag in the metadata and set the model into the correct mode.

        Returns:
            The in-place modified and restored model. If the modification failed, the entrypoint can
                    return None instead

        Raises:
            :meth:`ApplyModeError<ammo.torch.opt._conversion.ApplyModeError>` to indicate that the
            conversion process failed. This error can be caught by user-facing APIs if they want to
            enable a fall-back behavior.
        """

    @property
    def update(self) -> UpdateEntrypoint:
        """The mode's (optionally provided) entrypoint for updating a model's metadata.

        This is useful when the metadata changes from time to time and needs updating.
        Note that the metadata dictionary should be updated IN-PLACE.

        The function signature of the update entrypoint is described below:

        Args:
            model: Model to be restored.
            metadata: The metadata as described above. It can be modified IN-PLACE.

        Returns:
            None.
        """
        return lambda model, metadata: None


ModeType = Union[_ModeDescriptor, str]
ModeLike = Union[ModeType, List[ModeType], ModeConfigList]


class _ModeRegistryCls:
    """A registry to keep track of available modes."""

    T = TypeVar("T", bound=_ModeDescriptor)

    # global list to keep track of all registries we initialize
    _all_registries: List["_ModeRegistryCls"] = []

    def __init__(self) -> None:
        """Initialize the registry with the lookup dictionaries."""
        self._name2descriptor: Dict[str, _ModeDescriptor] = {}
        self._all_registries.append(self)

    def register_mode(self, cls_descriptor: Type[T]) -> Type[T]:
        """Register a new mode with the given descriptor."""
        # initialize descriptor and get name
        descriptor = cls_descriptor()
        name = descriptor.name

        # check if we have a descriptor instance already and use that instance instead.
        if self.contained_in_any(name):
            descriptor = self.get_from_any(name)

        # check if mode_name/value is already taken
        if name in self._name2descriptor:
            raise ValueError(f"Mode {name} already registered: {self._name2descriptor}")

        # register mode
        self._name2descriptor[name] = descriptor
        return cls_descriptor

    def remove_mode(self, mode: ModeType) -> None:
        """Remove a mode from the registry."""
        # remove mode
        del self._name2descriptor[str(mode)]

    def get(self, mode: ModeType) -> Optional[_ModeDescriptor]:
        """Get the mode by value or throw an error."""
        return self._name2descriptor.get(str(mode))

    def __getitem__(self, mode: ModeType) -> _ModeDescriptor:
        """Get the mode by value or throw an error."""
        return self._name2descriptor[str(mode)]

    def __contains__(self, mode: ModeType) -> bool:
        """Check if mode is registered in this registry."""
        return str(mode) in self._name2descriptor

    def __del__(self) -> None:
        """Remove the registry from the global list."""
        self._all_registries.remove(self)

    @classmethod
    def contained_in_any(cls, mode: ModeType) -> bool:
        """Check if mode is registered in any registry."""
        for registry in cls._all_registries:
            if str(mode) in registry._name2descriptor:
                return True
        return False

    @classmethod
    def get_from_any(cls, mode: ModeType) -> _ModeDescriptor:
        """Get the mode by value from any registry or throw a KeyError.

        Adds a sanity check to ensure that the mode is not ambiguous, i.e., there is only one
        instance.
        """
        mode_ds = [registry[mode] for registry in cls._all_registries if mode in registry]
        if not mode_ds:
            raise KeyError(f"Mode {mode} not found in any registry.")
        assert all(mode_ds[0] == m_d for m_d in mode_ds), f"Mode {mode} is ambiguous."
        return mode_ds[0]


def get_mode_config(mode_like: ModeLike) -> ModeConfigList:
    """Standardize mode to ModeConfigDict and return."""
    return [((m, {}) if isinstance(m, str) else (m[0], m[1] or {})) for m in val2list(mode_like)]
