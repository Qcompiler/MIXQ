# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions for loading CPP / CUDA extensions."""

import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Union

import torch
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from torch.utils.cpp_extension import load

__all__ = ["load_cpp_extension"]


def load_cpp_extension(
    name: str,
    sources: List[Union[str, Path]],
    cuda_version_specifiers: Optional[str],
    **load_kwargs: Any,
) -> Optional[ModuleType]:
    """Load a C++ / CUDA extension using torch.utils.cpp_extension.load() if the current CUDA version satisfies it.

    Loading first time may take a few mins because of the compilation, but subsequent loads are instantaneous.

    Args:
        name: Name of the extension.
        sources: Source files to compile.
        cuda_version_specifiers: Specifier (e.g. ">=11.8,<12") for CUDA versions required to enable the extension.
        **load_kwargs: Keyword arguments to torch.utils.cpp_extension.load().
    """
    if torch.version.cuda is None:
        return None

    if cuda_version_specifiers and Version(torch.version.cuda) not in SpecifierSet(
        cuda_version_specifiers
    ):
        return None

    try:
        print(f"Loading extension {name}...")
        return load(name, sources, **load_kwargs)
    except Exception as e:
        # RuntimeError can be raised if there are any errors while compiling the extension.
        # OSError can be raised if CUDA_HOME path is not set correctly.
        # subprocess.CalledProcessError can be raised on `-runtime` images where c++ is not installed.
        warnings.warn(f"{e}\nUnable to load extension {name} and falling back to CPU version.")
        return None
