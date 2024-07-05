# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions for logging."""
import contextlib
import os
from contextlib import contextmanager
from inspect import signature

import tqdm

from . import distributed as dist

__all__ = ["num2hrb", "no_stdout", "print_rank_0", "DeprecatedError"]


def num2hrb(num: float, suffix="") -> str:
    """Convert big floating number to human readable string."""
    step = 1000  # step between units is 1000
    units = ["", "K", "M", "G", "T", "P", "E"]
    while abs(num) >= step and len(units) > 1:
        num /= step
        units.pop(0)
    return f"{num:3.2f}{units[0]}{suffix}"


@contextmanager
def _monkeypatched(obj, name, patch):
    """Temporarily monkeypatch."""
    old_attr = getattr(obj, name)
    setattr(obj, name, patch(old_attr))
    try:
        yield
    finally:
        setattr(obj, name, old_attr)


@contextmanager
def _disable_tqdm():
    """Context manager to disable tqdm.

    tqdm progress bar outputs to sys.stderr.
    Silencing sys.stderr to silence tqdm will also prevent error messages from streamed out.

    Therefore, monkey patching is used to silence tqdm module without silencing sys.stderr.
    """

    def _patch(old_init):
        def _new_init(self, *args, **kwargs):
            # get the index of disable from function signature
            index_disable = list(signature(old_init).parameters.keys()).index("disable") - 1

            if len(args) >= index_disable:
                # if arg "disable" is passed as a positional arg,
                # overwrite pos args with disable = False
                args = args[: index_disable - 1] + (True,) + args[index_disable:]
            else:
                kwargs["disable"] = True

            # initialize tqdm with updated args reflecting "disable" = True
            old_init(self, *args, **kwargs)

        return _new_init

    with _monkeypatched(tqdm.std.tqdm, "__init__", _patch):
        yield


@contextmanager
def no_stdout():
    """Silences stdout within the invoked context."""
    # Special disable for tqdm
    with contextlib.redirect_stdout(open(os.devnull, "w")), _disable_tqdm():
        yield


def print_rank_0(*args, **kwargs):
    """Prints only on the master process."""
    if dist.is_master():
        print(*args, **kwargs, flush=True)


class DeprecatedError(NotImplementedError):
    """Error for deprecated functions."""
