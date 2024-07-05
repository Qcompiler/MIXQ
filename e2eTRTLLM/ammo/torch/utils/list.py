# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils for operating on lists."""
from typing import Any, Dict, List, Tuple, Union

import numpy as np

__all__ = [
    "list_closest_to_median",  # used
    "val2list",  # used
    "val2tuple",  # used
    "stats",  # used
]


def list_closest_to_median(x: List) -> Any:
    """Return element from list that's closest to list mean."""
    median = np.median(x)
    diff = [abs(elem - median) for elem in x]
    return x[diff.index(min(diff))]


def val2list(val: Union[List, Tuple, Any], repeat_time=1) -> List:
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val for _ in range(repeat_time)]


def val2tuple(val: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    val = val2list(val)

    # repeat elements if necessary
    if len(val) > 0:
        val[idx_repeat:idx_repeat] = [val[idx_repeat] for _ in range(min_len - len(val))]

    return tuple(val)


def stats(vals: List[float]) -> Dict[str, float]:
    """Compute min, max, avg, std of vals."""
    stats = {"min": np.min, "max": np.max, "avg": np.mean, "std": np.std}
    return {name: fn(vals) for name, fn in stats.items()} if vals else {}
