# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions for computational graph."""
import itertools
from typing import Callable, Dict, Optional, Sequence, Union

import torch
from torch import nn
from torch.fx import Node, symbolic_trace

__all__ = ["match"]

NodeTarget = Union[nn.Module, Callable]


def _get_node_target(node: Node, root: nn.Module) -> Optional[NodeTarget]:
    """Return node target depending on node operator."""
    target_extractor = {
        "call_module": lambda t: root.get_submodule(t),
        "call_method": lambda t: getattr(torch.Tensor, t),
        "call_function": lambda t: t,
        "output": lambda _: None,
        "placeholder": lambda _: None,
        "get_attr": lambda _: None,
    }
    return target_extractor[node.op](node.target)


def _local_match(nx: Node, ny: Node, mx: nn.Module, my: nn.Module) -> bool:
    # Check if input and output degrees match.
    if len(nx.all_input_nodes) != len(ny.all_input_nodes):
        return False
    if len(nx.users) != len(ny.users):
        return False

    # Check if the node operator matches.
    if nx.op != ny.op:
        return False

    # Check if the node target matches.
    tx = _get_node_target(nx, mx)
    ty = _get_node_target(ny, my)
    if isinstance(tx, nn.Module) and isinstance(ty, nn.Module):
        tx, ty = type(tx), type(ty)
    if tx != ty:
        return False

    return True


def _recursive_match(
    nx: Node, ny: Node, mx: nn.Module, my: nn.Module, maps: Dict[Node, Node]
) -> bool:
    # Check if we have already matched these nodes in the current traversal.
    if nx in maps:
        return maps[nx] == ny

    # Terminate early if the nodes do not match locally.
    if not _local_match(nx, ny, mx, my):
        return False

    # Optimistically mark `nx` as a match for `ny`.
    maps[nx] = ny

    # Both nodes are inputs. We have a match!
    if nx.op == "placeholder":
        return True

    # Enumerate all possible input node matches.
    ixs = nx.all_input_nodes
    for iys in itertools.permutations(ny.all_input_nodes):
        if not all(_local_match(ix, iy, mx, my) for ix, iy in zip(ixs, iys)):
            continue
        if all(_recursive_match(ix, iy, mx, my, maps) for ix, iy in zip(ixs, iys)):
            return True

    # No match found.
    del maps[nx]
    return False


def match(module: nn.Module, patterns: Sequence[nn.Module]) -> bool:
    """Check if a module matches any of the patterns.

    Args:
        module: The module to be checked.
        patterns: The patterns to be matched.

    Returns:
        True if the module matches any of the patterns, False otherwise.
    """
    try:
        module_g = symbolic_trace(module).graph
    except Exception:
        return False

    for pattern in patterns:
        pattern_g = symbolic_trace(pattern).graph

        # Check if the number of nodes match.
        if len(module_g.nodes) != len(pattern_g.nodes):
            continue

        # Extract output nodes from the graphs.
        *_, pattern_o = pattern_g.nodes
        *_, module_o = module_g.nodes

        # Check whether two graphs match recursively from the output node.
        if _recursive_match(pattern_o, module_o, pattern, module, {}):
            return True
    return False
