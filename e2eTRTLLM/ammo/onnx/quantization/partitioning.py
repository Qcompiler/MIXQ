# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities related to partitioning the ONNX model to place QDQ nodes."""
from typing import Dict, List, Set, Tuple

import onnx

from ammo.onnx.op_types import (
    is_copy_op,
    is_linear_op,
    is_pointwise_or_elementwise_op,
    is_pooling_or_window_op,
)
from ammo.onnx.quantization.graph_utils import (
    get_fusible_backbone,
    has_path_type,
)
from ammo.onnx.utils import (
    has_const_input,
    is_const_input,
)


def _build_fusible_partition(
    cur_node: onnx.onnx_ml_pb2.NodeProto,
    fusible_partition: List[str],
    partitioned_nodes: Set[str],
    non_residual_inputs: Dict[str, str],
    node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    graph_nodes: Dict[str, Tuple[int, onnx.onnx_ml_pb2.NodeProto]],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_consumers: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
) -> None:
    """Traverses the graph starting from cur_node and updates the fusible_partition list.

    Add a nodes to the partition if any of these holds:
    1. The node is a unary or binary pointwise operation and fusible by cask
    2. The node is BN or Relu and fusible with precedding Conv op
    3. The node is a residual Add and fusible with current partition

    Args:
        cur_node: Current candidate node for the partition.
        fusible_partition: Current fusible partition.
        partitioned_nodes: Set of already partitioned nodes.
        non_residual_inputs: Non-residual input map.
        node_parents: Nodes parent info map.
        graph_nodes: Nodes of the onnx model graph.
        tensor_producers: Tensor name vs producer node name map.
        tensor_consumers: Tensor name vs consumer node name map.

    Returns:
        Backbone node of the given pointwise op, None if not found.
    """

    def _is_on_non_residual_path(node: onnx.onnx_ml_pb2.NodeProto) -> bool:
        if (
            node.op_type == "Add"  # Input node should be an Add node
            # The Add node should have a non-residual input
            and non_residual_inputs[node.name]
            # Input from the current node is non-residual
            and cur_node.output[0] == non_residual_inputs[node.name]
        ):
            return True
        return False

    def _get_partition_node_outputs():
        # Collect tensor names produced by nodes in fusible_partition
        # TODO: cache sub-partition outputs and append after them
        partition_node_outputs = []
        for node_name in fusible_partition:
            partition_node = graph_nodes[node_name][1]
            for node_output in partition_node.output:
                partition_node_outputs.append(node_output)

        return partition_node_outputs

    def _is_cask_fusible(
        node: onnx.onnx_ml_pb2.NodeProto, partition_node_outputs: List[str]
    ) -> bool:
        for input_name in node.input:
            if input_name not in partition_node_outputs and not is_const_input(
                input_name, tensor_producers
            ):
                return False
        return True

    # Add current node to the partition
    fusible_partition.append(cur_node.name)
    partitioned_nodes.add(cur_node.name)

    # If on non-residual path, return after adding the node to the partition
    # TODO: can Myelin fuse pointwise ops followed by residual Add?
    if cur_node.op_type == "Add" and non_residual_inputs[cur_node.name]:
        return

    consumer_nodes = tensor_consumers[cur_node.output[0]]
    partition_node_outputs = _get_partition_node_outputs()

    # TODO: traverse consumer nodes in topologically sorted order
    for consumer in consumer_nodes:
        if consumer.name in partitioned_nodes:
            continue

        consumer_node = graph_nodes[consumer.name][1]
        if (
            (
                is_pointwise_or_elementwise_op(consumer_node.op_type)
                and _is_cask_fusible(consumer_node, partition_node_outputs)
            )
            or (
                consumer_node.op_type in ["BatchNormalization", "Relu"]
                and get_fusible_backbone(consumer_node, node_parents, tensor_producers)
            )
            or _is_on_non_residual_path(consumer_node)
        ):
            # DFS with the consumer and find more nodes for the partition
            _build_fusible_partition(
                consumer_node,
                fusible_partition,
                partitioned_nodes,
                non_residual_inputs,
                node_parents,
                graph_nodes,
                tensor_producers,
                tensor_consumers,
            )


def find_fusible_partitions(
    graph: onnx.onnx_ml_pb2.GraphProto,
    partitioned_nodes: Set[str],
    non_residual_inputs: Dict[str, str],
    node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    graph_nodes: Dict[str, Tuple[int, onnx.onnx_ml_pb2.NodeProto]],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_consumers: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Traverses the graph and collects all cask/kgen fusible partitions.

    Args:
        graph: Onnx model graph.
        partitioned_nodes: Set of already partitioned nodes.
        non_residual_inputs: Non-residual input map.
        node_parents: Nodes parent info map.
        graph_nodes: Nodes of the onnx model graph.
        tensor_producers: Tensor name vs producer node name map.
        tensor_consumers: Tensor name vs consumer node name map.

    Returns:
        List of partitions that are fusible by CASK with Conv/MatMul backbone.
        List of KGEN partitions with pointwise ops only.
    """

    def _partition_helper(fusible_root_type_checker):
        all_fusible_partitions = []  # Collects all individual partitions
        for node in graph.node:
            # Check if the node is already in some other partition
            if node.name in partitioned_nodes:
                continue

            # Start a partition with a linear op
            if not fusible_root_type_checker(node.op_type):
                continue

            # Try building a partition starting with this current linear op
            fusible_partition = []
            _build_fusible_partition(
                node,
                fusible_partition,
                partitioned_nodes,
                non_residual_inputs,
                node_parents,
                graph_nodes,
                tensor_producers,
                tensor_consumers,
            )

            # Gather the non-empty partitions
            assert fusible_partition
            partitioned_nodes.update(fusible_partition)
            all_fusible_partitions.append(fusible_partition)

        return all_fusible_partitions

    cask_fusible_partitions = _partition_helper(is_linear_op)
    kgen_partitions = _partition_helper(is_pointwise_or_elementwise_op)

    return cask_fusible_partitions, kgen_partitions


def get_skiped_output_layers(graph: onnx.onnx_ml_pb2.GraphProto) -> List[str]:
    """Returns the name of the non-quantizable output layers."""
    # TODO: see if input producer is already quantized or not
    # TODO: filter out input layers if consumer is not quantized already
    output_layers = []
    graph_output_names = [output_node.name for output_node in graph.output]

    for node in graph.node:
        for node_output_name in node.output:
            if node_output_name in graph_output_names:
                if node.op_type not in ["Conv", "Gemm", "MatMul"]:
                    output_layers.append(node.name)

    return output_layers


def find_quantizable_ops(
    graph: onnx.onnx_ml_pb2.GraphProto,
    nodes_to_quantize: List[str],
    partitioned_nodes: Set[str],
    quantizable_op_types: List[str],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    node_children: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
) -> List[str]:
    """Return the graph ops which are quantizable but not partitioned yet."""

    # TODO: Check if any BatchNormalization is un-partitioned
    # Note. Maxpool quantization has +/-
    # Note. Prologue fusion is not handled, so some pointwise ops might be unnecessarily quantized
    def _has_quantizable_consumer(node_name: str, quantizable_node_set: Set[str]) -> bool:
        children = node_children[node_name]
        for child_node in children:
            if (child_node.name in quantizable_node_set) or (
                is_copy_op(child_node.op_type)
                and _has_quantizable_consumer(child_node.name, quantizable_node_set)
            ):
                return True

        return False

    quantizable_ops = []
    pooling_and_window_ops = []
    for node in graph.node:
        if node.name in partitioned_nodes or node.op_type not in quantizable_op_types:
            continue

        # Collect pooling and window ops for second pass
        # as they need to check their neighbor's quantization status
        if is_pooling_or_window_op(node.op_type):
            pooling_and_window_ops.append(node.name)
            continue

        if is_pointwise_or_elementwise_op(node.op_type) and has_const_input(node, tensor_producers):
            continue

        quantizable_ops.append(node.name)

    quantizable_node_set = set(nodes_to_quantize + quantizable_ops)
    for node_name in pooling_and_window_ops:
        # TODO: Add or _has_quantizable_producer, ex. inception-v1-12.onnx
        if _has_quantizable_consumer(node_name, quantizable_node_set):
            quantizable_ops.append(node_name)

    return quantizable_ops


def find_hardcoded_patterns(
    graph: onnx.onnx_ml_pb2.GraphProto,
    node_children: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
) -> List[List[str]]:
    """Finds some non-quantizable pre-defined patterns!.

    Note. matching this tail pattern causes MTL_v1 -5.5%
    ["ReduceSum", "Add", "Div", "Mul", "ReduceSum", "Sub", "Pow", "Mul", "ReduceSum", "Sqrt"]
    """
    p1 = ["MatMul", "Transpose", "BatchNormalization", "Transpose", "Relu", "ReduceMax"]

    matched_node_names = []
    for node in graph.node:
        for path_type in [p1]:
            path_node_names = []
            if has_path_type(
                node,
                node_children,
                tensor_producers,
                path_type,
                wild_card_types=[],
                path_node_names=path_node_names,
            ):
                matched_node_names.extend(path_node_names)

    return [matched_node_names]


def find_layer_norm_partitions(
    graph: onnx.onnx_ml_pb2.GraphProto,
    node_children: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
) -> List[List[str]]:
    """Finds the layer norm patterns in the graph."""
    # The most common LayerNorm implementation looks like this:
    # t -> ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add -> output
    #   \________________/  \______________________________________/
    # For simplicity, we do not match the connection between Sub and Div nodes.
    layer_norm_chain_types = [
        "ReduceMean",
        "Sub",
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
        "Mul",
        "Add",
    ]
    wild_card_types = ["Cast"]
    layer_norm_partitions = []

    for node in graph.node:
        layer_norm_partition = []
        if node.op_type == "ReduceMean" and has_path_type(
            node,
            node_children,
            tensor_producers,
            layer_norm_chain_types,
            wild_card_types,
            layer_norm_partition,
        ):
            layer_norm_partitions.append(layer_norm_partition)

    return layer_norm_partitions


def find_mha_partitions(
    graph: onnx.onnx_ml_pb2.GraphProto,
    node_children: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
) -> List[List[str]]:
    """Finds the MHA patterns in the graph.

    The most common MHA inplementation looks like this:
    t -> MatMul -> "Add" -> Mul (Optional) -> Sub (Optional) -> Softmax -> MatMul -> output
    Note, we do not match the optional nodes yet.
    """
    mha_chain_types = [
        [
            "MatMul",
            "Add",
            "Softmax",
            "MatMul",
        ],
        [
            "MatMul",
            "Add",
            "Reshape",
            "Add",
            "Reshape",
            "Softmax",
            "MatMul",
        ],
    ]
    mha_partitions = []

    for node in graph.node:
        if node.op_type == "MatMul":
            for chain_type in mha_chain_types:
                mha_partition = []
                if has_path_type(
                    node, node_children, tensor_producers, chain_type, [], mha_partition
                ):
                    mha_partitions.append(mha_partition)

    return mha_partitions


def find_partitions_from_patterns(
    graph: onnx.onnx_ml_pb2.GraphProto,
    node_children: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
) -> List[List[str]]:
    """Finds fusible partition from fixed patterns.

    Certain fused kernel counterpart is often a subgraph of native ops in onnx.
    Those patterns are identified here and quantized to match compiler expectation.
    """
    hard_coded_partitions = find_hardcoded_patterns(graph, node_children, tensor_producers)
    layer_norm_partitions = find_layer_norm_partitions(graph, node_children, tensor_producers)
    # TODO: QDQ pattern for mha versions
    # mha_partitions = find_mha_partitions(graph, node_children, tensor_producers)

    return hard_coded_partitions + layer_norm_partitions
