# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Provides ONNX graph related utils for QDQ placement."""
import logging
from typing import Dict, List, Optional, Tuple

import onnx

from ammo.onnx.op_types import is_copy_op, is_linear_op
from ammo.onnx.utils import (
    find_lowest_common_ancestor,
    has_const_input,
    is_const_input,
)


def has_path_type(
    node: onnx.onnx_ml_pb2.NodeProto,
    next_level_nodes: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    path_type: List[str],
    wild_card_types: List[str] = [],
    path_node_names: List[str] = [],
) -> bool:
    """Checks if the given node is start of a given path type.

    Note, Path can be forward or backward wrt a node depending on the next level nodes.
    Additionally, this method can work with optional nodes and collect the traversed path.

    Args:
        node: Start node of the path.
        next_level_nodes: Adjacency list, either parent or child info map.
        tensor_producers: Tensor name vs producer node name map.
        path_type: Path types to match from the given node.
        wild_card_types: Wild card types, these type of nodes are skipped and not matched with the path_type.
        path_node_names: Accumulated node names in the searched path.

    Returns:
        Bool, whether the given node is start of the given path type.
    """
    optional_path_types = ["BiasAdd", "ConstMul"]
    if not path_type:
        # All types matched
        return True

    # Current node type and special type conversion for optional BiasAdd and ConstMul
    # Note, matching path with Add/Mul type nodes with const input will fail
    node_type = node.op_type
    if node_type == "Add" and has_const_input(node, tensor_producers):
        node_type = "BiasAdd"
    elif node_type == "Mul" and has_const_input(node, tensor_producers):
        node_type = "ConstMul"

    # Check if current non-wild node type does not matche the expected path type
    # And if path type is not optional (ex. BiasAdd)
    is_match = (node_type == path_type[0]) or (node.op_type == path_type[0])
    is_wild_match = node_type in wild_card_types
    if not is_match and not is_wild_match and (path_type[0] not in optional_path_types):
        return False

    # Add current node name in the path
    if is_match:
        path_node_names.append(node.name)

    # If current node type matches the expected path type or path type is optional (ex. BiasAdd), we have a type match
    # Update the remaining path types to match
    next_path_type = path_type[:]

    # Non-repeatable optional types should be consumed
    if is_match or (path_type[0] in ["BiasAdd", "ConstMul"]):
        next_path_type = path_type[1:]

    # If current node is not wild card and didn't match, go ahead and match with the
    # remaining path types starting with the current node
    if not is_match and not is_wild_match:
        assert path_type[0] in optional_path_types
        return has_path_type(
            node,
            next_level_nodes,
            tensor_producers,
            next_path_type,
            wild_card_types,
            path_node_names,
        )

    # Check if any parent can match the remaining path types
    for next_node in next_level_nodes[node.name]:
        sub_path = []
        if has_path_type(
            next_node,
            next_level_nodes,
            tensor_producers,
            next_path_type,
            wild_card_types,
            sub_path,
        ):
            path_node_names.extend(sub_path)
            return True

    # Path type matches if there is no remaining types to match
    return not next_path_type


def get_fusible_backbone(
    node: onnx.onnx_ml_pb2.NodeProto,
    node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
) -> Optional[onnx.onnx_ml_pb2.NodeProto]:
    """Returns whether the current node is on fusible Conv group path.

    Note. BiasAdd and ConstMul are optional in path types.

    Args:
        node: Start node of the path.
        node_parents: Nodes parent info map.
        tensor_producers: Tensor name vs producer node name map.

    Returns:
        Backbone node of the given pointwise op, None if not found.
    """

    def _get_backbone(root: onnx.onnx_ml_pb2.NodeProto):
        if root.op_type == "Conv":
            return root

        for parent_node in node_parents[root.name]:
            bb = _get_backbone(parent_node)
            if bb:
                return bb

    fusible_linear_path_types = [
        # ["Sigmoid", "Conv"],  # With following Mul
        # ["Resize", "Relu", "Conv"],   # Note. this causes regression in MTL_v1
        ["BiasAdd", "ConstMul", "Conv"],
        ["Relu", "BiasAdd", "ConstMul", "Conv"],
        ["BatchNormalization", "BiasAdd", "Conv"],
        ["Relu", "BatchNormalization", "BiasAdd", "Conv"],
    ]
    for idx, path_type in enumerate(fusible_linear_path_types):
        if has_path_type(node, node_parents, tensor_producers, path_type, wild_card_types=[]):
            return _get_backbone(node)

    return None


def filter_quantizable_kgen_heads(
    cask_fusible_partitions: List[List[str]],
    kgen_partitions: List[List[str]],
    graph_nodes: Dict[str, Tuple[int, onnx.onnx_ml_pb2.NodeProto]],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_consumers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    quantizable_op_types: List[str],
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Returns the list of kgen head names if it follows a CASK partition."""
    cask_partition_nodes = set()
    for partition in cask_fusible_partitions:
        cask_partition_nodes.update(partition)

    cask_partition_heads = [partition[0] for partition in cask_fusible_partitions]

    def _is_followed_by_cask_partition(node: onnx.onnx_ml_pb2.NodeProto):
        # Checking if cask fusible partition can be reached backward
        # ignoring the copy ops
        if node.name in cask_partition_nodes:
            return True

        if not is_copy_op(node.op_type):
            return False

        for parent in node_parents[node.name]:
            if _is_followed_by_cask_partition(parent):
                return True

        return False

    def _has_other_quantizable_consumer(
        tensor_name: str, quantizable_kgen_heads: List[str], head_name: str
    ):
        # Note. this is kinda approximate analysis,
        # all quantizable kgen heads may haven't got discovered yet
        quantizable_ops = cask_partition_heads + quantizable_kgen_heads

        # Look for other quantizable consumer than the currnet kgen head
        if head_name in quantizable_ops:
            quantizable_ops.remove(head_name)

        consumers = tensor_consumers[tensor_name]
        for consumer in consumers:
            if consumer.name in quantizable_ops:
                return True

        return False

    def _has_non_quantizable_consumer(tensor_name: str):
        consumers = tensor_consumers[tensor_name]
        for consumer in consumers:
            # Ex. coatnet_0.onnx to avoid mul quantization in BN -> sigmoid -> mul pattern
            # Ideally, this should be fixed by _is_cask_fusible logic which is not complete yet
            if consumer.op_type in ["Sigmoid"]:
                return True
        return False

    quantizable_kgen_heads = []
    no_quantize_inputs = []  # list of tuple [(src_node_name, dst_node_name, input_name), ...]
    output_quantization_candidates = [
        "AveragePool",
        "BatchNormalization",
        "GlobalAveragePool",
        "MaxPool",
    ]

    for partition in kgen_partitions:
        head_node = graph_nodes[partition[0]][1]
        # Check if partition head is of default quantizable type
        if head_node.op_type not in quantizable_op_types:
            continue

        # If the node has cost input, do not quantize
        if has_const_input(head_node, tensor_producers):
            continue

        head_parents = node_parents[partition[0]]
        no_quantize_inputs_of_head = []
        has_quantizable_input = False

        # Check each of the parent (input producer for partition head)
        # or predecessor nodes and see if output quantization is needed for them
        # and decide which input of kgen head needs quantization
        for parent in head_parents:
            if _has_non_quantizable_consumer(parent.output[0]):
                continue

            # If the head is followed by quantizable ops/groups
            if (
                _is_followed_by_cask_partition(parent)
                or parent.op_type in output_quantization_candidates
            ):
                quantizable_kgen_heads.append(partition[0])
                has_quantizable_input = True
            # If has no quantizable consumer of the input from parent
            elif parent.name != "root_0" and not _has_other_quantizable_consumer(
                parent.output[0], quantizable_kgen_heads, head_node.name
            ):
                no_quantize_inputs_of_head.append((parent.name, partition[0], parent.output[0]))

        # If at least one input is quantizable, collect if there is any non-quantizable inputs
        if has_quantizable_input:
            no_quantize_inputs.extend(no_quantize_inputs_of_head)

    return quantizable_kgen_heads, no_quantize_inputs


def classify_partition_nodes(
    graph_nodes: Dict[str, Tuple[int, onnx.onnx_ml_pb2.NodeProto]],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    partitions: List[List[str]],
) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
    """We should partially quantize the partition nodes with inputs outside of the partition.

    Args:
        graph_nodes: Nodes of the onnx model graph.
        tensor_producers: Tensor name vs producer node name map.
        partitions: Partitions created by ammo ptq algo.

    Returns:
        List of non-quantizable node names.
        List of quantizable node names.
        List of partially-quantizable inputs with non-quantizable input info as (src, dst, input_name)
    """
    non_quantizable_partition_nodes = []  # list of str [node_name, ...]
    quantizable_partition_nodes = []  # list of str [node_name, ...]
    no_quantize_inputs = []  # list of tuple [(src_node_name, dst_node_name, input_name), ...]

    for partition in partitions:
        partition_root_type = graph_nodes[partition[0]][1].op_type
        assert is_linear_op(partition_root_type)

        # Collect tensor names produced by partition nodes
        partition_node_outputs = []
        for node_name in partition:
            node = graph_nodes[node_name][1]
            for node_output in node.output:
                partition_node_outputs.append(node_output)

        for node_name in partition:
            node = graph_nodes[node_name][1]
            has_external_inputs = False
            internal_inputs = []  # Keeps (node_name, input_name)
            for input_name in node.input:
                if (
                    # If a KGEN op has external non-constant input, it is considered partially quantizable
                    input_name not in partition_node_outputs
                    and not is_const_input(input_name, tensor_producers)
                ):
                    # partition heads will be fully quantizable and added
                    has_external_inputs = True
                else:
                    producer_node = tensor_producers.get(input_name, None)
                    if producer_node:
                        # format: source, target, input
                        # Note. it might happen that this node was not quantized
                        # We just ignore it from no_quantize_inputs list in post-processing
                        internal_inputs.append((producer_node.name, node_name, input_name))

            if not has_external_inputs:
                non_quantizable_partition_nodes.append(node_name)
            elif has_external_inputs and internal_inputs:
                no_quantize_inputs.extend(internal_inputs)
            else:
                # partition head is quantizable
                quantizable_partition_nodes.append(node_name)

    return non_quantizable_partition_nodes, quantizable_partition_nodes, no_quantize_inputs


def build_non_residual_input_map(
    graph: onnx.onnx_ml_pb2.GraphProto,
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
) -> Dict[str, str]:
    """Builds a map of non-redidual Add input name to the Add node name from the given graph.

    We will refer to a subgraph which has a Convolution node whose output is summed (element-wise)
    with another non-constant input-tensor as a “residual-add” subgraph, because it occurs in modern
    convnets that use residual connections.

    Args:
        graph: Onnx model graph.
        tensor_producers: Tensor name vs producer node name map.
        node_parents: Nodes parent info map.

    Returns:
        Dictionary of Add node names vs their non-residual input name.
    """
    non_residual_inputs = {}
    for node in graph.node:
        if node.op_type in ["Add"]:
            input1_producer = tensor_producers.get(node.input[0], None)
            input2_producer = tensor_producers.get(node.input[1], None)

            # Add nodes with constant input does not have non-residual input
            if not input1_producer or not input2_producer:
                non_residual_inputs[node.name] = None
                continue

            backbone1 = get_fusible_backbone(input1_producer, node_parents, tensor_producers)
            backbone2 = get_fusible_backbone(input2_producer, node_parents, tensor_producers)

            if backbone1 == backbone2:
                non_residual_inputs[node.name] = None
                continue

            # Generally if both the inputs have a backbone then both backbones are of the same type
            if backbone1 and backbone2:
                assert backbone1.op_type == backbone2.op_type, (
                    f"{backbone1.name} and {backbone2.name} are different types of backbone for"
                    f" {node.name}!"
                )
                # Input in the longest path to LCA is the non-residual input
                _, d1, d2 = find_lowest_common_ancestor(
                    input1_producer.name, input2_producer.name, node_parents
                )
                if d1 >= d2:
                    non_residual_inputs[node.name] = node.input[0]
                else:
                    non_residual_inputs[node.name] = node.input[1]
            elif backbone1:
                non_residual_inputs[node.name] = node.input[0]
            elif backbone2:
                non_residual_inputs[node.name] = node.input[1]
            else:
                # Not a residual Add node
                non_residual_inputs[node.name] = None

    return non_residual_inputs


def remove_partial_input_qdq(
    graph: onnx.onnx_ml_pb2.GraphProto,
    graph_nodes: Dict[str, Tuple[int, onnx.onnx_ml_pb2.NodeProto]],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_consumers: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    no_quantize_inputs: List[Tuple[str, str, str]],
) -> None:
    """Modifies the onnx model by removing QDQ nodes from the marked inputs, ex. non-residual inputs etc.

    Args:
        graph: Onnx model graph.
        graph_nodes: Nodes of the onnx model graph.
        tensor_producers: Tensor name vs producer node name map.
        tensor_consumers: Tensor name vs consumer node name map.
        no_quantize_inputs: List non-quantizable input info as (src, dst, input_name)

    Returns:
        None
    """
    logging.info("Deleting QDQ nodes from marked inputs to make certain operations fusible ...")
    dangling_qdq_indices = []
    for source, target, non_qdq_input_name in no_quantize_inputs:
        source_idx = graph_nodes[source][0]
        target_idx = graph_nodes[target][0]

        assert source_idx is not None
        assert target_idx is not None

        for input_idx, input_name in enumerate(graph.node[target_idx].input):
            # Quantized inputs are sometimes copied with names like '*_DequantizeLinear_Output_1',
            # '*_DequantizeLinear_Output_2' etc., so we match substring to remove them
            if non_qdq_input_name + "_DequantizeLinear_Output" in input_name:
                # This input must be present in consumer map and we need to remove from it
                assert graph_nodes[target][1] in tensor_consumers[input_name]

                # Replace the quantized input with non-quantized input name for all consumer of this input
                for consumer_node in tensor_consumers[input_name]:
                    consumer_node_idx = graph_nodes[consumer_node.name][0]
                    qdq_removed = False
                    for consumer_input_idx, consumer_input_name in enumerate(
                        graph.node[consumer_node_idx].input
                    ):
                        if non_qdq_input_name + "_DequantizeLinear_Output" in consumer_input_name:
                            graph.node[consumer_node_idx].input[
                                consumer_input_idx
                            ] = non_qdq_input_name
                            qdq_removed = True
                            break

                    # For each consumer qdq node should be removed
                    assert qdq_removed

                # The quantized input is not consumed by other node(s), it can be deleted
                dq_node = tensor_producers[input_name]
                q_node = tensor_producers[
                    input_name.replace("_DequantizeLinear", "_QuantizeLinear")
                ]
                assert q_node
                assert dq_node
                dangling_qdq_indices.extend(
                    [graph_nodes[q_node.name][0], graph_nodes[dq_node.name][0]]
                )

    # Remove dangling Q/DQ nodes
    if dangling_qdq_indices:
        dangling_qdq_indices = sorted(dangling_qdq_indices, reverse=True)
        for node_idx in dangling_qdq_indices:
            del graph.node[node_idx]


def print_stat(
    graph: onnx.onnx_ml_pb2.GraphProto,
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    verbose: bool,
) -> None:
    """Collect and print stats of the quantized model."""
    count = 0
    quantized_node_types = set()
    quantized_nodes = []
    output_names = [output_node.name for output_node in graph.output]
    for node in graph.node:
        for input_name in node.input:
            producer_node = tensor_producers.get(input_name, None)

            if producer_node is None:
                continue

            if "_DequantizeLinear_Output" in input_name:
                assert producer_node.op_type == "DequantizeLinear"
                quantized_node_types.add(node.op_type)
                quantized_nodes.append(node.name)
                count += 1
                break
            else:
                # Sometimes "_DequantizeLinear_Output" is not suffix of the "DequantizeLinear" typed node,
                # if that node is also in final model output. Ex. CLIP-ViT-L-14-opset16.onnx
                assert input_name in output_names or producer_node.op_type != "DequantizeLinear"

    if verbose:
        logging.info(f"Quantized nodes: {quantized_nodes}")
    logging.info(f"Total number of quantized nodes: {count}")
    logging.info(f"Quantized node types: {quantized_node_types}")
