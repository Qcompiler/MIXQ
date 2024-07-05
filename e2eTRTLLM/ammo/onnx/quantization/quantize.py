# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Convert ONNX model without QDQ nodes + calib data into ONNX model with QDQ nodes.

Typically quantizing linear operations like Conv, MatMul etc. gives most of the performance boost.
But there are many other ops that are quantizable (aka low precision kernels available) and provides
optimal performance with lower accuracy drop. The default op types that this ONNX ptq tool quantizes
are: ['Add', 'AveragePool', 'BatchNormalization', 'Clip', 'Conv', 'ConvTranspose', 'Gemm',
'GlobalAveragePool', 'MatMul', 'MaxPool', 'Mul']. The tool inserts QDQ nodes following compiler
friendly patterns and generates an explicit ONNX model.
"""
import logging
import os
import re
import shutil
import tempfile
from typing import Dict, List, Tuple

import onnx
from onnx.external_data_helper import load_external_data_for_model
from onnxruntime.quantization import (
    CalibrationMethod,
    quantize_static,
)
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

from ammo.onnx.op_types import get_quantizable_op_types
from ammo.onnx.quantization.calib_utils import (
    CalibrationDataProvider,
    CalibrationDataType,
    RandomDataProvider,
)
from ammo.onnx.quantization.graph_utils import (
    build_non_residual_input_map,
    classify_partition_nodes,
    filter_quantizable_kgen_heads,
    print_stat,
    remove_partial_input_qdq,
)
from ammo.onnx.quantization.operators import QDQConvTranspose, QDQNormalization
from ammo.onnx.quantization.ort_patching import patch_ort_modules
from ammo.onnx.quantization.partitioning import (
    find_fusible_partitions,
    find_partitions_from_patterns,
    find_quantizable_ops,
    get_skiped_output_layers,
)
from ammo.onnx.utils import (
    duplicate_shared_conv_weights,
    get_child_nodes,
    get_parent_nodes,
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
    name_onnx_nodes,
)

__all__ = ["quantize"]

# Set logging level to info
logging.getLogger().setLevel(logging.INFO)


def _find_nodes_from_op_types_to_exclude(
    graph: onnx.onnx_ml_pb2.GraphProto, op_types_to_exclude=None
) -> List[str]:
    nodes_to_exclude = []
    if op_types_to_exclude:
        nodes_to_exclude = [node.name for node in graph.node if node.op_type in op_types_to_exclude]
    return nodes_to_exclude


def _expand_node_names_from_patterns(
    graph: onnx.onnx_ml_pb2.GraphProto, name_patterns: List[str]
) -> List[str]:
    matched_node_names = []
    for pattern in name_patterns:
        for node in graph.node:
            if re.match(pattern, node.name):
                matched_node_names.append(node.name)

    return matched_node_names


def _configure_ort(op_types: List[str], op_types_to_quantize: List[str]):
    # Register some new QDQ operators on top of ORT
    QDQRegistry["BatchNormalization"] = QDQNormalization
    QDQRegistry["ConvTranspose"] = QDQConvTranspose
    QDQRegistry["InstanceNormalization"] = QDQNormalization
    QDQRegistry["LRN"] = QDQNormalization  # Example: caffenet-12.onnx

    # Patch ORT modules to fix bugs and support some edge cases
    patch_ort_modules()

    # Remove copy, reduction and activation ops from ORT QDQ registry
    for op_type in [
        "ArgMax",
        "Concat",
        "EmbedLayerNormalization",
        "Gather",
        "InstanceNormalization",
        "LeakyRelu",
        "Pad",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
        "Softmax",
        "Split",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
        "Where",
    ]:
        if op_type in QLinearOpsRegistry:
            del QLinearOpsRegistry[op_type]
        if op_type in QDQRegistry:
            del QDQRegistry[op_type]

    # Prepare TensorRT friendly quantization settings
    trt_guided_options = {
        "QuantizeBias": False,
        "ActivationSymmetric": True,
        "OpTypesToExcludeOutputQuantization": op_types,  # No output quantization
        "AddQDQPairToWeight": True,  # Instead of quantizing the weights, add QDQ node
        "QDQOpTypePerChannelSupportToAxis": {
            "Conv": 0,
            "ConvTranspose": 1,
        },  # per_channel should be True
        # TODO: twins_svt_small_opset13_simplified.onnx need one QDQ node for /pos_block.0/Reshape_output_0?
        "DedicatedQDQPair": True,
        "ForceQuantizeNoInputCheck": (
            # By default, for some latent operators like MaxPool, Transpose, etc.,
            # ORT does not quantize if their input is not quantized already.
            True
        ),
    }

    quantizable_op_types = get_quantizable_op_types(op_types_to_quantize)
    return trt_guided_options, quantizable_op_types


def _find_nodes_to_quantize(
    graph: onnx.onnx_ml_pb2.GraphProto,
    node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    graph_nodes: Dict[str, Tuple[int, onnx.onnx_ml_pb2.NodeProto]],
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
    tensor_consumers: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    node_children: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    quantizable_op_types: List[str],
    verbose: bool,
):
    # Build a map of add nodes to their non-residual inputs, i.e. fusible with Conv group
    logging.info("Building non-residual Add input map ...")
    non_residual_inputs = build_non_residual_input_map(graph, tensor_producers, node_parents)

    logging.info("Searching for hard-coded patterns like MHA, LayerNorm etc. ...")
    hard_coded_partitions = find_partitions_from_patterns(graph, node_children, tensor_producers)

    logging.info("Building KGEN/CASK targeted partitions ...")
    # partitioned_nodes keeps track of nodes that are already part of some partition.
    # Certain nodes of those partitions are quantizable. For example, heads.
    partitioned_nodes = set(sum(hard_coded_partitions, []))
    cask_fusible_partitions, kgen_partitions = find_fusible_partitions(
        graph,
        partitioned_nodes,
        non_residual_inputs,
        node_parents,
        graph_nodes,
        tensor_producers,
        tensor_consumers,
    )
    if verbose:
        logging.info(f"CASK fusible partitions: {cask_fusible_partitions}")
        logging.info(f"KGEN partitions: {kgen_partitions}")

    logging.info("Classifying the partition nodes ...")
    _, quantizable_partition_nodes, no_quantize_inputs = classify_partition_nodes(
        graph_nodes,
        tensor_producers,
        cask_fusible_partitions,
    )
    quantizable_kgen_heads, no_quantize_kgen_inputs = filter_quantizable_kgen_heads(
        cask_fusible_partitions,
        kgen_partitions,
        graph_nodes,
        tensor_producers,
        tensor_consumers,
        node_parents,
        quantizable_op_types,
    )

    nodes_to_quantize = quantizable_kgen_heads + quantizable_partition_nodes
    nodes_to_quantize.extend([dst for _, dst, _ in no_quantize_inputs])
    nodes_to_quantize.extend(
        find_quantizable_ops(
            graph,
            nodes_to_quantize,
            partitioned_nodes,
            quantizable_op_types,
            tensor_producers,
            node_children,
        )
    )

    skip_list = get_skiped_output_layers(graph)
    nodes_to_quantize = [node_name for node_name in nodes_to_quantize if node_name not in skip_list]

    return nodes_to_quantize, no_quantize_inputs + no_quantize_kgen_inputs


def _find_nodes_to_exclude(
    graph: onnx.onnx_ml_pb2.GraphProto, nodes_to_exclude: List[str], op_types_to_exclude: List[str]
):
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_exclude = _expand_node_names_from_patterns(graph, nodes_to_exclude)
    nodes_to_exclude.extend(_find_nodes_from_op_types_to_exclude(graph, op_types_to_exclude))

    # Remove duplicates from the exclusion list
    nodes_to_exclude = [*set(nodes_to_exclude)]

    return nodes_to_exclude


def quantize(
    onnx_path: str,
    external_data_path: str = None,
    calibration_data: CalibrationDataType = None,
    calibration_method: str = "entropy",
    op_types_to_quantize: List[str] = None,
    op_types_to_exclude: List[str] = None,
    nodes_to_quantize: List[str] = None,
    nodes_to_exclude: List[str] = None,
    use_external_data_format: bool = False,
    keep_intermediate_files: bool = False,
    output_path: str = None,
    verbose: bool = False,
) -> None:
    """Quantize the given onnx model.

    Args:
        onnx_path:
            Path to the input onnx model.
        external_data_path:
            External data path for the model. If not None, this path will be used to load the weights of the model.
        calibration_data:
            Calibration data, either a numpy array or list/dict of numpy array.
        calibration_method:
            Calibration method. Options={entropy (default), minmax}.
        op_types_to_quantize:
            List of types of operators to quantize. When this list is not None, only the types in this list
            are quantized. Example: ['Conv'] indicates that only ops of type 'Conv' should be quantized.
            If this list is None (default), all supported operators are quantized.
            This flag does not support regular expression.
        op_types_to_exclude:
            List of types of operators to exclude from quantization.
            This flag does not support regular expression.
        nodes_to_quantize:
            List of node names to quantize. When this list is not None, only the nodes in this list
            are quantized. Example: ['Conv__224', 'Conv__252'].
            If this list is None (default), all supported nodes are quantized.
            This flag does not support regular expression.
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None. This flag supports regular expression.
        use_external_data_format:
            If not None, this path will be used to store the weights of the quantized model.
        keep_intermediate_files:
            If False, only save the converted ONNX files for the user. Otherwise, keep all intermediate files
             generated during the ONNX models' conversion/calibration.
        output_path:
            Output filename to save the converted ONNX model.
            If None, save in the same directory as the original ONNX model with .quant suffix.

    Returns:
        None, write the quantized onnx model in the same directory with filename like "<model_name>.quant.onnx".
    """
    # quantize_static creates a shape-inferred copy at the input model's directory
    # Needs to check if we have write permission to this directory
    assert onnx_path.endswith(".onnx")
    if not os.access(os.path.dirname(os.path.abspath(onnx_path)), os.W_OK):
        old_dir = os.path.dirname(os.path.abspath(onnx_path))
        tmp_dir = tempfile.mkdtemp()
        logging.info(f"Directory {old_dir} is not writable, store intermediate files in {tmp_dir}")
        new_onnx_path = os.path.join(tmp_dir, os.path.basename(onnx_path))
        shutil.copy(onnx_path, new_onnx_path)
        onnx_path = new_onnx_path

    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    if not output_path:
        output_dir = os.path.dirname(onnx_path)
        output_path = os.path.join(output_dir, f"{model_name}.quant.onnx")
        logging.info(f"No output path specified, save quantized model to {output_path}")
    intermediate_generated_files = []

    # Load the model and weights
    onnx_model = onnx.load(onnx_path, load_external_data=external_data_path is None)
    if external_data_path:
        load_external_data_for_model(onnx_model, external_data_path)

    # Per-Channel support with QDQ format requires onnx opset version 13 or above
    ai_onnx_domain = [
        opset for opset in onnx_model.opset_import if not opset.domain or opset.domain == "ai.onnx"
    ]
    opset_version = ai_onnx_domain[0].version
    logging.info(f"Model {onnx_path} with opset_version {opset_version} is loaded.")

    if opset_version < 13:
        opset_version = 13
        onnx_model = onnx.version_converter.convert_version(onnx_model, opset_version)
        output_dir = os.path.dirname(output_path)
        onnx_path = os.path.join(output_dir, f"{model_name}_opset{opset_version}.onnx")
        onnx.save(onnx_model, onnx_path, save_as_external_data=use_external_data_format)
        logging.info(f"Model is cloned to {onnx_path} with opset_version {opset_version}.")
        intermediate_generated_files.append(onnx_path)

    # Take the onnx graph
    graph = onnx_model.graph

    # Change the default configuration of ORT quantization
    op_types_to_quantize = op_types_to_quantize or []
    op_types = set([node.op_type for node in graph.node])
    trt_guided_options, quantizable_op_types = _configure_ort(list(op_types), op_types_to_quantize)
    logging.info(
        f"Quantizable op types in the model: {[t for t in quantizable_op_types if t in op_types]}"
    )

    # Sometimes input onnx model does not contain the node names
    # This tool depends on those names, so we assign names if needed
    is_named = name_onnx_nodes(graph)

    # Build graph helper maps
    logging.info("Preparing tensor producer map ...")
    tensor_producers = get_tensor_producer_nodes(graph)
    tensor_consumers = get_tensor_consumer_nodes(graph)

    logging.info("Preparing node and relationship maps ...")
    graph_nodes = {node.name: (idx, node) for idx, node in enumerate(graph.node)}
    node_parents = get_parent_nodes(graph, tensor_producers)
    node_children = get_child_nodes(graph, tensor_consumers)

    is_duplicated = duplicate_shared_conv_weights(graph, graph_nodes, tensor_consumers)
    if is_named or is_duplicated:
        output_dir = os.path.dirname(output_path)
        onnx_path = os.path.join(output_dir, f"{model_name}_named.onnx")
        onnx.save(onnx_model, onnx_path, save_as_external_data=use_external_data_format)
        logging.info(f"Model is cloned to {onnx_path} after naming the nodes.")
        intermediate_generated_files.append(onnx_path)

    no_quantize_inputs = []
    if not nodes_to_quantize:
        nodes_to_quantize, no_quantize_inputs = _find_nodes_to_quantize(
            graph,
            node_parents,
            graph_nodes,
            tensor_producers,
            tensor_consumers,
            node_children,
            quantizable_op_types,
            verbose,
        )

    # Collect node names to exclude from quantization
    nodes_to_exclude = _find_nodes_to_exclude(graph, nodes_to_exclude, op_types_to_exclude)

    logging.info(f"Total number of nodes: {len(graph.node)}")
    logging.info(f"Manually skipped node count: {len(nodes_to_exclude)}")
    if verbose:
        logging.info(f"Manually skipped nodes: {nodes_to_exclude}")

    # Use random scales if calibration data is not supplied
    if calibration_data is None:
        calibration_data_reader = RandomDataProvider(onnx_path)
    else:
        calibration_data_reader = CalibrationDataProvider(onnx_path, calibration_data)

    # Use ort api to quantize the onnx model
    quantize_static(
        onnx_path,
        output_path,
        calibration_data_reader,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=(opset_version >= 13),
        extra_options=trt_guided_options,
        use_external_data_format=use_external_data_format,
        calibrate_method=(
            CalibrationMethod.Entropy
            if calibration_method == "entropy"
            else CalibrationMethod.MinMax
        ),
    )

    # Post-processing of the onnx model after ort quantization
    onnx_model = onnx.load(output_path)
    graph = onnx_model.graph
    graph_nodes = {node.name: (idx, node) for idx, node in enumerate(graph.node)}
    tensor_producers = get_tensor_producer_nodes(graph)
    tensor_consumers = get_tensor_consumer_nodes(graph)
    remove_partial_input_qdq(
        graph,
        graph_nodes,
        tensor_producers,
        tensor_consumers,
        no_quantize_inputs,
    )

    # Collect and print stats of the quantized model
    print_stat(graph, tensor_producers, verbose)

    # Check if intermediate files should be deleted
    if not keep_intermediate_files:
        for file in intermediate_generated_files:
            os.remove(file)

    # Save the modified model
    onnx.save(onnx_model, output_path, save_as_external_data=use_external_data_format)
    logging.info(f"Quantized onnx model is saved as {output_path}")

    # Check if the quantized model is valid
    onnx.checker.check_model(onnx_model)
