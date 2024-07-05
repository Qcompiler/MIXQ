# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions related to onnx."""
import io
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
from onnx import numpy_helper

ModelType = Any


def get_input_names_from_bytes(model_bytes: bytes, external_inputs_only: bool = True) -> List[str]:
    """This function returns the inputs names of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        List of input names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_input_names(model, external_inputs_only)


def get_all_input_names(model: onnx.onnx_ml_pb2.ModelProto) -> List[str]:
    """This function returns the inputs names of the given onnx model."""
    return [graph_input.name for graph_input in model.graph.input]


def _get_initializer_names(model: onnx.onnx_ml_pb2.ModelProto) -> List[str]:
    return [initializer.name for initializer in model.graph.initializer]


def get_input_names(
    model: onnx.onnx_ml_pb2.ModelProto, external_inputs_only: bool = True
) -> List[str]:
    """This function returns the external inputs names of the given onnx model.

    Note: external_input_names = input_names - initializer_names

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of external input names of the model.
    """
    input_names = get_all_input_names(model)
    if not external_inputs_only:
        return input_names

    initializer_names = _get_initializer_names(model)
    external_input_names = list(np.setdiff1d(input_names, initializer_names))
    return external_input_names


def get_output_names_from_bytes(model_bytes: bytes) -> List[str]:
    """This function returns the output names of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        List of output names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_output_names(model)


def get_output_names(model: onnx.onnx_ml_pb2.ModelProto) -> List[str]:
    """This function returns the output names of the given onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of output names of the model.
    """
    return [output.name for output in model.graph.output]


def get_node_names_from_bytes(model_bytes: bytes) -> List[str]:
    """This function returns all node names from the given onnx model in bytes.

    Args:
        model: onnx model in bytes.

    Returns:
        List of node names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_node_names(model)


def get_node_names(model: onnx.onnx_ml_pb2.ModelProto) -> List[str]:
    """This function returns all node names from the given onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of node names of the model.
    """
    return [node.name for node in model.graph.node]


def _get_tensor_shape(tensor: onnx.onnx_ml_pb2.ValueInfoProto) -> List[int]:
    """This function returns the shape of the input onnx tensor.

    Onnx tensors are of type ValueInfoProto and their dimensions are stored in a
    RepeatedCompositeContainer. Each of these dimensions is of type onnx.onnx_ml_pb2.Dimension.
    In a loop we access each of the Dimension object and create a shape list to return.

    Args:
        tensor: Onnx tensor object for which the shape needs to be computed.

    Returns:
        Shape of the input tensor.
    """
    if not hasattr(tensor.type, "tensor_type"):
        raise NotImplementedError("Only tensor type inputs are supported.")

    dimensions = tensor.type.tensor_type.shape.dim
    shape = []
    for dim in dimensions:
        if dim.dim_value is not None:
            shape.append(dim.dim_value)
        else:
            shape.append(dim.dim_param)

    return shape


def _get_all_shapes(container: Any) -> Dict[str, List[int]]:
    """This method returns the shape of tensors within a RepeatedCompositeContainer.

    Args:
        container: Model graph input/output container.

    Returns:
        Dictionary of tensor names and shape of the tensors within the container.
    """
    results = {}
    for tensor in container:
        results[tensor.name] = _get_tensor_shape(tensor)
    return results


def _get_selected_shapes(container: Any, inputs_to_include: List[str]) -> Dict[str, List[int]]:
    """This method returns the shape tensors within a RepeatedCompositeContainer.

    It only computes the shape of the tensors with name containing in `inputs_to_include` list.

    Args:
        container: Model graph input/output container.

    Returns:
        Dictionary of tensor names in inputs_to_include and their shapes.
    """
    results = {}
    for tensor in container:
        if tensor.name in inputs_to_include:
            results[tensor.name] = _get_tensor_shape(tensor)
    return results


def get_input_shapes_from_bytes(model_bytes: bytes) -> Dict[str, List[int]]:
    """This function returns the input shapes of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        Dictionary of inputs names and shapes.
    """
    model = onnx.load_from_string(model_bytes)
    return get_input_shapes(model)


def get_input_shapes(
    model: onnx.onnx_ml_pb2.ModelProto, external_inputs_only: bool = True
) -> Dict[str, List[int]]:
    """This function returns the inputs shapes for the given onnx model."""
    if external_inputs_only:
        return _get_selected_shapes(model.graph.input, get_input_names(model))
    return _get_all_shapes(model.graph.input)


def get_output_shapes(model: onnx.onnx_ml_pb2.ModelProto) -> Dict[str, List[int]]:
    """This function returns the output shapes for the given onnx model."""
    return _get_all_shapes(model.graph.output)


def _get_tensor_type(tensor: onnx.onnx_ml_pb2.ValueInfoProto) -> int:
    if not hasattr(tensor.type, "tensor_type"):
        raise NotImplementedError("Only tensor type inputs are supported.")
    type_ = tensor.type.tensor_type.elem_type
    return type_


def _get_container_types(
    container, inputs_to_include: Union[None, List[str]] = None
) -> Dict[str, int]:
    results = {}
    for tensor in container:
        if inputs_to_include is not None:
            if tensor.name not in inputs_to_include:
                continue
        t_type = _get_tensor_type(tensor)
        results[tensor.name] = t_type
    return results


def _get_input_types(
    model: onnx.onnx_ml_pb2.ModelProto, external_inputs_only: bool = True
) -> Dict[str, int]:
    inputs_to_include = get_input_names(model, external_inputs_only)
    return _get_container_types(model.graph.input, inputs_to_include)


def _get_output_types(model: onnx.onnx_ml_pb2.ModelProto) -> Dict[str, int]:
    results = _get_container_types(model.graph.output)
    return results


def _convert_types_to_np(types: Union[Dict[str, int], List[int], int]) -> Any:
    if isinstance(types, dict):
        types_np = {}
        for name in types.keys():
            types_np[name] = onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[types[name]]
        return types_np
    elif isinstance(types, list):
        return [onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[type_] for type_ in types]
    else:
        return onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[types]


def gen_random_inputs(model: onnx.onnx_ml_pb2.ModelProto) -> Dict[str, np.ndarray]:
    """This function generates random inputs for an onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        Dictionary of numpy tensors.
    """
    input_dict = {}
    types = _get_input_types(model)
    types_np = _convert_types_to_np(types)

    for graph_input in model.graph.input:
        # Generate tensors for external inputs only
        if graph_input.name not in types_np:
            continue

        shape_arr = []
        dims = graph_input.type.tensor_type.shape.dim
        for dim in dims:
            if dim.HasField("dim_param"):
                shape_arr.append(1)
            if dim.HasField("dim_value"):
                shape_arr.append(dim.dim_value)

        input_dict[graph_input.name] = np.random.uniform(size=shape_arr).astype(
            types_np[graph_input.name]
        )

    return input_dict


def remove_weights_data(onnx_bytes: bytes) -> bytes:
    """Removes raw weight data from the onnx model."""
    model = onnx.load_from_string(onnx_bytes)
    inits = model.graph.initializer

    for idx, init in enumerate(inits):
        # Only remove arrays with dimension larger than 1
        if len(init.dims) > 1:
            dtype = onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]
            if dtype in ["float16", "float32", "float64"]:
                # Setting up some metadata to randomize the weights later
                np_tensor = np.frombuffer(init.raw_data, dtype=dtype)
                meta = model.metadata_props.add()
                meta.key = init.name + "_avg"
                meta.value = str(np.average(np_tensor))

                meta = model.metadata_props.add()
                meta.key = init.name + "_var"
                meta.value = str(np.var(np_tensor))

                # Note that, onnx.checker will fail due to data cleaning
                # We should not check the model till weights are reassigned
                model.graph.initializer[idx].raw_data = b""

    buffer = io.BytesIO()
    onnx.save(model, buffer)
    buffer.seek(0, 0)

    return buffer.read()


def randomize_weights(onnx_path: str) -> None:
    """Assigns random values to the onnx model weights."""
    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()
        onnx_bytes = randomize_weights_onnx_bytes(onnx_bytes)

    with open(onnx_path, "wb") as f:
        # Write the modified onnx model to the same path
        f.write(onnx_bytes)


def randomize_weights_onnx_bytes(onnx_bytes: bytes, seed: int = 0) -> bytes:
    """Assigns random values to the onnx model weights."""
    model = onnx.load_from_string(onnx_bytes)
    inits = model.graph.initializer
    np.random.seed(seed)
    weight_metadata = {item.key: item.value for item in model.metadata_props}

    for idx, init in enumerate(inits):
        # Randomize only the arrays with dimension larger than 1
        if len(init.dims) > 1:
            dtype = onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]
            if dtype in ["float16", "float32", "float64"]:
                avg = weight_metadata.get(init.name + "_avg", None)
                var = weight_metadata.get(init.name + "_var", None)
                if avg and var:
                    numpy_array = np.random.normal(float(avg), float(var), size=init.dims).astype(
                        dtype
                    )
                    tensor = numpy_helper.from_array(numpy_array, init.name)
                    model.graph.initializer[idx].CopyFrom(tensor)

    buffer = io.BytesIO()
    onnx.save(model, buffer)
    buffer.seek(0, 0)

    return buffer.read()


def validate_onnx(onnx_bytes: bytes) -> bool:
    """Returns True if the onnx_bytes is valid, else False."""
    if not onnx_bytes:
        return False

    try:
        onnx_model = onnx.load_from_string(onnx_bytes)
        return onnx_model is not None
    except Exception:
        return False


def validate_batch_size(onnx_bytes: bytes, batch_size: int) -> bool:
    """Returns True if all the model inputs has batch dimension equal to batch_size."""
    input_shapes = list(get_input_shapes_from_bytes(onnx_bytes).values())
    for shape in input_shapes:
        if shape[0] != batch_size:
            return False

    return True


def get_batch_size(model: onnx.onnx_ml_pb2.ModelProto) -> int:
    """Returns the batch size of the given onnx model.

    Assertion will fail if batch size is not same over all the inputs.
    """
    input_shapes = list(get_input_shapes(model).values())
    batch_size = input_shapes[0][0]
    for shape in input_shapes:
        if batch_size != shape[0]:
            # The model does not have the batch dimension
            return 1

    return batch_size


def get_batch_size_from_bytes(onnx_bytes: bytes) -> int:
    """Returns the batch size of the given onnx model.

    Assertion will fail if batch size is not same over all the inputs.
    """
    model = onnx.load_from_string(onnx_bytes)
    return get_batch_size(model)


def save_onnx_bytes_to_dir(onnx_bytes: bytes, onnx_dir: str, onnx_name: str) -> None:
    """Saves the onnx bytes to a directory with specified file name."""
    os.makedirs(onnx_dir, exist_ok=True)
    file_path = os.path.join(onnx_dir, onnx_name + ".onnx")

    try:
        with open(file_path, "wb") as f:
            f.write(onnx_bytes)
        print(f"Onnx model saved as {file_path}")
    except Exception as e:
        print(f"Onnx model exporting as {file_path} failed, error {str(e)}")


def name_onnx_nodes(graph: onnx.onnx_ml_pb2.GraphProto) -> bool:
    """Assigns name to the onnx nodes if not present and return the modified status."""
    is_modified = False
    for idx, node in enumerate(graph.node):
        if not node.name:
            is_modified = True
            node.name = f"{node.op_type}_{idx}"

    return is_modified


def duplicate_shared_conv_weights(
    graph: onnx.onnx_ml_pb2.GraphProto,
    graph_nodes: Dict[str, Tuple[int, onnx.onnx_ml_pb2.NodeProto]],
    tensor_consumers: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
) -> bool:
    """Duplicated weights of Conv operators if they are shared."""
    is_modified = False
    for initializer in graph.initializer:
        consumers = tensor_consumers[initializer.name]
        if not consumers:
            continue

        first_consumer_node = graph_nodes[consumers[0].name][1]
        assert first_consumer_node

        # Weight duplication is necessary for weighted layers
        if first_consumer_node.op_type not in ["Conv", "MatMul", "Gemm"]:
            continue

        dtype = onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
        if dtype not in ["bfloat16", "float16", "float32", "float64"]:
            continue

        # If the initializer has more than one consumer, duplicate it with renaming for different consumers
        for idx in range(1, len(consumers)):
            is_modified = True
            graph.initializer.append(initializer)
            graph.initializer[-1].name = initializer.name + "_" + str(idx)
            consumer_node = graph_nodes[consumers[idx].name][1]
            assert consumer_node
            inp_idx = [i for i, x in enumerate(consumer_node.input) if x == initializer.name][0]
            consumer_node.input[inp_idx] = graph.initializer[-1].name

    return is_modified


def has_const_input(
    node: onnx.onnx_ml_pb2.NodeProto,
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
) -> bool:
    """Returns whether the given node has at least one constant input or not."""
    for input_name in node.input:
        if is_const_input(input_name, tensor_producers):
            return True

    return False


def is_const_input(
    input_name: str,
    tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto],
):
    """Returns whether input_name is a const input to the node or not."""
    producer_node = tensor_producers.get(input_name, None)
    # Constant inputs are not saved in tensor_produces map
    if not producer_node:
        return True

    if producer_node.op_type in ["Constant", "Identity"]:
        return True

    if producer_node.op_type in ["Squeeze", "Unsqueeze"]:
        # Second axes input to Squeeze/Unsqueeze is a constant, we need to check the first input
        producer_node = tensor_producers.get(producer_node.input[0], None)
        # If the input is constant, tensor_producers map does not have it
        return not producer_node

    # Const -> Clip -> Exp -> Mul pattern matching for swin_v2
    if producer_node.op_type == "Exp":
        clip_node = tensor_producers[producer_node.input[0]]
        if (
            clip_node
            and clip_node.op_type == "Clip"
            and has_const_input(clip_node, tensor_producers)
        ):
            return True

    return False


def is_valid_onnx_model(file_path):
    """Checks if the given file is a valid ONNX model."""
    if not os.path.exists(file_path):
        print(f"No file found at {file_path}")
        return False

    try:
        # Load the ONNX model
        model = onnx.load(file_path)

        # Check the model
        onnx.checker.check_model(model)
        print(f"ONNX model at {file_path} is valid.")
        return True
    except C.ValidationError as e:
        print(f"The file is not a valid ONNX model. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return False


def get_tensor_producer_nodes(
    graph: onnx.onnx_ml_pb2.GraphProto,
) -> Dict[str, onnx.onnx_ml_pb2.NodeProto]:
    """Returns a dictionary of tensor name and their producer node object mapping.

    Note. we create a special Root type node as external inputs producer for ease of implementation.
    """
    # Create a dictionary to store tensor producer nodes
    tensor_producers = defaultdict(None)

    # Special Root type producer node
    root_node = onnx.helper.make_node(
        op_type="Root",
        inputs=[],
        outputs=[i.name for i in graph.input],
        name="root_0",
    )

    input_names = [graph_input.name for graph_input in graph.input]
    initializer_names = [initializer.name for initializer in graph.initializer]
    external_input_names = list(np.setdiff1d(input_names, initializer_names))

    # Note. We are marking external inputs as non-constant by adding a parent,
    # so that we can quantize the first node of the graph if appropriate
    for graph_input in external_input_names:
        tensor_producers[graph_input] = root_node

    # Traverse the graph to find producer nodes for each tensor
    for node in graph.node:
        for output_name in node.output:
            tensor_producers[output_name] = node

    return tensor_producers


def get_tensor_consumer_nodes(
    graph: onnx.onnx_ml_pb2.GraphProto,
) -> Dict[str, List[onnx.onnx_ml_pb2.NodeProto]]:
    """Returns a dictionary of tensor name and their consumer node object mapping."""
    # Create a dictionary to store tensor consumer nodes
    tensor_consumers = defaultdict(list)

    # Traverse the graph to find consumer nodes for each tensor
    for node in graph.node:
        for input_name in node.input:
            tensor_consumers[input_name].append(node)

    return tensor_consumers


def get_parent_nodes(
    graph: onnx.onnx_ml_pb2.GraphProto, tensor_producers: Dict[str, onnx.onnx_ml_pb2.NodeProto]
) -> Dict[str, List[onnx.onnx_ml_pb2.NodeProto]]:
    """Returns a dictionary of node name vs parent node object mapping."""
    # Create a dictionary to store node-parent relationships
    node_parents = defaultdict(list)  # Parent can be empty list

    # Traverse the graph to find parent nodes for each node
    for node in graph.node:
        for input_name in node.input:
            producer_node = tensor_producers.get(input_name, None)
            if producer_node:
                node_parents[node.name].append(producer_node)

    return node_parents


def get_child_nodes(
    graph: onnx.onnx_ml_pb2.GraphProto,
    tensor_consumers: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
) -> Dict[str, List[onnx.onnx_ml_pb2.NodeProto]]:
    """Returns a dictionary of node name vs child node object mapping."""
    # Create a dictionary to store node-child relationships
    node_children = defaultdict(list)  # Children can be empty list

    # Traverse the graph to find child nodes for each node
    for node in graph.node:
        for output_name in node.output:
            consumer_nodes = tensor_consumers[output_name]
            node_children[node.name].extend(consumer_nodes)

    return node_children


def find_lowest_common_ancestor(
    node1_name: str,
    node2_name: str,
    node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
) -> Tuple[Optional[str], int, int]:
    """Function to find the lowest common ancestor of two nodes.

    Args:
        node1_name: First node name.
        node2_name: Second node name.
        node_parents: Nodes parent info map.

    Returns:
        LCA node.
        Distance from first node.
        Distance from second node.
    """

    def _find_ancestors(
        node_name: str,
        node_parents: Dict[str, List[onnx.onnx_ml_pb2.NodeProto]],
    ):
        ancestors = {node_name: 0}
        stack = [(node_name, 0)]

        while stack:
            cur_node_name, distance = stack.pop()

            # Iterate over parents to search for LCA
            for parent_node in node_parents[cur_node_name]:
                if parent_node is None:
                    # Node with const input
                    continue

                if parent_node.name not in ancestors:
                    ancestors[parent_node.name] = distance + 1
                    stack.append((parent_node.name, distance + 1))

        return ancestors

    ancestors1 = _find_ancestors(node1_name, node_parents)
    ancestors2 = _find_ancestors(node2_name, node_parents)

    # Find the lowest common ancestor
    common_ancestors = set(ancestors1.keys()).intersection(ancestors2.keys())
    if common_ancestors:
        lowest_common_ancestor = common_ancestors.pop()
        distance1 = ancestors1[lowest_common_ancestor]
        distance2 = ancestors2[lowest_common_ancestor]
        return lowest_common_ancestor, distance1, distance2
    else:
        return None, -1, -1  # No common ancestor found
