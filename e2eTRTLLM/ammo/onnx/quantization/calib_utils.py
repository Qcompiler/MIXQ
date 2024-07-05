# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Provides basic calibration utils."""

from typing import Dict, List, Union

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader

from ammo.onnx.utils import gen_random_inputs, get_batch_size, get_input_names, get_input_shapes

CalibrationDataType = Union[np.ndarray, Dict[str, np.ndarray]]


class CalibrationDataProvider(CalibrationDataReader):
    """Calibration data provider class."""

    def __init__(self, onnx_path: str, calibration_data: CalibrationDataType):
        """Intializes the data provider class with the calibration data iterator."""
        onnx_model = onnx.load(onnx_path)
        input_names = get_input_names(onnx_model)
        input_shapes = get_input_shapes(onnx_model)
        batch_size = max(1, get_batch_size(onnx_model))

        # Validate calibration data against expected inputs by the model
        if isinstance(calibration_data, np.ndarray):
            assert len(input_names) == 1, "Calibration data has only one tensor."
            calibration_data = {input_names[0]: calibration_data}
        elif isinstance(calibration_data, dict):
            assert len(input_names) == len(
                calibration_data
            ), "Model input count and calibration data doesn't match."
            for input_name in input_names:
                assert input_name in calibration_data
        else:
            raise ValueError(
                f"calibration data must be numpy array or dict, got {type(calibration_data)}"
            )

        # Check against expected input shapes except batch dimension
        for input_name in input_names:
            assert (
                input_shapes[input_name][1:] == list(calibration_data[input_name].shape)[1:]
            ), f"Tensor shape does't match for {input_name}."

        # Validate calibration data size across different inputs
        calibration_data_size = calibration_data[input_names[0]].shape[0]
        for input_name, numpy_tensor in calibration_data.items():
            assert calibration_data_size == numpy_tensor.shape[0]
            assert calibration_data_size >= batch_size

        # Create list of model inputs with appropriate batch size
        # So that we can create an input iterator
        n_itr = int(calibration_data_size / batch_size)
        calibration_data_list = [{}] * n_itr

        for input_name in input_names:
            for idx, calib_data in enumerate(
                np.array_split(calibration_data[input_name], n_itr, axis=0)
            ):
                calibration_data_list[idx][input_name] = calib_data

        self.calibration_data_reader = iter(calibration_data_list)

    def get_next(self):
        """Returns the next available calibration input from the reader."""
        return next(self.calibration_data_reader, None)


class RandomDataProvider(CalibrationDataReader):
    """Calibration data reader class with random data provider."""

    def __init__(self, onnx_path: str):
        """Initializes the data reader class with random calibration data."""
        onnx_model = onnx.load(onnx_path)
        calibration_data_list: List[Dict[str, np.ndarray]] = [gen_random_inputs(onnx_model)]
        self.calibration_data_reader = iter(calibration_data_list)

    def get_next(self):
        """Returns the next available calibration input from the reader."""
        return next(self.calibration_data_reader, None)
