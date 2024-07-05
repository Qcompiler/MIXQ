# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Command-line entrypoint for ONNX PTQ."""

import argparse

import numpy as np
import onnx

from ..utils import gen_random_inputs
from .quantize import quantize

__all__ = ["main"]


def parse_args():
    argparser = argparse.ArgumentParser("python -m ammo.onnx.quantization")
    argparser.add_argument(
        "--onnx_path", required=True, type=str, help="Input onnx model without Q/DQ nodes."
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Output filename to save the converted ONNX model. If None, save it in the same dir as"
            " the original ONNX model with an appropriate suffix."
        ),
    )
    argparser.add_argument(
        "--calibration_data_path",
        "-c",
        type=str,
        default=None,
        help="Calibration data in npz/npy format. If None, use random data for calibration.",
    )
    argparser.add_argument(
        "--calibration_method",
        type=str,
        default="entropy",
        help="Calibration method. Options={entropy (default), minmax}.",
    )
    argparser.add_argument(
        "--op_types_to_quantize",
        "-t",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of types of nodes to quantize.",
    )
    argparser.add_argument(
        "--op_types_to_exclude",
        "-e",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of types of nodes to exclude from quantization.",
    )
    argparser.add_argument(
        "--nodes_to_quantize",
        "-q",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node names to quantize.",
    )
    argparser.add_argument(
        "--nodes_to_exclude",
        "-x",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node names to exclude from quantization.",
    )
    argparser.add_argument(
        "--keep_intermediate_files",
        action="store_true",
        help=(
            "If True, keep the files generated during the ONNX models' conversion/calibration."
            "Otherwise, only the converted ONNX file is kept for the user."
        ),
    )
    argparser.add_argument(
        "--use_external_data_format",
        action="store_true",
        help="If True, <MODEL_NAME>.onnx_data will be used to load weights and constants.",
    )
    argparser.add_argument(
        "--verbose",
        action="store_true",
        help="If verbose, print all the debug info.",
    )
    return argparser.parse_args()


def main():
    """Command-line entrypoint for ONNX PTQ."""
    args = parse_args()

    if args.calibration_data_path:
        calibration_data = np.load(args.calibration_data_path, allow_pickle=True)
    else:
        print("WARNING: No calibration data provided. Using random data for calibration.")
        calibration_data = gen_random_inputs(onnx.load(args.onnx_path))

    quantize(
        args.onnx_path,
        calibration_data=calibration_data,
        calibration_method=args.calibration_method,
        op_types_to_quantize=args.op_types_to_quantize,
        op_types_to_exclude=args.op_types_to_exclude,
        nodes_to_quantize=args.nodes_to_quantize,
        nodes_to_exclude=args.nodes_to_exclude,
        use_external_data_format=args.use_external_data_format,
        keep_intermediate_files=args.keep_intermediate_files,
        output_path=args.output_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
