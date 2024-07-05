# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This document lists the quantization formats supported by AMMO and example quantization configs.

.. _quantization-formats:

Quantization Formats
==========================================

The following table lists the quantization formats supported by AMMO, the corresponding quantization
config and use-cases. See :ref:`Quantization Configs <example-quantization-configs>` for the
specific quantization config definitions.

.. note::

    The recommended configs given below are for LLM models. For CNN models, only INT8 quantization
    is supported. Please use quantization config ``INT8_DEFAULT_CFG`` for CNN models.

=================================   =======================================================
Quantization  Format                Details
=================================   =======================================================
INT8                                * AMMO config: ``INT8_SMOOTHQUANT_CFG``
                                    * 8-bit integer quantization with a variant of
                                      `SmoothQuant <https://arxiv.org/pdf/2211.10438.pdf>`_
                                      calibration.

                                      - per-channel weight quantization, per-tensor activation
                                        quantization.

                                    * Compresses FP16/BF16 model to 50% of original size, faster
                                      inference than FP16/BF16.
                                    * Deploy using TensorRT, TRT-LLM. Supported on most GPUs.

FP8                                 * AMMO config: ``FP8_DEFAULT_CFG``
                                    * FP8 per-tensor weight & activation quantization with min-max
                                      calibration.
                                    * Compresses FP16/BF16 model to 50% of original size.
                                    * Least accuracy drop compared with other formats, faster
                                      inference than FP16/BF16.
                                    * Deploy via TensorRT, TRT-LLM. Supported GPUs: Ada, Hopper and
                                      later.

INT4 Weights only                   * AMMO config: ``INT4_AWQ_CFG``
                                    * 4-bit integer group-wise/block-wise weight only quantization
                                      with `AWQ <https://arxiv.org/pdf/2306.00978.pdf>`_
                                      calibration.
                                    * Compresses FP16 model to 25% of original size.
                                    * Faster inference for 'memory bound' applications than formats
                                      like FP8, INT8. Memory bound applications typically have low
                                      batch size.
                                    * Quantized model accuracy typically better than INT8.
                                    * Deploy via TRT-LLM. Supported GPUs: Ampere and later.

INT4 Weights, FP8 Activations       * AMMO config: ``W4A8_AWQ_BETA_CFG``
                                    * 4-bit integer group-wise/block-wise weight quantization & FP8
                                      per-tensor activation quantization.
                                      `AWQ <https://arxiv.org/pdf/2306.00978.pdf>`_ calibration.
                                    * Compresses FP16 model to 25% of original size.
                                    * Typically faster inference than all other formats. More
                                      effective for memory bound applications.
                                    * Quantized model accuracy similar to INT4 weights only
                                      quantization.
                                    * TRT-LLM support in development; Supported GPUs: Ada, Hopper
                                      and later.

=================================   =======================================================

.. _quantization-configs:

Quantization Configs
================================

Quantization config is dictionary specifying the values for keys ``"quant_cfg"`` and
``"algorithm"``. The ``"quant_cfg"`` key specifies the quantization configurations. The
``"algorithm"`` key specifies the ``algorithm`` argument to
:meth:`calibrate <ammo.torch.quantization.model_calib.calibrate>`.

Quantization configurations is a dictionary mapping wildcards or filter functions
to its quantizer attributes. The wildcards or filter functions  are matched
against the quantizer module names. The quantizer modules have names ending with
``weight_quantizer`` and ``input_quantizer`` and they perform weight quantization and
input quantization (or activation quantization) respectively. The quantizer modules are generally
instances of
:class:`TensorQuantizer <ammo.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>` and
the specified quantizer attributes describe its quantization behavior. Quantizer attributes is a
dictionary mapping quantizer attribute names to their values.

Quantizer attributes can also be a list of dictionaries. In this case, the matched quantizer module
is replaced with a
:class:`SequentialQuantizer <ammo.torch.quantization.nn.modules.tensor_quantizer.SequentialQuantizer>`
module which is used to quantize a tensor in multiple formats sequentially. Each quantizer attribute
dictionary in the list specifies the quantization formats for each quantization step of the
sequential quantizer. For example, `SequentialQuantizer` is used in 'INT4 Weights, FP8 Activations'
quantization in which the weights are quantized in INT4 followed by FP8.

.. _example-quantization-configs:

Here are examples quantization configs from AMMO:

.. code-block::

    INT8_DEFAULT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": "max",
    }

    INT8_SMOOTHQUANT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": "smoothquant",
    }

    FP8_DEFAULT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "default": {"num_bits": (4, 3), "axis": None},
        },
        "algorithm": "max",
    }

    INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "default": {"enable": False},
        },
        "algorithm": "max",
    }

    INT4_AWQ_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "default": {"enable": False},
        },
        "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
        # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
        # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
    }

    W4A8_AWQ_BETA_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*lm_head*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
    }

These config can be accessed as attributes of ``ammo.torch.quantization`` and can be given as
input to :meth:`atq.quantize() <ammo.torch.quantization.model_quant.quantize>`. For example:

.. code-block::

    import ammo.torch.quantization as atq
    atq.quantize(model, atq.INT8_DEFAULT_CFG, forward_loop)

You can also create your own config by following these examples.
For instance, if you want to quantize a model with int4 AWQ algorithm, but need to skip quantizing
the layer named ``lm_head``,  you can create a custom config and quantize your model as following:

.. code-block::

    # Create custom config
    CUSTOM_INT4_AWQ_CFG = copy.deepcopy(atq.INT4_AWQ_CFG)
    CUSTOM_INT4_AWQ_CFG["quant_cfg"]["*lm_head*"] = {"enable": False}

    # quantize model
    atq.quantize(model, CUSTOM_INT4_AWQ_CFG, forward_loop)

"""

INT8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",
}

INT8_SMOOTHQUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "smoothquant",
}

FP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "default": {"num_bits": (4, 3), "axis": None},
    },
    "algorithm": "max",
}

INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

INT4_AWQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
    # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
    # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
}

# W4A8 currently uses INT4 blockwise quantization (block size = 128) followed by FP8 quantization
# for weights. This could change in the future
W4A8_AWQ_BETA_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
}



INT8_MIX = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "mixq",
}