# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""torch.distribute utils."""

import json
from contextlib import contextmanager
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List

import torch

from .model_config_utils import (
    model_config_from_dict,
    model_config_to_dict,
    restore_model_config,
    split_config_and_weights,
)


def get_world_size() -> int:
    """Safe method to get world size."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        print("torch.distributed not initialized, assuming single world_size.")
        return 1


def get_rank() -> int:
    """Safe method to get local rank."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        print("torch.distributed not initialized, assuming single world_size.")
        return 0


def get_group(ranks: List[int]):
    """Returns the process group if torch.distributed.is_initialized()."""
    # NCCL has an issue with calling barrier. So we just use the gloo backebnd for group barriers.
    return (
        torch.distributed.new_group(ranks, backend="gloo")
        if torch.distributed.is_initialized()
        else None
    )


def barrier(group=None):
    """Set a parallel barrier."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group=group)


@contextmanager
def get_tensors_parallel(tensor: torch.Tensor, ranks: List[int], group=None):
    """Gathers the tensors across distributed processes using shm.

    Args:
        tensor: the tensor that each rank want to pass to the first rank.
            The tensors across the ranks need to have the same size.
        ranks: the list of the ranks
        group: the barrier sync group.

    Yields:
        the first rank in the ranks has the full access of the tensors across all the ranks.
        the other ranks returns an empty list

    The shm will be destroyed after consumption.
    """
    assert tensor is not None
    assert len(ranks) > 1
    local_rank = get_rank()
    shm_writer = None
    shm_readers = []
    tensor = tensor.cpu()

    is_merged_rank = local_rank == ranks[0]
    # Create shm and copy the tensor to the shm if not the merged rank.
    # Assume each tensor need up to 2KB additional space for metadata.
    if not is_merged_rank:
        shm_writer = SharedMemory(name=f"rank_{local_rank}", create=True, size=tensor.nbytes + 2048)
        torch.save(tensor, shm_writer._mmap)  # type: ignore[attr-defined]
    # All ranks wait for this to complete.
    barrier(group)

    tensors = []
    # The merged rank gather the tensor from the other ranks (including itself).
    if is_merged_rank:
        for rank in ranks:
            if rank == ranks[0]:
                tensors.append(tensor)
            else:
                shm = SharedMemory(name=f"rank_{rank}", create=False)
                shared_tensor = torch.load(BytesIO(shm.buf))
                tensors.append(shared_tensor)
                shm_readers.append(shm)
    try:
        # Send the tensor list to the consumer.
        # The merged rank will get a valid tensor while the other ranks an empty tensor.
        yield tensors
    finally:
        # Reader closes the shms.
        if shm_readers:
            for shm in shm_readers:
                shm.close()

        # All ranks wait for the reader to close the shms.
        barrier(group)

        # Writer frees the shm resource.
        if shm_writer is not None:
            shm_writer.close()
            shm_writer.unlink()


@contextmanager
def get_configs_parallel(config, ranks: List[int], group):
    """Gathers the layer config across distributed processes using shm.

    Args:
        config: the config (nullable) that each rank want to pass to the first rank.
        ranks: the list of the ranks
        group: the barrier sync group.

    Yields:
        the first rank in the ranks has the full access of the configs across all the ranks.
        the other ranks returns an empty list

    The shm will be destroyed after consumption.
    """
    assert len(ranks) > 1
    local_rank = get_rank()
    shm_writer = None
    shm_readers = []

    is_merged_rank = local_rank == ranks[0]

    def _get_weights_nbytes(weights_dict: Dict[str, torch.Tensor]):
        total_nbytes = 0
        for k, v in weights_dict.items():
            # Assume each tensor need up to 2KB additional space for metadata.
            # In reality this should be much smaller.
            total_nbytes = total_nbytes + len(k) + v.nbytes + 2048

        return total_nbytes

    # Create shm and copy the serialized config to the shm if not the merged rank.
    if not is_merged_rank:
        if config is not None:
            config_dict = model_config_to_dict(config)
            # Add additional config type name to the dict so we can later pick the right config type.
            config_dict["__name__"] = str(type(config).__name__)
            weights = {}
            split_config_and_weights(config_dict, weights)

            config_json = json.dumps(config_dict)

            # SHM data structure: 8B json size, serialized json bytes and the weights dict.
            shm_writer = SharedMemory(
                name=f"rank_{local_rank}_config",
                create=True,
                size=(8 + len(config_json) + _get_weights_nbytes(weights)),
            )

            # Write json length to the shm
            shm_writer.buf[:8] = len(config_json).to_bytes(8, "little")

            # Write json to the shm
            shm_writer.buf[8 : len(config_json) + 8] = config_json.encode()

            # Write np tensors to the shm.
            shm_writer._mmap.seek(len(config_json) + 8)  # type: ignore[attr-defined]
            torch.save(weights, shm_writer._mmap)  # type: ignore[attr-defined]
        else:
            # If the config is None, we just store the empty 0.
            shm_writer = SharedMemory(
                name=f"rank_{local_rank}_config",
                create=True,
                size=8,
            )

            shm_writer.buf[:8] = int(0).to_bytes(8, "little")

    # All ranks wait for this to complete.
    barrier(group)

    configs = []
    if is_merged_rank:
        for rank in ranks:
            if rank == ranks[0]:
                configs.append(config)
            else:
                shm = SharedMemory(name=f"rank_{rank}_config", create=False)
                len_json = int.from_bytes(shm.buf[:8], "little")

                if len_json != 0:
                    config_dict = json.loads(shm.buf[8 : 8 + len_json].tobytes().decode())
                    weights = torch.load(BytesIO(shm.buf[8 + len_json :]), allow_pickle=True)
                    restore_model_config(config_dict, weights)
                    config = model_config_from_dict(config_dict)

                    configs.append(config)
                    shm_readers.append(shm)
    try:
        # Send the config list to the consumer.
        # The merged rank will get a valid config list while the other ranks an empty list.
        yield configs
    finally:
        # Reader closes the shms.
        if shm_readers:
            for shm in shm_readers:
                shm.close()

        # All ranks wait for the reader to close the shms.
        barrier(group)

        # Writer frees the shm resource.
        if shm_writer is not None:
            shm_writer.close()
            shm_writer.unlink()
