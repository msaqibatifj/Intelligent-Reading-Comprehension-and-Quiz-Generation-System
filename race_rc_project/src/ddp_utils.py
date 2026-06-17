"""
ddp_utils.py — Distributed Data Parallel (DDP) setup & helpers for torchrun.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_ddp() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_ddp():
    """
    Initialise DDP from torchrun environment variables.

    If ``LOCAL_RANK`` is not set (i.e. not launched via torchrun), run in
    single-process mode and return rank=0, world_size=1, device=cuda:0 (or cpu).

    Returns
    -------
    (rank, world_size, device)
    """
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        rank = int(local_rank)
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(device)

    return rank, world_size, device


def destroy_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
