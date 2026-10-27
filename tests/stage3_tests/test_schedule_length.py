"""The cosine schedule's epoch length must follow the number of ranks.

Stage 3 builds its learning-rate schedule from ``args.traindata_len``, the
number of optimizer steps in an epoch. It used to be the single-process loader
length divided by ``devices_per_node``, which ignored ``num_nodes``: on N nodes
the warmup lasted N epochs and the decay was stretched N times. It also ran
before any distributed sampler existed, so it missed that each rank's shard is
truncated to ``num_samples // world_size``.

The first four cases below are real runs on Aurora, with the steps per epoch
those runs actually took.
"""

from argparse import Namespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from biom3.Stage3.run_PL_training import optimizer_steps_per_epoch, set_traindata_len


@pytest.mark.parametrize("num_samples, batch_size, world_size, expected", [
    (175, 16, 1, 11),    # 1 node x 1 device
    (175, 16, 24, 1),    # 2 nodes x 12 devices
    (800, 8, 24, 5),     # 2 nodes x 12 devices
    (800, 8, 48, 2),     # 4 nodes x 12 devices
    (800, 8, 12, 9),     # 1 node x 12 devices: 66 samples per rank
])
def test_steps_follow_the_world_size(num_samples, batch_size, world_size, expected):
    assert optimizer_steps_per_epoch(num_samples, batch_size, world_size) == expected


def test_gradient_accumulation_rounds_up():
    # 100 batches in groups of 3: 33 full groups and a final step for the last one.
    assert optimizer_steps_per_epoch(800, 8, 1, acc_grad_batches=3) == 34


@pytest.mark.parametrize("limit, expected", [(None, 100), (1.0, 100), (0.25, 25), (10.0, 10), (500.0, 100)])
def test_limit_train_batches(limit, expected):
    assert optimizer_steps_per_epoch(800, 8, 1, limit_train_batches=limit) == expected


def test_never_zero():
    """More ranks than samples used to give 0, which made the schedule flip the
    learning rate between full and zero on alternate steps."""
    assert optimizer_steps_per_epoch(10, 16, 24) == 1


class _DataModule:
    def __init__(self, num_samples, batch_size):
        self._loader = DataLoader(
            TensorDataset(torch.zeros(num_samples, 1)), batch_size=batch_size)

    def train_dataloader(self):
        return self._loader


@pytest.mark.parametrize("num_nodes, devices_per_node, expected", [(1, 1, 100), (1, 12, 9), (4, 12, 2)])
def test_set_traindata_len(num_nodes, devices_per_node, expected):
    args = Namespace(num_nodes=num_nodes, devices_per_node=devices_per_node,
                     acc_grad_batches=1, batch_size=8, limit_train_batches=None)

    set_traindata_len(args, _DataModule(800, 8))

    assert args.traindata_len == expected
