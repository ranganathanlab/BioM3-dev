"""Tests for biom3.core.distributed.gather_object_to_main.

Runs over gloo, which is available everywhere, so the collective's contract is
checked without a GPU or a launcher. The portability requirement this guards is
real: NCCL has no ``gather`` at all, so the helper must stay on ``all_gather``.
"""

import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from biom3.core.distributed import gather_object_to_main


def _worker(rank, world, port, payloads, out_q):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK=str(rank), WORLD_SIZE=str(world),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        result = gather_object_to_main(payloads[rank])
        out_q.put((rank, result))
    finally:
        dist.destroy_process_group()


def _run(payloads, port):
    world = len(payloads)
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    mp.spawn(_worker, args=(world, port, payloads, out_q), nprocs=world, join=True)
    return dict(out_q.get() for _ in range(world))


def test_gather_returns_all_shards_on_dst():
    payloads = [{"r": 0}, {"r": 1}, {"r": 2}]
    results = _run(payloads, 29601)
    assert results[0] == payloads


def test_gather_returns_none_off_dst():
    results = _run([{"r": 0}, {"r": 1}], 29602)
    assert results[1] is None


def test_gather_handles_empty_shards():
    """Ranks with no work contribute empty payloads and must not be dropped."""
    payloads = [{"r": 0}, {}, {"r": 2}, {}]
    results = _run(payloads, 29603)
    assert results[0] == payloads
    assert len(results[0]) == 4


@pytest.mark.usefixtures("no_process_group")
def test_gather_is_noop_without_launcher():
    assert gather_object_to_main({"a": 1}) == [{"a": 1}]
