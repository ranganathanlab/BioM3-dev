"""Pfam_DataModule.setup: a held-out validation set, and equal steps on every rank.

split_by_accession -- every caption variant of a protein lands on one side, and
every rank computes the same assignment whatever rows its shard lets it keep.

equalize_across_ranks -- ranks filter Swiss-Prot against different Pfam shards,
so their tables differ in length; unequal lengths mean unequal steps per epoch
and a hang at the first epoch boundary. Exercised here with a real 3-process
gloo group, since the bug only exists when there is more than one rank.
"""
import json
import os
import socket

import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from biom3.Stage1.preprocess import equalize_across_ranks, split_by_accession


def _frame(n_accessions, variants=9, start=0):
    rows = [(f"P{a:06d}", v) for a in range(start, start + n_accessions) for v in range(variants)]
    return pd.DataFrame(rows, columns=["primary_Accession", "variant"])


def test_no_accession_on_both_sides():
    train, valid = split_by_accession(_frame(2000), "primary_Accession", 0.1, seed=42)
    assert set(train["primary_Accession"]).isdisjoint(valid["primary_Accession"])
    assert len(train) + len(valid) == 2000 * 9


def test_every_rank_agrees_whatever_rows_it_keeps():
    """Two 'ranks' keep different, overlapping, differently ordered subsets."""
    full = _frame(3000)
    rank_a = full.iloc[: 2 * len(full) // 3].sample(frac=1.0, random_state=1)
    rank_b = full.iloc[len(full) // 3:].sample(frac=1.0, random_state=2)
    va = set(split_by_accession(rank_a, "primary_Accession", 0.1, 42)[1]["primary_Accession"])
    vb = set(split_by_accession(rank_b, "primary_Accession", 0.1, 42)[1]["primary_Accession"])
    shared = set(rank_a["primary_Accession"]) & set(rank_b["primary_Accession"])
    assert shared, "test needs overlapping accessions"
    assert va & shared == vb & shared


def test_valid_fraction_and_seed_dependence():
    df = _frame(20000, variants=1)
    v42 = set(split_by_accession(df, "primary_Accession", 0.1, 42)[1]["primary_Accession"])
    v7 = set(split_by_accession(df, "primary_Accession", 0.1, 7)[1]["primary_Accession"])
    assert 1700 < len(v42) < 2300, len(v42)          # expect 2000, sd ~42
    assert v42 != v7


@pytest.mark.usefixtures("no_process_group")
def test_single_process_is_a_no_op():
    train, valid = _frame(10), _frame(5, start=100)
    t, v, per_rank = equalize_across_ranks(train, valid, seed=0)
    assert t is train and v is valid
    assert per_rank.tolist() == [[90, 45]]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, out_dir, sizes):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        n_train, n_valid = sizes[rank]
        train = _frame(n_train, variants=1, start=rank * 10_000)
        valid = _frame(n_valid, variants=1, start=rank * 10_000 + 5_000)
        t, v, per_rank = equalize_across_ranks(train, valid, seed=123, device=torch.device("cpu"))
        with open(os.path.join(out_dir, f"{rank}.json"), "w") as fh:
            json.dump({
                "train": len(t), "valid": len(v), "per_rank": per_rank.tolist(),
                "train_subset": set(t["primary_Accession"]) <= set(train["primary_Accession"]),
                "valid_subset": set(v["primary_Accession"]) <= set(valid["primary_Accession"]),
                "not_just_head": list(t["primary_Accession"]) != list(train["primary_Accession"].iloc[: len(t)]),
            }, fh)
    finally:
        dist.destroy_process_group()


def test_ranks_end_up_with_equal_lengths(tmp_path):
    # unequal on purpose, and the minimum is NOT on rank 0
    sizes = [(1000, 120), (970, 131), (1012, 117)]
    world = len(sizes)
    mp.spawn(_worker, args=(world, _free_port(), str(tmp_path), sizes), nprocs=world, join=True)
    out = [json.load(open(tmp_path / f"{r}.json")) for r in range(world)]
    for r, o in enumerate(out):
        assert o["train"] == 970 and o["valid"] == 117, (r, o)
        assert o["per_rank"] == [list(s) for s in sizes], (r, o["per_rank"])
        assert o["train_subset"] and o["valid_subset"]
    # rank 0 had to drop 30 train rows; they must be a random 30, not the tail
    assert out[0]["not_just_head"]
