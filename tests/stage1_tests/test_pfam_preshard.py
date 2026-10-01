"""Pre-written Pfam shards: reused when they match, never silently overwritten."""
import json
import os
import types

import pandas as pd
import pytest

from biom3.Stage1 import preshard_pfam
from biom3.Stage1.preprocess import (
    PFAM_SPLITS_MANIFEST, Pfam_DataModule, pfam_splits_status, write_pfam_splits,
)

pytestmark = pytest.mark.usefixtures("no_process_group")


def _pfam(n=103):
    return pd.DataFrame({
        "id": [f"A{i}" for i in range(n)], "pfam_label": [f"PF{i % 5:05d}" for i in range(n)],
        "sequence": ["MKT"] * n, "[final]text_caption": ["cap"] * n,
    })


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "pfam.csv"
    _pfam().to_csv(p, index=False)
    return str(p)


def test_write_then_reuse(tmp_path, src):
    d = str(tmp_path / "splits")
    assert pfam_splits_status(d, src, 7) == "write"
    df = _pfam()
    write_pfam_splits(df, d, 7, src)
    for ii in range(7):
        got = pd.read_csv(f"{d}/split_pfam_rank_{ii}.csv")
        pd.testing.assert_frame_equal(got, df.iloc[ii::7].reset_index(drop=True))
    man = json.load(open(os.path.join(d, PFAM_SPLITS_MANIFEST)))
    assert man["num_shards"] == 7 and man["total_rows"] == 103
    assert pfam_splits_status(d, src, 7) == "reuse"


def test_refuses_other_world_size_changed_source_or_missing_shard(tmp_path, src):
    d = str(tmp_path / "splits")
    write_pfam_splits(_pfam(), d, 7, src)
    with pytest.raises(ValueError, match="num_shards"):
        pfam_splits_status(d, src, 8)
    os.remove(f"{d}/split_pfam_rank_3.csv")
    with pytest.raises(ValueError, match="missing"):
        pfam_splits_status(d, src, 7)
    with open(src, "a") as fh:
        fh.write("A999,PF00001,MKT,cap\n")
    with pytest.raises(ValueError, match="source_size"):
        pfam_splits_status(d, src, 7)


def _dm(src, splits_dir, world):
    args = types.SimpleNamespace(pfam_data_path=src, pfam_splits_dir=splits_dir, dataset_type="pfam")
    dm = Pfam_DataModule(args)
    dm.trainer = types.SimpleNamespace(world_size=world)
    return dm


def test_prepare_data_skips_loading_when_shards_match(tmp_path, src, monkeypatch):
    d = str(tmp_path / "splits")
    write_pfam_splits(_pfam(), d, 7, src)
    dm = _dm(src, d, 7)
    def boom(*a, **k):
        raise AssertionError("prepare_data loaded data despite a matching manifest")
    monkeypatch.setattr(dm, "load_pfam_database", boom)
    monkeypatch.setattr(dm, "load_swiss_prot", boom)
    dm.prepare_data()


def test_prepare_data_writes_manifest_when_none(tmp_path, src, monkeypatch):
    d = str(tmp_path / "splits")
    dm = _dm(src, d, 4)
    monkeypatch.setattr(dm, "load_pfam_database", lambda: _pfam())
    monkeypatch.setattr(dm, "load_swiss_prot", lambda: pd.DataFrame({"pfam_label": ["['PF00001']"]}))
    dm.prepare_data()
    assert pfam_splits_status(d, src, 4) == "reuse"


def test_cli_writes_once_then_reuses_and_rejects_per_run_dir(tmp_path, src, monkeypatch, capsys):
    monkeypatch.setattr(Pfam_DataModule, "load_pfam_database", lambda self: _pfam())
    cfg = tmp_path / "cfg.json"
    base = {"pfam_data_path": src, "dataset_type": "pfam", "num_nodes": 1, "devices_per_node": 6,
            "output_root": str(tmp_path / "out")}
    cfg.write_text(json.dumps(dict(base, pfam_splits_dir=str(tmp_path / "persist"))))
    assert preshard_pfam.main(["-c", str(cfg)]) == 0
    assert pfam_splits_status(str(tmp_path / "persist"), src, 6) == "reuse"
    assert preshard_pfam.main(["-c", str(cfg)]) == 0
    assert "nothing to do" in capsys.readouterr().out
    cfg.write_text(json.dumps(base))
    with pytest.raises(SystemExit):
        preshard_pfam.main(["-c", str(cfg)])
