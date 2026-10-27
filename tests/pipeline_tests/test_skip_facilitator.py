"""Tests for --skip_facilitator in biom3_embedding_pipeline.

The option replaces Stage 2 with the identity map, for weight sets that have no
Facilitator: z_c is written as a copy of z_t and the rest of the pipeline runs
on it as usual. Stages 1 to 3 are stubbed, so these run under `pytest --quick`
without downloaded weights; the HDF5 compile is real.
"""

import json
import os

import h5py
import numpy as np
import pytest
import torch

import biom3.Stage1.run_PenCL_inference as stage1_mod
import biom3.Stage2.run_Facilitator_sample as stage2_mod
import biom3.Stage3.run_ProteoScribe_sample as stage3_mod
from biom3.pipeline.embedding_pipeline import main, parse_arguments

PENCL_CONFIG = "configs/inference/stage1_PenCL.json"
FACILITATOR_CONFIG = "configs/inference/stage2_Facilitator.json"
PROTEOSCRIBE_CONFIG = "configs/inference/stage3_ProteoScribe_sample.json"


@pytest.fixture
def pencl_only_weight_set(tmp_path):
    path = tmp_path / "pencl_only.json"
    path.write_text(json.dumps({"pencl_weights": "weights/PenCL/only_pencl.bin"}))
    return str(path)


def _argv(tmp_path, weight_set, *extra):
    return [
        "-i", "tests/_data/stage1_inputs/sample_text_seqs1.csv",
        "-o", str(tmp_path / "out"),
        "--prefix", "run",
        "--weight_set", weight_set,
        "--pencl_config", PENCL_CONFIG,
        "--device", "cpu",
        *extra,
    ]


def _stage1_output():
    """What Stage 1 writes, at toy size."""
    torch.manual_seed(0)
    return {
        "z_t": torch.randn(3, 4),
        "z_p": torch.randn(3, 4),
        "text_prompts": np.array([b"a", b"b", b"c"], dtype=object),
        "sequence": ["MKT", "MKTA", "MKTAY"],
        "acc_id": ["P1", "P2", "P3"],
    }


@pytest.fixture
def stub_stages(monkeypatch):
    """Stub Stage 1 to write a toy output; Stage 2 must never run."""
    calls = {"stage1": [], "stage3": []}

    def fake_stage1(args, _setup_logging=True):
        calls["stage1"].append(args)
        torch.save(_stage1_output(), args.output_path)

    def stage2_must_not_run(*_args, **_kwargs):
        raise AssertionError("Stage 2 must not run with --skip_facilitator")

    def fake_stage3(args, _setup_logging=True):
        calls["stage3"].append(args)

    monkeypatch.setattr(stage1_mod, "main", fake_stage1)
    monkeypatch.setattr(stage2_mod, "main", stage2_must_not_run)
    monkeypatch.setattr(stage3_mod, "main", fake_stage3)
    return calls


def test_parses_with_a_pencl_only_weight_set(tmp_path, pencl_only_weight_set):
    args = parse_arguments(_argv(tmp_path, pencl_only_weight_set, "--skip_facilitator"))

    assert args.skip_facilitator
    assert args.pencl_weights == "weights/PenCL/only_pencl.bin"
    assert args.facilitator_weights is None
    assert args.facilitator_config is None


def test_facilitator_still_required_without_the_flag(tmp_path, pencl_only_weight_set):
    with pytest.raises(SystemExit):
        parse_arguments(_argv(tmp_path, pencl_only_weight_set))
    with pytest.raises(SystemExit):
        parse_arguments(_argv(tmp_path, pencl_only_weight_set,
                              "--facilitator_config", FACILITATOR_CONFIG))


def test_zc_is_a_copy_of_zt(tmp_path, pencl_only_weight_set, stub_stages):
    main(parse_arguments(_argv(tmp_path, pencl_only_weight_set, "--skip_facilitator")))

    out = tmp_path / "out"
    assert len(stub_stages["stage1"]) == 1
    assert stub_stages["stage1"][0].model_path == "weights/PenCL/only_pencl.bin"
    assert sorted(os.listdir(out)) == [
        "run.Facilitator_emb.pt", "run.PenCL_emb.pt", "run.build_manifest.json",
        "run.compiled_emb.hdf5", "run.run.log",
    ]

    expected = _stage1_output()
    embeddings = torch.load(out / "run.Facilitator_emb.pt", weights_only=False)
    assert torch.equal(embeddings["z_t"], expected["z_t"])
    assert torch.equal(embeddings["z_c"], embeddings["z_t"])
    assert embeddings["z_c"].data_ptr() != embeddings["z_t"].data_ptr()
    assert torch.equal(embeddings["z_p"], expected["z_p"])

    with h5py.File(out / "run.compiled_emb.hdf5") as compiled:
        z_c = torch.from_numpy(compiled["MMD_data/text_to_protein_embedding"][:])
    assert torch.equal(z_c, expected["z_t"])

    manifest = json.loads((out / "run.build_manifest.json").read_text())
    assert list(manifest["outputs"]) == ["pencl_output", "facilitator_output", "hdf5_output"]
    assert list(manifest["config_contents"]) == ["pencl"]
    assert not any("facilitator" in key for key in manifest["resolved_paths"])


def test_generate_samples_from_the_copy(tmp_path, stub_stages):
    main(parse_arguments([
        "-i", "tests/_data/stage1_inputs/sample_text_seqs1.csv",
        "-o", str(tmp_path / "out"),
        "--prefix", "run",
        "--pencl_weights", "weights/PenCL/only_pencl.bin",
        "--pencl_config", PENCL_CONFIG,
        "--device", "cpu",
        "--skip_facilitator",
        "--generate",
        "--proteoscribe_weights", "weights/ProteoScribe/x.bin",
        "--proteoscribe_config", PROTEOSCRIBE_CONFIG,
    ]))

    out = tmp_path / "out"
    assert len(stub_stages["stage3"]) == 1
    assert stub_stages["stage3"][0].input_path == str(out / "run.Facilitator_emb.pt")
    embeddings = torch.load(out / "run.Facilitator_emb.pt", weights_only=False)
    assert torch.equal(embeddings["z_c"], embeddings["z_t"])
    assert not (out / "run.compiled_emb.hdf5").exists()
