"""--text_attention_mask is checked against what the weight set records.

Whether to pass the caption attention mask is a property of the PenCL weights.
A weight set can record it as pencl_trained_with_text_attention_mask; the flag
still decides what the run does, and the pipeline warns loudly when the two
disagree. Stage 1 is stubbed, so these run under `pytest --quick`.
"""

import json

import pytest
import torch

import biom3.Stage1.run_PenCL_inference as stage1_mod
from biom3.core.weight_sets import pencl_trained_with_mask
from biom3.pipeline.embedding_pipeline import main, parse_arguments

PENCL = "weights/PenCL/some_pencl.bin"
WARNING = "CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS"


def _weight_set(tmp_path, **extra):
    path = tmp_path / "weight_set.json"
    path.write_text(json.dumps({"pencl_weights": PENCL, **extra}))
    return str(path)


@pytest.mark.parametrize("recorded", [True, False])
def test_reads_the_recorded_value(tmp_path, recorded):
    path = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=recorded)

    assert pencl_trained_with_mask(path) is recorded
    assert pencl_trained_with_mask(path, PENCL) is recorded


def test_nothing_recorded(tmp_path):
    assert pencl_trained_with_mask(None) is None
    assert pencl_trained_with_mask(_weight_set(tmp_path)) is None


def test_says_nothing_about_other_pencl_weights(tmp_path):
    path = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=True)

    assert pencl_trained_with_mask(path, "weights/PenCL/another.bin") is None


def test_recorded_value_must_be_a_boolean(tmp_path):
    path = _weight_set(tmp_path, pencl_trained_with_text_attention_mask="yes")

    with pytest.raises(ValueError, match="must be true or false"):
        pencl_trained_with_mask(path)


@pytest.mark.parametrize("name", ["run0_nm_base", "run1_base"])
def test_shipped_weight_sets_record_no_mask(name):
    assert pencl_trained_with_mask(f"configs/weights/{name}.json") is False


def _run(tmp_path, monkeypatch, weight_set, *extra):
    """Run the pipeline with Stage 1 stubbed and return its run log."""
    def fake_stage1(args, _setup_logging=True):
        torch.save({"z_t": torch.zeros(1, 2), "sequence": ["MK"], "acc_id": ["P1"]},
                   args.output_path)

    monkeypatch.setattr(stage1_mod, "main", fake_stage1)
    main(parse_arguments([
        "-i", "tests/_data/stage1_inputs/sample_text_seqs1.csv",
        "-o", str(tmp_path / "out"),
        "--prefix", "run",
        "--weight_set", weight_set,
        "--pencl_config", "configs/inference/stage1_PenCL.json",
        "--device", "cpu",
        "--skip_facilitator",
        *extra,
    ]))
    return (tmp_path / "out" / "run.run.log").read_text()


@pytest.mark.parametrize("recorded, flag, trained, run", [
    (True, [], "with the mask", "--text_attention_mask off"),
    (False, ["--text_attention_mask"], "without the mask", "--text_attention_mask on"),
])
def test_warns_when_the_flag_disagrees(tmp_path, monkeypatch, recorded, flag, trained, run):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=recorded)

    log = _run(tmp_path, monkeypatch, weight_set, *flag)

    assert log.count(WARNING) == 2, "expected at the start and again at the end"
    assert log.index(WARNING) < log.index("Stage 1: PenCL inference")
    assert log.rindex(WARNING) > log.index("Pipeline complete")
    assert trained in log and run in log


@pytest.mark.parametrize("recorded, flag", [
    (True, ["--text_attention_mask"]),
    (False, []),
])
def test_quiet_when_the_flag_agrees(tmp_path, monkeypatch, recorded, flag):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=recorded)

    assert WARNING not in _run(tmp_path, monkeypatch, weight_set, *flag)


@pytest.mark.parametrize("flag", [[], ["--text_attention_mask"]])
def test_quiet_when_nothing_is_recorded(tmp_path, monkeypatch, flag):
    log = _run(tmp_path, monkeypatch, _weight_set(tmp_path), *flag)

    assert WARNING not in log
    assert "how PenCL was trained is not recorded" in log


def test_quiet_when_pencl_weights_are_overridden(tmp_path, monkeypatch):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=True)

    log = _run(tmp_path, monkeypatch, weight_set, "--pencl_weights", "weights/PenCL/another.bin")

    assert WARNING not in log
