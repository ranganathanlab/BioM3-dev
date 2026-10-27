"""normalize_zc is checked against what the weight set records.

A ProteoScribe trained on unit-length conditioning vectors has to be generated
from, and finetuned on, unit-length vectors, and the weights themselves do not
say how they were trained. A weight set can record it as
proteoscribe_trained_with_normalized_zc. The flag still decides what a run does;
the sampler, the pipeline's --generate and both Stage 3 training scripts warn
loudly when it disagrees with the record. RL cannot normalise, so it warns
whenever the record says the weights need it. Stages are stubbed.
"""

import json
from argparse import Namespace

import pytest
import torch

import biom3.Stage1.run_PenCL_inference as stage1_mod
import biom3.Stage3.run_PL_training as base
import biom3.Stage3.run_ProteoScribe_finetuning as run_ft
import biom3.Stage3.run_ProteoScribe_sample as stage3_mod
from biom3.core.weight_sets import check_normalize_zc, proteoscribe_trained_with_normalized_zc
from biom3.pipeline.embedding_pipeline import main as pipeline_main
from biom3.pipeline.embedding_pipeline import parse_arguments as parse_pipeline_args
from biom3.rl.io import configure_conditioning

WARNING = "Z_C NORMALISATION DOES NOT MATCH THE WEIGHTS"
PROTEOSCRIBE = "weights/ProteoScribe/s.bin"


def _weight_set(tmp_path, **extra):
    path = tmp_path / "weight_set.json"
    path.write_text(json.dumps({
        "pencl_weights": "weights/PenCL/p.bin",
        "facilitator_weights": "weights/Facilitator/f.bin",
        "proteoscribe_weights": "./" + PROTEOSCRIBE,
        **extra,
    }))
    return str(path)


@pytest.mark.parametrize("recorded", [True, False])
def test_reads_the_record(tmp_path, recorded):
    path = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=recorded)

    assert proteoscribe_trained_with_normalized_zc(path) is recorded
    assert proteoscribe_trained_with_normalized_zc(path, PROTEOSCRIBE) is recorded
    assert proteoscribe_trained_with_normalized_zc(path, "weights/ProteoScribe/other.bin") is None


def test_nothing_recorded(tmp_path):
    assert proteoscribe_trained_with_normalized_zc(None) is None
    assert proteoscribe_trained_with_normalized_zc(_weight_set(tmp_path)) is None


@pytest.mark.parametrize("name", ["run0_nm_base", "run1_base"])
def test_shipped_weight_sets_record_no_normalisation(name):
    assert proteoscribe_trained_with_normalized_zc(f"configs/weights/{name}.json") is False


@pytest.mark.parametrize("recorded, flag, warns", [
    (True, False, True), (False, True, True), (True, True, False), (False, False, False),
    (None, True, False), (None, False, False),
])
def test_check(tmp_path, recorded, flag, warns):
    extra = {} if recorded is None else {"proteoscribe_trained_with_normalized_zc": recorded}
    path = _weight_set(tmp_path, **extra)

    summary, warning = check_normalize_zc(path, PROTEOSCRIBE, flag)

    assert summary.startswith("Conditioning vector scaled to unit length: %s" % ("yes" if flag else "no"))
    assert (WARNING in warning) is warns


def test_check_says_when_the_record_is_for_another_file(tmp_path):
    path = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=True)

    summary, warning = check_normalize_zc(path, "weights/ProteoScribe/other.pth", False)

    assert warning == []
    assert ("record is for ./%s, not the weights in use (weights/ProteoScribe/other.pth)"
            % PROTEOSCRIBE) in summary


def test_sampler_accepts_a_weight_set():
    args = stage3_mod.parse_arguments(
        ["-i", "emb.pt", "-c", "cfg.json", "-m", PROTEOSCRIBE, "-o", "out.pt",
         "--weight_set", "configs/weights/run1_base.json"])

    assert args.weight_set == "configs/weights/run1_base.json"
    assert stage3_mod.parse_arguments(
        ["-i", "emb.pt", "-c", "cfg.json", "-m", PROTEOSCRIBE, "-o", "out.pt"]).weight_set is None


class _Stop(Exception):
    pass


@pytest.fixture
def stop_before_data(monkeypatch):
    def _stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(run_ft, "load_embedder_configs", lambda args: (Namespace(), Namespace()))
    monkeypatch.setattr(run_ft, "load_data", _stop)


@pytest.mark.parametrize("recorded, flag, warns", [
    (True, [], True),
    (False, ["--normalize_zc", "True"], True),
    (True, ["--normalize_zc", "True"], False),
    (False, [], False),
])
def test_finetuning_warns_when_the_flag_disagrees(tmp_path, stop_before_data, recorded, flag, warns):
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=recorded)
    args = run_ft.parse_arguments([
        "--record_schema", json.dumps({"sequence": {"from": "sequence"}}),
        "--output_root", str(tmp_path), "--run_id", "norm", "--checkpoints_folder", "checkpoints",
        "--device", "cpu", "--weight_set", weight_set, *flag,
    ])

    with pytest.raises(_Stop):
        run_ft.main(args)

    log = (tmp_path / "runs" / "norm" / "artifacts" / "run.log").read_text()
    assert (WARNING in log) is warns


def _run_pipeline(tmp_path, monkeypatch, weight_set, *extra):
    def fake_stage1(args, _setup_logging=True):
        torch.save({"z_t": torch.zeros(1, 2), "sequence": ["MK"], "acc_id": ["P1"]},
                   args.output_path)

    monkeypatch.setattr(stage1_mod, "main", fake_stage1)
    monkeypatch.setattr(stage3_mod, "main", lambda args, _setup_logging=True: None)
    pipeline_main(parse_pipeline_args([
        "-i", "tests/_data/stage1_inputs/sample_text_seqs1.csv",
        "-o", str(tmp_path / "out"), "--prefix", "run",
        "--weight_set", weight_set,
        "--pencl_config", "configs/inference/stage1_PenCL.json",
        "--device", "cpu", "--skip_facilitator", *extra,
    ]))
    return (tmp_path / "out" / "run.run.log").read_text()


GENERATE = ["--generate", "--proteoscribe_config", "configs/inference/stage3_ProteoScribe_sample.json"]


@pytest.mark.parametrize("recorded, flag, count", [
    (True, [], 2), (False, ["--normalize_zc"], 2), (True, ["--normalize_zc"], 0), (False, [], 0),
])
def test_pipeline_warns_at_the_start_and_the_end(tmp_path, monkeypatch, recorded, flag, count):
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=recorded)

    log = _run_pipeline(tmp_path, monkeypatch, weight_set, *GENERATE, *flag)

    assert log.count(WARNING) == count


def test_pipeline_that_does_not_generate_is_not_checked(tmp_path, monkeypatch):
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=True)

    log = _run_pipeline(tmp_path, monkeypatch, weight_set)

    assert WARNING not in log
    assert "Conditioning vector scaled to unit length" not in log


@pytest.mark.parametrize("recorded, warns", [(True, True), (False, False)])
def test_rl_warns_about_weights_that_need_normalising(tmp_path, recorded, warns):
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=recorded)
    args = Namespace(weight_set=weight_set, text_attention_mask=False,
                     stage1_weights=None, stage2_weights=None, stage3_init_weights=None)

    warning = configure_conditioning(args, Namespace())

    assert (WARNING in warning) is warns


def _train_args(tmp_path, *extra):
    return base.parse_arguments([
        "--output_root", str(tmp_path), "--run_id", "train",
        "--checkpoints_folder", "checkpoints", "--device", "cpu", *extra,
    ])


def test_train_stage3_takes_its_weights_from_a_weight_set(tmp_path):
    weight_set = _weight_set(tmp_path)

    assert _train_args(tmp_path).pretrained_weights is None
    assert _train_args(tmp_path, "--weight_set", weight_set).pretrained_weights == "./" + PROTEOSCRIBE
    explicit = _train_args(tmp_path, "--weight_set", weight_set, "--pretrained_weights", "mine.bin")
    assert explicit.pretrained_weights == "mine.bin"
    base.require_finetune_weights(_train_args(tmp_path, "--weight_set", weight_set, "--finetune", "True"))


@pytest.mark.parametrize("recorded, flag, warns", [
    (True, [], True),
    (False, ["--normalize_zc", "True"], True),
    (True, ["--normalize_zc", "True"], False),
    (False, [], False),
])
def test_train_stage3_warns_when_the_flag_disagrees(tmp_path, monkeypatch, recorded, flag, warns):
    def _stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(base, "load_data", _stop)
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=recorded)

    with pytest.raises(_Stop):
        base.main(_train_args(tmp_path, "--weight_set", weight_set, "--finetune", "True", *flag))

    log = (tmp_path / "runs" / "train" / "artifacts" / "run.log").read_text()
    assert (WARNING in log) is warns
    assert "Conditioning vector scaled to unit length" in log


def _config(tmp_path, **values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    return str(path)


def test_train_stage3_weight_set_on_the_command_line_beats_the_config(tmp_path, monkeypatch):
    config = _config(tmp_path, pretrained_weights="weights/ProteoScribe/old.pth")
    weight_set = _weight_set(tmp_path, proteoscribe_trained_with_normalized_zc=True)

    from_config = _train_args(tmp_path, "--config_path", config)
    from_weight_set = _train_args(tmp_path, "--config_path", config, "--weight_set", weight_set)
    explicit = _train_args(tmp_path, "--config_path", config, "--weight_set", weight_set,
                           "--pretrained_weights", "mine.bin")

    assert from_config.pretrained_weights == "weights/ProteoScribe/old.pth"
    assert from_weight_set.pretrained_weights == "./" + PROTEOSCRIBE
    assert explicit.pretrained_weights == "mine.bin"

    def _stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(base, "load_data", _stop)
    with pytest.raises(_Stop):
        base.main(_train_args(tmp_path, "--config_path", config, "--weight_set", weight_set,
                              "--finetune", "True"))
    log = (tmp_path / "runs" / "train" / "artifacts" / "run.log").read_text()
    assert "replaces the config's pretrained_weights: weights/ProteoScribe/old.pth" in log
    assert WARNING in log


def test_weight_set_named_in_the_config_does_not_override_its_paths(tmp_path):
    weight_set = _weight_set(tmp_path)
    config = _config(tmp_path, pretrained_weights="weights/ProteoScribe/old.pth",
                     weight_set=weight_set)

    assert _train_args(tmp_path, "--config_path", config).pretrained_weights == "weights/ProteoScribe/old.pth"


def test_finetuning_weight_set_on_the_command_line_beats_the_config(tmp_path):
    config = _config(tmp_path, pretrained_weights="cfg/s.bin", pencl_weights="cfg/p.bin",
                     facilitator_weights="cfg/f.bin",
                     record_schema={"sequence": {"from": "sequence"}})

    args = run_ft.parse_arguments(["--config_path", config, "--run_id", "x",
                                   "--weight_set", _weight_set(tmp_path),
                                   "--facilitator_weights", "mine/f.bin"])

    assert args.pretrained_weights == "./" + PROTEOSCRIBE
    assert args.pencl_weights == "weights/PenCL/p.bin"
    assert args.facilitator_weights == "mine/f.bin"


def test_rl_weight_set_on_the_command_line_beats_the_config(tmp_path):
    args = Namespace(weight_set=_weight_set(tmp_path), text_attention_mask=False,
                     stage1_weights="cfg/p.ckpt", stage2_weights="cfg/f.ckpt",
                     stage3_init_weights="cfg/s.bin",
                     _argv=["--weight_set", "ws.json", "--stage3_init_weights", "cfg/s.bin"])

    configure_conditioning(args, Namespace())

    assert args.stage1_weights == "weights/PenCL/p.bin"
    assert args.stage2_weights == "weights/Facilitator/f.bin"
    assert args.stage3_init_weights == "cfg/s.bin"
