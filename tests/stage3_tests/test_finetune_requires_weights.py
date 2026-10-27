"""Stage 3 finetuning refuses to start from random weights.

Both finetuning entry points used to accept a run with neither pretrained
weights nor a checkpoint to resume from: biom3_finetune_stage3 froze and trained
a randomly initialised ProteoScribe, and biom3_train_stage3 --finetune True
logged a warning and trained from scratch.
"""

import json
from argparse import Namespace

import pytest

import biom3.Stage3.run_ProteoScribe_finetuning as run_ft
from biom3.Stage3.run_PL_training import require_finetune_weights


def test_train_stage3_finetune_without_weights_is_refused():
    args = Namespace(finetune=True, pretrained_weights=None, resume_from_checkpoint=None)

    with pytest.raises(ValueError, match="randomly initialised"):
        require_finetune_weights(args)


@pytest.mark.parametrize("args", [
    Namespace(finetune=True, pretrained_weights="w.bin", resume_from_checkpoint=None),
    Namespace(finetune=True, pretrained_weights=None, resume_from_checkpoint="last.ckpt"),
    Namespace(finetune=False, pretrained_weights=None, resume_from_checkpoint=None),
])
def test_train_stage3_check_passes(args):
    require_finetune_weights(args)


class _Stop(Exception):
    pass


def _finetune_args(tmp_path, *extra):
    return run_ft.parse_arguments([
        "--record_schema", json.dumps({"sequence": {"from": "sequence"}}),
        "--output_root", str(tmp_path),
        "--run_id", "weights",
        "--checkpoints_folder", "checkpoints",
        "--device", "cpu",
        *extra,
    ])


@pytest.fixture
def stop_before_data(monkeypatch):
    def _stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(run_ft, "load_embedder_configs", lambda args: (Namespace(), Namespace()))
    monkeypatch.setattr(run_ft, "load_data", _stop)


def test_finetune_stage3_without_weights_is_refused(tmp_path, stop_before_data):
    with pytest.raises(ValueError, match="randomly initialised"):
        run_ft.main(_finetune_args(tmp_path))


@pytest.mark.parametrize("extra", [
    ["--pretrained_weights", "weights/ProteoScribe/x.bin"],
    ["--resume_from_checkpoint", "checkpoints/last.ckpt"],
])
def test_finetune_stage3_check_passes(tmp_path, stop_before_data, extra):
    with pytest.raises(_Stop):
        run_ft.main(_finetune_args(tmp_path, *extra))
