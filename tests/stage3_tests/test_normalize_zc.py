"""--normalize_zc scales the conditioning vector to unit length.

The option is off by default. With it on, Stage 3 training and validation give
the model the conditioning vector divided by its length, after any z_p blend,
and the sampler does the same so that a model trained this way is generated
from the way it was trained.
"""

import json
from argparse import Namespace

import pytest
import torch
from torch import nn

import biom3.Stage3.run_ProteoScribe_finetuning as run_ft
from biom3.pipeline.embedding_pipeline import _build_stage3_argv
from biom3.pipeline.embedding_pipeline import parse_arguments as parse_pipeline_args
from biom3.Stage3.PL_wrapper import PL_ProtARDM, PL_ProtARDM_Finetune
from biom3.Stage3.preprocess import encode_protein_sequence
from biom3.Stage3.run_ProteoScribe_sample import normalize_conditioning
from biom3.Stage3.run_ProteoScribe_sample import parse_arguments as parse_sample_args


class _RecordingModel(nn.Module):
    """Stand-in for ProteoScribe that keeps the conditioning it was given."""

    def __init__(self, num_classes=29):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.seen = []

    def forward(self, x, t, y_c):
        self.seen.append(y_c)
        return self.bias.view(1, -1, 1).expand(x.size(0), -1, x.size(-1))


def _quiet(module):
    module.performance_step = lambda **kwargs: tuple(torch.tensor(0.0) for _ in range(10))
    module.log = lambda *args, **kwargs: None
    return module


def _sequences():
    return torch.tensor([encode_protein_sequence(s, 4) for s in ("ACD", "EFGHIK")])


Z = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0]])


@pytest.mark.parametrize("args", [{}, {"normalize_zc": False}])
def test_conditioning_is_untouched_by_default(args):
    model = _RecordingModel()
    module = _quiet(PL_ProtARDM(args=Namespace(task="proteins", **args), model=model))

    module.common_step([_sequences(), Z.clone()], 0, stage="train")

    assert torch.equal(model.seen[0], Z)


@pytest.mark.parametrize("stage", ["train", "val"])
def test_conditioning_is_scaled_to_unit_length(stage):
    model = _RecordingModel()
    module = _quiet(PL_ProtARDM(args=Namespace(task="proteins", normalize_zc=True), model=model))

    module.common_step([_sequences(), Z.clone()], 0, stage=stage)

    given = model.seen[0]
    assert torch.allclose(given.norm(dim=-1), torch.ones(2))
    assert torch.allclose(given, torch.tensor([[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]))


class _Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, input_ids):
        return self.scale * torch.ones(input_ids.size(0), 4)


def test_finetuning_normalises_what_the_embedder_returns():
    model = _RecordingModel()
    module = _quiet(PL_ProtARDM_Finetune(
        args=Namespace(task="proteins", normalize_zc=True), model=model, embedder=_Embedder()))

    batch = module.on_after_batch_transfer((_sequences(), torch.ones(2, 3, dtype=torch.long)), 0)
    module.common_step(batch, 0, stage="train")

    assert torch.allclose(batch[1].norm(dim=-1), torch.full((2,), 10.0))
    assert torch.allclose(model.seen[0].norm(dim=-1), torch.ones(2))


def _train_args(tmp_path, *extra):
    return run_ft.parse_arguments([
        "--record_schema", json.dumps({"sequence": {"from": "sequence"}}),
        "--output_root", str(tmp_path), "--run_id", "norm", *extra,
    ])


@pytest.mark.parametrize("extra, expected", [
    ([], False), (["--normalize_zc", "True"], True), (["--normalize_zc", "false"], False),
])
def test_training_option_parses(tmp_path, extra, expected):
    assert _train_args(tmp_path, *extra).normalize_zc is expected


def test_sampler_normalises_each_vector():
    out = normalize_conditioning(Z)

    assert torch.allclose(out.norm(dim=-1), torch.ones(2))
    assert torch.allclose(out[0], torch.tensor([0.6, 0.8, 0.0, 0.0]))
    assert torch.isfinite(normalize_conditioning(torch.zeros(1, 4))).all()


@pytest.mark.parametrize("extra, expected", [([], False), (["--normalize_zc"], True)])
def test_sampler_option_parses(extra, expected):
    args = parse_sample_args(["-i", "emb.pt", "-c", "cfg.json", "-m", "w.bin", "-o", "out.pt"] + extra)

    assert args.normalize_zc is expected


@pytest.mark.parametrize("extra, expected", [([], False), (["--normalize_zc"], True)])
def test_pipeline_hands_the_option_to_the_sampler(tmp_path, extra, expected):
    args = parse_pipeline_args([
        "-i", "in.csv", "-o", str(tmp_path), "--prefix", "run",
        "--pencl_weights", "p.bin", "--facilitator_weights", "f.bin",
        "--pencl_config", "configs/inference/stage1_PenCL.json",
        "--facilitator_config", "configs/inference/stage2_Facilitator.json",
        "--generate", "--proteoscribe_weights", "s.bin",
        "--proteoscribe_config", "configs/inference/stage3_ProteoScribe_sample.json",
    ] + extra)

    argv = _build_stage3_argv(args, "emb.pt", "gen.pt")

    assert ("--normalize_zc" in argv) is expected
    assert parse_sample_args(argv).normalize_zc is expected
