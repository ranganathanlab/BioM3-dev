"""Tests for how Stage 1 inference loads PenCL weights.

Regression cover for the silent no-op load: inference chose its loader from the
file extension, and anything not named ``.ckpt`` went through a bare
``strict=False`` load that kept a Lightning checkpoint's ``model.`` prefix. For
a Lightning checkpoint named ``.bin`` no key matched, nothing was reported, and
the run embedded with an untrained projection head.

Stub encoders stand in for ESM-2 and BioBERT so these run under
`pytest --quick` without downloaded weights.
"""

import logging
from argparse import Namespace

import pytest
import torch
from torch import nn

import biom3.Stage1.model as stage1_mod
from biom3.Stage1.io import load_pencl_weights
from biom3.Stage1.run_PenCL_inference import prepare_model


PROTEIN_EMB = 12
TEXT_EMB = 16
PROJ_DIM = 8
DEVICE = torch.device("cpu")


class _StubEncoder(nn.Module):
    """Stand-in for an encoder: a few parameters and one persistent buffer."""

    embedding_attr = None

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(4, getattr(args, self.embedding_attr))
        self.register_buffer("position_ids", torch.arange(4))


class _StubProteinEncoder(_StubEncoder):
    embedding_attr = "protein_encoder_embedding"


class _StubTextEncoder(_StubEncoder):
    embedding_attr = "text_encoder_embedding"


@pytest.fixture(autouse=True)
def stub_encoders(monkeypatch):
    monkeypatch.setattr(stage1_mod, "ProteinEncoder", _StubProteinEncoder)
    monkeypatch.setattr(stage1_mod, "TextEncoder", _StubTextEncoder)


@pytest.fixture
def io_log(caplog):
    """Capture biom3.Stage1.io records; BioM3 loggers do not propagate."""
    io_logger = logging.getLogger("biom3.Stage1.io")
    io_logger.addHandler(caplog.handler)
    yield caplog
    io_logger.removeHandler(caplog.handler)


def _args(model_type="pfam"):
    return Namespace(
        protein_encoder_embedding=PROTEIN_EMB,
        text_encoder_embedding=TEXT_EMB,
        proj_embedding_dim=PROJ_DIM,
        temperature=0.8,
        dropout=0.0,
        model_type=model_type,
    )


@pytest.fixture
def reference():
    """A model with distinctive weights, standing in for the trained one."""
    torch.manual_seed(1234)
    ref = stage1_mod.pfam_PEN_CL(_args())
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p))
    return ref


def _write(path, state_dict, *, lightning):
    """Save ``state_dict`` raw, or wrapped and prefixed as a PL wrapper does."""
    if lightning:
        payload = {"state_dict": {f"model.{k}": v for k, v in state_dict.items()}}
    else:
        payload = dict(state_dict)
    torch.save(payload, path)
    return str(path)


def _assert_same_weights(model, reference):
    expected = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        assert torch.equal(param, expected[name]), f"{name} was not loaded"


@pytest.mark.parametrize("lightning", [False, True])
@pytest.mark.parametrize("filename, load_from_checkpoint", [
    ("weights.bin", False),
    ("weights.ckpt", True),
])
def test_format_comes_from_the_file_not_the_name(
        tmp_path, reference, lightning, filename, load_from_checkpoint):
    """Raw and Lightning files both load, under either name and loader path."""
    path = _write(tmp_path / filename, reference.state_dict(), lightning=lightning)

    model = prepare_model(_args(), path, DEVICE, load_from_checkpoint)

    _assert_same_weights(model, reference)
    assert not model.training


@pytest.mark.parametrize("model_type, model_class", [
    ("pfam", stage1_mod.pfam_PEN_CL),
    ("default", stage1_mod.PEN_CL),
])
def test_checkpoint_path_follows_model_type(tmp_path, reference, model_type, model_class):
    path = _write(tmp_path / "weights.ckpt", reference.state_dict(), lightning=True)

    model = prepare_model(_args(model_type), path, DEVICE, True)

    assert type(model) is model_class
    _assert_same_weights(model, reference)


@pytest.mark.parametrize("load_from_checkpoint", [False, True])
def test_file_with_no_matching_keys_raises(tmp_path, reference, load_from_checkpoint):
    """Doubly prefixed keys match nothing once one prefix is stripped."""
    doubled = {f"model.{k}": v for k, v in reference.state_dict().items()}
    path = _write(tmp_path / "weights.bin", doubled, lightning=True)

    with pytest.raises(RuntimeError, match="did not populate"):
        prepare_model(_args(), path, DEVICE, load_from_checkpoint)


def test_file_missing_some_parameters_raises(tmp_path, reference):
    partial = {k: v for k, v in reference.state_dict().items()
               if not k.startswith("text_projection.")}
    path = _write(tmp_path / "weights.bin", partial, lightning=False)

    with pytest.raises(RuntimeError, match="text_projection"):
        prepare_model(_args(), path, DEVICE, False)


def test_missing_buffers_are_reported_not_fatal(tmp_path, reference, io_log):
    """Buffers such as BERT's position_ids come and go with library versions."""
    no_buffers = {k: v for k, v in reference.state_dict().items()
                  if not k.endswith("position_ids")}
    path = _write(tmp_path / "weights.bin", no_buffers, lightning=False)
    model = stage1_mod.pfam_PEN_CL(_args())

    with io_log.at_level(logging.WARNING, logger="biom3.Stage1.io"):
        load_pencl_weights(model, path, device=DEVICE)

    _assert_same_weights(model, reference)
    assert "2 buffer(s) not present" in io_log.text


def test_unused_keys_are_reported_not_fatal(tmp_path, reference, io_log):
    extra = dict(reference.state_dict())
    extra["text_encoder.model.bert.embeddings.position_ids"] = torch.arange(4)
    path = _write(tmp_path / "weights.bin", extra, lightning=True)
    model = stage1_mod.pfam_PEN_CL(_args())

    with io_log.at_level(logging.WARNING, logger="biom3.Stage1.io"):
        load_pencl_weights(model, path, device=DEVICE)

    _assert_same_weights(model, reference)
    assert "1 key(s)" in io_log.text and "not used by the model" in io_log.text


class _Tied(nn.Module):
    """One parameter under two names, as ESM-2 and BERT tie their output heads."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 3)
        self.head = nn.Linear(3, 5, bias=False)
        self.head.weight = self.embed.weight


@pytest.mark.parametrize("kept", ["embed.weight", "head.weight"])
def test_tied_parameter_loads_from_either_name(tmp_path, kept, io_log):
    reference = _Tied()
    path = _write(tmp_path / "weights.bin",
                  {kept: reference.embed.weight.detach().clone()}, lightning=False)
    model = _Tied()

    with io_log.at_level(logging.WARNING, logger="biom3.Stage1.io"):
        load_pencl_weights(model, path, device=DEVICE)

    assert torch.equal(model.embed.weight, reference.embed.weight)
    assert "not present" not in io_log.text


def test_tied_parameter_under_neither_name_raises(tmp_path):
    path = _write(tmp_path / "weights.bin", {"other": torch.zeros(1)}, lightning=False)

    with pytest.raises(RuntimeError, match="1/1 parameters"):
        load_pencl_weights(_Tied(), path, device=DEVICE)
