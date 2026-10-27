"""RL loads its frozen PenCL and Facilitator, and its policy, or stops.

The loaders used a non-strict load that only warned about missing keys, and
built an untrained model when no path was given. An RL run then embedded its
prompts with untrained weights, or optimised a randomly initialised policy,
without an error. Stub encoders stand in for ESM-2 and BioBERT.
"""

from argparse import Namespace

import pytest
import torch
from torch import nn

import biom3.Stage1.model as stage1_mod
from biom3.rl.io import load_facilitator_frozen, load_pencl_frozen, load_proteoscribe_trainable

CFG2 = Namespace(emb_dim=4, hid_dim=8, dropout=0.0)


class _StubEncoder(nn.Module):
    embedding_attr = None

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(4, getattr(args, self.embedding_attr))


class _StubProteinEncoder(_StubEncoder):
    embedding_attr = "protein_encoder_embedding"


class _StubTextEncoder(_StubEncoder):
    embedding_attr = "text_encoder_embedding"


@pytest.fixture
def cfg1(monkeypatch):
    monkeypatch.setattr(stage1_mod, "ProteinEncoder", _StubProteinEncoder)
    monkeypatch.setattr(stage1_mod, "TextEncoder", _StubTextEncoder)
    return Namespace(protein_encoder_embedding=12, text_encoder_embedding=16,
                     proj_embedding_dim=8, temperature=0.8, dropout=0.0, model_type="pfam")


def _save(path, state_dict, lightning=False):
    payload = ({"state_dict": {f"model.{k}": v for k, v in state_dict.items()}}
               if lightning else dict(state_dict))
    torch.save(payload, path)
    return str(path)


def _facilitator():
    torch.manual_seed(3)
    return stage1_mod.Facilitator(in_dim=4, hid_dim=8, out_dim=4, dropout=0.0)


def _assert_frozen_copy(model, reference):
    expected = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        assert torch.equal(param.cpu(), expected[name]), f"{name} was not loaded"
        assert not param.requires_grad
    assert not model.training


@pytest.mark.parametrize("lightning", [False, True])
def test_facilitator_loads_raw_and_lightning_files(tmp_path, lightning):
    reference = _facilitator()
    path = _save(tmp_path / "facilitator.bin", reference.state_dict(), lightning)

    _assert_frozen_copy(load_facilitator_frozen(CFG2, path, device="cpu"), reference)


def test_facilitator_with_missing_tensors_is_refused(tmp_path):
    state_dict = _facilitator().state_dict()
    state_dict.pop(sorted(state_dict)[0])
    path = _save(tmp_path / "facilitator.bin", state_dict)

    with pytest.raises(RuntimeError, match="did not populate"):
        load_facilitator_frozen(CFG2, path, device="cpu")


def test_facilitator_from_the_wrong_file_is_refused(tmp_path):
    path = _save(tmp_path / "other.bin", {"some.weight": torch.zeros(2)})

    with pytest.raises(RuntimeError, match="did not populate"):
        load_facilitator_frozen(CFG2, path, device="cpu")


@pytest.mark.parametrize("lightning", [False, True])
def test_pencl_loads_raw_and_lightning_files(tmp_path, cfg1, lightning):
    torch.manual_seed(5)
    reference = stage1_mod.pfam_PEN_CL(args=cfg1)
    path = _save(tmp_path / "pencl.bin", reference.state_dict(), lightning)

    _assert_frozen_copy(load_pencl_frozen(cfg1, path, device="cpu"), reference)


def test_pencl_from_the_wrong_file_is_refused(tmp_path, cfg1):
    path = _save(tmp_path / "facilitator.bin", _facilitator().state_dict())

    with pytest.raises(RuntimeError, match="did not populate"):
        load_pencl_frozen(cfg1, path, device="cpu")


@pytest.mark.parametrize("missing", [None, "None", ""])
def test_missing_paths_are_refused(cfg1, missing):
    with pytest.raises(ValueError, match="stage1_weights is required"):
        load_pencl_frozen(cfg1, missing)
    with pytest.raises(ValueError, match="stage2_weights is required"):
        load_facilitator_frozen(CFG2, missing)
    with pytest.raises(ValueError, match="stage3_init_weights is required"):
        load_proteoscribe_trainable(Namespace(), missing)
