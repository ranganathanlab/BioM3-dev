"""Tests for the frozen text->z_c embedder used by Stage 3 finetuning.

Regression cover for the silent no-op load: PenCL Lightning checkpoints are
keyed ``model.<submodule>...`` (PL wrappers do ``self.model = model``), so
loading one into the bare ``TextToZcEmbedder`` with ``strict=False`` and no
prefix stripping made every key unexpected and every parameter missing. The
build succeeded, logged nothing, and finetuning ran against a randomly
initialised projection head on stock BioBERT.

A stub TextEncoder stands in for BioBERT so these run under `pytest --quick`
without downloaded weights.
"""

import os
from argparse import Namespace

import pytest
import torch
from torch import nn

import biom3.Stage1.model as stage1_mod
from biom3.core.io import load_state_dict_unwrap_pl, strip_pl_model_prefix
from biom3.Stage3.finetune_embedder import (
    TextToZcEmbedder,
    build_text_to_zc_embedder,
)


TEXT_EMB = 16
PROJ_DIM = 8
VOCAB = 32
SEQ_LEN = 6


class _StubTextEncoder(nn.Module):
    """Stand-in for the BioBERT TextEncoder with the same call signature."""

    def __init__(self, args):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, args.text_encoder_embedding)
        self.dense = nn.Linear(args.text_encoder_embedding,
                               args.text_encoder_embedding)

    def forward(self, inputs, compute_logits=False):
        return self.dense(self.embed(inputs.squeeze(1)).mean(dim=1))


@pytest.fixture(autouse=True)
def stub_text_encoder(monkeypatch):
    monkeypatch.setattr(stage1_mod, "TextEncoder", _StubTextEncoder)


@pytest.fixture
def args():
    stage1 = Namespace(
        text_encoder_embedding=TEXT_EMB,
        proj_embedding_dim=PROJ_DIM,
        dropout=0.0,
    )
    stage2 = Namespace(emb_dim=PROJ_DIM, hid_dim=2 * PROJ_DIM, dropout=0.0)
    return stage1, stage2


@pytest.fixture
def reference(args):
    """An embedder with distinctive weights, standing in for the trained model."""
    torch.manual_seed(1234)
    ref = TextToZcEmbedder(*args)
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p))
    return ref


def _text_branch_keys(state_dict):
    return [k for k in state_dict if not k.startswith("facilitator.")]


def _write_pencl_ckpt(path, reference, *, pl_prefix=True, include_text=True):
    """Serialise a PenCL-shaped checkpoint: text branch + protein branch."""
    ref_sd = reference.state_dict()
    sd = {}
    if include_text:
        sd.update({k: v.clone() for k, v in ref_sd.items()
                   if k in _text_branch_keys(ref_sd)})
    # Protein branch: present in every real PenCL checkpoint, absent from this
    # module, and must be tolerated rather than raise.
    sd["protein_encoder.embed_tokens.weight"] = torch.randn(4, 4)
    sd["protein_projection.projection.weight"] = torch.randn(4, 4)
    if pl_prefix:
        sd = {"model." + k: v for k, v in sd.items()}
        torch.save({"state_dict": sd, "epoch": 3, "global_step": 187000}, path)
    else:
        torch.save(sd, path)
    return path


def _write_facilitator_ckpt(path, reference, *, pl_prefix=True):
    sd = {k[len("facilitator."):]: v.clone()
          for k, v in reference.state_dict().items()
          if k.startswith("facilitator.")}
    if pl_prefix:
        torch.save({"state_dict": {"model." + k: v for k, v in sd.items()}}, path)
    else:
        torch.save(sd, path)
    return path


# --------------------------- core loader helpers ----------------------------

class TestStripPlModelPrefix:
    def test_strips_prefix(self):
        out = strip_pl_model_prefix({"model.a.weight": 1, "model.b": 2})
        assert set(out) == {"a.weight", "b"}

    def test_noop_on_raw_state_dict(self):
        raw = {"main.0.weight": 1, "main.3.bias": 2}
        assert strip_pl_model_prefix(raw) is raw

    def test_partial_prefix_left_alone(self):
        out = strip_pl_model_prefix({"model.a": 1, "modelling.b": 2})
        assert set(out) == {"a", "modelling.b"}


class TestLoadStateDictUnwrapPl:
    def test_lightning_checkpoint(self, tmp_path):
        path = tmp_path / "x.ckpt"
        torch.save({"state_dict": {"model.w": torch.zeros(2)}, "epoch": 1}, path)
        assert list(load_state_dict_unwrap_pl(str(path))) == ["w"]

    def test_raw_state_dict(self, tmp_path):
        path = tmp_path / "x.bin"
        torch.save({"w": torch.zeros(2)}, path)
        assert list(load_state_dict_unwrap_pl(str(path))) == ["w"]

    def test_checkpoint_directory_prefers_last(self, tmp_path):
        d = tmp_path / "run.ckpt"
        d.mkdir()
        torch.save({"state_dict": {"model.from_last": torch.zeros(2)}},
                   d / "last.ckpt")
        torch.save({"state_dict": {"model.from_epoch": torch.zeros(2)}},
                   d / "epoch=01-step=10.ckpt")
        assert list(load_state_dict_unwrap_pl(str(d))) == ["from_last"]

    def test_empty_directory_raises(self, tmp_path):
        d = tmp_path / "empty.ckpt"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            load_state_dict_unwrap_pl(str(d))


# ------------------------------ embedder build ------------------------------

class TestBuildTextToZcEmbedder:
    def test_pl_prefixed_checkpoint_populates_text_branch(self, tmp_path, args,
                                                          reference):
        """The regression: `model.`-prefixed keys must actually land."""
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)

        embedder = build_text_to_zc_embedder(*args, str(pencl), str(fac))

        got, want = embedder.state_dict(), reference.state_dict()
        text_keys = _text_branch_keys(want)
        assert text_keys, "reference has no text-branch keys"
        for k in text_keys:
            assert torch.equal(got[k], want[k]), f"{k} did not load"

    def test_facilitator_loads_from_pl_checkpoint(self, tmp_path, args, reference):
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)

        embedder = build_text_to_zc_embedder(*args, str(pencl), str(fac))

        got, want = embedder.state_dict(), reference.state_dict()
        for k in want:
            if k.startswith("facilitator."):
                assert torch.equal(got[k], want[k]), f"{k} did not load"

    def test_zc_matches_reference(self, tmp_path, args, reference):
        """End-to-end: a correctly loaded embedder reproduces reference z_c."""
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)

        embedder = build_text_to_zc_embedder(*args, str(pencl), str(fac))

        input_ids = torch.randint(0, VOCAB, (3, SEQ_LEN))
        reference.eval()
        with torch.no_grad():
            assert torch.allclose(embedder(input_ids), reference(input_ids),
                                  atol=1e-6)

    def test_raw_unprefixed_weights_still_load(self, tmp_path, args, reference):
        """`.bin` raw state dicts have no `model.` prefix; stripping is a no-op."""
        pencl = _write_pencl_ckpt(tmp_path / "pencl.bin", reference, pl_prefix=False)
        fac = _write_facilitator_ckpt(tmp_path / "fac.bin", reference, pl_prefix=False)

        embedder = build_text_to_zc_embedder(*args, str(pencl), str(fac))

        got, want = embedder.state_dict(), reference.state_dict()
        for k in _text_branch_keys(want):
            assert torch.equal(got[k], want[k]), f"{k} did not load"

    def test_protein_branch_keys_tolerated(self, tmp_path, args, reference):
        """Protein-branch keys are expected in PenCL checkpoints and must not raise."""
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)
        sd = torch.load(pencl, map_location="cpu", weights_only=False)["state_dict"]
        assert any("protein_encoder" in k for k in sd)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)
        build_text_to_zc_embedder(*args, str(pencl), str(fac))

    def test_checkpoint_without_text_branch_raises(self, tmp_path, args, reference):
        """A no-op load must fail loudly rather than train on random weights."""
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference,
                                  include_text=False)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)

        with pytest.raises(RuntimeError, match="did not populate"):
            build_text_to_zc_embedder(*args, str(pencl), str(fac))

    def test_double_prefixed_checkpoint_raises(self, tmp_path, args, reference):
        """Guard the general no-match case, not just the specific prefix bug."""
        ref_sd = reference.state_dict()
        sd = {"encoder.wrapper." + k: v for k, v in ref_sd.items()}
        pencl = tmp_path / "pencl.ckpt"
        torch.save({"state_dict": sd}, pencl)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)

        with pytest.raises(RuntimeError, match="did not populate"):
            build_text_to_zc_embedder(*args, str(pencl), str(fac))

    def test_facilitator_mismatch_raises(self, tmp_path, args, reference):
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)
        fac = tmp_path / "fac.ckpt"
        torch.save({"state_dict": {"model.unrelated": torch.zeros(2)}}, fac)

        with pytest.raises(RuntimeError, match="did not populate"):
            build_text_to_zc_embedder(*args, str(pencl), str(fac))

    @pytest.mark.parametrize("missing", ["pencl", "facilitator"])
    def test_null_weights_raise(self, tmp_path, args, reference, missing):
        """A null path was previously a silent no-op; it must be rejected."""
        pencl = str(_write_pencl_ckpt(tmp_path / "pencl.ckpt", reference))
        fac = str(_write_facilitator_ckpt(tmp_path / "fac.ckpt", reference))
        if missing == "pencl":
            pencl = None
        else:
            fac = None

        with pytest.raises(ValueError, match="required for finetuning"):
            build_text_to_zc_embedder(*args, pencl, fac)

    def test_returned_module_is_frozen_and_eval(self, tmp_path, args, reference):
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)

        embedder = build_text_to_zc_embedder(*args, str(pencl), str(fac))

        assert not embedder.training
        assert all(not p.requires_grad for p in embedder.parameters())


# --------------------------- no-Facilitator path ----------------------------

class TestWithoutFacilitator:
    """``use_facilitator=False``: z_c is the raw z_t.

    A ProteoScribe pretrained on raw z_t has never seen facilitated vectors, so
    the front-end has to be able to stop after the projection head.
    """

    def test_no_facilitator_is_built(self, args):
        stage1, _ = args
        embedder = TextToZcEmbedder(stage1, None)
        assert embedder.facilitator is None
        assert not any(k.startswith("facilitator.")
                       for k in embedder.state_dict())

    def test_forward_returns_z_t(self, args):
        """The bypass output equals the projection head's own output."""
        stage1, _ = args
        torch.manual_seed(7)
        embedder = TextToZcEmbedder(stage1, None).eval()
        tokens = torch.randint(0, VOCAB, (3, 1, SEQ_LEN))

        with torch.no_grad():
            got = embedder(tokens)
            want = embedder.text_projection(embedder.text_encoder(tokens))

        assert got.shape == (3, PROJ_DIM)
        assert torch.equal(got, want)

    def test_builds_without_facilitator_weights(self, tmp_path, args, reference):
        stage1, _ = args
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)

        embedder = build_text_to_zc_embedder(
            stage1, None, str(pencl), None, use_facilitator=False)

        assert embedder.facilitator is None
        assert not embedder.training
        assert all(not p.requires_grad for p in embedder.parameters())

    def test_text_branch_still_required(self, tmp_path, args, reference):
        """Bypassing the Facilitator must not weaken the PenCL load."""
        stage1, _ = args
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference,
                                  include_text=False)

        with pytest.raises(RuntimeError, match="did not populate"):
            build_text_to_zc_embedder(
                stage1, None, str(pencl), None, use_facilitator=False)

    def test_null_pencl_still_raises(self, args):
        stage1, _ = args
        with pytest.raises(ValueError, match="required for finetuning"):
            build_text_to_zc_embedder(
                stage1, None, None, None, use_facilitator=False)

    def test_facilitator_weights_with_bypass_raise(self, tmp_path, args, reference):
        """Given both, refuse rather than silently ignore the weights."""
        stage1, _ = args
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)
        fac = _write_facilitator_ckpt(tmp_path / "fac.ckpt", reference)

        with pytest.raises(ValueError, match="would not be used"):
            build_text_to_zc_embedder(
                stage1, None, str(pencl), str(fac), use_facilitator=False)

    def test_missing_facilitator_weights_names_the_flag(self, tmp_path, args,
                                                        reference):
        """The default path's error points at the way out."""
        stage1, stage2 = args
        pencl = _write_pencl_ckpt(tmp_path / "pencl.ckpt", reference)

        with pytest.raises(ValueError, match="--use_facilitator False"):
            build_text_to_zc_embedder(stage1, stage2, str(pencl), None)


# ------------------- shipped run1_base checkpoint (opt-in) -------------------

_RUN1_PENCL = "weights/PenCL/run1_base_pencl.ckpt"
_RUN1_FACILITATOR = "weights/Facilitator/run1_base_facilitator.ckpt"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_RUN1_PENCL),
                    reason=f"{_RUN1_PENCL} not present")
def test_run1_base_checkpoint_is_pl_prefixed():
    """The shipped checkpoint the finetune configs point at is PL-prefixed.

    Pins the premise of the regression above: if a future export drops the
    prefix this test flags that the fixtures no longer match reality.
    """
    sd = torch.load(_RUN1_PENCL, map_location="cpu", mmap=True,
                    weights_only=False)["state_dict"]
    assert all(k.startswith("model.") for k in sd)
    stripped = strip_pl_model_prefix(sd)
    assert any(k.startswith("text_projection.") for k in stripped)
    assert any(k.startswith("text_encoder.") for k in stripped)
