"""Tests for alpha-blended z_c/z_p conditioning in Stage 3 finetuning and sampling.

ProteoScribe is normally conditioned on z_c (text). The blend conditions on
``y = alpha * z_p + (1 - alpha) * z_c``, where z_p is PenCL's protein-branch
embedding of the sequence. Both live in the same joint space, so a convex
combination is meaningful.

Covers the alpha spec parsing, the sampling schedule (including the invariant
that validation never resamples), the blend arithmetic, the z_p lookup keying,
and that the protein-branch loader refuses a checkpoint that matches nothing.
A stub ESM-2 keeps these runnable under `pytest --quick`.
"""

from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

import json

import biom3.Stage1.model as stage1_mod
import biom3.Stage3.PL_wrapper as plw
from biom3.Stage3.PL_wrapper import (
    ALPHA_BLEND, alpha_spec_uses_zp, normalize_alpha_spec,
)
from biom3.Stage3.preprocess import make_seq_caption_collate_fn
from biom3.Stage3.run_ProteoScribe_sample import blend_conditioning


PROT_EMB = 20
PROJ_DIM = 8


# ----------------------------- alpha spec parsing ----------------------------

class TestNormalizeAlphaSpec:
    @pytest.mark.parametrize("value,expected", [
        ("zc", 0.0), ("zp", 1.0), ("ZC", 0.0), ("  zp  ", 1.0),
        ("blend", ALPHA_BLEND), ("BLEND", ALPHA_BLEND),
        ("0", 0.0), ("1", 1.0), ("0.25", 0.25), (0.5, 0.5), (1, 1.0),
    ])
    def test_accepted(self, value, expected):
        assert normalize_alpha_spec(value) == expected

    @pytest.mark.parametrize("value", ["turbo", "z_p", "", "none"])
    def test_rejects_unknown_names(self, value):
        with pytest.raises(ValueError, match="train_alpha must be"):
            normalize_alpha_spec(value)

    @pytest.mark.parametrize("value", ["-0.1", "1.5", 2.0, -1])
    def test_rejects_out_of_range(self, value):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            normalize_alpha_spec(value)

    def test_numeric_string_is_a_constant_not_the_schedule(self):
        """'0.5' must mean a constant 0.5, never the blend schedule."""
        assert normalize_alpha_spec("0.5") == 0.5
        assert normalize_alpha_spec("0.5") != ALPHA_BLEND


class TestAlphaSpecUsesZp:
    @pytest.mark.parametrize("spec,expected", [
        (0.0, False), (1.0, True), (0.25, True), (ALPHA_BLEND, True),
    ])
    def test_detection(self, spec, expected):
        assert alpha_spec_uses_zp(spec) is expected

    def test_zc_needs_no_zp(self):
        assert alpha_spec_uses_zp(normalize_alpha_spec("zc")) is False


# ------------------------------- collate wiring ------------------------------

class _FakeTokenizer:
    def __call__(self, prompts, *, truncation, max_length, padding,
                          return_tensors, return_attention_mask,
                          return_token_type_ids):
        return {"input_ids": torch.zeros(len(prompts), max_length, dtype=torch.long)}


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(path):
        return _FakeTokenizer()


class TestCollateIncludeSequences:
    BATCH = [{"sequence": "ACDEF", "caption": "x"},
             {"sequence": "GH-IK", "caption": "y"}]

    def _collate(self, **kw):
        return make_seq_caption_collate_fn(
            text_tokenizer=_FakeTokenizer(), text_max_length=8, image_size=4, **kw,
        )

    def test_default_stays_a_two_tuple(self):
        """The established batch contract must not change by default."""
        assert len(self._collate()(self.BATCH)) == 2

    def test_opt_in_appends_raw_sequences(self):
        num_seqs, input_ids, seqs = self._collate(include_sequences=True)(self.BATCH)
        assert num_seqs.shape == (2, 16)
        assert input_ids.shape == (2, 8)
        assert seqs == ["ACDEF", "GH-IK"]  # verbatim, gaps intact

    def test_sequences_are_lookup_keys_not_tensors(self):
        _, _, seqs = self._collate(include_sequences=True)(self.BATCH)
        assert all(isinstance(s, str) for s in seqs)


# --------------------------- alpha sampling schedule -------------------------

class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)


class _FakeEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, input_ids):
        return torch.ones(input_ids.shape[0], self.dim)


_UNSET = object()


def _make_wrapper(train_alpha=0.0, eval_alpha=_UNSET, zp_lookup=None, dim=PROJ_DIM):
    args = Namespace(
        lr=1e-4, choose_optim="AdamW", weight_decay=0.0, scheduler_gamma=None,
        epochs=1, traindata_len=1, acc_grad_batches=1, batch_size=2,
        diffusion_steps=4, image_size=4, num_classes=25, text_emb_dim=dim,
        scale_learning_rate=False, seed=0,
    )
    # Leave eval_alpha at the constructor default unless a test sets it.
    kw = {} if eval_alpha is _UNSET else {"eval_alpha": eval_alpha}
    return plw.PL_ProtARDM_Finetune(
        args=args, model=_StubModel(), embedder=_FakeEmbedder(dim),
        zp_lookup=zp_lookup, train_alpha=train_alpha, **kw,
    )


class TestDataModuleUniqueSequences:
    """The z_p precompute keys off unique_sequences(); those keys must match
    the raw sequences the collate emits, or the batch lookup misses."""

    RECORDS = [
        {"sequence": "ACDEFG", "fields": {"n": "a"}, "sequence_length": 6},
        {"sequence": "GHIKLM", "fields": {"n": "b"}, "sequence_length": 6},
        {"sequence": "ACDEFG", "fields": {"n": "c"}, "sequence_length": 6},  # dup seq
    ] * 8
    SCHEMA = {
        "sequence": {"from": "sequence"},
        "caption": {"compose": "fields_to_caption",
                    "args": {"fields_key": "fields", "shuffle": False}},
    }

    def _dm(self, tmp_path, monkeypatch, needs_unique):
        monkeypatch.setattr(plw, "AutoTokenizer", _FakeAutoTokenizer)
        path = tmp_path / "d.jsonl"
        with open(path, "w") as fh:
            for r in self.RECORDS:
                fh.write(json.dumps(r) + "\n")
        dm = plw.GeneralizedDataModule(
            jsonl_path=str(path), record_schema=self.SCHEMA,
            text_model_path="unused", text_max_length=16, batch_size=4,
            num_workers=0, valid_size=0.25, seed=1, diffusion_steps=8, image_size=4,
            needs_unique_sequences=needs_unique,
        )
        dm.setup()
        return dm

    def test_unique_sequences_deduplicated(self, tmp_path, monkeypatch):
        dm = self._dm(tmp_path, monkeypatch, needs_unique=True)
        uniq = dm.unique_sequences()
        assert set(uniq) == {"ACDEFG", "GHIKLM"}
        assert len(uniq) == len(set(uniq))

    def test_collate_emits_keys_present_in_lookup(self, tmp_path, monkeypatch):
        dm = self._dm(tmp_path, monkeypatch, needs_unique=True)
        lookup = set(dm.unique_sequences())
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 3
        for seq in batch[2]:
            assert seq in lookup, f"collate emitted {seq!r} absent from z_p lookup"

    def test_disabled_leaves_two_tuple_and_raises(self, tmp_path, monkeypatch):
        dm = self._dm(tmp_path, monkeypatch, needs_unique=False)
        assert len(next(iter(dm.train_dataloader()))) == 2
        with pytest.raises(RuntimeError, match="needs_unique_sequences=True"):
            dm.unique_sequences()


class TestTrainAlpha:
    def test_constant_spec(self):
        m = _make_wrapper(train_alpha=0.25)
        a = m._train_alpha(6, torch.device("cpu"))
        assert a.shape == (6, 1)
        assert torch.equal(a, torch.full((6, 1), 0.25))

    def test_zc_is_all_zero(self):
        m = _make_wrapper(train_alpha=normalize_alpha_spec("zc"))
        assert torch.equal(m._train_alpha(4, torch.device("cpu")), torch.zeros(4, 1))

    def test_zp_is_all_one(self):
        m = _make_wrapper(train_alpha=normalize_alpha_spec("zp"))
        assert torch.equal(m._train_alpha(4, torch.device("cpu")), torch.ones(4, 1))

    def test_blend_schedule_proportions(self):
        """{alpha=1: .25, alpha=0: .25, U(0,1): .5}."""
        torch.manual_seed(0)
        m = _make_wrapper(train_alpha=ALPHA_BLEND)
        a = m._train_alpha(20000, torch.device("cpu")).squeeze(1)
        assert ((a >= 0.0) & (a <= 1.0)).all()
        frac_one = (a == 1.0).float().mean().item()
        frac_zero = (a == 0.0).float().mean().item()
        assert frac_one == pytest.approx(0.25, abs=0.02)
        assert frac_zero == pytest.approx(0.25, abs=0.02)
        interior = a[(a > 0.0) & (a < 1.0)]
        assert interior.numel() / a.numel() == pytest.approx(0.5, abs=0.02)

    def test_blend_varies_per_example(self):
        torch.manual_seed(1)
        m = _make_wrapper(train_alpha=ALPHA_BLEND)
        a = m._train_alpha(64, torch.device("cpu"))
        assert a.unique().numel() > 1


class TestEvalAlpha:
    def _seqs(self, n):
        return [f"SEQ{i:04d}" for i in range(n)]

    def test_spread_varies_across_datapoints(self):
        """The whole point: not a single alpha for the whole val set."""
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="spread")
        a = m._eval_alpha(self._seqs(200), torch.device("cpu"))
        assert a.shape == (200, 1)
        assert ((a >= 0.0) & (a < 1.0)).all()
        assert a.unique().numel() > 150  # genuinely spread, not clustered

    def test_spread_covers_the_range(self):
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="spread")
        a = m._eval_alpha(self._seqs(500), torch.device("cpu")).squeeze(1)
        assert a.min() < 0.1 and a.max() > 0.9      # reaches both ends
        assert 0.35 < a.mean() < 0.65               # ~centred, ~uniform

    def test_spread_is_deterministic_across_epochs(self):
        """Same examples -> identical alpha every epoch, so val loss is stable."""
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="spread")
        seqs = self._seqs(64)
        a1 = m._eval_alpha(seqs, torch.device("cpu"))
        a2 = m._eval_alpha(seqs, torch.device("cpu"))
        assert torch.equal(a1, a2)

    def test_spread_is_keyed_on_sequence_not_position(self):
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="spread")
        a = m._eval_alpha(["ACD", "GHI", "ACD"], torch.device("cpu"))
        assert torch.equal(a[0], a[2])          # same sequence -> same alpha
        assert not torch.equal(a[0], a[1])      # different sequence -> different

    def test_spread_independent_of_batching(self):
        """A val example's alpha must not depend on its batch neighbours (DDP-safe)."""
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="spread")
        seqs = self._seqs(10)
        full = m._eval_alpha(seqs, torch.device("cpu"))
        halves = torch.cat([m._eval_alpha(seqs[:4], torch.device("cpu")),
                            m._eval_alpha(seqs[4:], torch.device("cpu"))])
        assert torch.equal(full, halves)

    def test_default_is_spread(self):
        """Constructed without eval_alpha -> spread, not a single point."""
        m = _make_wrapper(train_alpha=ALPHA_BLEND)  # helper leaves it unset
        assert m.eval_alpha == "spread"

    def test_constant_applies_one_alpha(self):
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha=0.5)
        a = m._eval_alpha(self._seqs(8), torch.device("cpu"))
        assert torch.equal(a, torch.full((8, 1), 0.5))

    def test_constant_string_alias(self):
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="zp")
        a = m._eval_alpha(self._seqs(4), torch.device("cpu"))
        assert torch.equal(a, torch.ones(4, 1))

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            _make_wrapper(eval_alpha=1.5)

    def test_rejects_blend_schedule(self):
        with pytest.raises(ValueError, match="eval_alpha must be"):
            _make_wrapper(eval_alpha=ALPHA_BLEND)


class TestFinetuneArgConversions:
    """train_alpha / eval_alpha normalization through the real arg parser."""

    @staticmethod
    def _args(**overrides):
        import json as _json
        import biom3.Stage3.run_ProteoScribe_finetuning as run_ft
        argv = ["--record_schema", _json.dumps({"sequence": {"from": "sequence"}})]
        for k, v in overrides.items():
            argv += [f"--{k}", str(v)]
        return run_ft.parse_arguments(argv)

    def test_default_is_text_only(self):
        args = self._args()
        assert args.train_alpha == 0.0
        assert alpha_spec_uses_zp(args.train_alpha) is False

    @pytest.mark.parametrize("value,expected", [
        ("zp", 1.0), ("blend", ALPHA_BLEND), ("0.3", 0.3), ("0", 0.0),
    ])
    def test_normalized_on_the_namespace(self, value, expected):
        assert self._args(train_alpha=value).train_alpha == expected

    def test_invalid_train_alpha_rejected(self):
        with pytest.raises(ValueError, match="train_alpha must be"):
            self._args(train_alpha="sequence")

    def test_eval_alpha_default_is_spread(self):
        assert self._args().eval_alpha == "spread"

    @pytest.mark.parametrize("value,expected", [
        ("spread", "spread"), ("zc", 0.0), ("zp", 1.0), ("0.5", 0.5),
    ])
    def test_eval_alpha_normalized(self, value, expected):
        assert self._args(eval_alpha=value).eval_alpha == expected

    def test_eval_alpha_rejects_a_schedule(self):
        with pytest.raises(ValueError, match="eval_alpha must be"):
            self._args(eval_alpha="blend")


# ------------------------------ blend in the wrapper -------------------------

class TestOnAfterBatchTransferBlend:
    def _batch(self, seqs, dim=PROJ_DIM):
        return (torch.zeros(len(seqs), 16), torch.zeros(len(seqs), 8, dtype=torch.long),
                list(seqs))

    def test_no_lookup_returns_zc_untouched(self):
        m = _make_wrapper()
        num_seqs = torch.zeros(2, 16)
        out = m.on_after_batch_transfer((num_seqs, torch.zeros(2, 8, dtype=torch.long)), 0)
        assert len(out) == 2
        assert torch.equal(out[1], torch.ones(2, PROJ_DIM))  # the fake z_c

    def test_alpha_one_yields_pure_zp(self):
        lookup = {"AC": torch.full((PROJ_DIM,), 5.0),
                  "DE": torch.full((PROJ_DIM,), 7.0)}
        m = _make_wrapper(train_alpha=1.0, zp_lookup=lookup)
        m.train()
        _, y = m.on_after_batch_transfer(self._batch(["AC", "DE"]), 0)
        assert torch.equal(y[0], torch.full((PROJ_DIM,), 5.0))
        assert torch.equal(y[1], torch.full((PROJ_DIM,), 7.0))

    def test_alpha_zero_yields_pure_zc_even_with_lookup(self):
        lookup = {"AC": torch.full((PROJ_DIM,), 5.0)}
        m = _make_wrapper(train_alpha=0.0, zp_lookup=lookup)
        m.train()
        _, y = m.on_after_batch_transfer(self._batch(["AC"]), 0)
        assert torch.equal(y, torch.ones(1, PROJ_DIM))

    def test_constant_alpha_is_a_convex_combination(self):
        lookup = {"AC": torch.full((PROJ_DIM,), 3.0)}
        m = _make_wrapper(train_alpha=0.25, zp_lookup=lookup)
        m.train()
        _, y = m.on_after_batch_transfer(self._batch(["AC"]), 0)
        # 0.25 * 3.0 + 0.75 * 1.0
        assert torch.allclose(y, torch.full((1, PROJ_DIM), 1.5))

    def test_lookup_keyed_on_gapped_sequence(self):
        """Keys are the raw record strings, gaps included."""
        lookup = {"A-C": torch.full((PROJ_DIM,), 9.0)}
        m = _make_wrapper(train_alpha=1.0, zp_lookup=lookup)
        m.train()
        _, y = m.on_after_batch_transfer(self._batch(["A-C"]), 0)
        assert torch.equal(y, torch.full((1, PROJ_DIM), 9.0))

    def test_missing_sequences_in_batch_raises(self):
        """A 2-tuple batch with a lookup set is a wiring error, not silent z_c."""
        m = _make_wrapper(train_alpha=1.0, zp_lookup={"AC": torch.zeros(PROJ_DIM)})
        with pytest.raises(RuntimeError, match="needs_unique_sequences"):
            m.on_after_batch_transfer(
                (torch.zeros(1, 16), torch.zeros(1, 8, dtype=torch.long)), 0)

    def test_validation_constant_eval_alpha(self):
        lookup = {"AC": torch.full((PROJ_DIM,), 3.0)}
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha=0.0, zp_lookup=lookup)
        m.eval()
        _, y1 = m.on_after_batch_transfer(self._batch(["AC"]), 0)
        _, y2 = m.on_after_batch_transfer(self._batch(["AC"]), 0)
        assert torch.equal(y1, y2)
        assert torch.equal(y1, torch.ones(1, PROJ_DIM))  # pure z_c at eval_alpha=0

    def test_validation_spread_is_deterministic_blend(self):
        """Under the default spread, val output is a genuine blend yet stable."""
        from biom3.Stage3.PL_wrapper import deterministic_alpha
        lookup = {"AC": torch.full((PROJ_DIM,), 3.0)}
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="spread", zp_lookup=lookup)
        m.eval()
        _, y1 = m.on_after_batch_transfer(self._batch(["AC"]), 0)
        _, y2 = m.on_after_batch_transfer(self._batch(["AC"]), 0)
        assert torch.equal(y1, y2)  # deterministic across epochs
        a = deterministic_alpha("AC")
        expected = a * 3.0 + (1 - a) * 1.0
        assert torch.allclose(y1, torch.full((1, PROJ_DIM), expected))
        assert 0.0 < a < 1.0  # a real blend, not a degenerate endpoint

    def test_training_and_validation_differ_under_blend(self):
        """Training resamples; validation is fixed. They should not coincide."""
        lookup = {f"S{i}": torch.full((PROJ_DIM,), 3.0) for i in range(64)}
        seqs = list(lookup)
        m = _make_wrapper(train_alpha=ALPHA_BLEND, eval_alpha="spread", zp_lookup=lookup)
        m.eval()
        _, ve1 = m.on_after_batch_transfer(self._batch(seqs), 0)
        _, ve2 = m.on_after_batch_transfer(self._batch(seqs), 0)
        assert torch.equal(ve1, ve2)  # val stable
        m.train()
        torch.manual_seed(0)
        _, vt1 = m.on_after_batch_transfer(self._batch(seqs), 0)
        torch.manual_seed(1)
        _, vt2 = m.on_after_batch_transfer(self._batch(seqs), 0)
        assert not torch.equal(vt1, vt2)  # train resamples


# ------------------------------ sampling-side blend --------------------------

class TestBlendConditioning:
    def _ds(self, n=3, dim=4):
        return {"z_c": torch.ones(n, dim), "z_p": torch.full((n, dim), 5.0)}

    def test_alpha_zero_returns_zc_object(self):
        ds = self._ds()
        assert blend_conditioning(ds, 0.0) is ds["z_c"]

    def test_alpha_one_returns_zp_values(self):
        assert torch.equal(blend_conditioning(self._ds(), 1.0),
                           torch.full((3, 4), 5.0))

    def test_midpoint(self):
        assert torch.allclose(blend_conditioning(self._ds(), 0.5),
                              torch.full((3, 4), 3.0))

    @pytest.mark.parametrize("alpha", [-0.01, 1.01, 2.0])
    def test_out_of_range_rejected(self, alpha):
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            blend_conditioning(self._ds(), alpha)

    def test_alpha_zero_works_without_zp(self):
        """Text-only generation must not require z_p to be present."""
        ds = {"z_c": torch.ones(2, 4)}
        assert torch.equal(blend_conditioning(ds, 0.0), torch.ones(2, 4))

    def test_missing_zp_raises_with_available_keys(self):
        ds = {"z_c": torch.ones(2, 4), "z_t": torch.ones(2, 4)}
        with pytest.raises(KeyError, match="z_t"):
            blend_conditioning(ds, 0.5)

    def test_shape_mismatch_raises(self):
        ds = {"z_c": torch.ones(3, 4), "z_p": torch.ones(2, 4)}
        with pytest.raises(ValueError, match="row-aligned"):
            blend_conditioning(ds, 0.5)


# --------------------------- protein-branch embedder -------------------------

class _StubAlphabet:
    def get_batch_converter(self):
        def convert(batch):
            longest = max(len(s) for _, s in batch)
            toks = torch.zeros(len(batch), longest + 2, dtype=torch.long)
            for i, (_, s) in enumerate(batch):
                for j, ch in enumerate(s):
                    toks[i, j + 1] = (ord(ch) % 20) + 1
            return None, None, toks
        return convert


class _StubProteinEncoder(nn.Module):
    """Stand-in for ESM-2 with the attributes PenCL's protein branch exposes.

    Pools over non-pad positions only. Real ESM-2 masks padding in attention
    and PenCL reads the CLS token, so its output does not depend on what else
    shares the batch; the stub has to match or batching would change results.
    """

    def __init__(self, args):
        super().__init__()
        self.alphabet = _StubAlphabet()
        self.embed = nn.Embedding(32, args.protein_encoder_embedding)
        self.dense = nn.Linear(args.protein_encoder_embedding,
                               args.protein_encoder_embedding)

    def forward(self, tokens, compute_logits=False):
        mask = (tokens != 0).unsqueeze(-1).to(self.embed.weight.dtype)
        pooled = (self.embed(tokens) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.dense(pooled)


@pytest.fixture
def stub_protein_encoder(monkeypatch):
    monkeypatch.setattr(stage1_mod, "ProteinEncoder", _StubProteinEncoder)


@pytest.fixture
def stage1_args():
    return Namespace(protein_encoder_embedding=PROT_EMB,
                     proj_embedding_dim=PROJ_DIM, dropout=0.0)


class TestProteinToZpEmbedder:
    def _reference(self, stage1_args):
        from biom3.Stage3.finetune_embedder import ProteinToZpEmbedder
        torch.manual_seed(7)
        ref = ProteinToZpEmbedder(stage1_args)
        with torch.no_grad():
            for p in ref.parameters():
                p.add_(torch.randn_like(p))
        return ref

    def _write_ckpt(self, path, ref, *, include_protein=True):
        sd = dict(ref.state_dict()) if include_protein else {}
        # Text-branch keys are present in every real PenCL checkpoint.
        sd["text_projection.projection.weight"] = torch.randn(4, 4)
        torch.save({"state_dict": {"model." + k: v for k, v in sd.items()}}, path)
        return path

    def test_pl_prefixed_checkpoint_populates_protein_branch(
            self, tmp_path, stub_protein_encoder, stage1_args):
        from biom3.Stage3.finetune_embedder import build_protein_to_zp_embedder
        ref = self._reference(stage1_args)
        ckpt = self._write_ckpt(tmp_path / "pencl.ckpt", ref)

        emb = build_protein_to_zp_embedder(stage1_args, str(ckpt))

        got, want = emb.state_dict(), ref.state_dict()
        assert want, "reference has no parameters"
        for k in want:
            assert torch.equal(got[k], want[k]), f"{k} did not load"

    def test_checkpoint_without_protein_branch_raises(
            self, tmp_path, stub_protein_encoder, stage1_args):
        """A silent no-op would give a random projection of stock ESM-2."""
        from biom3.Stage3.finetune_embedder import build_protein_to_zp_embedder
        ref = self._reference(stage1_args)
        ckpt = self._write_ckpt(tmp_path / "pencl.ckpt", ref, include_protein=False)

        with pytest.raises(RuntimeError, match="did not populate"):
            build_protein_to_zp_embedder(stage1_args, str(ckpt))

    def test_null_weights_raise(self, stub_protein_encoder, stage1_args):
        from biom3.Stage3.finetune_embedder import build_protein_to_zp_embedder
        with pytest.raises(ValueError, match="pencl_weights is required"):
            build_protein_to_zp_embedder(stage1_args, None)

    def test_frozen_and_eval(self, tmp_path, stub_protein_encoder, stage1_args):
        from biom3.Stage3.finetune_embedder import build_protein_to_zp_embedder
        ref = self._reference(stage1_args)
        ckpt = self._write_ckpt(tmp_path / "pencl.ckpt", ref)
        emb = build_protein_to_zp_embedder(stage1_args, str(ckpt))
        assert not emb.training
        assert all(not p.requires_grad for p in emb.parameters())

    def test_embed_sequences_order_and_shape(
            self, tmp_path, stub_protein_encoder, stage1_args):
        from biom3.Stage3.finetune_embedder import build_protein_to_zp_embedder
        ref = self._reference(stage1_args)
        ckpt = self._write_ckpt(tmp_path / "pencl.ckpt", ref)
        emb = build_protein_to_zp_embedder(stage1_args, str(ckpt))

        seqs = ["ACDEF", "GHIK", "MNPQRST"]
        z_p = emb.embed_sequences(seqs, batch_size=2)  # forces >1 chunk
        assert z_p.shape == (3, PROJ_DIM)
        # Row i must correspond to seqs[i] across the chunk boundary.
        for i, s in enumerate(seqs):
            assert torch.allclose(z_p[i], emb.embed_sequences([s])[0], atol=1e-6)

    def test_embed_sequences_independent_of_batch_size(
            self, tmp_path, stub_protein_encoder, stage1_args):
        """zp_batch_size must be a speed knob only, never change z_p."""
        from biom3.Stage3.finetune_embedder import build_protein_to_zp_embedder
        ref = self._reference(stage1_args)
        ckpt = self._write_ckpt(tmp_path / "pencl.ckpt", ref)
        emb = build_protein_to_zp_embedder(stage1_args, str(ckpt))

        seqs = ["ACDEF", "GHIK", "MNPQRST", "AC", "WYVTS"]
        assert torch.allclose(emb.embed_sequences(seqs, batch_size=1),
                              emb.embed_sequences(seqs, batch_size=5), atol=1e-6)

    def test_embed_sequences_strips_gaps(
            self, tmp_path, stub_protein_encoder, stage1_args):
        from biom3.Stage3.finetune_embedder import build_protein_to_zp_embedder
        ref = self._reference(stage1_args)
        ckpt = self._write_ckpt(tmp_path / "pencl.ckpt", ref)
        emb = build_protein_to_zp_embedder(stage1_args, str(ckpt))
        assert torch.allclose(emb.embed_sequences(["AC-DE-F"]),
                              emb.embed_sequences(["ACDEF"]), atol=1e-6)


# ------------------------------ blend on the HDF5 path -----------------------

def _write_hdf5_and_zp(tmp_path, n=8, dim=PROJ_DIM, group="MMD_data",
                       shift_zp_rows=False, drop_sequence_key=False):
    """A tiny compiled-HDF5 / Facilitator-.pt pair, row-aligned by construction.

    ``shift_zp_rows`` rolls the .pt rows to imitate pointing at a different
    Facilitator run of the same shape.
    """
    import h5py

    seqs = [f"AC{'D' * (i + 1)}EF" for i in range(n)]
    z_c = torch.arange(n * dim, dtype=torch.float32).reshape(n, dim)
    z_p = z_c + 1000.0

    h5_path = tmp_path / "data.hdf5"
    with h5py.File(h5_path, "w") as fh:
        grp = fh.create_group(group)
        grp.create_dataset("sequence", data=[s.encode() for s in seqs])
        grp.create_dataset("acc_id",
                           data=[f"P{i:05d}".encode() for i in range(n)])
        grp.create_dataset("sequence_length",
                           data=np.array([len(s) for s in seqs], dtype=int))
        grp.create_dataset("text_to_protein_embedding", data=z_c.numpy())

    payload = {"z_p": torch.roll(z_p, 1, dims=0) if shift_zp_rows else z_p,
               "z_c": z_c}
    if not drop_sequence_key:
        payload["sequence"] = seqs[1:] + seqs[:1] if shift_zp_rows else seqs
    pt_path = tmp_path / "facilitator.pt"
    torch.save(payload, pt_path)
    return str(h5_path), str(pt_path), seqs, z_c, z_p


DS_ARGS = {"diffusion_steps": 1024, "image_size": 32}


def _dataset(h5_path, **kwargs):
    return plw.HDF5Dataset(args=DS_ARGS, file_path=h5_path,
                           group_name="MMD_data", **kwargs)


class TestHDF5DatasetBlend:
    def test_pure_zc_needs_no_zp(self, tmp_path):
        h5_path, _, _, z_c, _ = _write_hdf5_and_zp(tmp_path)
        ds = _dataset(h5_path, train_alpha="zc")
        assert ds.z_p is None
        assert torch.allclose(ds[0][1], z_c[0])
        ds.close()

    def test_zp_required_when_alpha_uses_it(self, tmp_path):
        h5_path, _, _, _, _ = _write_hdf5_and_zp(tmp_path)
        with pytest.raises(ValueError, match="zp_path is required"):
            _dataset(h5_path, train_alpha="blend")

    def test_constant_alpha_blends_arithmetically(self, tmp_path):
        h5_path, pt_path, _, z_c, z_p = _write_hdf5_and_zp(tmp_path)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha=0.25)
        assert torch.allclose(ds[3][1], 0.25 * z_p[3] + 0.75 * z_c[3])
        ds.close()

    def test_alpha_zp_returns_pure_zp(self, tmp_path):
        h5_path, pt_path, _, _, z_p = _write_hdf5_and_zp(tmp_path)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha="zp")
        assert torch.allclose(ds[2][1], z_p[2])
        ds.close()

    def test_numeric_string_is_a_constant_not_the_schedule(self, tmp_path):
        """The sentinel footgun: '0.5' must not select the blend schedule."""
        h5_path, pt_path, _, z_c, z_p = _write_hdf5_and_zp(tmp_path)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha="0.5")
        expected = 0.5 * z_p[1] + 0.5 * z_c[1]
        assert all(torch.allclose(ds[1][1], expected) for _ in range(20))
        ds.close()

    def test_train_rows_resample_under_blend(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha="blend")
        torch.manual_seed(0)
        draws = {float(ds[0][1][0]) for _ in range(50)}
        assert len(draws) > 1
        ds.close()


class TestHDF5DatasetEvalIndices:
    def test_eval_rows_are_stable_across_accesses(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha="blend")
        ds.set_eval_indices([0, 1])
        torch.manual_seed(0)
        first = ds[0][1].clone()
        assert all(torch.allclose(ds[0][1], first) for _ in range(50))
        ds.close()

    def test_eval_alpha_is_deterministic_across_instances(self, tmp_path):
        """Two ranks building the dataset independently must agree on val."""
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path)
        a = _dataset(h5_path, zp_path=pt_path, train_alpha="blend")
        b = _dataset(h5_path, zp_path=pt_path, train_alpha="blend")
        a.set_eval_indices([2])
        b.set_eval_indices([2])
        torch.manual_seed(0)
        left = a[2][1].clone()
        torch.manual_seed(999)
        assert torch.allclose(b[2][1], left)
        a.close()
        b.close()

    def test_spread_covers_a_range_of_alphas(self, tmp_path):
        h5_path, pt_path, seqs, _, _ = _write_hdf5_and_zp(tmp_path, n=32)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha="blend")
        ds.set_eval_indices(range(32))
        alphas = [ds._alpha_for(i) for i in range(32)]
        assert min(alphas) < 0.25 and max(alphas) > 0.75
        ds.close()

    def test_constant_eval_alpha_applies_to_eval_rows_only(self, tmp_path):
        h5_path, pt_path, _, z_c, z_p = _write_hdf5_and_zp(tmp_path)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha="zp",
                      eval_alpha="zc")
        ds.set_eval_indices([5])
        assert torch.allclose(ds[5][1], z_c[5])
        assert torch.allclose(ds[4][1], z_p[4])
        ds.close()

    def test_rejects_blend_as_eval_alpha(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path)
        with pytest.raises(ValueError, match="eval_alpha must be"):
            _dataset(h5_path, zp_path=pt_path, train_alpha="blend",
                     eval_alpha="blend")


class TestHDF5ZpAlignment:
    def test_rejects_row_count_mismatch(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path, n=8)
        payload = torch.load(pt_path, weights_only=False)
        payload["z_p"] = payload["z_p"][:6]
        payload["sequence"] = payload["sequence"][:6]
        torch.save(payload, pt_path)
        with pytest.raises(ValueError, match="row-aligned"):
            _dataset(h5_path, zp_path=pt_path, train_alpha="blend")

    def test_rejects_mismatched_facilitator_run(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(
            tmp_path, n=16, shift_zp_rows=True)
        with pytest.raises(ValueError, match="does not match"):
            _dataset(h5_path, zp_path=pt_path, train_alpha="blend")

    def test_rejects_pt_without_zp(self, tmp_path):
        h5_path, pt_path, _, z_c, _ = _write_hdf5_and_zp(tmp_path)
        torch.save({"z_c": z_c}, pt_path)
        with pytest.raises(KeyError, match="no 'z_p' key"):
            _dataset(h5_path, zp_path=pt_path, train_alpha="blend")

    def test_accepts_pt_without_sequences(self, tmp_path):
        """Row count is the only check available; it must still load."""
        h5_path, pt_path, _, _, z_p = _write_hdf5_and_zp(
            tmp_path, drop_sequence_key=True)
        ds = _dataset(h5_path, zp_path=pt_path, train_alpha="zp")
        assert torch.allclose(ds[0][1], z_p[0])
        ds.close()


class TestHDF5DataModuleBlend:
    def _dm(self, h5_path, **kwargs):
        return plw.HDF5DataModule(
            batch_size=4, num_workers=0, valid_size=0.25, seed=0,
            diffusion_steps=1024, image_size=32,
            primary_path=h5_path, group_name="MMD_data", **kwargs)

    def test_setup_marks_val_rows_as_eval(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path, n=16)
        dm = self._dm(h5_path, zp_path=pt_path, train_alpha="blend")
        dm.setup()
        dataset = dm.val_dataset.datasets[0].dataset
        assert dataset._eval_indices == frozenset(dm.split_info[0]["val_indices"])
        dm.teardown(stage="fit")

    def test_val_batches_are_identical_across_epochs(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path, n=16)
        dm = self._dm(h5_path, zp_path=pt_path, train_alpha="blend")
        dm.setup()
        torch.manual_seed(0)
        first = torch.cat([b[1] for b in dm.val_dataloader()])
        torch.manual_seed(1)
        second = torch.cat([b[1] for b in dm.val_dataloader()])
        assert torch.allclose(first, second)
        dm.teardown(stage="fit")

    def test_rejects_secondary_paths_when_blending(self, tmp_path):
        h5_path, pt_path, _, _, _ = _write_hdf5_and_zp(tmp_path)
        dm = self._dm(h5_path, zp_path=pt_path, train_alpha="blend",
                      secondary_paths=[h5_path])
        with pytest.raises(ValueError, match="single HDF5"):
            dm.setup()

    def test_secondary_paths_allowed_without_blending(self, tmp_path):
        h5_path, _, _, _, _ = _write_hdf5_and_zp(tmp_path, n=16)
        dm = self._dm(h5_path, secondary_paths=[h5_path])
        dm.setup()
        assert len(dm.split_info) == 2
        dm.teardown(stage="fit")


class TestTrainingArgConversions:
    """train_alpha / eval_alpha / zp_path through the biom3_train_stage3 parser."""

    @staticmethod
    def _args(**overrides):
        import biom3.Stage3.run_PL_training as base
        argv = []
        for k, v in overrides.items():
            argv += [f"--{k}", str(v)]
        return base.retrieve_all_args(argv)

    def test_defaults_are_text_only(self):
        args = self._args()
        assert args.train_alpha == 0.0
        assert args.eval_alpha == "spread"
        assert args.zp_path is None
        assert alpha_spec_uses_zp(args.train_alpha) is False

    @pytest.mark.parametrize("value,expected", [
        ("zp", 1.0), ("blend", ALPHA_BLEND), ("0.3", 0.3), ("zc", 0.0),
    ])
    def test_train_alpha_normalized(self, value, expected):
        assert self._args(train_alpha=value).train_alpha == expected

    def test_eval_alpha_rejects_a_schedule(self):
        with pytest.raises(ValueError, match="eval_alpha must be"):
            self._args(eval_alpha="blend")

    def test_invalid_train_alpha_rejected(self):
        with pytest.raises(ValueError, match="train_alpha must be"):
            self._args(train_alpha="sequence")

    def test_zp_path_none_sentinel(self):
        assert self._args(zp_path="None").zp_path is None
        assert self._args(zp_path="/tmp/f.pt").zp_path == "/tmp/f.pt"


class TestLoadDataWiring:
    """load_data must hand the blend settings to the data module."""

    def _capture(self, monkeypatch, **overrides):
        import biom3.Stage3.run_PL_training as base
        seen = {}

        class _Recorder:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def setup(self):
                pass

        monkeypatch.setattr(base.PL_mod, "HDF5DataModule", _Recorder)
        args = TestTrainingArgConversions._args(**overrides)
        args.batch_size, args.num_workers = 4, 0
        args.valid_size, args.seed = 0.2, 0
        args.diffusion_steps, args.image_size = 1024, 32
        base.load_data(args, primary_data_path="x.hdf5",
                       secondary_data_paths=None, facilitator="MMD")
        return seen

    def test_defaults_reach_the_data_module(self, monkeypatch):
        seen = self._capture(monkeypatch)
        assert seen["train_alpha"] == 0.0
        assert seen["eval_alpha"] == "spread"
        assert seen["zp_path"] is None

    def test_blend_settings_reach_the_data_module(self, monkeypatch):
        seen = self._capture(monkeypatch, train_alpha="blend",
                             eval_alpha="0.5", zp_path="/tmp/f.pt")
        assert seen["train_alpha"] == ALPHA_BLEND
        assert seen["eval_alpha"] == 0.5
        assert seen["zp_path"] == "/tmp/f.pt"
