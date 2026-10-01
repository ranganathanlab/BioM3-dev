"""Tests for ProteoScribe generalized finetuning (CPU, no weights required).

Covers:
1. encode_protein_sequence: tokenization shapes / lengths / gap handling.
2. make_seq_caption_collate_fn: max_length padding and tensor shapes/dtypes.
3. GeneralizedDataModule: deterministic split, length filtering via length_field
   (with fallback), eager vs lazy equivalence, and schema-driven caption wiring.
4. PL_ProtARDM_Finetune: on-device z_c embedding wiring, and the design
   invariant that the frozen embedder stays out of self.parameters() and out
   of train() mode.

The schema-driven composition engine itself (GeneralizedRecordDataset,
JsonlRecordStore, fields_to_caption) is covered by tests/core_tests/. A real
BioBERT tokenizer / weights are not needed; a fake tokenizer/embedder is used so
these run under `pytest --quick`.
"""

import json
import logging
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

import biom3.Stage3.preprocess as prep
import biom3.Stage3.run_ProteoScribe_finetuning as run_ft
from biom3.Stage3.preprocess import encode_protein_sequence, make_seq_caption_collate_fn
from biom3.split.manifest import write_manifest
from biom3.split.run_stratified_split import build_stratified_split_manifest


# ----------------------------- fixtures / fakes -----------------------------

def _write_jsonl(path, records):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# Cleaned records: {sequence, fields: {key: raw_description}, sequence_length}
_SAMPLE_RECORDS = [
    {"sequence": "ACDEF", "fields": {"protein_name": "SH3", "function": "binding"},
     "sequence_length": 5},
    {"sequence": "GHIKLM", "fields": {"protein_name": "PDZ", "function": "signaling"},
     "sequence_length": 6},
    {"sequence": "NPQRS", "fields": {"protein_name": "WW", "domain": "WW"},
     "sequence_length": 5},
    {"sequence": "TVWY", "fields": {"protein_name": "SH2"},
     "sequence_length": 4},
]

_NO_DROPOUT_SCHEMA = {
    "sequence": {"from": "sequence"},
    "caption": {"compose": "fields_to_caption",
                "args": {"fields_key": "fields", "shuffle": False, "add_label": True}},
}


class _FakeTokenizer:
    """Minimal stand-in for calling a HF tokenizer on a batch."""

    def __call__(self, prompts, *, truncation, max_length, padding,
                          return_tensors, return_attention_mask,
                          return_token_type_ids):
        assert padding == "max_length"
        assert return_tensors == "pt"
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            toks = [ord(c) % 100 + 1 for c in p[:max_length]]
            if toks:
                ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
        return {"input_ids": ids}


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(path):
        return _FakeTokenizer()


# --------------------------- sequence encoding ------------------------------

class TestEncodeSequence:
    def test_padded_to_image_size_squared(self):
        nums = encode_protein_sequence("ACDEF", image_size=4)
        assert len(nums) == 16  # 4*4
        # START token index is 1, END is 22 in the vocab
        assert nums[0] == 1
        assert nums[6] == 22  # <START> A C D E F <END> -> END at idx 6

    def test_gaps_stripped(self):
        assert encode_protein_sequence("AC-DE", image_size=4) == \
            encode_protein_sequence("ACDE", image_size=4)


# ------------------------------- collate fn ---------------------------------

class TestSeqCaptionCollate:
    def test_padding_and_shapes(self):
        collate = make_seq_caption_collate_fn(
            text_tokenizer=_FakeTokenizer(), text_max_length=32, image_size=4,
        )
        batch = [
            {"sequence": "ACDEF", "caption": "PROTEIN_NAME: SH3."},
            {"sequence": "GHIKLM", "caption": "PROTEIN_NAME: PDZ."},
            {"sequence": "NPQRS", "caption": "PROTEIN_NAME: WW."},
        ]
        num_seqs, input_ids = collate(batch)
        assert num_seqs.shape == (3, 16)
        assert num_seqs.dtype == torch.float32
        assert input_ids.shape == (3, 32)  # padded to max_length
        assert input_ids.dtype == torch.long

    def test_custom_output_keys(self):
        collate = make_seq_caption_collate_fn(
            text_tokenizer=_FakeTokenizer(), text_max_length=16, image_size=4,
            sequence_key="seq", caption_key="text",
        )
        num_seqs, input_ids = collate([{"seq": "AC", "text": "x"}])
        assert num_seqs.shape == (1, 16)
        assert input_ids.shape == (1, 16)


# ------------------------------ data module ---------------------------------

class TestGeneralizedDataModule:
    def _make_dm(self, tmp_path, monkeypatch, records=None, **overrides):
        import biom3.Stage3.PL_wrapper as plw
        monkeypatch.setattr(plw, "AutoTokenizer", _FakeAutoTokenizer)
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, records if records is not None else _SAMPLE_RECORDS * 10)
        kwargs = dict(
            jsonl_path=str(path),
            record_schema=_NO_DROPOUT_SCHEMA,
            text_model_path="unused",
            text_max_length=32,
            batch_size=4,
            num_workers=0,
            valid_size=0.25,
            seed=42,
            diffusion_steps=8,   # min_seq_length = 6
            image_size=4,
        )
        kwargs.update(overrides)
        dm = plw.GeneralizedDataModule(**kwargs)
        dm.setup()
        return dm

    def test_split_sizes_and_filter(self, tmp_path, monkeypatch):
        dm = self._make_dm(tmp_path, monkeypatch)
        info = dm.split_info[0]
        # all sample sequences have raw length <= 6 == min_seq_length, none filtered
        assert len(info["train_indices"]) + len(info["val_indices"]) == 40
        assert len(info["val_indices"]) == 10  # 25% of 40

    def test_split_deterministic(self, tmp_path, monkeypatch):
        dm1 = self._make_dm(tmp_path, monkeypatch, seed=7)
        dm2 = self._make_dm(tmp_path, monkeypatch, seed=7)
        np.testing.assert_array_equal(
            dm1.split_info[0]["train_indices"],
            dm2.split_info[0]["train_indices"],
        )

    def test_length_filter_drops_long_sequences(self, tmp_path, monkeypatch):
        records = [
            {"sequence": "AC", "fields": {"a": "x"}, "sequence_length": 2},
            {"sequence": "DEFGHIKLMN", "fields": {"a": "y"}, "sequence_length": 10},
        ] * 10  # 20 records, half exceed min_seq_length=6
        dm = self._make_dm(tmp_path, monkeypatch, records=records, valid_size=0.0)
        kept = dm.split_info[0]["train_indices"]
        assert len(kept) == 10  # only the length-2 records survive

    def test_missing_length_field_falls_back_to_sequence(self, tmp_path, monkeypatch):
        records = [
            {"sequence": "AC", "fields": {"a": "x"}},                       # no length
            {"sequence": "DEFGHIKLMN", "fields": {"a": "y"}},               # no length, len 10
        ] * 10
        dm = self._make_dm(tmp_path, monkeypatch, records=records, valid_size=0.0)
        # fallback computes ungapped length; the length-10 records are filtered out
        assert len(dm.split_info[0]["train_indices"]) == 10

    def test_lazy_matches_eager(self, tmp_path, monkeypatch):
        eager = self._make_dm(tmp_path, monkeypatch, lazy=False)
        lazy = self._make_dm(tmp_path, monkeypatch, lazy=True)
        np.testing.assert_array_equal(
            eager.split_info[0]["train_indices"],
            lazy.split_info[0]["train_indices"],
        )
        e_seqs, e_ids = next(iter(eager.train_dataloader()))
        l_seqs, l_ids = next(iter(lazy.train_dataloader()))
        assert e_seqs.shape == l_seqs.shape == (4, 16)
        assert e_ids.shape == l_ids.shape == (4, 32)

    def test_dataloader_batch_shapes(self, tmp_path, monkeypatch):
        dm = self._make_dm(tmp_path, monkeypatch)
        num_seqs, input_ids = next(iter(dm.train_dataloader()))
        assert num_seqs.shape == (4, 16)
        assert input_ids.shape == (4, 32)

    def test_schema_composes_caption(self, tmp_path, monkeypatch):
        dm = self._make_dm(tmp_path, monkeypatch)
        # the underlying GeneralizedRecordDataset composes a labeled caption
        underlying = dm.train_dataset.dataset
        sample = underlying[0]
        assert "PROTEIN_NAME" in sample["caption"]
        assert isinstance(sample["sequence"], str)


# --------------------- data module: manifest-driven split -------------------

_MANIFEST_RECORDS = [
    {"sequence": "AC" * (i % 3 + 1),
     "fields": {"protein_name": f"p{i}"},
     "sequence_length": 2 * (i % 3 + 1),
     "source": "pfam" if i % 3 else "swissprot"}
    for i in range(24)
]


def _make_manifest(tmp_path, records, ratios):
    data = tmp_path / "records.jsonl"
    _write_jsonl(data, records)
    tsv = tmp_path / "clusters.tsv"
    with open(tsv, "w") as fh:
        for row in range(len(records)):
            fh.write(f"0-{row}\t0-{row}\n")  # singleton clusters
    manifest, _ = build_stratified_split_manifest(
        data_path=str(data), ratios=ratios,
        clusters_tsv=str(tsv), min_seq_length=None,
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), manifest)
    return str(data), str(manifest_path), manifest


class TestGeneralizedDataModuleManifest:
    def _dm(self, tmp_path, monkeypatch, data_path, manifest_path, **overrides):
        import biom3.Stage3.PL_wrapper as plw
        monkeypatch.setattr(plw, "AutoTokenizer", _FakeAutoTokenizer)
        kwargs = dict(
            jsonl_path=data_path,
            record_schema=_NO_DROPOUT_SCHEMA,
            text_model_path="unused",
            text_max_length=32,
            batch_size=4,
            num_workers=0,
            valid_size=0.25,   # ignored when a manifest is given
            seed=42,
            diffusion_steps=8,
            image_size=4,
            split_manifest_path=manifest_path,
        )
        kwargs.update(overrides)
        dm = plw.GeneralizedDataModule(**kwargs)
        dm.setup()
        return dm

    def test_split_indices_come_from_manifest(self, tmp_path, monkeypatch):
        ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
        data, manifest_path, manifest = _make_manifest(tmp_path, _MANIFEST_RECORDS, ratios)
        dm = self._dm(tmp_path, monkeypatch, data, manifest_path)
        entry = manifest["files"][0]
        info = dm.split_info[0]
        assert info["train_indices"] == entry["train"]
        assert info["val_indices"] == entry["val"]
        assert info["test_indices"] == entry["test"]

    def test_test_split_held_out(self, tmp_path, monkeypatch):
        ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
        data, manifest_path, manifest = _make_manifest(tmp_path, _MANIFEST_RECORDS, ratios)
        dm = self._dm(tmp_path, monkeypatch, data, manifest_path)
        test_idx = set(dm.split_info[0]["test_indices"])
        assert test_idx  # non-empty
        train_val = set(dm.train_dataset.indices) | set(dm.val_dataset.indices)
        assert test_idx.isdisjoint(train_val)

    def test_lazy_matches_eager_under_manifest(self, tmp_path, monkeypatch):
        ratios = {"train": 0.7, "val": 0.3}
        data, manifest_path, _ = _make_manifest(tmp_path, _MANIFEST_RECORDS, ratios)
        eager = self._dm(tmp_path, monkeypatch, data, manifest_path, lazy=False)
        lazy = self._dm(tmp_path, monkeypatch, data, manifest_path, lazy=True)
        assert eager.split_info[0]["train_indices"] == lazy.split_info[0]["train_indices"]
        assert eager.split_info[0]["val_indices"] == lazy.split_info[0]["val_indices"]

    def test_fingerprint_mismatch_raises(self, tmp_path, monkeypatch):
        ratios = {"train": 0.8, "val": 0.2}
        data, manifest_path, _ = _make_manifest(tmp_path, _MANIFEST_RECORDS, ratios)
        # Mutate the JSONL after the manifest was built -> fingerprint no longer matches.
        mutated = [dict(r) for r in _MANIFEST_RECORDS]
        mutated[0]["sequence"] = "WWWWWW"
        _write_jsonl(tmp_path / "records.jsonl", mutated)
        with pytest.raises(ValueError, match="fingerprint"):
            self._dm(tmp_path, monkeypatch, data, manifest_path)


# --------------------------- finetune PL wrapper ----------------------------

class _FakeEmbedder(nn.Module):
    def __init__(self, text_emb_dim):
        super().__init__()
        self.lin = nn.Linear(1, text_emb_dim)

    def forward(self, input_ids):
        return self.lin(input_ids[:, :1].float())


class TestFinetuneWrapper:
    def _make_wrapper(self, text_emb_dim=8):
        from biom3.Stage3.PL_wrapper import PL_ProtARDM_Finetune
        args = Namespace()
        stub_model = nn.Linear(4, 4)
        embedder = _FakeEmbedder(text_emb_dim)
        return PL_ProtARDM_Finetune(args=args, model=stub_model, embedder=embedder), embedder

    def test_embedder_excluded_from_parameters(self):
        pl_model, embedder = self._make_wrapper()
        wrapper_param_ids = {id(p) for p in pl_model.parameters()}
        embedder_param_ids = {id(p) for p in embedder.parameters()}
        assert wrapper_param_ids.isdisjoint(embedder_param_ids)

    def test_embedder_frozen_and_eval(self):
        pl_model, embedder = self._make_wrapper()
        assert all(not p.requires_grad for p in embedder.parameters())
        pl_model.train()  # must NOT flip the frozen front-end into train mode
        assert not embedder.training

    def test_on_after_batch_transfer_produces_zc(self):
        pl_model, _ = self._make_wrapper(text_emb_dim=8)
        num_seqs = torch.zeros(3, 16)
        input_ids = torch.randint(1, 50, (3, 32))
        out = pl_model.on_after_batch_transfer((num_seqs, input_ids), 0)
        assert isinstance(out, list) and len(out) == 2
        ret_seqs, z_c = out
        assert ret_seqs is num_seqs
        assert z_c.shape == (3, 8)
        assert not z_c.requires_grad  # computed under no_grad with a frozen front-end


# --------------------------- runtime arg wiring -----------------------------

class _StopAfterRuntimeSetup(Exception):
    """Aborts main() once the runtime settings have been applied."""


@pytest.fixture
def isolated_matmul_precision():
    """Pin the global to torch's default, then restore it (and any log handlers)."""
    original = torch.get_float32_matmul_precision()
    biom3_logger = logging.getLogger("biom3")
    original_handlers = list(biom3_logger.handlers)
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(original)
    for handler in list(biom3_logger.handlers):
        if handler not in original_handlers:
            biom3_logger.removeHandler(handler)
            handler.close()


class TestFloat32MatmulPrecision:
    """--float32_matmul_precision was inherited from run_PL_training but dead:
    it is applied inside that module's main(), which the finetune main() does
    not call, so finetune runs stayed at torch's "highest" default."""

    @staticmethod
    def _args(tmp_path, precision):
        return run_ft.parse_arguments([
            "--record_schema", json.dumps({"sequence": {"from": "sequence"}}),
            "--output_root", str(tmp_path),
            "--run_id", "matmul_precision",
            "--checkpoints_folder", "checkpoints",
            "--float32_matmul_precision", precision,
        ])

    @pytest.mark.parametrize("precision", ["high", "medium"])
    def test_applied_by_main(self, tmp_path, monkeypatch,
                             isolated_matmul_precision, precision):
        args = self._args(tmp_path, precision)

        def _stop(*a, **kw):
            raise _StopAfterRuntimeSetup

        monkeypatch.setattr(run_ft, "load_embedder_configs", _stop)

        with pytest.raises(_StopAfterRuntimeSetup):
            run_ft.main(args)

        assert torch.get_float32_matmul_precision() == precision

    def test_invalid_value_rejected(self, tmp_path, isolated_matmul_precision):
        """argparse `choices` guards the CLI, but JSON configs reach the
        namespace via set_defaults() and are not validated there."""
        args = self._args(tmp_path, "high")
        args.float32_matmul_precision = "turbo"
        with pytest.raises(ValueError, match="float32_matmul_precision"):
            run_ft.main(args)
