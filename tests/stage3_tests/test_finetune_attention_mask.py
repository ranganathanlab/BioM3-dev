"""Generalized finetuning can pass the caption attention mask.

The frozen text->z_c embedder used to call BERT with input_ids alone. That
suits PenCL weights trained without the mask (run1_base) and is wrong for
weights trained with it. --text_attention_mask (default off) threads the mask
from the collate to BERT; with it off nothing about the batch or the calls
changes. Also covered: --weight_set and the check of the flag against it. Fakes
stand in for the tokenizer and the encoder, so these run under `pytest --quick`.
"""

import json
from argparse import Namespace

import pytest
import torch
from torch import nn

import biom3.Stage1.model as stage1_mod
import biom3.Stage3.run_ProteoScribe_finetuning as run_ft
from biom3.Stage3.finetune_embedder import TextToZcEmbedder
from biom3.Stage3.PL_wrapper import PL_ProtARDM_Finetune
from biom3.Stage3.preprocess import make_seq_caption_collate_fn

WARNING = "CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS"
BATCH = [{"sequence": "ACDEF", "caption": "ab"}, {"sequence": "GHIK", "caption": "abcd"}]


class _Tokenizer:
    """Records how it was called; returns a mask only when asked for one."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, prompts, **kwargs):
        self.kwargs = kwargs
        ids = torch.zeros(len(prompts), kwargs["max_length"], dtype=torch.long)
        for i, prompt in enumerate(prompts):
            ids[i, : len(prompt)] = 7
        out = {"input_ids": ids}
        if kwargs["return_attention_mask"]:
            out["attention_mask"] = (ids != 0).long()
        return out


def _collate(tokenizer, **kwargs):
    return make_seq_caption_collate_fn(
        text_tokenizer=tokenizer, text_max_length=6, image_size=4, **kwargs)


def test_collate_without_the_mask_is_unchanged():
    tokenizer = _Tokenizer()

    out = _collate(tokenizer)(BATCH)

    assert tokenizer.kwargs["return_attention_mask"] is False
    assert tokenizer.kwargs["padding"] == "max_length"
    assert len(out) == 2 and out[1].shape == (2, 6)
    assert len(_collate(tokenizer, include_sequences=True)(BATCH)) == 3


def test_collate_emits_the_mask_after_input_ids():
    num_seqs, input_ids, mask = _collate(_Tokenizer(), include_mask=True)(BATCH)

    assert torch.equal(mask, (input_ids != 0).long())
    assert mask.sum(dim=1).tolist() == [2, 4]


def test_collate_keeps_sequences_last():
    out = _collate(_Tokenizer(), include_mask=True, include_sequences=True)(BATCH)

    assert len(out) == 4
    assert out[3] == ["ACDEF", "GHIK"]


class _TextEncoder(nn.Module):
    """Stand-in for BioBERT's TextEncoder that records the mask it was given."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(1, args.text_encoder_embedding)
        self.calls = []

    def forward(self, inputs, compute_logits=False, **kwargs):
        self.calls.append(kwargs)
        return self.dense(inputs[:, :1].float())


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setattr(stage1_mod, "TextEncoder", _TextEncoder)
    stage1 = Namespace(text_encoder_embedding=6, proj_embedding_dim=4, dropout=0.0)
    stage2 = Namespace(emb_dim=4, hid_dim=8, dropout=0.0)
    return TextToZcEmbedder(stage1, stage2).eval()


def test_embedder_passes_the_mask_only_when_given(embedder):
    input_ids = torch.tensor([[7, 7, 0], [7, 0, 0]])
    mask = (input_ids != 0).long()

    embedder(input_ids)
    embedder(input_ids, attention_mask=mask)

    assert embedder.text_encoder.calls[0] == {}
    assert torch.equal(embedder.text_encoder.calls[1]["attention_mask"], mask)


class _Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 4)
        self.masks = []

    def forward(self, input_ids, **kwargs):
        self.masks.append(kwargs.get("attention_mask"))
        return self.lin(input_ids[:, :1].float())


def _wrapper(**kwargs):
    embedder = _Embedder()
    wrapper = PL_ProtARDM_Finetune(
        args=Namespace(), model=nn.Linear(2, 2), embedder=embedder, **kwargs)
    return wrapper, embedder


def test_wrapper_gives_the_mask_to_the_embedder():
    wrapper, embedder = _wrapper(text_attention_mask=True)
    num_seqs, input_ids = torch.zeros(2, 16), torch.tensor([[7, 7, 0], [7, 0, 0]])
    mask = (input_ids != 0).long()

    out = wrapper.on_after_batch_transfer((num_seqs, input_ids, mask), 0)

    assert len(out) == 2 and out[1].shape == (2, 4)
    assert torch.equal(embedder.masks[0], mask)


def test_wrapper_without_the_flag_passes_no_mask():
    wrapper, embedder = _wrapper()

    wrapper.on_after_batch_transfer((torch.zeros(2, 16), torch.ones(2, 3, dtype=torch.long)), 0)

    assert embedder.masks == [None]


def test_wrapper_finds_sequences_after_the_mask():
    z_p = {"AC": torch.zeros(4), "GH": torch.ones(4)}
    wrapper, _ = _wrapper(text_attention_mask=True, zp_lookup=z_p, train_alpha=1.0, eval_alpha=1.0)
    input_ids = torch.ones(2, 3, dtype=torch.long)

    _, y = wrapper.on_after_batch_transfer(
        (torch.zeros(2, 16), input_ids, torch.ones_like(input_ids), ["AC", "GH"]), 0)

    assert torch.equal(y, torch.stack([z_p["AC"], z_p["GH"]]))


def test_wrapper_needs_the_mask_in_the_batch():
    wrapper, _ = _wrapper(text_attention_mask=True)

    with pytest.raises(RuntimeError, match="caption attention mask"):
        wrapper.on_after_batch_transfer((torch.zeros(2, 16), torch.ones(2, 3, dtype=torch.long)), 0)


def _args(tmp_path, *extra):
    return run_ft.parse_arguments([
        "--record_schema", json.dumps({"sequence": {"from": "sequence"}}),
        "--output_root", str(tmp_path),
        "--run_id", "mask",
        "--checkpoints_folder", "checkpoints",
        *extra,
    ])


@pytest.mark.parametrize("extra, expected", [
    ([], False), (["--text_attention_mask", "True"], True), (["--text_attention_mask", "false"], False),
])
def test_flag_parses(tmp_path, extra, expected):
    assert _args(tmp_path, *extra).text_attention_mask is expected


def _weight_set(tmp_path, **extra):
    path = tmp_path / "weight_set.json"
    path.write_text(json.dumps({
        "pencl_weights": "weights/PenCL/p.bin",
        "facilitator_weights": "weights/Facilitator/f.bin",
        "proteoscribe_weights": "weights/ProteoScribe/s.bin",
        **extra,
    }))
    return str(path)


def test_weight_set_fills_unset_weights(tmp_path):
    args = _args(tmp_path, "--weight_set", _weight_set(tmp_path),
                 "--facilitator_weights", "weights/Facilitator/mine.bin")

    assert args.pencl_weights == "weights/PenCL/p.bin"
    assert args.facilitator_weights == "weights/Facilitator/mine.bin"
    assert args.pretrained_weights == "weights/ProteoScribe/s.bin"


@pytest.mark.parametrize("name", ["finetune_generalized_v1", "finetune_generalized_aurora"])
def test_shipped_configs_are_checked_against_their_weight_set(name):
    args = run_ft.parse_arguments(
        ["--config_path", f"configs/stage3_training/{name}.json", "--run_id", "x"])

    summary, warning = run_ft.check_text_attention_mask(
        args.weight_set, args.pencl_weights, args.text_attention_mask)

    assert "trained without it" in summary
    assert warning == []


class _Stop(Exception):
    pass


@pytest.fixture
def stop_before_data(monkeypatch):
    def _stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(run_ft, "load_embedder_configs", lambda args: (Namespace(), Namespace()))
    monkeypatch.setattr(run_ft, "load_data", _stop)


def _run_log(tmp_path):
    return (tmp_path / "runs" / "mask" / "artifacts" / "run.log").read_text()


@pytest.mark.parametrize("recorded, flag, warns", [
    (True, [], True),
    (False, ["--text_attention_mask", "True"], True),
    (False, [], False),
])
def test_main_warns_when_the_flag_disagrees(tmp_path, stop_before_data, recorded, flag, warns):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=recorded)

    with pytest.raises(_Stop):
        run_ft.main(_args(tmp_path, "--device", "cpu", "--weight_set", weight_set, *flag))

    assert (WARNING in _run_log(tmp_path)) is warns
