"""The RL prompt encoder passes the caption attention mask only when asked.

Covers the encoder itself, the shared --weight_set / --text_attention_mask
handling, and that the three RL entry points accept the arguments. The Stage 1
model and dataset are stubbed, so no weights are needed.
"""

import json
from argparse import Namespace

import pytest
import torch

import biom3.rl.run_dpo_train as run_dpo
import biom3.rl.run_gdpo_train as run_gdpo
import biom3.rl.run_grpo_train as run_grpo
from biom3.rl.grpo import _PromptEncoder
from biom3.rl.io import configure_conditioning

WARNING = "CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS"
MASK = torch.tensor([[1, 1, 1, 0, 0]])


class _Dataset:
    """What _PromptEncoder touches on TextSeqPairing_Dataset."""

    def __getitem__(self, idx):
        return torch.tensor([[5, 6, 7, 0, 0]]), torch.tensor([[0, 4, 2]])

    def caption_tokenizer(self, batch_captions):
        return {"attention_mask": MASK}


class _Stage1:
    def __init__(self):
        self.calls = []

    def __call__(self, x_t, x_p, compute_masked_logits=False, **kwargs):
        self.calls.append(kwargs)
        return {"text_joint_latent": torch.ones(1, 4)}


def _encoder(cfg1):
    s1 = _Stage1()
    encoder = _PromptEncoder(s1, lambda z_t: 2 * z_t, cfg1,
                             torch.device("cpu"))
    encoder._dataset = _Dataset()
    return encoder, s1


def _cfg1(**extra):
    return Namespace(sequence_keyword="sequence", id_keyword="id", **extra)


@pytest.mark.parametrize("cfg1", [_cfg1(), _cfg1(text_attention_mask=False)])
def test_mask_is_not_passed_by_default(cfg1):
    encoder, s1 = _encoder(cfg1)

    z_c = encoder("a prompt")

    assert s1.calls == [{}]
    assert torch.equal(z_c, 2 * torch.ones(1, 4))


def test_mask_is_passed_when_configured():
    encoder, s1 = _encoder(_cfg1(text_attention_mask=True))

    encoder("a prompt")

    assert list(s1.calls[0]) == ["x_t_mask"]
    assert torch.equal(s1.calls[0]["x_t_mask"], MASK)


def _weight_set(tmp_path, **extra):
    path = tmp_path / "weight_set.json"
    path.write_text(json.dumps({
        "pencl_weights": "weights/PenCL/p.bin",
        "facilitator_weights": "weights/Facilitator/f.bin",
        "proteoscribe_weights": "weights/ProteoScribe/s.bin",
        **extra,
    }))
    return str(path)


def test_weight_set_fills_the_stage_weights(tmp_path):
    args = Namespace(weight_set=_weight_set(tmp_path), text_attention_mask=False,
                     stage1_weights=None, stage2_weights=None,
                     stage3_init_weights="weights/ProteoScribe/blend.bin")
    cfg1 = Namespace()

    warning = configure_conditioning(args, cfg1)

    assert args.stage1_weights == "weights/PenCL/p.bin"
    assert args.stage2_weights == "weights/Facilitator/f.bin"
    assert args.stage3_init_weights == "weights/ProteoScribe/blend.bin"
    assert cfg1.text_attention_mask is False
    assert warning == []


@pytest.mark.parametrize("recorded, flag, warns", [
    (True, False, True), (False, True, True), (True, True, False), (False, False, False),
])
def test_flag_is_checked_against_the_weight_set(tmp_path, recorded, flag, warns):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=recorded)
    args = Namespace(weight_set=weight_set, text_attention_mask=flag,
                     stage1_weights=None, stage2_weights=None, stage3_init_weights=None)
    cfg1 = Namespace()

    warning = configure_conditioning(args, cfg1)

    assert cfg1.text_attention_mask is flag
    assert (WARNING in warning) is warns


def test_other_stage1_weights_are_not_checked(tmp_path):
    weight_set = _weight_set(tmp_path, pencl_trained_with_text_attention_mask=True)
    args = Namespace(weight_set=weight_set, text_attention_mask=False,
                     stage1_weights="weights/Run1_frozen_ckpts/other.ckpt",
                     stage2_weights=None, stage3_init_weights=None)

    assert configure_conditioning(args, Namespace()) == []


@pytest.mark.parametrize("module", [run_grpo, run_gdpo, run_dpo])
def test_entry_points_accept_the_arguments(module):
    default = module.parse_arguments([])
    given = module.parse_arguments(
        ["--text_attention_mask", "--weight_set", "configs/weights/run1_base.json"])

    assert default.text_attention_mask is False and default.weight_set is None
    assert given.text_attention_mask is True
    assert given.weight_set == "configs/weights/run1_base.json"
