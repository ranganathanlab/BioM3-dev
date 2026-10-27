"""The Stage 3 loss, split by whether the target is padding.

A sequence is padded at its tail with pad tokens up to the model's fixed length,
so the loss over every unsampled position mixes the sequence itself with its
padding. Training now computes that loss three ways, over all unsampled
positions, over the non-pad ones and over the pad ones, logs all three, and
lets --loss_positions choose which one drives the gradients.

Padding is not masking. The mask token (0) marks positions the model has not
been shown yet; a pad (23) is the true token at a tail position. An unsampled
pad is a padding target, a pad that has been sampled is context, and an
unsampled residue is a sequence target.

The non-pad and pad terms are means over the positions a sequence actually
contributes. The unsampled positions are drawn from the whole window, so that
count depends on the sequence's length and on the draw, not on the diffusion
time: dividing by anything tied to the time alone would make the reported
per-position loss depend on how long the sequence is.
"""

import json
import math
from argparse import Namespace

import pytest
import torch
from torch import nn

import biom3.Stage3.run_ProteoScribe_finetuning as run_ft
import biom3.Stage3.transformer_training_helper as helper
from biom3.rl.grpo import MASK_ID, PAD_ID
from biom3.Stage3.PL_wrapper import PL_ProtARDM
from biom3.Stage3.preprocess import PAD_TOKEN_ID, create_num_seqs, encode_protein_sequence

START, END, PAD, MASK = 1, 22, PAD_TOKEN_ID, 0


def test_pad_token_id_matches_the_vocabulary():
    assert create_num_seqs(['-']) == [PAD_TOKEN_ID]
    assert PAD_TOKEN_ID == PAD_ID
    assert MASK_ID == MASK != PAD_TOKEN_ID


def test_pads_fill_only_the_tail():
    tokens = encode_protein_sequence("ACD", image_size=3)

    assert tokens[0] == START and tokens[4] == END
    assert tokens[5:] == [PAD] * 4
    assert PAD not in tokens[:5] and MASK not in tokens


def test_terms_on_a_worked_example():
    real = torch.tensor([[START, 2, 3, END, PAD, PAD]])
    # shown to the model: START, the residue at 2 and the last pad; the rest are masked
    model_input = torch.tensor([[START, MASK, 3, MASK, MASK, PAD]])
    log_prob = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])

    terms = helper.unsampled_loss_terms(log_prob, model_input, real, PAD)

    # unsampled sequence positions: 1 and 3 (the <END>); unsampled padding: 4
    assert terms['non_pad'].item() == pytest.approx((2.0 + 4.0) / 2)
    assert terms['pad'].item() == pytest.approx(5.0 / 1)


def test_a_sampled_pad_is_context_not_a_target():
    real = torch.tensor([[START, 2, END, PAD, PAD]])
    log_prob = torch.full((1, 5), -1.0)
    every_pad_shown = torch.tensor([[START, MASK, MASK, PAD, PAD]])

    terms = helper.unsampled_loss_terms(log_prob, every_pad_shown, real, PAD)

    assert terms['pad'].item() == 0.0
    assert terms['non_pad'].item() == pytest.approx(2.0 / 2)


def test_each_sequence_is_divided_by_its_own_count():
    """Two sequences at the same diffusion time, four positions unsampled in
    each. The long one has all four on the sequence; the short one has one."""
    real = torch.tensor([[START, 2, 3, 4, 5, END, PAD, PAD],
                         [START, 2, END, PAD, PAD, PAD, PAD, PAD]])
    model_input = torch.tensor([[START, MASK, MASK, MASK, MASK, END, PAD, PAD],
                                [START, 2, MASK, MASK, MASK, MASK, PAD, PAD]])
    assert (model_input == MASK).sum(1).tolist() == [4, 4]
    log_prob = torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, -6.0, -1.0, -1.0, -1.0, 0.0, 0.0]])

    terms = helper.unsampled_loss_terms(log_prob, model_input, real, PAD)

    long_sequence = (1.0 + 2.0 + 3.0 + 4.0) / 4
    short_sequence = 6.0 / 1
    assert terms['non_pad'].item() == pytest.approx((long_sequence + short_sequence) / 2)
    # the long one has no unsampled padding, so only the short one is in the pad term
    assert terms['pad'].item() == pytest.approx((1.0 + 1.0 + 1.0) / 3)


def test_a_sequence_with_nothing_to_predict_is_left_out():
    real = torch.tensor([[START, 2, END, PAD],
                         [START, 2, END, PAD]])
    model_input = torch.tensor([[START, 2, END, MASK],
                                [START, MASK, END, PAD]])
    log_prob = torch.full((2, 4), -2.0)

    terms = helper.unsampled_loss_terms(log_prob, model_input, real, PAD)

    assert terms['non_pad'].item() == pytest.approx(2.0)
    assert terms['pad'].item() == pytest.approx(2.0)


@pytest.mark.parametrize("length", [3, 20, 60])
def test_reported_value_does_not_depend_on_sequence_length(length):
    """A model that charges the same for every token must report that cost,
    whatever the sequence length and wherever the diffusion time falls."""
    torch.manual_seed(length)
    window, batch, cost = 64, 512, 1.5
    real = torch.tensor(encode_protein_sequence("A" * length, 8)).repeat(batch, 1)
    order = torch.stack([torch.randperm(window) for _ in range(batch)])
    idx = torch.randint(0, window + 1, (batch, 1))
    model_input = torch.where(order < idx, real, torch.zeros_like(real))
    log_prob = torch.full((batch, window), -cost)

    terms = helper.unsampled_loss_terms(log_prob, model_input, real, PAD)

    assert terms['non_pad'].item() == pytest.approx(cost)
    assert terms['pad'].item() == pytest.approx(cost)


def test_gradient_reaches_only_the_chosen_targets():
    real = torch.tensor([[START, 2, 3, END, PAD, PAD]])
    model_input = torch.tensor([[START, MASK, 3, MASK, MASK, PAD]])
    log_prob = torch.full((1, 6), -1.0, requires_grad=True)

    terms = helper.unsampled_loss_terms(log_prob, model_input, real, PAD)
    (non_pad_grad,) = torch.autograd.grad(terms['non_pad'], log_prob, retain_graph=True)
    (pad_grad,) = torch.autograd.grad(terms['pad'], log_prob)

    assert (non_pad_grad != 0).tolist() == [[False, True, False, True, False, False]]
    assert (pad_grad != 0).tolist() == [[False, False, False, False, True, False]]


class _UniformModel(nn.Module):
    """Stand-in for ProteoScribe: the same logits everywhere, with a parameter."""

    def __init__(self, num_classes=29):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, t, y_c):
        return self.bias.view(1, -1, 1).expand(x.size(0), -1, x.size(-1))


def _module(**args):
    module = PL_ProtARDM(args=Namespace(task='proteins', **args), model=_UniformModel())
    module.performance_step = lambda **kwargs: tuple(torch.tensor(0.0) for _ in range(10))
    return module


def _batch(length=64):
    sequences = ["ACDEFGHIKL", "MNPQRSTVWYACDEFGHIKLMNPQ"]
    return torch.tensor([encode_protein_sequence(s, 8) for s in sequences]).view(2, 1, length)


@pytest.mark.parametrize("args, chosen", [({}, 'all'), ({'loss_positions': 'all'}, 'all'),
                                          ({'loss_positions': 'non_pad'}, 'non_pad')])
def test_module_returns_every_term_and_uses_the_chosen_one(args, chosen):
    torch.manual_seed(0)
    module = _module(**args)

    loss, _, terms = module.cond_elbo_objective(
        realization=_batch(), y_c=torch.zeros(2, 4), realization_idx=0, stage='train')

    assert sorted(terms) == ['all', 'non_pad', 'pad']
    assert loss is terms[chosen]
    # uniform logits: every target costs log(29), and each term reports exactly that
    assert terms['non_pad'].item() == pytest.approx(math.log(29))
    assert terms['pad'].item() == pytest.approx(math.log(29))
    assert terms['all'].item() < math.log(29)


def test_module_logs_every_term(monkeypatch):
    module = _module(loss_positions='non_pad')
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.update({name: value}))

    out = module.common_step([_batch().view(2, 64), torch.zeros(2, 4)], 0, stage='train')

    assert {'train_loss', 'train_loss_all', 'train_loss_non_pad', 'train_loss_pad'} <= set(logged)
    assert logged['train_loss'] is logged['train_loss_non_pad'] is out['loss']


def _args(tmp_path, *extra):
    return run_ft.parse_arguments([
        "--record_schema", json.dumps({"sequence": {"from": "sequence"}}),
        "--output_root", str(tmp_path), "--run_id", "loss", *extra,
    ])


def test_option_parses(tmp_path):
    assert _args(tmp_path).loss_positions == 'all'
    assert _args(tmp_path, "--loss_positions", "non_pad").loss_positions == 'non_pad'


def test_option_from_a_config_is_validated(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"loss_positions": "sequence",
                                  "record_schema": {"sequence": {"from": "sequence"}}}))

    with pytest.raises(ValueError, match="loss_positions"):
        run_ft.parse_arguments(["--config_path", str(config), "--run_id", "loss"])
