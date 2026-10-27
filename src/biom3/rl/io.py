"""Model loading helpers for GRPO.

Wraps the existing per-stage I/O so the GRPO trainer can build the frozen
Stage 1 + Stage 2 conditioning chain and a trainable Stage 3 policy from a
single config.
"""

from argparse import Namespace
from typing import Optional

import torch.nn as nn

import biom3.Stage1.model as S1mod
from biom3.Stage1.io import load_pencl_weights
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.core.distributed import is_main_process
from biom3.core.io import load_state_dict_unwrap_pl as _load_state_dict_unwrap_pl
from biom3.core.weight_sets import (
    WEIGHT_KEYS,
    check_normalize_zc,
    check_text_attention_mask,
    merge_weight_set,
)
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

_WEIGHT_SET_ARGS = {
    "pencl_weights": "stage1_weights",
    "facilitator_weights": "stage2_weights",
    "proteoscribe_weights": "stage3_init_weights",
}


def add_conditioning_args(parser):
    """Arguments shared by the RL entry points for the prompt -> z_c chain."""
    parser.add_argument("--weight_set", type=str, default=None,
                        help="Weight-set bundle JSON (e.g. configs/weights/run1_base.json). "
                             "Fills --stage1_weights, --stage2_weights and "
                             "--stage3_init_weights when they are not given, and its "
                             "record of how PenCL was trained is checked against "
                             "--text_attention_mask.")
    parser.add_argument("--text_attention_mask", action="store_true",
                        help="Pass the caption attention mask to BERT when embedding "
                             "prompts. Set it to match how the PenCL weights were "
                             "trained; off (default) for run1_base.")
    return parser


def log_mask_warning(lines):
    """Log the mask mismatch warning, if there is one, on the main rank."""
    if is_main_process():
        for line in lines:
            logger.warning(line)


def configure_conditioning(args, cfg1):
    """Apply --weight_set and --text_attention_mask to an RL run.

    Fills the stage weights the run does not name from the weight set, tells
    the prompt encoder (through ``cfg1``) whether to pass the caption attention
    mask, and logs how that compares with the weight set's record. Returns the
    lines of the mismatch warning, empty unless the two disagree, so the caller
    can repeat them when the run ends.
    """
    replaced = merge_weight_set(args, args.weight_set, keys=WEIGHT_KEYS,
                                rename=_WEIGHT_SET_ARGS, argv=getattr(args, "_argv", None))
    if is_main_process():
        for attr, old, new in replaced:
            logger.info("--weight_set on the command line replaces the config's %s: %s -> %s",
                        attr, old, new)
    cfg1.text_attention_mask = bool(args.text_attention_mask)
    summary, warning = check_text_attention_mask(
        args.weight_set, args.stage1_weights, args.text_attention_mask)
    # RL has no normalize_zc: it always conditions on z_c at its own length.
    _, zc_warning = check_normalize_zc(args.weight_set, args.stage3_init_weights, False)
    warning = warning + zc_warning
    if is_main_process():
        logger.info(summary)
    log_mask_warning(warning)
    return warning


def _require_weights(weights_path, name, consequence):
    if not weights_path or str(weights_path) == "None":
        raise ValueError(f"{name} is required: without it {consequence}.")


def _freeze(model: nn.Module, device) -> nn.Module:
    if device is not None:
        model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_pencl_frozen(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    _require_weights(weights_path, "stage1_weights",
                     "prompts would be embedded by an untrained PenCL")
    model = S1mod.pfam_PEN_CL(args=cfg)
    load_pencl_weights(model, weights_path, device=device)
    return _freeze(model, device)


def load_facilitator_frozen(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    _require_weights(weights_path, "stage2_weights",
                     "z_c would come from an untrained Facilitator")
    model = S1mod.Facilitator(
        in_dim=cfg.emb_dim,
        hid_dim=cfg.hid_dim,
        out_dim=cfg.emb_dim,
        dropout=cfg.dropout,
    )
    sd = _load_state_dict_unwrap_pl(weights_path, device=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"Facilitator weights at {weights_path} did not populate "
            f"{len(missing)}/{len(model.state_dict())} tensors (e.g. {missing[:5]}). "
            f"File keys look like {sorted(sd)[:3]}."
        )
    if unexpected:
        logger.warning(
            "Facilitator: %d key(s) in %s are not used by the model: %s",
            len(unexpected), weights_path, unexpected[:5],
        )
    return _freeze(model, device)


def load_proteoscribe_trainable(
    cfg: Namespace,
    weights_path: Optional[str],
    device: Optional[str] = None,
) -> nn.Module:
    _require_weights(weights_path, "stage3_init_weights",
                     "the policy would start from a randomly initialised ProteoScribe")
    model = prepare_model_ProteoScribe(
        config_args=cfg,
        model_fpath=weights_path,
        device=device,
        strict=True,
        eval=False,
        attempt_correction=True,
        verbosity=2
    )
    return model
