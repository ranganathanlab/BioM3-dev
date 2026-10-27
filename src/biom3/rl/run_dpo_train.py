#!/usr/bin/env python3
"""BioM3 Stage 3 DPO fine-tuning runner.

Offline Direct Preference Optimization. Loads three configs (Stage 1/2/3) plus
a DPO config, builds the trainable policy + frozen reference, ingests a scored
preference CSV (as produced by ``data/rl/convert_rl_data_to_csv.py``), and runs
the loop in ``biom3.rl.dpo.dpo_train``.

Configuration precedence (high -> low):  CLI  >  --config_path JSON  >  defaults

Example:

biom3_dpo_train \\
    --config_path configs/dpo/example_dpo_paired.json \\
    --run_id dpo_001 \\
    --data_csv data/rl/processed/biom3_designs.csv
"""

import argparse
import os
import sys
from typing import List, Optional

import torch

from biom3.backend.device import get_device, setup_logger
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.rl.dpo import DPOConfig, dpo_train
from biom3.rl.preference_data import PreferenceSampler, load_groups
from biom3.rl.io import add_conditioning_args, configure_conditioning, log_mask_warning

logger = setup_logger(__name__)


def _parse_float_list(s) -> Optional[List[float]]:
    if s is None or isinstance(s, list):
        return [float(x) for x in s] if s else None
    s = s.strip()
    return [float(x) for x in s.split(",")] if s else None


def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--run_id", type=str, default="dpo_run")
    parser.add_argument("--output_root", type=str, default="./outputs/dpo")

    parser.add_argument("--stage1_config", type=str, required=False)
    parser.add_argument("--stage2_config", type=str, required=False)
    parser.add_argument("--stage3_config", type=str, required=False)
    parser.add_argument("--stage1_weights", type=str, default=None)
    parser.add_argument("--stage2_weights", type=str, default=None)
    parser.add_argument("--stage3_init_weights", type=str, default=None)

    # Preference data
    parser.add_argument("--data_csv", type=str, required=False,
                        help="Scored preference CSV (dataset,source,prompt,prompt_text,"
                             "functional,sequence,score).")
    parser.add_argument("--dataset_filter", type=str, default=None,
                        help="Restrict to one dataset value (e.g. 'biom3' or 'vae').")
    parser.add_argument("--group_by", type=str, default="prompt_text",
                        help="Column whose values define per-caption groups (Case B). "
                             "If absent/empty, all rows share --default_caption (Case C).")
    parser.add_argument("--default_caption", type=str, default="SH3 domain protein.")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Drop sequences longer than this many residues.")
    parser.add_argument("--min_group_size", type=int, default=2)

    # DPO objective
    parser.add_argument("--loss_type", type=str, default="paired",
                        choices=["paired", "weighted"])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--length_normalize", dest="length_normalize",
                        action="store_true", default=True)
    parser.add_argument("--no_length_normalize", dest="length_normalize",
                        action="store_false")
    # paired
    parser.add_argument("--pairing", type=str, default="margin",
                        choices=["margin", "label"])
    parser.add_argument("--gap_level", type=float, default=0.5)
    parser.add_argument("--min_margin", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    # weighted
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)

    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["constant", "linear"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # SDMC ELBO
    parser.add_argument("--n_quadrature", type=int, default=3)
    parser.add_argument("--quadrature_grid", type=str, default="uniform",
                        choices=["uniform", "explicit"])
    parser.add_argument("--quadrature_points", type=str, default=None)
    parser.add_argument("--quadrature_weights", type=str, default=None)
    parser.add_argument("--inner_mc", type=int, default=1)
    parser.add_argument("--eps_t", type=float, default=1e-3)
    parser.add_argument("--no_gradient_checkpoint", dest="gradient_checkpoint",
                        action="store_false", default=True)

    # pre-unmask
    parser.add_argument("--pre_unmask", action="store_true", default=False)
    parser.add_argument("--pre_unmask_config", type=str, default=None)

    parser.add_argument("--device", type=str, default=None)
    add_conditioning_args(parser)
    return parser


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="BioM3 Stage 3 DPO fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = get_args(parser)
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config_path is not None:
        parser.set_defaults(**load_json_config(pre_args.config_path))
    args = parser.parse_args(argv)
    args._argv = list(argv)
    return args


def _required(value, name):
    if value is None:
        raise ValueError(f"Missing required arg: --{name} (or set in --config_path JSON)")
    return value


def main(args):
    device = torch.device(args.device) if args.device else get_device()
    logger.info("Device: %s", device)

    cfg1 = convert_to_namespace(load_json_config(_required(args.stage1_config, "stage1_config")))
    mask_warning = configure_conditioning(args, cfg1)
    cfg2 = convert_to_namespace(load_json_config(_required(args.stage2_config, "stage2_config")))
    cfg3 = convert_to_namespace(load_json_config(_required(args.stage3_config, "stage3_config")))

    groups = load_groups(
        _required(args.data_csv, "data_csv"),
        dataset=args.dataset_filter,
        group_by=args.group_by,
        default_caption=args.default_caption,
        max_len=args.max_len,
        min_group_size=args.min_group_size,
    )
    logger.info("Loaded %d preference group(s) from %s", len(groups), args.data_csv)
    sampler = PreferenceSampler(groups, image_size=cfg3.image_size, seed=args.seed)

    dpo_cfg = DPOConfig(
        output_dir=os.path.join(args.output_root, args.run_id),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        steps=args.steps,
        batch_size=args.batch_size,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        loss_type=args.loss_type,
        pairing=args.pairing,
        gap_level=args.gap_level,
        min_margin=args.min_margin,
        label_smoothing=args.label_smoothing,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        length_normalize=args.length_normalize,
        n_quadrature=args.n_quadrature,
        quadrature_grid=args.quadrature_grid,
        quadrature_points=_parse_float_list(args.quadrature_points),
        quadrature_weights=_parse_float_list(args.quadrature_weights),
        inner_mc=args.inner_mc,
        eps_t=args.eps_t,
        gradient_checkpoint=args.gradient_checkpoint,
        pre_unmask=args.pre_unmask,
        pre_unmask_config=args.pre_unmask_config,
        default_caption=args.default_caption,
    )

    dpo_train(
        dpo_cfg=dpo_cfg, cfg1=cfg1, cfg2=cfg2, cfg3=cfg3,
        sampler=sampler, device=device,
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        stage3_init_weights=args.stage3_init_weights,
    )
    log_mask_warning(mask_warning)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
