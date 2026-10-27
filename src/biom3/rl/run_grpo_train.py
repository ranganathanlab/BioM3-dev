#!/usr/bin/env python3
"""BioM3 Stage 3 GRPO fine-tuning runner.

Single-GPU GRPO training. Loads three configs (Stage 1, Stage 2, Stage 3)
plus a GRPO config, builds the policy + frozen reference, and runs the
GRPO loop in ``biom3.rl.grpo.grpo_train``.

Configuration precedence (high → low):
    CLI args  >  --config_path JSON  >  argparse defaults

Example:

biom3_grpo_train \\
    --config_path configs/grpo/example_grpo.json \\
    --run_id grpo_001 \\
    --steps 100 --num_generations 4
"""

import argparse
import os
import sys

import torch

from biom3.backend.device import get_device, setup_logger
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.rl.grpo import GRPOConfig, grpo_train, load_prompts
from biom3.rl.rewards import CompositeReward, DiversityReward, build_reward
from biom3.rl.io import add_conditioning_args, configure_conditioning, log_mask_warning

logger = setup_logger(__name__)


def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config_path", type=str, default=None,
                        help="GRPO experiment config JSON.")
    parser.add_argument("--run_id", type=str, default="grpo_run")
    parser.add_argument("--output_root", type=str, default="./outputs/grpo")

    # Conditioning + policy configs
    parser.add_argument("--stage1_config", type=str, required=False)
    parser.add_argument("--stage2_config", type=str, required=False)
    parser.add_argument("--stage3_config", type=str, required=False)

    # Weights
    parser.add_argument("--stage1_weights", type=str, default=None)
    parser.add_argument("--stage2_weights", type=str, default=None)
    parser.add_argument("--stage3_init_weights", type=str, default=None)

    # Prompts
    parser.add_argument("--prompts_path", type=str, required=False)

    # GRPO hyperparameters
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="K — sequences per prompt (group size).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Prompts per gradient update.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL coefficient.")
    parser.add_argument("--eps", type=float, default=0.20,
                        help="PPO clip epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Reward
    parser.add_argument("--reward", type=str, default="esmfold_plddt",
                        choices=["esmfold_plddt", "stub"])

    # Diversity reward (composite). When --diversity_weight > 0 the base
    # reward is wrapped in CompositeReward({reward: 1, diversity: w}); the
    # combined scalar feeds the trainer's group-relative advantage.
    parser.add_argument("--diversity_weight", type=float, default=0.0,
                        help="Weight on the diversity component of the composite "
                             "reward. 0 disables (default).")
    parser.add_argument("--diversity_mode", type=str, default="monotone",
                        choices=["monotone", "targeted"],
                        help="'monotone' rewards higher diversity linearly; "
                             "'targeted' peaks at --diversity_target.")
    parser.add_argument("--diversity_target", type=float, default=0.5,
                        help="Target divergence for --diversity_mode=targeted.")

    # Debug log
    parser.add_argument("--no_debug_log", dest="debug_log", action="store_false",
                        help="Disable per-step debug.out dump (sequences, ratios).")
    parser.set_defaults(debug_log=True)

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device override; defaults to backend.get_device().")

    add_conditioning_args(parser)
    return parser


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="BioM3 Stage 3 GRPO fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = get_args(parser)
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)
    args = parser.parse_args(argv)
    args._argv = list(argv)
    return args


def _required(value, name):
    if value is None:
        raise ValueError(f"Missing required arg: --{name} (or set in --config_path JSON)")
    return value


def main(args):
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info("Device: %s", device)

    cfg1 = convert_to_namespace(load_json_config(_required(args.stage1_config, "stage1_config")))
    mask_warning = configure_conditioning(args, cfg1)
    cfg2 = convert_to_namespace(load_json_config(_required(args.stage2_config, "stage2_config")))
    cfg3 = convert_to_namespace(load_json_config(_required(args.stage3_config, "stage3_config")))

    prompts = load_prompts(_required(args.prompts_path, "prompts_path"))
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts_path)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_path}")

    output_dir = os.path.join(args.output_root, args.run_id)

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        eps=args.eps,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        steps=args.steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        debug_log=args.debug_log,
    )

    reward_fn = build_reward(args.reward, device=device)
    if args.diversity_weight > 0:
        base_reward = reward_fn
        div_reward = DiversityReward(
            group_size=args.num_generations,
            scale=100.0,
            mode=args.diversity_mode,
            target=args.diversity_target,
        )
        reward_fn = CompositeReward(
            {
                args.reward: (base_reward, 1.0),
                "diversity": (div_reward, float(args.diversity_weight)),
            },
            reduction="weighted_sum",
        )
        logger.info(
            "Composite reward enabled: %s (w=1.0) + diversity (w=%.3f, mode=%s, target=%.3f)",
            args.reward, args.diversity_weight, args.diversity_mode, args.diversity_target,
        )

    grpo_train(
        grpo_cfg=grpo_cfg,
        cfg1=cfg1,
        cfg2=cfg2,
        cfg3=cfg3,
        prompts=prompts,
        reward_fn=reward_fn,
        device=device,
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        stage3_init_weights=args.stage3_init_weights,
    )
    log_mask_warning(mask_warning)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
