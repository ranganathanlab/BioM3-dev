#!/usr/bin/env python3

"""Training script for BioM3 Stage 1 (PenCL).

Follows the Stage 3 convention: config composition via JSON + _base_configs,
per-run output layout under {output_root}/{checkpoints_folder|runs_folder}/{run_id}/,
build manifest + args.json + run.log in the artifacts dir. Dispatches across
dataset_type in {default, masked, pfam, pfam_ablated}.
"""

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger, set_float32_matmul_precision
from biom3.backend.device import DEVICE_CHOICES, resolve_device, check_devices_per_node

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning import Trainer
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
    from lightning.pytorch.callbacks import (
        LearningRateMonitor, DeviceStatsMonitor, EarlyStopping,
    )
    from lightning.pytorch.strategies import SingleDeviceStrategy, DDPStrategy
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import (
        LearningRateMonitor, DeviceStatsMonitor, EarlyStopping,
    )
    from pytorch_lightning.strategies import SingleDeviceStrategy, DDPStrategy

import biom3.Stage1.preprocess as prep
import biom3.Stage1.model as mod
import biom3.Stage1.PL_wrapper as PL_mod
from biom3.core.dry_run import coerce_dry_run_output, run_dry_run
from biom3.core.helpers import coerce_limit_batches, load_json_config
from biom3.core.run_utils import (
    backup_if_exists, collect_training_env, resolve_devices_per_node,
    setup_file_logging, teardown_file_logging, write_manifest,
)
from biom3.backend.device import print_gpu_initialization
from biom3.core.distributed import get_global_rank

logger = setup_logger(__name__)

_LOGS_SUBDIR = "logs"
_ARTIFACTS_SUBDIR = "artifacts"


def parse_lr_scaling(value):
    """Normalize --scale_learning_rate to a scaling mode.

    Ported from Stage3/run_PL_training.py so both stages accept the same
    vocabulary. Returns ``None`` (no scaling), ``'linear'`` (lr * n), or
    ``'sqrt'`` (lr * sqrt(n)). Accepts booleans and the case-insensitive
    strings true/false/linear/sqrt; ``True``/``'true'`` map to ``'linear'``
    for backward compatibility.
    """
    if isinstance(value, bool):
        return 'linear' if value else None
    s = str(value).strip().lower()
    if s in ('false', 'none', ''):
        return None
    if s in ('true', 'linear'):
        return 'linear'
    if s == 'sqrt':
        return 'sqrt'
    raise ValueError(
        f"scale_learning_rate must be one of true/false/linear/sqrt, got {value!r}")


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Input must be 'True' or 'False'")


def none_or_int(s):
    """None / 'None' -> None; otherwise an int. For the checkpoint cadences.

    These had no argparse type, so `--checkpoint_every_n_epochs 1` arrived as
    the string '1' and reached ModelCheckpoint(every_n_epochs='1'), while an
    integer from a JSON config raised in nonestr_to_none. No job had set either
    until the run2 configs asked for a checkpoint every epoch.
    """
    if s is None or (isinstance(s, str) and s.lower() == 'none'):
        return None
    return int(s)


def nonestr_to_none(s):
    if isinstance(s, str):
        if s.lower() == 'none':
            return None
        return s
    elif s is None:
        return None
    else:
        raise ValueError("Input must be string or 'None'")


def get_args(parser):
    parser.add_argument('--description', type=str, default="",
                        help='Free-text description of the run.')
    parser.add_argument('--tags', type=str, nargs="*", default=[],
                        help='Tags for the run.')
    parser.add_argument('--notes', type=str, nargs="*", default=[],
                        help='Notes for the run.')

    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to Swiss-Prot CSV.')
    parser.add_argument('--pfam_data_path', type=str, default='None',
                        help='Path to Pfam CSV (required for dataset_type=pfam).')
    parser.add_argument('--pfam_splits_dir', type=str, default=None,
                        help='Where to write per-rank Pfam shards. '
                             'Default: under the run directory.')
    parser.add_argument('--dataset_type', type=str, default='default',
                        choices=['default', 'masked', 'pfam', 'pfam_ablated'],
                        help='Which dataset/loader variant to use. Selects the '
                             'PenCL training mode and matching collate/data flow.')
    parser.add_argument('--model_type', type=str, default='default',
                        choices=['default', 'pfam'],
                        help='Model class: default (PEN_CL) or pfam (pfam_PEN_CL). '
                             'Auto-set from dataset_type when unspecified.')

    parser.add_argument('--output_root', type=str, default='./outputs/Stage1/pretraining',
                        help='Base directory for all training outputs '
                             '(checkpoints/, runs/).')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints',
                        help='Subdirectory under output_root for checkpoint files.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='None',
                        help='Path to a Lightning .ckpt to resume training from. '
                             "Pass 'None' (string) or omit to start fresh.")
    parser.add_argument('--pretrained_weights', type=str, default='None',
                        help="Path to a raw weights file to load before training "
                             "(no optimizer state). Pass 'None' to skip.")

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling, weight init, and '
                             'reproducibility.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker count. 0 = main process only.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-device mini-batch size.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--valid_size', type=float, default=0.1,
                        help='Fraction of dataset held out for validation '
                             '(default 0.1 = 90/10 train/val split).')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Validation cadence (1.0 = once per epoch).')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Cap validation batches per epoch. Values >1 are an '
                             'absolute batch count; values in (0,1] are a fraction. '
                             'Default 1.0 uses the full val set, appropriate for '
                             'typical Stage 1 dataset sizes.')
    parser.add_argument('--limit_train_batches', type=float, default=None,
                        help='Cap training batches per epoch. Values >1 are an '
                             'absolute batch count; values in (0,1] are a fraction. '
                             'None = use full training dataset.')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                        help='Lightning Trainer log_every_n_steps (TensorBoard / wandb).')

    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='AdamW weight decay.')
    parser.add_argument('--head_lr', type=float, default=1e-3,
                        help='Learning rate for the projection head (z_t/z_p heads).')
    parser.add_argument('--protein_encoder_lr', type=float, default=1e-4,
                        help='Learning rate for the protein encoder (ESM-2) trunk.')
    parser.add_argument('--text_encoder_lr', type=float, default=1e-6,
                        help='Learning rate for the text encoder (BERT) trunk.')
    parser.add_argument('--choose_optim', type=str, default='AdamW',
                        help="Optimizer name (e.g. 'AdamW').")
    parser.add_argument('--scale_learning_rate', type=str, default='False',
                        help="'false'/'linear'/'sqrt' (Stage3-compatible; 'true' "
                             "means 'linear'). Scales all three learning rates by "
                             'num_nodes * devices_per_node, relative to '
                             '--lr_scale_baseline_ranks.')
    parser.add_argument('--contrastive_impl', type=str, default='dense',
                        choices=['dense', 'sharded'],
                        help="Contrastive loss implementation. 'dense' builds the "
                             'full M x M similarity matrix on every rank, which is '
                             'O(world_size^2): ~2 GB at 512 ranks, 32 GB at 2048. '
                             "'sharded' computes only this rank's rows against all "
                             'candidates, O(world_size). Mathematically identical '
                             '(tests/stage1_tests/test_sharded_contrastive.py checks '
                             'values and gradients); prefer it past a few hundred ranks.')
    parser.add_argument('--lr_scale_baseline_ranks', type=int, default=1,
                        help='World size the base learning rates are defined at. '
                             'Stage3 convention is 1. Run-1 Track B behaves as 96: '
                             'its factor 2.3 = sqrt(512/96).')

    parser.add_argument('--precision', type=str, default='32',
                        help="Training precision: '32', '16', 'bf16', 'bf16-mixed'.")
    parser.add_argument('--float32_matmul_precision', type=str, default='medium',
                        choices=['highest', 'high', 'medium'],
                        help="fp32 matmul precision. 'medium' (default) uses the bf16 "
                             "path; 'high' uses TF32 tensor cores; 'highest' keeps full "
                             "fp32. CLI overrides the config value.")
    parser.add_argument('--device', type=str, default='auto',
                        choices=list(DEVICE_CHOICES),
                        help='Compute device for training; auto = the detected '
                             'GPU backend (CUDA, then XPU; never falls back to CPU).')
    parser.add_argument('--devices_per_node', type=int, default=None,
                        help='Number of GPUs (CUDA) or tiles (XPU) per node. Default 1.')
    parser.add_argument('--gpu_devices', type=int, default=None,
                        help='(deprecated, use --devices_per_node) preserved for '
                             'backward compatibility with older configs and scripts.')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='Number of nodes participating in training.')
    parser.add_argument('--acc_grad_batches', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch size = '
                             'batch_size * acc_grad_batches * num_nodes * devices_per_node.')

    parser.add_argument('--save_metrics_history', type=str, default='True',
                        help="'True'/'False'. Save MetricsHistoryCallback "
                             'JSONL files alongside checkpoints.')
    parser.add_argument('--metrics_history_every_n_steps', type=int, default=1,
                        help='Record training metrics every N global steps.')
    parser.add_argument('--metrics_history_every_n_epochs', type=int, default=None,
                        help='Also record training metrics every N epochs (captures '
                             'epoch-averaged values). None disables.')
    parser.add_argument('--metrics_history_ranks', type=int, nargs='*', default=[0],
                        help='Rank indices that write metrics JSONL '
                             '(default: rank 0 only).')
    parser.add_argument('--metrics_history_all_ranks_val_loss', type=str, default='False',
                        help="'True'/'False'. Diagnostic: dump val_loss per "
                             'rank at epoch end to check sync_dist consistency.')

    parser.add_argument('--save_benchmark', type=str, default='False',
                        help="'True'/'False'. Save per-epoch wall-clock + memory "
                             'benchmark to artifacts/benchmark_history.json.')
    parser.add_argument('--benchmark_skip_first_epoch', type=str, default='True',
                        help="'True'/'False'. Exclude first epoch from "
                             'benchmark summary stats (treats it as warmup).')
    parser.add_argument('--benchmark_all_ranks_memory', type=str, default='False',
                        help="'True'/'False'. All-gather peak memory from every "
                             'rank (vs rank-0 only).')
    parser.add_argument('--benchmark_per_step', type=str, default='False',
                        help='Write per-step wall-clock timing to benchmark_steps.jsonl')

    parser.add_argument('--early_stopping_metric', type=str, default='None',
                        help="Metric to monitor for early stopping (e.g. 'val_loss'). "
                             "'None' disables early stopping.")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Validation checks with no improvement before stopping.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum change in monitored metric to qualify as improvement.')
    parser.add_argument('--early_stopping_mode', type=str, default='min',
                        help="'min' or 'max' — whether smaller or larger is better.")

    parser.add_argument('--checkpoint_monitors', default=None,
                        help='JSON list of {"metric","mode"} dicts, or None for default val_loss.')
    parser.add_argument('--checkpoint_every_n_steps', default=None,
                        help='Periodic snapshot every N training steps '
                             '(orthogonal to best-metric saves).')
    parser.add_argument('--checkpoint_every_n_epochs', default=None,
                        help='Periodic snapshot every N epochs '
                             '(orthogonal to best-metric saves).')
    parser.add_argument('--checkpoint_periodic_max_keep', type=int, default=-1,
                        choices=[-1, 0, 1],
                        help='Periodic snapshot retention: -1 = keep all '
                             '(default), 0 = disable, 1 = keep only most recent.')
    parser.add_argument('--use_sync_safe_checkpoint', type=str, default='False',
                        help="'True'/'False'. Use SyncSafeModelCheckpoint to "
                             'bypass reduce_boolean_decision (workaround for '
                             'XPU/CCL integer all-reduce bug).')
    return parser


def get_model_args(parser):
    parser.add_argument('--seq_model_path', type=str,
                        default='./weights/LLMs/esm2_t33_650M_UR50D.pt',
                        help='Path to the ESM-2 protein LM weights.')
    parser.add_argument('--pretrained_seq', type=str, default='True',
                        help="'True'/'False'. Load pretrained ESM-2 weights "
                             '(False = random init).')
    parser.add_argument('--trainable_seq', type=str, default='True',
                        help="'True'/'False'. Allow ESM-2 parameters to update.")
    parser.add_argument('--pLM_n_layers_to_finetune', type=int, default=1,
                        help='Number of last ESM-2 layers to train when '
                             'trainable_seq=True (rest are frozen).')
    parser.add_argument('--rep_layer', type=int, default=33,
                        help='ESM-2 layer index used as the protein representation.')
    parser.add_argument('--protein_encoder_embedding', type=int, default=1280,
                        help='ESM-2 hidden dim (must match seq_model_path).')

    parser.add_argument('--text_model_path', type=str,
                        default='./weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                        help='Path to the BioBERT/BiomedBERT text encoder weights.')
    parser.add_argument('--pretrained_text', type=str, default='True',
                        help="'True'/'False'. Load pretrained text encoder weights.")
    parser.add_argument('--trainable_text', type=str, default='True',
                        help="'True'/'False'. Allow text encoder parameters to update.")
    parser.add_argument('--bLM_n_layers_to_finetune', type=int, default=1,
                        help='Number of last text-encoder layers to train when '
                             'trainable_text=True.')
    parser.add_argument('--text_encoder_embedding', type=int, default=768,
                        help='Text encoder hidden dim (must match text_model_path).')
    parser.add_argument('--text_padding', type=str, default='max_padding',
                        choices=['max_padding', 'dynamic'],
                        help="Caption padding for training. 'max_padding' pads every "
                             "caption to text_max_length (Run 1); 'dynamic' crops each "
                             "batch to its longest caption, rounded up to a multiple of "
                             "64. BERT gets the attention mask either way, so z_t is the "
                             "same to float rounding; dynamic just skips the [PAD] compute.")
    parser.add_argument('--text_max_length', type=int, default=512,
                        help='Max BERT token length. Must be 512 (training '
                             'pad-to-max contract — see docs/bug_reports/).')

    parser.add_argument('--proj_embedding_dim', type=int, default=512,
                        help='Output dim of the projection heads (z_t, z_p).')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate inside the projection heads.')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Softmax temperature for the contrastive loss.')
    parser.add_argument('--mask_same_sequence', type=str, default='False',
                        help="'True'/'False'. Exclude candidates whose protein "
                             'sequence is identical to the anchor\'s from the '
                             'contrastive denominator. Swiss-Prot carries ~8.9 '
                             'caption variants per accession, so ~5.6% of rows '
                             'have such a false negative in the pooled batch.')
    parser.add_argument('--mask_same_family', type=str, default='False',
                        help="'True'/'False'. Exclude candidates drawn on the "
                             'same Pfam family as the anchor. At M = 49,152 a '
                             'row has ~25 of these against the single (i, i+-N) '
                             'pair the homolog index rule already covers; for '
                             'L_PFC they are the very items it exists to pull '
                             'together. Defaults off so Run 1 is reproducible.')
    parser.add_argument('--uniformity_weight', type=float, default=0.0,
                        help='Weight of the Wang & Isola (2020) uniformity term '
                             'on the L2-normalized joint embeddings (pfam dataset '
                             'types only). 0 disables it.')
    parser.add_argument('--uniformity_t', type=float, default=2.0,
                        help='Gaussian-potential scale t in the uniformity term. '
                             'Its minimum in high dimension is about -2t.')
    parser.add_argument('--log_uniformity', type=str, default='False',
                        help="'True'/'False'. Compute and log the uniformity term "
                             'without adding it to the loss, so a weight-0 control '
                             'run reports the same metric as a weighted one. '
                             'Implied when --uniformity_weight > 0.')
    parser.add_argument('--uniformity_on', type=str, default='both',
                        choices=['protein', 'text', 'both'],
                        help='Which joint embeddings the uniformity term spreads: '
                             'z_p, z_t, or the mean of both.')

    parser.add_argument('--sequence_keyword', type=str, default='protein_sequence',
                        help='CSV column name for the protein sequence.')
    parser.add_argument('--id_keyword', type=str, default='primary_Accession',
                        help='CSV column name for the protein accession ID.')
    return parser


def get_path_args(parser):
    parser.add_argument('--run_id', default=None, type=str,
                        help='Unique identifier for this training run.')
    parser.add_argument('--runs_folder', default='runs', type=str,
                        help='Subdirectory under output_root for per-run logs and artifacts.')
    return parser


def get_wrapper_args(parser):
    parser.add_argument('--wandb', type=str, default='False',
                        help="'True'/'False'. Enable Weights & Biases logging. "
                             'Requires WANDB_API_KEY in env when True.')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Optional wandb run display name.')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb entity (team/user) the run belongs to.')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='wandb project the run belongs to.')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                        help='wandb tags applied to the run.')
    parser.add_argument('--dry_run', default='False', type=str,
                        help='Print effective config (with provenance), output '
                             'paths, distributed/batch math, and an a-priori '
                             'memory estimate, then exit without training.')
    parser.add_argument('--dry_run_output', default='False', type=str,
                        help="Where to write the JSON dry-run report. 'False' "
                             "(default) prints to stdout only; 'True' writes "
                             "to <artifacts_dir>/dry_run_report.json; any "
                             "other string is treated as a filepath.")
    return parser


def retrieve_all_args(args):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_path', '-c', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(description='Stage 1: PenCL training')
    parser.add_argument('--config_path', '-c', type=str, default=None,
                        help='Path to JSON config file. CLI overrides JSON.')
    get_args(parser)
    get_model_args(parser)
    get_path_args(parser)
    get_wrapper_args(parser)

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    argv = list(args)
    args = parser.parse_args(args)
    args._argv = argv  # consumed by core.dry_run for CLI provenance attribution

    # Type conversions (idempotent)
    args.pretrained_seq = str_to_bool(args.pretrained_seq)
    args.trainable_seq = str_to_bool(args.trainable_seq)
    args.pretrained_text = str_to_bool(args.pretrained_text)
    args.trainable_text = str_to_bool(args.trainable_text)
    args.scale_learning_rate = parse_lr_scaling(args.scale_learning_rate)
    args.wandb = str_to_bool(args.wandb)
    args.save_metrics_history = str_to_bool(args.save_metrics_history)
    args.log_uniformity = str_to_bool(args.log_uniformity)
    args.mask_same_sequence = str_to_bool(args.mask_same_sequence)
    args.mask_same_family = str_to_bool(args.mask_same_family)
    args.metrics_history_all_ranks_val_loss = str_to_bool(
        args.metrics_history_all_ranks_val_loss
    )
    args.use_sync_safe_checkpoint = str_to_bool(args.use_sync_safe_checkpoint)
    args.save_benchmark = str_to_bool(args.save_benchmark)
    args.benchmark_skip_first_epoch = str_to_bool(args.benchmark_skip_first_epoch)
    args.benchmark_all_ranks_memory = str_to_bool(args.benchmark_all_ranks_memory)
    args.benchmark_per_step = str_to_bool(args.benchmark_per_step)

    args.data_path = nonestr_to_none(args.data_path) if args.data_path is not None else None
    args.pfam_data_path = nonestr_to_none(args.pfam_data_path)
    args.pfam_splits_dir = nonestr_to_none(args.pfam_splits_dir)
    args.resume_from_checkpoint = nonestr_to_none(args.resume_from_checkpoint)
    args.pretrained_weights = nonestr_to_none(args.pretrained_weights)
    args.early_stopping_metric = nonestr_to_none(args.early_stopping_metric)
    args.checkpoint_every_n_steps = none_or_int(args.checkpoint_every_n_steps)
    args.checkpoint_every_n_epochs = none_or_int(args.checkpoint_every_n_epochs)

    # Auto-derive model_type from dataset_type if user left it at the default
    if args.dataset_type in ('pfam', 'pfam_ablated') and args.model_type == 'default':
        logger.info("Auto-setting model_type='pfam' for dataset_type=%s", args.dataset_type)
        args.model_type = 'pfam'
    if args.uniformity_weight > 0 and args.dataset_type not in ('pfam', 'pfam_ablated'):
        raise ValueError(
            f"--uniformity_weight is only applied for dataset_type pfam/pfam_ablated, "
            f"got dataset_type={args.dataset_type}")

    # Construct a default run_id if missing
    if args.run_id is None:
        args.run_id = f"stage1_{args.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Default pfam_splits_dir under the run directory so writes don't land next to input
    if args.dataset_type in ('pfam', 'pfam_ablated') and args.pfam_splits_dir is None:
        args.pfam_splits_dir = os.path.join(
            args.output_root, args.runs_folder, args.run_id, 'pfam_splits',
        )

    resolve_devices_per_node(args)

    args.dry_run = str_to_bool(args.dry_run)
    args.dry_run_output = coerce_dry_run_output(args.dry_run_output)

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_gpu_cache():
    gc.collect()


def get_dataloaders_models(args):
    data_module_options = {
        'default': prep.Default_DataModule,
        'masked': prep.Default_DataModule,
        'pfam': prep.Pfam_DataModule,
        'pfam_ablated': prep.Pfam_DataModule,
    }
    data_module_class = data_module_options.get(args.dataset_type, prep.Default_DataModule)
    data_module = data_module_class(args=args)

    model_options = {
        'default': mod.PEN_CL,
        'pfam': mod.pfam_PEN_CL,
    }
    model_class = model_options.get(args.model_type, mod.PEN_CL)
    model = model_class(args=args)

    wrapper_options = {
        'default': PL_mod.PL_PEN_CL,
        'masked': PL_mod.mask_PL_PEN_CL,
        'pfam': PL_mod.pfam_PL_PEN_CL,
        'pfam_ablated': PL_mod.pfam_PL_PEN_CL,
    }
    wrapper_class = wrapper_options.get(args.dataset_type, PL_mod.PL_PEN_CL)

    PL_model = wrapper_class(
        args=args,
        model=model,
        text_tokenizer=model.text_encoder.tokenizer,
        sequence_tokenizer=model.protein_encoder.alphabet,
    )
    return data_module, PL_model


def load_pretrained_weights(PL_model, checkpoint_path: str):
    logger.info("Loading pretrained weights from %s", checkpoint_path)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
        sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
    missing, unexpected = PL_model.model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading pretrained weights: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys when loading pretrained weights: %d", len(unexpected))
    return PL_model


def save_model(args, checkpoint_path, artifacts_path, PL_model, trainer):
    if not trainer.is_global_zero:
        return
    try:
        best_ckpt_fpath = trainer.checkpoint_callback.best_model_path
    except Exception:
        best_ckpt_fpath = None

    best_state_dict_fpath = os.path.join(checkpoint_path, 'state_dict.best.pth')
    backup_if_exists(best_state_dict_fpath)

    if best_ckpt_fpath and os.path.exists(best_ckpt_fpath):
        logger.info("Loading best checkpoint from %s", best_ckpt_fpath)
        ckpt = torch.load(best_ckpt_fpath, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
        torch.save(sd, best_state_dict_fpath)
        logger.info("Saved best state_dict to %s", best_state_dict_fpath)

        artifacts_copy = os.path.join(artifacts_path, 'state_dict.best.pth')
        backup_if_exists(artifacts_copy)
        torch.save(sd, artifacts_copy)
        logger.info("Copied best state_dict to %s", artifacts_copy)
    else:
        logger.warning("No best checkpoint path reported by trainer; skipping save_model")


def train_model(args, PL_model, data_module):
    logger.info("Beginning Stage 1 training...")

    output_root = args.output_root
    checkpoints_folder = args.checkpoints_folder
    runs_folder = args.runs_folder
    run_id = args.run_id
    log_every_n_steps = args.log_every_n_steps
    devices_per_node = args.devices_per_node
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    val_check_interval = args.val_check_interval
    limit_val_batches = args.limit_val_batches
    resume_from_checkpoint = args.resume_from_checkpoint
    precision = args.precision
    use_wandb = args.wandb

    if args.scale_learning_rate:
        n = (num_nodes * devices_per_node) / max(1, args.lr_scale_baseline_ranks)
        factor = n ** 0.5 if args.scale_learning_rate == 'sqrt' else n
        logger.info(
            "Scaling LRs (%s) by num_nodes x devices_per_node = %s x %s, "
            "baseline_ranks=%s -> factor %.4g",
            args.scale_learning_rate, num_nodes, devices_per_node,
            args.lr_scale_baseline_ranks, factor,
        )
        args.head_lr *= factor
        args.protein_encoder_lr *= factor
        args.text_encoder_lr *= factor

    checkpoint_dir = os.path.join(output_root, checkpoints_folder, run_id)
    run_dir = os.path.join(output_root, runs_folder, run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    if get_global_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

    loggers = [TensorBoardLogger(save_dir=logs_dir, version="")]
    if use_wandb:
        loggers.append(WandbLogger(
            name=args.wandb_name or run_id,
            save_dir=logs_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            tags=args.wandb_tags,
            group=run_id,
        ))

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [lr_monitor]
    if args.device != 'cpu':
        callbacks.append(DeviceStatsMonitor())

    from biom3.Stage3.callbacks import build_checkpoint_callbacks
    monitored_callbacks, periodic_callback = build_checkpoint_callbacks(
        checkpoint_dir=checkpoint_dir,
        checkpoint_monitors=args.checkpoint_monitors
            if args.checkpoint_monitors is not None
            else [{"metric": "valid_loss", "mode": "min"}],
        periodic_every_n_steps=args.checkpoint_every_n_steps,
        periodic_every_n_epochs=args.checkpoint_every_n_epochs,
        periodic_max_keep=args.checkpoint_periodic_max_keep,
        use_sync_safe=args.use_sync_safe_checkpoint,
    )
    checkpoint_callbacks = monitored_callbacks + (
        [periodic_callback] if periodic_callback is not None else []
    )
    callbacks = checkpoint_callbacks + callbacks

    if args.save_metrics_history:
        # Stage 1 subclass: explicit metric list. Stage 3's version harvests by
        # prefix and silently drops Stage 1's `valid_*` metrics and the LR.
        from biom3.Stage1.callbacks import Stage1MetricsHistoryCallback
        callbacks.append(Stage1MetricsHistoryCallback(
            output_dir=artifacts_dir,
            save_ranks=args.metrics_history_ranks,
            every_n_steps=args.metrics_history_every_n_steps,
            every_n_epochs=args.metrics_history_every_n_epochs,
            all_ranks_val_loss=args.metrics_history_all_ranks_val_loss,
        ))

    if args.save_benchmark:
        from biom3.Stage3.callbacks import TrainingBenchmarkCallback
        callbacks.append(TrainingBenchmarkCallback(
            output_dir=artifacts_dir,
            batch_size=args.batch_size,
            acc_grad_batches=acc_grad_batches,
            devices_per_node=devices_per_node,
            num_nodes=num_nodes,
            precision=precision,
            training_strategy='primary_only',
            num_workers=args.num_workers,
            skip_first_epoch=args.benchmark_skip_first_epoch,
            all_ranks_memory=args.benchmark_all_ranks_memory,
            per_step=args.benchmark_per_step,
        ))

    if args.early_stopping_metric is not None:
        callbacks.append(EarlyStopping(
            monitor=args.early_stopping_metric,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=args.early_stopping_mode,
            verbose=True,
        ))

    _world = num_nodes * devices_per_node
    if _world > 1:
        # Explicitly configured DDP, mirroring Stage3/run_PL_training.py. Do NOT
        # fall back to strategy='auto' on XPU: Lightning's _choose_strategy()
        # does not know about 'xpu', and an unconfigured multi-device XPU run
        # hangs in Trainer.__init__ (observed on 2 nodes x 12 tiles, 8+ minutes
        # with idle GPUs).
        #
        # process_group_backend='xccl': frameworks/2025.3.1 removed
        #   oneccl-bindings-for-pytorch and replaced the 'ccl' backend with
        #   torch's native 'xccl'.
        # static_graph=True: the autograd graph is stable across iterations, so
        #   DDP precomputes the gradient-bucket ready order at step 0. Without
        #   it, ranks fire bucket allreduces in different orders on oneCCL and
        #   deadlock on a mismatched collective.
        # gradient_as_bucket_view=True: small memory win, pairs with static_graph.
        strategy = DDPStrategy(
            process_group_backend='xccl' if args.device == 'xpu' else None,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
    elif args.device == 'xpu':
        # Single device. Lightning's _choose_strategy() falls back to
        # SingleDeviceStrategy(device="cpu") for XPU, which makes
        # trainer.strategy.root_device report CPU and breaks DeviceStatsMonitor.
        strategy = SingleDeviceStrategy(device=torch.device('xpu'))
    else:
        strategy = 'auto'

    # XPU manual bf16 autocast (Stage 3 / Rama pattern): PL's bf16 AMP plugin is
    # broken on Aurora, so we wrap training_step/validation_step ourselves.
    if args.device == 'xpu' and isinstance(precision, str) and 'bf16' in precision:
        import types as _types
        _orig_training = PL_model.training_step
        _orig_validation = PL_model.validation_step

        def _bf16_training_step(_self, batch, batch_idx):
            with torch.autocast(device_type='xpu', dtype=torch.bfloat16):
                return _orig_training(batch, batch_idx)

        def _bf16_validation_step(_self, batch, batch_idx):
            with torch.autocast(device_type='xpu', dtype=torch.bfloat16):
                return _orig_validation(batch, batch_idx)

        PL_model.training_step = _types.MethodType(_bf16_training_step, PL_model)
        PL_model.validation_step = _types.MethodType(_bf16_validation_step, PL_model)
        trainer_precision = '32'
        logger.info("Manual XPU bf16 autocast enabled; Trainer precision forced to '32'.")
    else:
        trainer_precision = precision

    trainer_params = {
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'enable_checkpointing': True,
        'devices': devices_per_node,
        'num_nodes': num_nodes,
        'accelerator': args.device,
        'strategy': strategy,
        'accumulate_grad_batches': acc_grad_batches,
        'logger': loggers,
        'log_every_n_steps': log_every_n_steps,
        'callbacks': callbacks,
        'precision': trainer_precision,
        'max_epochs': epochs,
        'val_check_interval': val_check_interval,
        'limit_val_batches': coerce_limit_batches(limit_val_batches),
        'num_sanity_val_steps': 0,
    }
    if args.limit_train_batches is not None:
        trainer_params['limit_train_batches'] = coerce_limit_batches(
            args.limit_train_batches
        )

    logger.info("Initializing Trainer with strategy=%s, precision=%s", strategy, trainer_precision)
    trainer = Trainer(**trainer_params)

    if resume_from_checkpoint is None:
        logger.info("Train from scratch")
        trainer.fit(PL_model, data_module)
    else:
        logger.info("Resume from checkpoint: %s", resume_from_checkpoint)
        trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)

    if get_global_rank() == 0:
        print_gpu_initialization()

    save_model(
        args=args,
        checkpoint_path=checkpoint_callbacks[0].dirpath,
        artifacts_path=artifacts_dir,
        PL_model=PL_model,
        trainer=trainer,
    )


def main(args):
    start_time = datetime.now()

    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    logging.getLogger("tensorboardX.x2num").setLevel(logging.ERROR)

    # A dry run only probes config, data and model, so it may land on CPU; a
    # real run must find a GPU unless --device cpu was asked for explicitly.
    dry_run = getattr(args, 'dry_run', False)
    args.device = resolve_device(args.device, allow_cpu=dry_run)
    if not dry_run:
        check_devices_per_node(args.device, args.devices_per_node)

    if getattr(args, 'dry_run', False):
        return run_dry_run(
            args,
            stage="stage1",
            dataset_probe=lambda: _stage1_dataset_probe(args),
            model_probe=lambda: _stage1_model_probe(args),
        )

    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    checkpoint_dir = os.path.join(args.output_root, args.checkpoints_folder, args.run_id)
    if get_global_rank() == 0:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
    log_path, file_handler = setup_file_logging(artifacts_dir)
    try:
        set_float32_matmul_precision(args.float32_matmul_precision)
        clear_gpu_cache()

        seed = args.seed
        if seed <= 0:
            seed = np.random.randint(2 ** 32)
            args.seed = seed
        set_seed(seed)
        logger.info("Using seed: %s", seed)

        data_module, PL_model = get_dataloaders_models(args=args)
        logger.info("PenCL parameters: %s", sum(p.numel() for p in PL_model.model.parameters()))

        if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
            PL_model = load_pretrained_weights(PL_model, args.pretrained_weights)
        elif args.pretrained_weights is not None:
            logger.warning("Pretrained weights path does not exist: %s", args.pretrained_weights)

        train_model(args=args, PL_model=PL_model, data_module=data_module)

        if get_global_rank() == 0:
            elapsed = datetime.now() - start_time

            args_path = os.path.join(artifacts_dir, "args.json")
            backup_if_exists(args_path)
            with open(args_path, "w") as f:
                json.dump({k: v for k, v in vars(args).items() if not k.startswith("_")},
                          f, indent=2, default=str)
            logger.info("Args written to %s", args_path)

            total_params = sum(p.numel() for p in PL_model.model.parameters())
            trainable_params = sum(
                p.numel() for p in PL_model.model.parameters() if p.requires_grad
            )

            outputs = {
                "seed": args.seed,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "batch_size": args.batch_size,
                "head_lr": args.head_lr,
                "protein_encoder_lr": args.protein_encoder_lr,
                "text_encoder_lr": args.text_encoder_lr,
                "precision": args.precision,
                "devices_per_node": args.devices_per_node,
                "num_nodes": args.num_nodes,
                "acc_grad_batches": args.acc_grad_batches,
                "epochs": args.epochs,
                "dataset_type": args.dataset_type,
                "model_type": args.model_type,
            }

            resolved_paths = {
                "checkpoint_dir": os.path.abspath(checkpoint_dir),
                "artifacts_dir": os.path.abspath(artifacts_dir),
            }
            if args.data_path is not None:
                resolved_paths["data_path"] = os.path.abspath(args.data_path)
            if args.pfam_data_path is not None:
                resolved_paths["pfam_data_path"] = os.path.abspath(args.pfam_data_path)
            if args.pretrained_weights is not None:
                resolved_paths["pretrained_weights"] = os.path.abspath(args.pretrained_weights)
            if args.resume_from_checkpoint is not None:
                resolved_paths["resume_from_checkpoint"] = os.path.abspath(args.resume_from_checkpoint)

            manifest_path = write_manifest(
                args, artifacts_dir, start_time, elapsed,
                outputs=outputs,
                resolved_paths=resolved_paths,
                environment=collect_training_env(),
            )
            logger.info("Build manifest written to %s", manifest_path)
    finally:
        teardown_file_logging("biom3", file_handler)


_DRY_RUN_CACHE = {}


def _stage1_dataset_probe(args):
    """CPU-only data-module probe for the dry-run report."""
    if 'pair' not in _DRY_RUN_CACHE:
        _DRY_RUN_CACHE['pair'] = get_dataloaders_models(args=args)
    data_module, _ = _DRY_RUN_CACHE['pair']
    if hasattr(data_module, 'setup'):
        try:
            data_module.setup()
        except Exception:
            pass
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader() if hasattr(data_module, 'val_dataloader') else None
    train_len = len(train_dl.dataset) if hasattr(train_dl, 'dataset') and hasattr(train_dl.dataset, '__len__') else None
    val_len = len(val_dl.dataset) if val_dl is not None and hasattr(val_dl, 'dataset') and hasattr(val_dl.dataset, '__len__') else None
    sample = next(iter(train_dl), None)
    return train_len, val_len, sample


def _stage1_model_probe(args):
    """CPU-only model probe for the dry-run report."""
    if 'pair' not in _DRY_RUN_CACHE:
        _DRY_RUN_CACHE['pair'] = get_dataloaders_models(args=args)
    _, PL_model = _DRY_RUN_CACHE['pair']
    return PL_model.model


def parse_arguments(args):
    return retrieve_all_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
