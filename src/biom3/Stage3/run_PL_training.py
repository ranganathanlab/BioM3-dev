#!/usr/bin/env python3

"""BioM3 Stage 3: ProteoScribe training and finetuning

Trains the conditional diffusion transformer (ProteoScribe) that generates
protein sequences from text-conditioned embeddings produced by Stage 2.
Supports pretraining from scratch, continuation with secondary data,
resumption from checkpoint, and selective-layer finetuning.

Config file:
    configs/stage3_training/*.json  (uses _base_configs composition)

Configuration precedence (high → low):
    CLI args  >  --config_path JSON  >  argparse defaults

Example: pretrain from scratch (epoch-based, primary data only)

biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id my_run_001 \
    --epochs 100

Example: continue with secondary data (step-based, combine strategy)

biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_phase2.json \
    --run_id my_run_phase2 \
    --resume_from_checkpoint /path/to/checkpoints/my_run_001/last.ckpt \
    --start_secondary True \
    --secondary_data_paths ./data/pfam_embeddings.hdf5 \
    --max_steps 3000000

Example: finetune selected blocks/layers from pretrained weights

biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --run_id finetune_001 \
    --finetune True \
    --pretrained_weights /path/to/state_dict.best.pth \
    --finetune_last_n_blocks 4 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True

Outputs are organized under {output_root}/{checkpoints_folder|runs_folder}/
{run_id}/ with checkpoints, derived state_dict.pth files, args.json, and
build_manifest.json. See docs/stage3_training.md for the full output
layout, metric definitions, and per-machine submission examples. See
docs/CLI_reference.md for the complete argument list.
"""

import sys
import os
import io
import json
import shutil
import contextlib
import logging
import math
import time
import warnings
import numpy as np
import random
import gc
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ----- Retrieve available device -----
from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger, set_float32_matmul_precision
from biom3.backend.device import DEVICE_CHOICES, resolve_device, check_devices_per_node

# Import pytorch lightning based on device
if BACKEND_NAME == _XPU:
    # lightning imports (from local installation)
    import lightning as pl
    from lightning import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy, SingleDeviceStrategy
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, EarlyStopping
    # Additional necessities
    from lightning.pytorch.plugins.environments import ClusterEnvironment, MPIEnvironment
    from lightning.pytorch.utilities.model_summary import ModelSummary
else:
    # PyTorch Lightning imports
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy, SingleDeviceStrategy
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, EarlyStopping
    # Additional necessities
    from pytorch_lightning.plugins.environments import ClusterEnvironment

# DeepSpeed is needed for the DeepSpeedStrategy
import deepspeed

# WandB
import wandb

# Custom modules
import biom3.Stage3.preprocess as prep
import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.PL_wrapper as PL_mod
from biom3.Stage3.callbacks import (
    BestArtifactSyncCallback,
    EpochProgressCallback,
    MetricsHistoryCallback,
    StepProgressCallback,
    TimeLimitCallback,
    TrainingBenchmarkCallback,
    build_checkpoint_callbacks,
)
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.core.dry_run import coerce_dry_run_output, run_dry_run
from biom3.core.helpers import coerce_limit_batches, load_json_config
from biom3.core.weight_sets import check_normalize_zc, merge_weight_set
from biom3.core.run_utils import (
    backup_if_exists, collect_training_env, resolve_devices_per_node,
    setup_file_logging, teardown_file_logging, write_manifest,
)
from biom3.backend.device import print_gpu_initialization, get_device
from biom3.core.distributed import get_global_rank

logger = setup_logger(__name__)

_LOGS_SUBDIR = "logs"
_ARTIFACTS_SUBDIR = "artifacts"

_BACKUP_HISTORY: dict[str, str] = {}

_MAIN_START_MONOTONIC: float | None = None

_LAST_TRAINER = None


def get_args(parser):
    """
    Configure argument parser with all training, model, and data parameters.
    
    This function adds a comprehensive set of arguments to the provided parser,
    including data paths, training hyperparameters, checkpointing options,
    model architecture settings, diffusion parameters, and dataset configurations.
    It serves as the central configuration point for the entire training pipeline.
    
    Args:
        parser: ArgumentParser object to which arguments will be added
        
    Returns:
        The parser with the complete set of arguments added
    """
    parser.add_argument('--description', default="", type=str,
                        help='human-readable description of this config (stored in args.json)')
    parser.add_argument('--tags', type=str, nargs='+', default=[],
                        help='tags for categorizing this run (stored in args.json)')
    parser.add_argument('--notes', type=str, nargs='+', default=[],
                        help='free-form notes about this run (stored in args.json)')

    parser.add_argument('--data_root', default="./data/ARDM_temp_homolog_family_dataset.csv", type=Path,
                        help='path to dataset root directory')

    parser.add_argument('--output_root', default=None, type=str,
                        help='base directory for all training outputs')
    parser.add_argument('--checkpoints_folder', default=None, type=str,
                        help='subdirectory under output_root for checkpoints')
    parser.add_argument('--resume_from_checkpoint', default='None', type=str,
                        help='checkpoint path to last model iteration (usually last.ckpt)')


    parser.add_argument('--dataset', default="normal", type=str,
                        choices=['normal', 'sequence'],
                        help='which dataset to train on')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loader workers')
    parser.add_argument('--warmup_steps', default=500, type=int,
                        help='number of learning rate warmup steps')
    parser.add_argument('--total_steps', default=1000, type=int,
                        help='total number of steps of minibatch gradient descent')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='mini-batch size')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
                        help='weight decay')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate')
    parser.add_argument('--ema_inv_gamma', default=1.0, type=float,
                        help='inverse gamma parameter for exponential moving average')
    parser.add_argument('--ema_power', default=0.75, type=float,
                        help='power parameter for exponential moving average')
    parser.add_argument('--ema_max_value', default=0.999, type=float,
                        help='max value parameter for exponential moving average')
    parser.add_argument('--precision', default='no', type=str, choices=['no', 'fp16', 'bf16', '32'],
                        help='whether to use 16-bit or 32-bit training')
    parser.add_argument('--float32_matmul_precision', default='medium', type=str,
                        choices=['highest', 'high', 'medium'],
                        help="fp32 matmul precision. 'medium' (default) uses the bf16 "
                             "path; 'high' uses TF32 tensor cores; 'highest' keeps full "
                             "fp32. CLI overrides the config value.")
    parser.add_argument('--seed', default=0, type=int,
                        help='random number seed')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path,
                        help='path to checkpoint directory')
    parser.add_argument('--checkpoint_prefix', default='channels',
                        help='prefix for local checkpoint')
    parser.add_argument('--device', default='auto', type=str,
                        choices=list(DEVICE_CHOICES),
                        help='computational device; auto = the detected GPU '
                             'backend (CUDA, then XPU; never falls back to CPU)')
    parser.add_argument('--model_option', default='transformer', type=str,
                        choices=['Unet', 'transformer'],
                        help='Choose model architecture')
    parser.add_argument('--download', default='True', type=str,
                        help='Download dataset')
    # Dataset paths (generalized)
    parser.add_argument('--primary_data_path', default='None', type=str,
                        help='path to primary training HDF5 dataset')
    parser.add_argument('--secondary_data_paths', default=None, type=str, nargs='+',
                        help='one or more paths to secondary HDF5 datasets')
    parser.add_argument('--training_strategy', default='auto', type=str,
                        choices=['auto', 'primary_only', 'combine'],
                        help='data handling strategy (auto: primary_only if no secondary, combine otherwise)')
    parser.add_argument('--split_manifest_path', default=None, type=str,
                        help='path to a curated train/val/test split manifest '
                             '(from biom3_cluster_split). When set, the random '
                             '80/20 split is bypassed and the test split is held out.')
    parser.add_argument('--loss_positions', default='all', type=str,
                        choices=['all', 'non_pad'],
                        help="unsampled positions whose loss drives the gradients. 'all' "
                             "(default) includes the padding at the tail of each sequence; "
                             "'non_pad' counts only the sequence itself (<START>, residues, "
                             "<END>). Either way all of loss_all, loss_non_pad and loss_pad "
                             "are computed and logged. Predicting pads is how the model "
                             "learns where a sequence ends, so 'non_pad' removes that signal; "
                             "its gradients are also larger, by about the model length over "
                             "the sequence length")
    # Conditioning blend: y = alpha * z_p + (1 - alpha) * z_c, alpha = weight on z_p
    parser.add_argument('--train_alpha', default='zc', type=str,
                        help="conditioning blend during training. 'zc' (default) "
                             "= text only, 'zp' = sequence only, 'blend' = the "
                             "per-example schedule {alpha=1: .25, alpha=0: .25, "
                             "U(0,1): .5}, or a constant in [0, 1].")
    parser.add_argument('--eval_alpha', default='spread', type=str,
                        help="blend used for validation batches. 'spread' (default) "
                             "gives each val example its own deterministic alpha "
                             "covering [0, 1], so best-checkpoint selection reflects "
                             "the whole operating range rather than one point. A "
                             "constant ('zc', 'zp', or a number in [0, 1]) evaluates "
                             "at a single alpha. Either way it is fixed across epochs; "
                             "the 'blend' training schedule is not allowed here.")
    parser.add_argument('--normalize_zc', default='False', type=str,
                        help="scale the conditioning vector to unit length before the "
                             "model sees it, after any z_p blend, in training and "
                             "validation. Generation has to match: a model trained this "
                             "way needs --normalize_zc in biom3_ProteoScribe_sample too")
    parser.add_argument('--zp_path', default=None, type=str,
                        help='Stage 2 Facilitator output (.pt) holding z_p row-aligned '
                             'with --primary_data_path. Required by biom3_train_stage3 '
                             'when --train_alpha puts weight on z_p. Ignored by '
                             'biom3_finetune_stage3, which precomputes z_p for every '
                             "unique train/val sequence via PenCL's protein branch.")
    # Deprecated aliases (mapped to primary/secondary in retrieve_all_args)
    parser.add_argument('--swissprot_data_root', default='None', type=str,
                        help='(deprecated, use --primary_data_path) path to SwissProt data')
    parser.add_argument('--pfam_data_root', default='None', type=str,
                        help='(deprecated, use --secondary_data_paths) path to Pfam data')

    parser.add_argument('--pretrained_weights', default='None', type=str,
                        help='path to .bin weight or checkpoint file containing model weights')
    parser.add_argument('--weight_set', default=None, type=str,
                        help='weight-set bundle JSON (e.g. configs/weights/run1_base.json). '
                             'Its proteoscribe_weights fill --pretrained_weights when that '
                             'is not given, and its record of how ProteoScribe was trained '
                             'is checked against --normalize_zc')
    
    parser.add_argument('--scale_learning_rate', default='True', type=str,
                        help="scale the learning rate by the total number of devices "
                             "(num_nodes x devices_per_node): 'true'/'linear' scales "
                             "linearly, 'sqrt' scales by its square root, 'false' "
                             "disables scaling")

    parser.add_argument('--distributed_strategy', default='deepspeed_zero2', type=str,
                        choices=['deepspeed_zero2', 'ddp'],
                        help='Lightning trainer strategy. deepspeed_zero2 (default): '
                             'DeepSpeed ZeRO Stage 2 with CPU offload. ddp: plain DDP '
                             'with static_graph=True, or a single-device strategy when '
                             'there is one rank. Distinct from --training_strategy '
                             '(which selects primary_only vs combine *data* mixing).')

    # Finetuning
    parser.add_argument('--finetune', default='False', type=str,
                        help='flag to run finetuning')
    parser.add_argument('--finetune_last_n_blocks', default=-2, type=int,
                        help='Number of last transformer blocks to finetune. '
                             '-1: all blocks, 0: no blocks, -2 (default): '
                             'unspecified → coerced to -1 (all blocks).')
    parser.add_argument('--finetune_last_n_layers', default=-2, type=int,
                        help='Number of last transformer layers per block to '
                             'finetune. -1: all layers, 0: no layers, -2 '
                             '(default): unspecified → coerced to -1 (all '
                             'layers).')
    parser.add_argument('--finetune_output_layers', default="True", type=str,
                        help='Whether to finetune the transformer output layers (norm and out)')

    # diffusion param
    parser.add_argument('--diffusion_steps', default=256, type=int,
                        help='number of timesteps, should be as long as the sequence')
    parser.add_argument('--task', default='MNIST', type=str,
                        help='problem system: MNIST or proteins')
    parser.add_argument('--enter_eval', default=1000, type=int,
                        help='iteration step to evaluation performance.')


    # number of epochs
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs for training...')
    parser.add_argument('--sequence_keyname', default='seq', type=str,
                        help='key name that belongs to the sequence list..')
    parser.add_argument('--facilitator', default='None', type=str,
                choices=['MSE', 'MMD', 'Default', 'None'],
                help='Option whether facilitator was used')
    parser.add_argument('--valid_size', default=0.1, type=float,
                        help='Validation dataset size...')
    parser.add_argument('--num_workers', default=0, type=int,  #  NOTE: CHANGED TO 0 TO PREVENT CUDA OOM ERROR
                        help='Number of dataloader workers...')

    # training on pfam database...
    parser.add_argument('--max_steps', default=100000, type=int,
                        help='number of iteration steps for training...')
    parser.add_argument('--val_check_interval', default=10000, type=int,
                        help='number of steps before starting evaluation on validation...')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int,
                        help='run validation (and, since checkpoints are monitored, '
                             'checkpoint saving) every N epochs in primary_only mode. '
                             'Default 1 = every epoch.')
    parser.add_argument('--limit_val_batches', default=200, type=float,
                        help='Cap validation batches per epoch. Values >1 are an '
                             'absolute batch count (predictable wall time across '
                             'dataset sizes); values in (0,1] are a fraction of '
                             'the val set (scales with dataset size). '
                             'Default 200 batches gives a stable val signal across '
                             'small (~10K) to large (10M+) datasets.')
    parser.add_argument('--limit_train_batches', default=None, type=float,
                        help='Cap training batches per epoch. Values >1 are an '
                             'absolute batch count; values in (0,1] are a fraction. '
                             'None = use full training dataset.')
    parser.add_argument('--log_every_n_steps', default=None, type=int,
                        help='Trainer(log_every_n_steps=N): cadence in '
                             'training batches at which Lightning flushes '
                             'metrics to its attached loggers (TensorBoard, '
                             'WandB). Also reused as the default '
                             'periodic-checkpoint cadence under combine '
                             'training mode. Default: once per epoch '
                             '(num_training_batches). Explicit values larger '
                             'than num_training_batches are clamped down.')
    parser.add_argument('--start_secondary', default='False', type=str,
                        help='flag for phase transition: load primary weights then train on primary+secondary')
    # Deprecated alias
    parser.add_argument('--start_pfam_trainer', default='False', type=str,
                        help='(deprecated, use --start_secondary)')

    # Metrics history
    parser.add_argument('--save_metrics_history', default='True', type=str,
                        help='Save training/validation metrics history to artifacts dir')
    parser.add_argument('--metrics_history_ranks', type=int, nargs='+', default=[0],
                        help='Rank indices on which to save metrics history')
    parser.add_argument('--metrics_history_every_n_steps', default=10, type=int,
                        help='Record training metrics every N global steps. '
                             'Records are buffered in memory and flushed to '
                             'metrics_history.train.jsonl in one batched '
                             'write at each train epoch end (and on '
                             'exception/train end for crash recovery).')
    parser.add_argument('--metrics_history_every_n_epochs', default=None, type=int,
                        help='Also record training metrics at the end of every N epochs '
                             '(captures epoch-averaged values). None disables.')
    parser.add_argument('--metrics_history_all_ranks_val_loss', default='False',
                        type=str,
                        help='Diagnostic: dump val_loss per rank at epoch end '
                             '(one file per rank) to check sync_dist consistency')

    # Early stopping
    parser.add_argument('--early_stopping_metric', default=None, type=str,
                        help='Metric to monitor for early stopping (e.g. val_loss). None to disable.')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Number of checks with no improvement before stopping')
    parser.add_argument('--early_stopping_min_delta', default=0.0, type=float,
                        help='Minimum change to qualify as an improvement')
    parser.add_argument('--early_stopping_mode', default='min', type=str,
                        choices=['min', 'max'],
                        help='Whether to minimize or maximize the monitored metric')

    # Periodic checkpoint saving (orthogonal to best-metric saves; written
    # to a separate `<checkpoint_dir>/periodic` subdirectory and never pruned
    # by the monitored callback's save_top_k cap).
    parser.add_argument('--checkpoint_every_n_steps', default=None, type=int,
                        help='Save periodic snapshot every N training steps '
                             '(in addition to best-metric saves)')
    parser.add_argument('--checkpoint_every_n_epochs', default=None, type=int,
                        help='Save periodic snapshot every N epochs '
                             '(in addition to best-metric saves)')
    parser.add_argument('--checkpoint_periodic_max_keep', default=-1, type=int,
                        choices=[-1, 0, 1],
                        help='Periodic snapshot retention: -1 = keep all '
                             '(default), 0 = disable, 1 = keep only most recent. '
                             "Lightning's ModelCheckpoint forbids other values "
                             'when monitor=None.')
    # Mid-training artifact sync: re-runs the DeepSpeed→fp32 conversion +
    # state_dict.best.pth copy on each new best checkpoint so that a
    # timeout-killed run leaves a ready-to-use artifact on disk.
    parser.add_argument('--artifact_sync_on_best', default='True', type=str,
                        help='Re-emit state_dict.best*.pth + checkpoint_summary.json '
                             'each time a new best checkpoint is saved. Disable '
                             'if the DeepSpeed→fp32 conversion is too costly.')
    parser.add_argument('--artifact_sync_every_n_val', default=1, type=int,
                        help='Throttle: sync at most every Nth validation epoch.')
    parser.add_argument('--backup_artifacts', default='False', type=str,
                        help='When True, rolling backup of best/summary files: '
                             'each mid-training sync renames the existing file '
                             'to <name>.bak.<timestamp>, and the next sync '
                             'deletes the previous backup it created. Only one '
                             'prior version on disk at a time, and only backups '
                             'this run created are ever deleted. Default False: '
                             'overwrite in place, no backups. Raw Lightning '
                             '.ckpt files (save_top_k) remain the authoritative '
                             'recovery surface either way.')

    # Multi-metric checkpoint monitors (JSON config only)
    parser.add_argument('--checkpoint_monitors', default=None, type=str,
                        help='JSON list of {metric, mode} dicts for checkpoint saving (set via config)')

    parser.add_argument('--use_sync_safe_checkpoint', default='False', type=str,
                        help='Use SyncSafeModelCheckpoint to bypass reduce_boolean_decision '
                             '(workaround for XPU/CCL integer all-reduce bug)')

    # Benchmark
    parser.add_argument('--save_benchmark', default='False', type=str,
                        help='Save per-epoch training benchmark to artifacts dir')
    parser.add_argument('--benchmark_skip_first_epoch', default='True', type=str,
                        help='Exclude first epoch from benchmark summary stats (warmup)')
    parser.add_argument('--benchmark_all_ranks_memory', default='False', type=str,
                        help='All-gather peak memory from every rank and save '
                             'per-rank lists in benchmark_history.json')
    parser.add_argument('--benchmark_per_step', default='False', type=str,
                        help='Write per-step wall-clock timing to '
                             'benchmark_steps.jsonl (rank 0 only)')

    # Within-epoch progress logging
    parser.add_argument('--log_progress_fraction', default=0.5, type=float,
                        help='Fraction-of-epoch cadence for the StepProgress '
                             'log line. 0.5 (default) fires at 50%% and 100%%; '
                             '1.0 only at the final step; 0.25 at '
                             '25/50/75/100. Set to 0 to disable.')

    # Wall-time budget
    parser.add_argument('--time_limit', default='None', type=str,
                        help='Optional hh:mm:ss wall-time budget measured '
                             'from the start of main(). When exceeded, '
                             'training stops gracefully at the next check '
                             '(~every 50 train batches on rank 0). All '
                             'end-of-run artifacts still get written, and '
                             'run_summary.json records '
                             'exit_reason=time_limit_exceeded.')

    # Progress bar
    parser.add_argument('--progress_bar', default='auto', type=str,
                        help="Show Lightning's tqdm progress bar. 'True' "
                             "forces on, 'False' forces off, 'auto' (default) "
                             "enables it only when stdout is a TTY (off under "
                             "PBS / redirected output to avoid log spam).")

    # Dry-run preview (no training executed)
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


def get_model_args(parser):
    """
    Configure argument parser with model-specific parameters.
    
    This function adds arguments to the provided parser for configuring 
    model training settings including image size, class counts, embedding dimensions,
    optimization parameters, and hardware utilization options.
    
    Args:
        parser: ArgumentParser object to which arguments will be added
        
    Returns:
        The parser with added model-specific arguments
    """
    parser.add_argument('--image_size', default=16, type=int,
                        help='size of training images, for rescaling')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='No. of classes for transformer')
    parser.add_argument('--text_emb_dim', default=256, type=int,
                        help='size of the text embedding')
    parser.add_argument('--num_y_class_labels', default=10, type=int,
            help='No. of y class labels for conditioning tranformer')
    parser.add_argument('--choose_optim', default='AdamW', type=str,
            help='Choose optimizer for training')
    parser.add_argument('--acc_grad_batches', default=4, type=int,
            help='Choose how many gradient steps to accumulate')
    parser.add_argument('--devices_per_node', default=None, type=int,
            help='Number of GPUs (CUDA) or tiles (XPU) per node. Default 1.')
    parser.add_argument('--gpu_devices', default=None, type=int,
            help='(deprecated, use --devices_per_node) preserved for '
                 'backward compatibility with older configs and scripts.')
    parser.add_argument('--num_nodes', default=1, type=int,
            help='Number of nodes to used for training')
    def float_or_str(value):
        try:
            return float(value)
        except ValueError:
            return value
    parser.add_argument('--scheduler_gamma', default=None, type=float_or_str,
                        help='Define the learning rate scheduler')
    return parser


def get_path_args(parser):
    """
    Configure argument parser with path-related parameters.
    
    This function adds arguments to the provided parser for specifying 
    various file paths and directories needed for training outputs, 
    checkpoints, logs, and version tracking.
    
    Args:
        parser: ArgumentParser object to which arguments will be added
        
    Returns:
        The parser with added path-specific arguments
    """
    parser.add_argument('--run_id', default=None, type=str,
                        help='unique identifier for this training run')
    parser.add_argument('--runs_folder', default='runs', type=str,
                        help='subdirectory under output_root for per-run logs and artifacts')
    parser.add_argument('--resume_from_checkpoint_state_dict_path', default=None, type=str,
                        help='Path to the deepspeed pytorch ckpt saved as state dict...')
    return parser


def get_wrapper_args(parser):
    """

    """
    parser.add_argument('--hydra', action="store_true", 
                        help='Whether to run with Hydra.')
    parser.add_argument('--wandb', type=str, default="False",
                        help='Flag to use Weights&Biases.')
    parser.add_argument('--wandb_name', type=str, default=None, 
                        help='Weights&Biases run name.')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='Weights&Biases entity.')
    parser.add_argument('--wandb_project', type=str, default=None, 
                        help='Weights&Biases project.')
    parser.add_argument('--wandb_tags', type=str, nargs="*", default=[],
                        help='Weights&Biases tags.')
    

    return parser


def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.
    
    This function ensures deterministic behavior by setting identical random
    seeds for PyTorch, NumPy, and Python's built-in random module based on
    the seed value specified in the arguments.
    
    Args:
        seed: Random seed
        
    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


def compile_model(
        args: any,
    ) -> pl.LightningModule:
    """
    Create and compile a PyTorch Lightning model for training.
    
    This function instantiates a base model using the provided arguments
    and wraps it in a PyTorch Lightning module (PL_ProtARDM) to enable
    distributed training, checkpointing, and other Lightning features.
    
    Args:
        args: Configuration object containing model parameters
        data_shape: Tuple specifying input data dimensions (default: (16,16))
        num_classes: Number of output classes (default: 3)
        
    Returns:
        A PyTorch Lightning module ready for training
    """
    model = prepare_model_ProteoScribe(
        config_args=args,
        model_fpath=args.pretrained_weights,
        device=args.device,
        strict=True,
        eval=False,
        attempt_correction=True,
        verbosity=2
    )

    PL_model = PL_mod.PL_ProtARDM(
        args=args,
        model=model,
    )
    return PL_model



def get_model_params(
        model_param_df_path,
        model: nn.Module
        ):
    """
    Calculate and save model parameter statistics to a CSV file.
    
    This function computes the total number of parameters in the provided
    neural network model, displays this count on the console, and saves
    the information to a CSV file at the specified path.
    
    Args:
        model_param_df_path: File path where the parameter statistics CSV will be saved
        model: PyTorch neural network module to analyze
        
    Returns:
        None
    """
    # save csv file for model param description:
    total_params = sum(
            param.numel() for param in model.parameters()
    )
    logger.info('Total number of model parameters: %s', total_params)
    model_param = {}
    model_param['total_params'] = [total_params]
    model_param_df = pd.DataFrame(model_param)
    model_param_df.to_csv(model_param_df_path, index=False)
    return


def get_protein_dataloader(args=any) -> DataLoader:
    """
    Create a DataLoader for protein sequence data.

    This function loads protein data from the specified path, prepares it
    for model training by converting sequences to numerical format and
    extracting text embeddings, then creates and returns a DataLoader
    with the appropriate batch size and shuffle settings.

    Args:
        args: Configuration object containing data_root path and batch_size

    Returns:
        DataLoader object with prepared protein sequence data
    """
    data = torch.load(args.data_root)
    num_seq_list, text_emb = prep.prepare_protein_data(
            args=args,
            data_dict=data
    )
    train_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list,
            text_emb=text_emb
    )
    protein_dataloader = DataLoader(
            #subset_dataset,
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True
    )
    return protein_dataloader


def get_deepspeed_model(args: any, PL_model) -> pl.LightningModule:
    """
    Load a model from a DeepSpeed checkpoint.

    This function handles the conversion of a DeepSpeed ZeRO checkpoint into a
    standard PyTorch state dictionary format and then loads it into the
    provided PyTorch Lightning model. The conversion process consolidates
    distributed model parameters into a single file.

    Args:
        args: Configuration object containing a 'resume_from_checkpoint' path
        PL_model: PyTorch Lightning model structure to load weights into

    Returns:
        A PyTorch Lightning module with loaded weights from the checkpoint
    """
    PL_model.model = prepare_model_ProteoScribe(
        config_args=args,
        model_fpath=args.resume_from_checkpoint,
        device=args.device,
        strict=True,
        eval=False,
        attempt_correction=True,
    )
    return PL_model


def _convert_or_copy_checkpoint(src_ckpt, dst_single_ckpt, label):
    """Materialize a single-file PL-loadable .ckpt from a checkpoint path.

    DeepSpeed Stage 2 writes a sharded directory; DDP / single-device DeepSpeed
    writes a single .ckpt file. The directory form needs ZeRO→fp32 conversion;
    the single-file form is already loadable — just copy it.
    """
    if os.path.isdir(src_ckpt):
        logger.info('Converting DeepSpeed checkpoint to fp32 (%s)...', label)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            convert_zero_checkpoint_to_fp32_state_dict(src_ckpt, dst_single_ckpt)
    else:
        logger.info('Copying single-file checkpoint (%s)...', label)
        shutil.copy2(src_ckpt, dst_single_ckpt)


def _sync_best_artifact(
    args,
    checkpoint_path: str,
    artifacts_path: str,
    primary_callback,
    extra_callbacks=None,
    expected_dtype=None,
) -> dict:
    """Materialize ``state_dict.best*.pth`` artifacts + ``checkpoint_summary.json``.

    Idempotent: safe to call mid-training (e.g. from BestArtifactSyncCallback)
    or at end-of-training (from save_model). Existing files are backed up
    rather than overwritten in place. Runs on rank 0 — caller is responsible
    for the rank guard.

    Returns the ``checkpoint_summary`` dict that was written, so the caller
    can stash it for further composition.
    """
    image_size = args.image_size
    num_classes = args.num_classes

    best_ckpt_fpath = primary_callback.best_model_path
    if not best_ckpt_fpath or not os.path.exists(best_ckpt_fpath):
        logger.warning(
            "Skipping best-artifact sync: no best checkpoint yet "
            "(best_model_path=%r)", best_ckpt_fpath,
        )
        return {}

    def _maybe_backup(fpath):
        if not getattr(args, 'backup_artifacts', False):
            return
        prior = _BACKUP_HISTORY.get(fpath)
        if prior is not None:
            try:
                os.remove(prior)
            except FileNotFoundError:
                pass
            _BACKUP_HISTORY.pop(fpath, None)
        new_backup = backup_if_exists(fpath)
        if new_backup is not None:
            _BACKUP_HISTORY[fpath] = new_backup

    best_single_ckpt_fpath = os.path.join(checkpoint_path, 'single_model.best.pth')
    best_state_dict_fpath = os.path.join(checkpoint_path, 'state_dict.best.pth')
    best_state_dict_ema_fpath = os.path.join(checkpoint_path, 'state_dict_ema.best.pth')

    for fpath in (best_single_ckpt_fpath, best_state_dict_fpath,
                  best_state_dict_ema_fpath,
                  os.path.join(checkpoint_path, 'params.csv')):
        _maybe_backup(fpath)

    _convert_or_copy_checkpoint(best_ckpt_fpath, best_single_ckpt_fpath, "best")
    logger.info('Save model (best)')
    new_temp_model = mod.get_model(
        args=args, data_shape=(image_size, image_size), num_classes=num_classes,
    ).cpu()
    loaded_model = PL_mod.PL_ProtARDM.load_from_checkpoint(
        best_single_ckpt_fpath, args=args, model=new_temp_model,
    )
    if expected_dtype is not None and expected_dtype != loaded_model.dtype:
        raise AssertionError(
            f"Data types are not matching. Expected {expected_dtype}. "
            f"Loaded: {loaded_model.dtype}"
        )
    torch.save(loaded_model.model.state_dict(), best_state_dict_fpath)
    if hasattr(loaded_model, 'ema_model'):
        logger.info('Also saving EMA model...')
        torch.save(loaded_model.ema_model.state_dict(), best_state_dict_ema_fpath)
    get_model_params(
        os.path.join(checkpoint_path, 'params.csv'), model=loaded_model.model,
    )

    # Copy best state_dict to artifacts directory
    os.makedirs(artifacts_path, exist_ok=True)
    artifact_best = os.path.join(artifacts_path, 'state_dict.best.pth')
    _maybe_backup(artifact_best)
    shutil.copy2(best_state_dict_fpath, artifact_best)
    logger.info("Copied best state_dict to %s", artifact_best)

    checkpoint_summary = {
        "primary": {
            "metric": primary_callback.monitor or "val_loss",
            "best_path": artifact_best,
            "best_score": float(primary_callback.best_model_score)
                if primary_callback.best_model_score is not None else None,
        },
        "additional": [],
    }
    for cb in (extra_callbacks or []):
        if not cb.best_model_path or not os.path.exists(cb.best_model_path):
            logger.warning("No best checkpoint found for monitor %s", cb.monitor)
            continue
        metric_slug = cb.monitor.replace("/", "_")
        extra_sd_fname = f"state_dict.best_{metric_slug}.pth"
        extra_sd_fpath = os.path.join(checkpoint_path, extra_sd_fname)
        extra_single_fpath = os.path.join(
            checkpoint_path, f"single_model.best_{metric_slug}.pth"
        )
        _maybe_backup(extra_sd_fpath)
        _maybe_backup(extra_single_fpath)
        _convert_or_copy_checkpoint(cb.best_model_path, extra_single_fpath, cb.monitor)
        extra_temp_model = mod.get_model(
            args=args, data_shape=(image_size, image_size), num_classes=num_classes,
        ).cpu()
        extra_loaded = PL_mod.PL_ProtARDM.load_from_checkpoint(
            extra_single_fpath, args=args, model=extra_temp_model,
        )
        torch.save(extra_loaded.model.state_dict(), extra_sd_fpath)
        artifact_extra = os.path.join(artifacts_path, extra_sd_fname)
        _maybe_backup(artifact_extra)
        shutil.copy2(extra_sd_fpath, artifact_extra)
        logger.info("Saved best state_dict for %s to %s", cb.monitor, artifact_extra)
        checkpoint_summary["additional"].append({
            "metric": cb.monitor,
            "mode": cb.mode,
            "best_score": float(cb.best_model_score)
                if cb.best_model_score is not None else None,
            "artifact": extra_sd_fname,
        })

    summary_path = os.path.join(artifacts_path, 'checkpoint_summary.json')
    _maybe_backup(summary_path)
    with open(summary_path, 'w') as f:
        json.dump(checkpoint_summary, f, indent=2)
    logger.info("Checkpoint summary written to %s", summary_path)
    return checkpoint_summary


def save_model(
        args: any,
        checkpoint_path: str,
        artifacts_path: str,
        PL_model: pl.LightningModule,
        trainer: Trainer,
        extra_checkpoint_callbacks=None,
    ) -> None:
    """
    Save a PyTorch Lightning model trained with DeepSpeed.

    Converts DeepSpeed ZeRO checkpoints into standard PyTorch state
    dictionaries. All derived weight files are written to checkpoint_path
    (alongside the raw .ckpt dirs). A copy of the best state_dict is
    also placed in artifacts_path for convenient downstream access.

    Only executes on the global_zero process in distributed environments.

    Args:
        args: Configuration object containing model parameters
        checkpoint_path: Directory containing Lightning checkpoints
        artifacts_path: Directory for run artifacts (receives best state_dict copy)
        PL_model: PyTorch Lightning model to be saved
        trainer: Trainer instance with checkpoint callback info
    """
    image_size = args.image_size
    num_classes = args.num_classes

    # once saved via the model checkpoint callback...
    # we have a saved folder containing the deepspeed checkpoint rather than a single file
    last_ckpt_fpath = os.path.join(checkpoint_path, 'last.ckpt')
    best_ckpt_fpath = trainer.checkpoint_callback.best_model_path
    last_single_ckpt_fpath = os.path.join(checkpoint_path, 'single_model.last.pth')
    best_single_ckpt_fpath = os.path.join(checkpoint_path, 'single_model.best.pth')
    best_state_dict_fpath = os.path.join(checkpoint_path, 'state_dict.best.pth')
    last_state_dict_fpath = os.path.join(checkpoint_path, 'state_dict.last.pth')
    last_state_dict_ema_fpath = os.path.join(checkpoint_path, 'state_dict_ema.last.pth')
    best_state_dict_ema_fpath = os.path.join(checkpoint_path, 'state_dict_ema.best.pth')

    symlink_to = "best"  # symlink from state_dict.pth to either best or last

    if trainer.is_global_zero:
        # Back up "last" derived files from a previous run; "best" backups are
        # handled inside _sync_best_artifact.
        for fpath in (
            last_single_ckpt_fpath, last_state_dict_fpath, last_state_dict_ema_fpath,
            os.path.join(checkpoint_path, 'state_dict.pth'),
            os.path.join(checkpoint_path, 'state_dict_ema.pth'),
        ):
            backup_if_exists(fpath)

        # ---------------- BEST MODEL + checkpoint_summary.json ----------------
        _sync_best_artifact(
            args=args,
            checkpoint_path=checkpoint_path,
            artifacts_path=artifacts_path,
            primary_callback=trainer.checkpoint_callback,
            extra_callbacks=extra_checkpoint_callbacks,
            expected_dtype=PL_model.dtype,
        )

        # No checkpoint at all: the monitored metric never fired, so
        # ModelCheckpoint saved nothing. The best-artifact sync above already
        # skips in that case; without the same guard here a run that trained to
        # completion dies in post-processing on a last.ckpt that was never
        # written, losing the whole run over a missing side artifact.
        if not os.path.exists(last_ckpt_fpath):
            logger.warning(
                "Skipping last-artifact sync: no checkpoint at %s. Nothing was "
                "saved this run -- the monitored metric never produced a value "
                "(e.g. validation did not run). Enable a periodic checkpoint "
                "(--checkpoint_every_n_epochs) or ensure validation runs.",
                last_ckpt_fpath,
            )
            return

        # Check whether last and best point to the same checkpoint
        # (last.ckpt is a symlink when save_last="link")
        last_real = os.path.realpath(last_ckpt_fpath)
        best_real = os.path.realpath(best_ckpt_fpath)
        same_checkpoint = (last_real == best_real)

        # ---------------- LAST MODEL ----------------
        if same_checkpoint:
            logger.info('Last checkpoint is same as best — creating symlinks')
            os.symlink(best_single_ckpt_fpath, last_single_ckpt_fpath)
            os.symlink(best_state_dict_fpath, last_state_dict_fpath)
            if os.path.exists(best_state_dict_ema_fpath):
                os.symlink(best_state_dict_ema_fpath, last_state_dict_ema_fpath)
        else:
            if os.path.isdir(last_ckpt_fpath):
                logger.info('Converting DeepSpeed checkpoint to fp32 (last)...')
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    convert_zero_checkpoint_to_fp32_state_dict(
                        last_ckpt_fpath, last_single_ckpt_fpath
                    )
            else:
                logger.info('Copying single-file checkpoint (last)...')
                shutil.copy2(last_ckpt_fpath, last_single_ckpt_fpath)
            logger.info('Save model (last)')
            new_temp_model = mod.get_model(
                args=args,
                data_shape=(image_size, image_size),
                num_classes=num_classes
            ).cpu()
            loaded_model = PL_mod.PL_ProtARDM.load_from_checkpoint(
                last_single_ckpt_fpath,
                args=args, model=new_temp_model
            )
            if PL_model.dtype != loaded_model.dtype:
                msg = "Data types are not matching."
                msg += f" Expected {PL_model.dtype}. Loaded: {loaded_model.dtype}"
                assert False, msg
            else:
                torch.save(loaded_model.model.state_dict(), last_state_dict_fpath)
                if hasattr(loaded_model, 'ema_model'):
                    logger.info('Also saving EMA model...')
                    torch.save(loaded_model.ema_model.state_dict(), last_state_dict_ema_fpath)
                get_model_params(
                        os.path.join(checkpoint_path, 'params.csv'),
                        model=loaded_model.model
                )
        
        symlink_path = os.path.join(checkpoint_path, 'state_dict.pth')
        symlink_ema_path = os.path.join(checkpoint_path, 'state_dict_ema.pth')
        if symlink_to == "best":
            os.symlink(best_state_dict_fpath, symlink_path)
            if os.path.exists(best_state_dict_ema_fpath):
                os.symlink(best_state_dict_ema_fpath, symlink_ema_path)
        elif symlink_to == "last":
            os.symlink(last_state_dict_fpath, symlink_path)
            if os.path.exists(last_state_dict_ema_fpath):
                os.symlink(last_state_dict_ema_fpath, symlink_ema_path)

    return


def str_to_bool(s):
    """
    Convert string representation of boolean values to Python bool type.
    
    This utility function handles conversion of case-insensitive string 
    values 'true' and 'false' to their corresponding Python boolean values. 
    It raises an error for any other input string.
    
    Args:
        s: String to convert, expected to be 'true' or 'false' (case-insensitive)
        
    Returns:
        bool: True if input is 'true' (case-insensitive), False if 'false'
        
    Raises:
        ValueError: If the input is anything other than 'true' or 'false'
    """
    if isinstance(s, bool):
        return s
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Input must be 'True' or 'False'")


def parse_lr_scaling(value):
    """Normalize --scale_learning_rate to a scaling mode.

    Returns ``None`` (no scaling), ``'linear'`` (lr * total_devices), or
    ``'sqrt'`` (lr * sqrt(total_devices)). Accepts booleans and the
    case-insensitive strings true/false/linear/sqrt; ``True``/``'true'`` map to
    ``'linear'`` for backward compatibility. Raises ValueError on anything else.
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


def nonestr_to_none(s):
    if isinstance(s, str):
        return None if s.lower() == 'none' else s
    return s


# === main functions ===

def clear_gpu_cache():
    """
    Free up GPU memory by clearing caches and running garbage collection.
    
    This utility function performs the following operations to optimize GPU memory usage:
    1. Empties the CUDA memory cache to release unused memory
    2. Runs Python's garbage collector to remove unreferenced objects

    This function is useful to call between training runs or when switching between
    memory-intensive operations to prevent out-of-memory errors.

    Args:
        None

    Returns:
        None
    """
    # torch.cuda.empty_cache()
    # torch.xpu.empty_cache()
    gc.collect()
    return


def retrieve_all_args(args):
    """
    Collect and consolidate all command line arguments from various components.

    Supports an optional ``--config_path`` argument pointing to a JSON file.
    When provided, the JSON values override argparse defaults but are themselves
    overridden by any explicitly passed CLI arguments (i.e. CLI > JSON > defaults).

    Args:
        args: list of CLI argument strings (typically ``sys.argv[1:]``)

    Returns:
        argparse.Namespace: A namespace object containing all parsed arguments
    """
    # Pre-parse to extract --config_path before building the full parser,
    # so we can set JSON values as defaults before the real parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_path', '-c', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(description='Stage 3: ProteoScribe')
    parser.add_argument('--config_path', '-c', type=str, default=None,
                        help='Path to JSON config file. Values are overridden by CLI args.')
    get_args(parser=parser)
    get_model_args(parser=parser)
    mod.add_model_args(parser=parser)
    get_path_args(parser=parser)
    get_wrapper_args(parser=parser)

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    argv = list(args)
    args = parser.parse_args(args)
    args._argv = argv  # consumed by core.dry_run for CLI provenance attribution

    apply_arg_type_conversions(args)

    return args


def apply_arg_type_conversions(args):
    """Normalize string/None-sentinel args in place (CLI > JSON > defaults).

    Extracted so alternative entrypoints (e.g. run_ProteoScribe_finetuning) can
    reuse the exact same coercions after building their own parser.
    """
    # Type conversions (idempotent — pass through values already of the target type)
    args.resume_from_checkpoint = nonestr_to_none(args.resume_from_checkpoint)
    args.download = str_to_bool(args.download)

    # New generalized dataset args
    args.primary_data_path = nonestr_to_none(args.primary_data_path)
    args.split_manifest_path = nonestr_to_none(getattr(args, 'split_manifest_path', None))
    args.start_secondary = str_to_bool(args.start_secondary)

    # Conditioning blend (idempotent, so entrypoints may re-normalize)
    args.zp_path = nonestr_to_none(getattr(args, 'zp_path', None))
    args.train_alpha = PL_mod.normalize_alpha_spec(getattr(args, 'train_alpha', 'zc'))
    args.eval_alpha = PL_mod.resolve_eval_alpha(getattr(args, 'eval_alpha', 'spread'))

    # Map deprecated aliases to new names
    args.swissprot_data_root = nonestr_to_none(args.swissprot_data_root)
    args.pfam_data_root = nonestr_to_none(args.pfam_data_root)
    args.start_pfam_trainer = str_to_bool(args.start_pfam_trainer)
    if args.primary_data_path is None and args.swissprot_data_root is not None:
        logger.warning("--swissprot_data_root is deprecated; use --primary_data_path")
        args.primary_data_path = args.swissprot_data_root
    if args.secondary_data_paths is None and args.pfam_data_root is not None:
        logger.warning("--pfam_data_root is deprecated; use --secondary_data_paths")
        args.secondary_data_paths = [args.pfam_data_root]
    if not args.start_secondary and args.start_pfam_trainer:
        logger.warning("--start_pfam_trainer is deprecated; use --start_secondary")
        args.start_secondary = args.start_pfam_trainer

    # Resolve training_strategy
    if args.training_strategy == 'auto':
        args.training_strategy = 'combine' if args.secondary_data_paths else 'primary_only'

    args.finetune = str_to_bool(args.finetune)
    args.finetune_output_layers = str_to_bool(args.finetune_output_layers)
    args.pretrained_weights = nonestr_to_none(args.pretrained_weights)
    args.weight_set = nonestr_to_none(getattr(args, 'weight_set', None))
    args._weight_set_replaced = merge_weight_set(
        args, args.weight_set, keys=("proteoscribe_weights",),
        rename={"proteoscribe_weights": "pretrained_weights"},
        argv=getattr(args, '_argv', None))
    args.wandb = str_to_bool(args.wandb)
    args.scale_learning_rate = parse_lr_scaling(args.scale_learning_rate)
    args.normalize_zc = str_to_bool(args.normalize_zc)
    if args.loss_positions not in ('all', 'non_pad'):
        raise ValueError(
            f"loss_positions must be 'all' or 'non_pad', got {args.loss_positions!r}")
    args.save_metrics_history = str_to_bool(args.save_metrics_history)
    args.metrics_history_all_ranks_val_loss = str_to_bool(
        args.metrics_history_all_ranks_val_loss
    )
    args.use_sync_safe_checkpoint = str_to_bool(args.use_sync_safe_checkpoint)
    args.save_benchmark = str_to_bool(args.save_benchmark)
    args.benchmark_skip_first_epoch = str_to_bool(args.benchmark_skip_first_epoch)
    args.benchmark_all_ranks_memory = str_to_bool(args.benchmark_all_ranks_memory)
    args.benchmark_per_step = str_to_bool(args.benchmark_per_step)
    args.early_stopping_metric = nonestr_to_none(args.early_stopping_metric)
    args.checkpoint_every_n_steps = nonestr_to_none(args.checkpoint_every_n_steps)
    args.checkpoint_every_n_epochs = nonestr_to_none(args.checkpoint_every_n_epochs)
    args.artifact_sync_on_best = str_to_bool(args.artifact_sync_on_best)
    args.backup_artifacts = str_to_bool(args.backup_artifacts)

    resolve_devices_per_node(args)

    args.dry_run = str_to_bool(args.dry_run)
    args.dry_run_output = coerce_dry_run_output(args.dry_run_output)

    if isinstance(args.progress_bar, str) and args.progress_bar.lower() == 'auto':
        args.progress_bar = None
    elif args.progress_bar is not None:
        args.progress_bar = str_to_bool(args.progress_bar)

    args.time_limit_seconds = None
    if isinstance(args.time_limit, str) and args.time_limit.lower() != 'none' and args.time_limit:
        h, m, s = (int(x) for x in args.time_limit.split(':'))
        args.time_limit_seconds = h * 3600 + m * 60 + s

    return args


def load_data(
        args, *,
        primary_data_path,
        secondary_data_paths,
        facilitator,
    ):
    """
    Initialize and prepare a data module for protein sequence datasets.

    Creates an :class:`HDF5DataModule` from one primary HDF5 file and zero
    or more secondary HDF5 files.  The *facilitator* name determines the
    HDF5 group name (e.g. ``MMD_data``).

    Args:
        args: Configuration namespace (batch_size, num_workers, etc.)
        primary_data_path: Path to the primary HDF5 training dataset.
        secondary_data_paths: List of paths to secondary HDF5 datasets, or None.
        facilitator: Facilitator name used as HDF5 group prefix.

    Returns:
        A configured PyTorch Lightning data module ready for training.
    """
    data_module = PL_mod.HDF5DataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        image_size=args.image_size,
        primary_path=primary_data_path,
        secondary_paths=secondary_data_paths,
        group_name=facilitator + '_data',
        split_manifest_path=getattr(args, 'split_manifest_path', None),
        zp_path=getattr(args, 'zp_path', None),
        train_alpha=getattr(args, 'train_alpha', 0.0),
        eval_alpha=getattr(args, 'eval_alpha', PL_mod.EVAL_SPREAD),
    )
    data_module.setup()
    return data_module


def log_weight_set_replacements(args):
    """Report each config path that --weight_set on the command line replaced."""
    for attr, replaced, new in getattr(args, '_weight_set_replaced', []):
        logger.info("--weight_set on the command line replaces the config's %s: %s -> %s",
                    attr, replaced, new)


def require_finetune_weights(args):
    """Refuse to finetune a model that would start from random weights."""
    if (args.finetune and args.pretrained_weights is None
            and args.resume_from_checkpoint is None):
        raise ValueError(
            "--finetune True needs --pretrained_weights (or "
            "--resume_from_checkpoint): without them the model would be "
            "trained from randomly initialised weights."
        )


def optimizer_steps_per_epoch(num_samples, batch_size, world_size,
                              acc_grad_batches=1, limit_train_batches=None):
    """Optimizer steps one rank takes per epoch.

    Under distributed training each rank gets ``num_samples // world_size``
    samples (see ``_make_distributed_sampler``), the DataLoader keeps the last
    partial batch, and Lightning steps once per ``acc_grad_batches`` batches
    and once more at the end of the epoch for any remainder.
    """
    per_rank = num_samples // world_size if world_size > 1 else num_samples
    batches = math.ceil(per_rank / batch_size)
    limit = coerce_limit_batches(limit_train_batches)
    if isinstance(limit, int):
        batches = min(batches, limit)
    elif limit is not None:
        batches = int(batches * limit)
    return max(1, math.ceil(batches / acc_grad_batches))


def set_traindata_len(args, data_module):
    """Set ``args.traindata_len``, the length of an epoch in optimizer steps.

    Computed from the dataset and the world size because the dataloader has no
    distributed sampler yet: ``torch.distributed`` is only initialized once the
    trainer starts fitting.
    """
    train_loader = data_module.train_dataloader()
    num_samples = len(train_loader.dataset)
    batch_size = train_loader.batch_size or args.batch_size
    world_size = args.num_nodes * args.devices_per_node
    args.traindata_len = optimizer_steps_per_epoch(
        num_samples=num_samples,
        batch_size=batch_size,
        world_size=world_size,
        acc_grad_batches=args.acc_grad_batches,
        limit_train_batches=getattr(args, 'limit_train_batches', None),
    )
    logger.info('Training samples: %s', num_samples)
    logger.info('Numer of devices: %s', args.devices_per_node)
    logger.info('Number of nodes: %s', args.num_nodes)
    logger.info('Batch size: %s', batch_size)
    logger.info('Length of a training epoch in batch gradient updates: %s',
                args.traindata_len)


def load_model(
    args, *,
    data_module
    ):
    """
    Initialize and configure a model based on dataset characteristics.
    
    This function performs several important setup steps:
    1. Calculates training epoch length adjusted for distributed training
    2. Validates diffusion step count against data dimensionality
    3. Logs GPU information
    4. Instantiates the model with appropriate parameters
    5. Reports model size
    
    The function ensures all requirements for training are met before
    returning the configured PyTorch Lightning model.
    
    Args:
        args: Configuration object containing model parameters and training settings
        data_module: PyTorch Lightning data module with dataset information
        
    Returns:
        A configured PyTorch Lightning model ready for training
    """
    diffusion_steps = args.diffusion_steps
    image_size = args.image_size

    set_traindata_len(args, data_module)
    w, h = image_size, image_size
    # Ensure diffusion steps are sufficient for data dimensions
    if diffusion_steps < int(w*h):
        logger.warning('Make sure that the number of diffusion steps is equal to or greather than the data cardinality')
    if get_global_rank() == 0:
        print_gpu_initialization()
    # Compile model architecture
    PL_model = compile_model(
        args=args,
    )
    logger.info('Model size: %s', sum(p.numel() for p in PL_model.model.parameters()))
    return PL_model


def load_pretrained_weights(
        PL_model,
        checkpoint_path: str,
        args=None,
    ):
    """
    Load pretrained model weights into a PyTorch Lightning model.

    Supports raw state dicts (.bin, .pth, .pt), Lightning checkpoints (.ckpt),
    and sharded DeepSpeed checkpoint directories. Format is auto-detected.
    Parameter renaming/correction is applied automatically.

    Args:
        PL_model: PyTorch Lightning model to load weights into
        checkpoint_path: Path to model weights (file or directory)
        args: Configuration namespace (used to rebuild model graph if needed).
              If None, uses PL_model.script_args.

    Returns:
        The model with loaded pretrained weights
    """
    if args is None:
        args = PL_model.script_args

    logger.info("Loading pretrained weights from: %s", checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    PL_model.model = prepare_model_ProteoScribe(
        config_args=args,
        model_fpath=checkpoint_path,
        device='cpu',
        strict=True,
        eval=False,
        attempt_correction=True,
    )

    logger.info("Pretrained weights loaded successfully")
    return PL_model


def freeze_except_last_n_blocks_and_layers(
        PL_model,
        n_blocks,
        n_layers,
        finetune_output_layers=True
    ):
    """
    Freeze all model parameters except those in the last n transformer blocks and
    layers.
    
    This enables efficient finetuning by only updating parameters in the last
    n transformer blocks while keeping all other layers frozen. This approach
    reduces computational cost and can help prevent catastrophic forgetting.
    
    Args:
        PL_model: PyTorch Lightning model with loaded pretrained weights
        n_blocks: Number of last transformer blocks to keep trainable (default: 1)
                  If n_blocks=0, no blocks will be trainable (complete freezing)
                  If n_blocks=-1, all blocks will be trainable (no freezing)
                  If n_blocks >= total blocks, all blocks will be trainable
        n_layers: Number of last transformer layers to keep trainable (default: 1)
                  If n_layers=0, no layers will be trainable (complete freezing)
                  If n_layers=-1, all layers will be trainable (no freezing)
                  If n_layers >= total layers, all layers will be trainable
        
    Returns:
        The model with frozen parameters (except last n blocks)
    """
    if n_blocks == -1 and n_layers == -1:
        logger.info("n_blocks=n_layers=-1: All parameters will remain trainable (no freezing)")
        # return PL_model

    if n_blocks == 0 and n_layers == 0:
        logger.info("n_blocks=n_layers=0: All parameters will be frozen (no training)")
    else:
        msg = f"Freezing all parameters except the last {n_layers} layer(s) "
        msg += f"of each of the last {n_blocks} transformer block(s)..."
        logger.info(msg)
    
    # First, freeze all parameters
    for param in PL_model.model.parameters():
        param.requires_grad = False
    
    # Then unfreeze only the last k layers of the last n transformer blocks
    # The model structure is: PL_model.model.transformer.transformer_blocks[bidx][depth]
    if hasattr(PL_model.model, 'transformer'):
        transformer = PL_model.model.transformer
        if hasattr(transformer, 'transformer_blocks') and len(transformer.transformer_blocks) > 0:
            total_blocks = len(transformer.transformer_blocks)
            if n_blocks == -1:
                n_blocks = total_blocks
            blocks_to_unfreeze = min(n_blocks, total_blocks)
            # Unfreeze the last k layers of the last n blocks
            logger.info("Attemtping to unfreeze last %s blocks...", blocks_to_unfreeze)
            for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                logger.debug("***** Freezing block %s...", i)
                block = transformer.transformer_blocks[i]
                if len(block) > 0:
                    total_layers = len(block)
                    if n_layers == -1:
                        n_layers = total_layers
                    layers_to_unfreeze = min(n_layers, total_layers)
                    for j in range(total_layers - layers_to_unfreeze, total_layers):
                        layer = block[j]
                        logger.debug("*** Freezing block %s layer %s", i, j)
                        for param in layer.parameters():
                            param.requires_grad = True
                else:
                    for param in block.parameters():
                        param.requires_grad = True
            
            logger.info("Unfroze last %s transformer block(s) out of %s total blocks", blocks_to_unfreeze, total_blocks)
            if blocks_to_unfreeze < n_blocks:
                logger.info("Note: Requested %s blocks but only %s exist in model", n_blocks, total_blocks)
        else:
            logger.warning("Could not find transformer_blocks in model")
    else:
        logger.warning("Could not find transformer attribute in model")
    
    # Also unfreeze the final output layers (norm and out) for better finetuning
    if finetune_output_layers:
        if hasattr(PL_model.model, 'transformer'):
            transformer = PL_model.model.transformer
            if hasattr(transformer, 'norm'):
                for param in transformer.norm.parameters():
                    param.requires_grad = True
                logger.info("Unfroze final LayerNorm")
            if hasattr(transformer, 'out'):
                for param in transformer.out.parameters():
                    param.requires_grad = True
                logger.info("Unfroze final output layer")
    
    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in PL_model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in PL_model.model.parameters())
    frozen_params = total_params - trainable_params
    
    logger.info("Trainable parameters: %s (%.2f%%)", f"{trainable_params:,}", 100 * trainable_params / total_params)
    logger.info("Frozen parameters: %s (%.2f%%)", f"{frozen_params:,}", 100 * frozen_params / total_params)
    
    return PL_model


###########################
##  Training Entrypoint  ##
###########################

def train_model(
        args,
        PL_model,
        data_module,
        ds_config=None,
        verbosity=1,
    ):
    """
    Train a PyTorch Lightning model with DeepSpeed distributed optimization.
    
    This function sets up the full training infrastructure including DeepSpeed 
    configuration, logging, checkpointing, and handles different training scenarios 
    (new training, resumed training, or phase-transfer training). Training parameters 
    adapt based on the dataset type, with special handling for Pfam datasets.
    
    The function supports two training modes:
    1. Epoch-based training (standard mode when not using Pfam)
    2. Step-based training with validation interval monitoring (for Pfam)
    
    After training completes, model weights are automatically saved in various formats.
    
    Args:
        args: Configuration object containing training parameters and paths
        PL_model: PyTorch Lightning model to be trained
        data_module: Data module providing training and validation data
        ds_config: DeepSpeed configurations. If None, uses defaults.
        
    Returns:
        None - results are saved to the paths specified in args
    """
    logger.info('Beginning Training...')
    # Note: Nested args must be accessed via dictionaries, not attributes.
    output_root = args.output_root
    checkpoints_folder = args.checkpoints_folder
    runs_folder = args.runs_folder
    run_id = args.run_id
    log_every_n_steps = args.log_every_n_steps
    num_training_batches = getattr(args, 'traindata_len', None)
    if log_every_n_steps is None:
        log_every_n_steps = max(1, num_training_batches or 1)
        logger.info("Defaulted log_every_n_steps to %d (once per epoch)",
                    log_every_n_steps)
    elif num_training_batches and log_every_n_steps > num_training_batches:
        log_every_n_steps = max(1, num_training_batches)
        logger.info("Clamped log_every_n_steps to %d (number of training batches)",
                     log_every_n_steps)
    training_strategy = args.training_strategy
    devices_per_node = args.devices_per_node
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    start_secondary = args.start_secondary  # Expect bool
    assert isinstance(start_secondary, bool), "start_secondary not bool"
    max_steps = args.max_steps
    val_check_interval = args.val_check_interval
    limit_val_batches = args.limit_val_batches
    resume_from_checkpoint = args.resume_from_checkpoint  # Expect str or None
    assert resume_from_checkpoint is None or (
            isinstance(resume_from_checkpoint, str) and 
            resume_from_checkpoint != "None"
        ), f"resume_from_checkpoint should be str or None (not str `None`)." + \
           f" Got {resume_from_checkpoint} (type {type(resume_from_checkpoint)})"
    precision = args.precision
    use_wandb = args.wandb

    # Scale the learning rate with number of total devices (None disables it).
    if args.scale_learning_rate:
        n = num_nodes * devices_per_node
        factor = n ** 0.5 if args.scale_learning_rate == 'sqrt' else n
        logger.info("Scaling learning rate (%s) by num_nodes x devices_per_node "
                    "= %s x %s -> factor %.4g", args.scale_learning_rate,
                    num_nodes, devices_per_node, factor)
        args.lr = args.lr * factor
    logger.info("Effective learning rate: %s", args.lr)
    
    # Configure DeepSpeed optimization settings
    if ds_config is None:
        ds_config = {
            "zero_optimization": {
                "stage": 1,
                "allgather_bucket_size": 5e8,
                "reduce_bucket_size": 5e8,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        }
    
    # strategy = DeepSpeedStrategy(
    #     config=ds_config
    # )

    # Derived output paths
    checkpoint_dir = os.path.join(output_root, checkpoints_folder, run_id)
    run_dir = os.path.join(output_root, runs_folder, run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)

    loggers = []

    # Set up TensorBoard logging
    logger.info("Setting up TensorBoard logging...")
    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        version="",
    )
    loggers.append(tb_logger)

    # Set up Weights&Biases logging
    logger.info("Setting up Weights&Biases logging...")
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            name=args.wandb_name if args.wandb_name else run_id,
            save_dir=logs_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            tags=args.wandb_tags,
            group=run_id,
            log_model="all",
        )  # TODO: investigate "all"
        loggers.append(wandb_logger)

    # Monitor learning rate changes
    logger.info("Setting up LearningRateMonitor...")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Monitor GPU usage
    logger.info("Setting up DeviceStatesMonitor...")
    gpu_logger = DeviceStatsMonitor()

    # ---- Checkpoint callbacks ----
    logger.info("Configuring ModelCheckpoint...")

    # Optional periodic saving (orthogonal to monitored top-k saves).
    # For step-based training (combine strategy), default periodic saving to
    # log_every_n_steps so that the previous behavior is preserved.
    periodic_every_n_steps = getattr(args, 'checkpoint_every_n_steps', None)
    periodic_every_n_epochs = getattr(args, 'checkpoint_every_n_epochs', None)
    if training_strategy == 'combine' and periodic_every_n_steps is None:
        periodic_every_n_steps = log_every_n_steps

    monitored_callbacks, periodic_callback = build_checkpoint_callbacks(
        checkpoint_dir=checkpoint_dir,
        checkpoint_monitors=getattr(args, 'checkpoint_monitors', None),
        periodic_every_n_steps=periodic_every_n_steps,
        periodic_every_n_epochs=periodic_every_n_epochs,
        periodic_max_keep=getattr(args, 'checkpoint_periodic_max_keep', -1),
        use_sync_safe=getattr(args, 'use_sync_safe_checkpoint', False),
    )
    checkpoint_callbacks = monitored_callbacks + (
        [periodic_callback] if periodic_callback is not None else []
    )

    # ---- Metrics history ----
    if getattr(args, 'save_metrics_history', True):
        metrics_cb = MetricsHistoryCallback(
            output_dir=artifacts_dir,
            save_ranks=getattr(args, 'metrics_history_ranks', [0]),
            every_n_steps=getattr(args, 'metrics_history_every_n_steps', 1),
            every_n_epochs=getattr(args, 'metrics_history_every_n_epochs', None),
            all_ranks_val_loss=getattr(
                args, 'metrics_history_all_ranks_val_loss', False
            ),
        )
    else:
        metrics_cb = None

    # ---- Training benchmark ----
    if getattr(args, 'save_benchmark', False):
        benchmark_cb = TrainingBenchmarkCallback(
            output_dir=artifacts_dir,
            batch_size=args.batch_size,
            acc_grad_batches=acc_grad_batches,
            devices_per_node=devices_per_node,
            num_nodes=num_nodes,
            precision=precision,
            training_strategy=training_strategy,
            num_workers=args.num_workers,
            skip_first_epoch=getattr(args, 'benchmark_skip_first_epoch', True),
            all_ranks_memory=getattr(args, 'benchmark_all_ranks_memory', False),
            per_step=getattr(args, 'benchmark_per_step', False),
        )
    else:
        benchmark_cb = None

    # ---- Early stopping ----
    early_stopping_metric = getattr(args, 'early_stopping_metric', None)
    if isinstance(early_stopping_metric, str) and early_stopping_metric.lower() == 'none':
        early_stopping_metric = None

    callbacks = checkpoint_callbacks + [lr_monitor, gpu_logger,
                                        EpochProgressCallback()]
    if metrics_cb is not None:
        callbacks.append(metrics_cb)
    if benchmark_cb is not None:
        callbacks.append(benchmark_cb)
    if getattr(args, 'log_progress_fraction', 0) > 0:
        callbacks.append(StepProgressCallback(fraction=args.log_progress_fraction))
    if getattr(args, 'time_limit_seconds', None) is not None:
        callbacks.append(TimeLimitCallback(
            deadline_seconds=args.time_limit_seconds,
            start_monotonic=_MAIN_START_MONOTONIC,
        ))

    # ---- Best-artifact mid-training sync (Tier 2/3) ----
    # Re-emits state_dict.best.pth + state_dict.best_<metric>.pth +
    # checkpoint_summary.json each time ModelCheckpoint promotes a new best,
    # so a SIGTERM/timeout leaves a ready-to-use artifact on disk.
    if getattr(args, 'artifact_sync_on_best', True):
        def _sync_fn(primary_callback, extra_callbacks):
            _sync_best_artifact(
                args=args,
                checkpoint_path=checkpoint_dir,
                artifacts_path=artifacts_dir,
                primary_callback=primary_callback,
                extra_callbacks=extra_callbacks,
                expected_dtype=None,  # PL_model.dtype not yet known here; safe to skip the strict check mid-training
            )

        callbacks.append(BestArtifactSyncCallback(
            sync_fn=_sync_fn,
            primary_callback=monitored_callbacks[0],
            extra_callbacks=monitored_callbacks[1:],
            every_n_val=getattr(args, 'artifact_sync_every_n_val', 1),
        ))

    if early_stopping_metric is not None:
        logger.info("Enabling early stopping on %s (patience=%d)",
                     early_stopping_metric, args.early_stopping_patience)
        callbacks.append(EarlyStopping(
            monitor=early_stopping_metric,
            patience=args.early_stopping_patience,
            min_delta=getattr(args, 'early_stopping_min_delta', 0.0),
            mode=getattr(args, 'early_stopping_mode', 'min'),
            verbose=True,
        ))

    # Define common trainer parameters.
    # overlap_comm=False on XPU: default True fires grad allreduces asynchronously
    # during backward, producing nondeterministic bucket ordering across ranks on
    # oneCCL and causing mismatched-collective deadlocks. Disabling serializes
    # grad reduction after backward — slightly slower, but stable.
    # process_group_backend='xccl' on XPU: frameworks/2025.3.1 removed the
    # `oneccl-bindings-for-pytorch` module and replaced the 'ccl' backend with
    # torch's native 'xccl' — breaking change documented in ALCF system-updates.
    deepspeed_strategy = DeepSpeedStrategy(
        stage=2,
        allgather_bucket_size=int(5e8),
        reduce_bucket_size=int(5e8),
        contiguous_gradients=True,
        overlap_comm=(BACKEND_NAME != _XPU),
        process_group_backend='xccl' if BACKEND_NAME == _XPU else None,
    )
    # Plain DDP with static_graph=True. The static_graph flag tells DDP that
    # the autograd graph structure is stable across iterations, so it can
    # precompute the gradient-bucket ready order at iteration 0 and reuse it
    # for every subsequent step. This removes the dynamic-hook race that
    # produced the mismatched-collective deadlocks we observed on Aurora xccl
    # (some ranks fired bucket-K allreduce while others moved on to the next
    # collective). gradient_as_bucket_view=True is a small memory win that
    # pairs naturally with static_graph.
    ddp_strategy = DDPStrategy(
        process_group_backend='xccl' if BACKEND_NAME == _XPU else None,
        static_graph=True,
        gradient_as_bucket_view=True,
    )
    # Strategy is selected by --distributed_strategy. DeepSpeed Stage 2 is
    # the production default; 'ddp' selects plain DDP with static_graph=True.
    # save_model handles both the sharded ZeRO checkpoint dir and the
    # single-file DDP .ckpt via the os.path.isdir() guard in
    # _convert_or_copy_checkpoint.
    if args.distributed_strategy != 'ddp':
        strategy = deepspeed_strategy
    elif num_nodes * devices_per_node > 1:
        strategy = ddp_strategy
    elif args.device == 'xpu':
        # One rank needs no process group. Explicit device, as in Stage 1:
        # Lightning's 'auto' resolves an XPU run to a CPU root device.
        strategy = SingleDeviceStrategy(device=torch.device('xpu'))
    else:
        strategy = 'auto'
    logger.info("Using distributed_strategy=%s", args.distributed_strategy)
    if strategy is not deepspeed_strategy and strategy is not ddp_strategy:
        logger.info("One rank: single-device strategy in place of DDP")

    progress_bar = getattr(args, 'progress_bar', None)
    if progress_bar is None:
        progress_bar = sys.stdout.isatty()

    trainer_params = {
        'enable_progress_bar': progress_bar,
        'enable_model_summary': True,
        'enable_checkpointing': True,
        'devices': devices_per_node,
        'num_nodes': num_nodes,
        'accelerator': args.device,
        'strategy': strategy,
        # We construct DistributedSampler(drop_last=True) ourselves in
        # PL_wrapper.{train,val}_dataloader so every rank gets exactly the
        # same number of samples; see PL_wrapper._make_distributed_sampler.
        # use_distributed_sampler=False prevents Lightning from auto-wrapping
        # our explicit sampler with its own (non-drop_last) one.
        'use_distributed_sampler': False,
        'accumulate_grad_batches': acc_grad_batches,
        'logger': loggers,
        'log_every_n_steps': log_every_n_steps,
        'callbacks': callbacks,
    }

    # Configure training mode: epoch-based (primary_only) or step-based (combine)
    trainer_params['limit_val_batches'] = coerce_limit_batches(limit_val_batches)
    if training_strategy == 'primary_only':
        trainer_params['max_epochs'] = epochs
        trainer_params['check_val_every_n_epoch'] = max(
            1, int(getattr(args, 'check_val_every_n_epoch', 1) or 1)
        )
    else:
        trainer_params['max_steps'] = max_steps
        trainer_params['val_check_interval'] = val_check_interval

    limit_train_batches = getattr(args, 'limit_train_batches', None)
    if limit_train_batches is not None:
        trainer_params['limit_train_batches'] = coerce_limit_batches(
            limit_train_batches
        )

    # Initialize trainer with configured parameters
    logger.info('Initializing Trainer...')
    trainer = Trainer(**trainer_params)
    global _LAST_TRAINER
    _LAST_TRAINER = trainer

    # wrap optimizer and model with intel extension for pytorch 
    # optimizer = torch.optim.AdamW(PL_model.parameters(), lr=lr)
    # PL_model, optimizer = ipex.optimize(PL_model, optimizer=optimizer, dtype=torch.float32)

    # Handle different training scenarios
    if args.finetune:
        # Finetuning
        if resume_from_checkpoint is None:
            logger.info("Start finetuning")
            trainer.fit(PL_model, data_module)
        else:
            # NOTE: the exact string "Resume finetuning from checkpoint" is
            # asserted by test_resume_finetune_ignores_pretrained_weights in
            # tests/stage3_tests/test_stage3_run_PL_training.py. Changing it
            # will break that regression test.
            logger.info("Resume finetuning from checkpoint: %s",
                        resume_from_checkpoint)
            trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)
    else:
        # Pretraining
        if resume_from_checkpoint is None:
            logger.info("Train from scratch")
            trainer.fit(PL_model, data_module)
        elif start_secondary:
            logger.info('Start training ProteoScribe with secondary data ...')
            trainer.fit(PL_model, data_module)
        else:
            logger.info('Continue training ProteoScribe from checkpoint ...')
            trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)

    # Save dataset split indices
    if get_global_rank() == 0 and hasattr(data_module, 'split_info'):
        splits_path = os.path.join(artifacts_dir, "dataset_splits.pt")
        torch.save(data_module.split_info, splits_path)
        logger.info("Saved dataset splits to %s", splits_path)

    # Save trained model in multiple formats. Only the *monitored* callbacks
    # (best-by-metric) get state-dict conversion; the periodic snapshots stay
    # as raw .ckpts under checkpoint_dir/periodic and can be converted
    # post-hoc if needed.
    save_model(
            args=args,
            checkpoint_path=monitored_callbacks[0].dirpath,
            artifacts_path=artifacts_dir,
            PL_model=PL_model,
            trainer=trainer,
            extra_checkpoint_callbacks=monitored_callbacks[1:],
    )


def _write_build_manifest(args, artifacts_dir, checkpoint_dir, PL_model,
                          start_time):
    if get_global_rank() != 0:
        return

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
        "effective_lr": args.lr,
        "precision": args.precision,
        "devices_per_node": args.devices_per_node,
        "num_nodes": args.num_nodes,
        "acc_grad_batches": args.acc_grad_batches,
        "distributed_strategy": args.distributed_strategy,
        "train_alpha": args.train_alpha,
        "eval_alpha": args.eval_alpha,
    }
    if PL_mod.alpha_spec_uses_zp(args.train_alpha):
        outputs["zp_path"] = getattr(args, 'zp_path', None)
    if args.distributed_strategy == "deepspeed_zero2":
        outputs["deepspeed_stage"] = "2"
    outputs["training_strategy"] = args.training_strategy
    if args.training_strategy == 'combine':
        outputs["max_steps"] = args.max_steps
        outputs["val_check_interval"] = args.val_check_interval
    else:
        outputs["epochs"] = args.epochs

    if args.finetune:
        outputs["finetune"] = True
        outputs["finetune_last_n_blocks"] = args.finetune_last_n_blocks
        outputs["finetune_last_n_layers"] = args.finetune_last_n_layers

    resolved_paths = {
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "artifacts_dir": os.path.abspath(artifacts_dir),
    }
    if args.primary_data_path is not None:
        resolved_paths["primary_data_path"] = os.path.abspath(
            args.primary_data_path
        )
    if args.secondary_data_paths is not None:
        resolved_paths["secondary_data_paths"] = [
            os.path.abspath(p) for p in args.secondary_data_paths
        ]
    if args.pretrained_weights is not None:
        resolved_paths["pretrained_weights"] = os.path.abspath(
            args.pretrained_weights
        )
    if args.resume_from_checkpoint is not None:
        resolved_paths["resume_from_checkpoint"] = os.path.abspath(
            args.resume_from_checkpoint
        )

    manifest_path = write_manifest(
        args, artifacts_dir, start_time, timedelta(0),
        outputs=outputs,
        resolved_paths=resolved_paths,
        environment=collect_training_env(),
    )
    logger.info("Build manifest written to %s", manifest_path)


def _write_run_summary(artifacts_dir, start_time, exit_reason, exception=None,
                       completed_epochs=None, completed_steps=None):
    if get_global_rank() != 0:
        return
    elapsed = datetime.now() - start_time
    summary = {
        "exit_reason": exit_reason,
        "elapsed_seconds": elapsed.total_seconds(),
        "end_time": datetime.now().isoformat(),
        "completed_epochs": completed_epochs,
        "completed_steps": completed_steps,
    }
    if exception is not None:
        summary["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
    summary_path = os.path.join(artifacts_dir, "run_summary.json")
    backup_if_exists(summary_path)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Run summary written to %s", summary_path)


def main(args, use_hydra=False, ds_config=None,):

    global _MAIN_START_MONOTONIC, _LAST_TRAINER
    _MAIN_START_MONOTONIC = time.perf_counter()
    _LAST_TRAINER = None
    start_time = datetime.now()
    _BACKUP_HISTORY.clear()

    # ----- Suppress noisy library warnings -----
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
    warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings(
        "once",
        message=r".*barrier\(\): using the device under current context.*",
    )
    logging.getLogger("tensorboardX.x2num").setLevel(logging.ERROR)

    # ----- Dry-run preview (no training executed) -----
    # A dry run only probes config, data and model, so it may land on CPU; a
    # real run must find a GPU unless --device cpu was asked for explicitly.
    dry_run = getattr(args, 'dry_run', False)
    args.device = resolve_device(args.device, allow_cpu=dry_run)
    if not dry_run:
        check_devices_per_node(args.device, args.devices_per_node)

    if getattr(args, 'dry_run', False):
        return run_dry_run(
            args,
            stage="stage3",
            dataset_probe=lambda: _stage3_dataset_probe(args),
            model_probe=lambda: _stage3_model_probe(args),
        )

    # ----- Set up output directories and file logging -----
    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    checkpoint_dir = os.path.join(
        args.output_root, args.checkpoints_folder, args.run_id,
    )
    if get_global_rank() == 0:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    log_path, file_handler = setup_file_logging(artifacts_dir)
    try:
        # ----- Process passed parameters -----
        seed = args.seed
        primary_data_path = args.primary_data_path
        secondary_data_paths = args.secondary_data_paths
        facilitator = args.facilitator

        # ----- Clear the GPU cache -----
        set_float32_matmul_precision(args.float32_matmul_precision)
        clear_gpu_cache()

        # ----- For reproducibility -----
        if seed <= 0:
            seed = np.random.randint(2**32)
            args.seed = seed
        set_seed(seed)
        logger.info("Using seed: %s", seed)

        require_finetune_weights(args)
        zc_summary, zc_warning = check_normalize_zc(
            args.weight_set, args.pretrained_weights, args.normalize_zc)
        if get_global_rank() == 0:
            log_weight_set_replacements(args)
            logger.info(zc_summary)
            for line in zc_warning:
                logger.warning(line)

        # ----- Load Data -----
        data_module = load_data(
            args=args,
            primary_data_path=primary_data_path,
            secondary_data_paths=secondary_data_paths,
            facilitator=facilitator,
        )

        # ----- Load Model -----
        PL_model = load_model(
            args=args,
            data_module=data_module
        )

        # ----- Load pretrained weights and freeze if finetuning -----
        finetuning = args.finetune
        if finetuning:
            finetune_last_n_blocks = args.finetune_last_n_blocks
            finetune_last_n_layers = args.finetune_last_n_layers
            finetune_output_layers = args.finetune_output_layers
            pretrained_weights = args.pretrained_weights
            resume_from_checkpoint = args.resume_from_checkpoint
            if finetune_last_n_layers == -2:
                # If flag is set to finetune and layers not specified (default -2)
                # set to -1 (all layers trainable)
                finetune_last_n_layers = -1
            if finetune_last_n_blocks == -2:
                # If flag is set to finetune and blocks not specified (default -2)
                # set to -1 (all blocks trainable)
                finetune_last_n_blocks = -1
            # When resuming, weights (and optimizer state) are restored from the
            # Lightning checkpoint by trainer.fit(ckpt_path=...), so loading
            # pretrained_weights would be wasted work and misleading. Still apply
            # the freeze logic, since requires_grad flags are not persisted in
            # Lightning checkpoints.
            skip_pretrained = resume_from_checkpoint is not None
            if skip_pretrained and pretrained_weights is not None:
                # NOTE: the exact string "Ignoring --pretrained_weights" is
                # asserted by test_resume_finetune_ignores_pretrained_weights in
                # tests/stage3_tests/test_stage3_run_PL_training.py. Changing it
                # will break that regression test.
                logger.info(
                    "Ignoring --pretrained_weights (%s) because "
                    "--resume_from_checkpoint (%s) is set; weights will be "
                    "restored from the checkpoint.",
                    pretrained_weights, resume_from_checkpoint,
                )
            if skip_pretrained:
                PL_model = freeze_except_last_n_blocks_and_layers(
                    PL_model=PL_model,
                    n_blocks=finetune_last_n_blocks,
                    n_layers=finetune_last_n_layers,
                    finetune_output_layers=finetune_output_layers
                )
            else:
                PL_model = load_pretrained_weights(
                    PL_model=PL_model,
                    checkpoint_path=pretrained_weights
                )
                # Freeze parameters based on user configuration
                PL_model = freeze_except_last_n_blocks_and_layers(
                    PL_model=PL_model,
                    n_blocks=finetune_last_n_blocks,
                    n_layers=finetune_last_n_layers,
                    finetune_output_layers=finetune_output_layers
                )
        else:
            pass

        # ----- Write build manifest before training (rank 0 only) -----
        _write_build_manifest(
            args=args,
            artifacts_dir=artifacts_dir,
            checkpoint_dir=checkpoint_dir,
            PL_model=PL_model,
            start_time=start_time,
        )

        # ----- Train Model -----
        exit_reason = "completed"
        exception = None
        try:
            train_model(
                args=args,
                PL_model=PL_model,
                data_module=data_module,
                ds_config=ds_config,
            )
        except KeyboardInterrupt as e:
            exit_reason = "interrupted"
            exception = e
            raise
        except BaseException as e:
            exit_reason = "exception"
            exception = e
            raise
        finally:
            time_limit_seconds = getattr(args, 'time_limit_seconds', None)
            if exit_reason == "completed" and time_limit_seconds is not None:
                elapsed = time.perf_counter() - _MAIN_START_MONOTONIC
                if elapsed >= time_limit_seconds:
                    exit_reason = "time_limit_exceeded"
            completed_epochs = (
                _LAST_TRAINER.current_epoch if _LAST_TRAINER is not None else None
            )
            completed_steps = (
                _LAST_TRAINER.global_step if _LAST_TRAINER is not None else None
            )
            _write_run_summary(
                artifacts_dir=artifacts_dir,
                start_time=start_time,
                exit_reason=exit_reason,
                exception=exception,
                completed_epochs=completed_epochs,
                completed_steps=completed_steps,
            )
            if get_global_rank() == 0:
                for line in zc_warning:
                    logger.warning(line)
            if _MAIN_START_MONOTONIC is not None:
                total = int(time.perf_counter() - _MAIN_START_MONOTONIC)
                h, rem = divmod(total, 3600)
                m, s = divmod(rem, 60)
                logger.info(
                    "Program exiting. Total elapsed time: %d:%02d:%02d", h, m, s,
                )
    finally:
        teardown_file_logging("biom3", file_handler)


def _stage3_dataset_probe(args):
    """CPU-only data-module probe for the dry-run report.

    Returns ``(train_len, val_len, sample_batch)``. Failures are
    surfaced as exceptions and rendered as notes in the report.
    """
    data_module = load_data(
        args=args,
        primary_data_path=args.primary_data_path,
        secondary_data_paths=args.secondary_data_paths,
        facilitator=args.facilitator,
    )
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()
    train_len = len(train_dl.dataset) if hasattr(train_dl.dataset, '__len__') else None
    val_len = len(val_dl.dataset) if hasattr(val_dl.dataset, '__len__') else None
    sample = next(iter(train_dl), None)
    return train_len, val_len, sample


def _stage3_model_probe(args):
    """CPU-only model probe for the dry-run report."""
    args.device = 'cpu'
    return mod.get_model(
        args=args,
        data_shape=(args.image_size, args.image_size),
        num_classes=args.num_classes,
    )


def parse_arguments(args):
    return retrieve_all_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
