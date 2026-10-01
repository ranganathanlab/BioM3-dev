#!/usr/bin/env python3

"""Training script for BioM3 Stage 2 (Facilitator).

Loads pre-computed Stage 1 embeddings (z_t, z_p) and trains the Facilitator
to align z_t to z_p. After fitting, runs a predict pass to emit the final
{text, protein, text_to_protein}-embedding dicts consumed by Stage 3.
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch

from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger, set_float32_matmul_precision
from biom3.backend.device import DEVICE_CHOICES, resolve_device, check_devices_per_node

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning import Trainer
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
    from lightning.pytorch.callbacks import (
        LearningRateMonitor, DeviceStatsMonitor, EarlyStopping,
    )
    from lightning.pytorch.strategies import SingleDeviceStrategy
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import (
        LearningRateMonitor, DeviceStatsMonitor, EarlyStopping,
    )
    from pytorch_lightning.strategies import SingleDeviceStrategy

from biom3.Stage2.PL_wrapper import PL_Facilitator
from biom3.Stage2.preprocess import Facilitator_DataModule
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


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Input must be 'True' or 'False'")


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
                        help='Free-text description of the run (stored in args.json).')
    parser.add_argument('--tags', type=str, nargs="*", default=[],
                        help='Tags categorizing this run (stored in args.json).')
    parser.add_argument('--notes', type=str, nargs="*", default=[],
                        help='Free-form notes about this run (stored in args.json).')

    parser.add_argument('--swissprot_data_path', type=str, default='None',
                        help='Path to Stage 1 SwissProt embeddings .pt dict.')
    parser.add_argument('--pfam_data_path', type=str, default='None',
                        help='Path to Stage 1 Pfam embeddings .pt dict.')
    parser.add_argument('--output_swissprot_dict_path', type=str, default=None,
                        help='Where to save Stage 2 SwissProt embeddings dict.')
    parser.add_argument('--output_pfam_dict_path', type=str, default=None,
                        help='Where to save Stage 2 Pfam embeddings dict.')

    parser.add_argument('--output_root', type=str, default='./outputs/Stage2/pretraining',
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Per-device mini-batch size.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='Fraction of dataset held out for validation '
                             '(default 0.2 = 80/20 train/val split).')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Validation cadence (1.0 = once per epoch).')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Cap validation batches per epoch. Values >1 are an '
                             'absolute batch count; values in (0,1] are a fraction. '
                             'Default 1.0 uses the full val set.')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                        help='Lightning Trainer log_every_n_steps (TensorBoard / wandb).')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Base learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='AdamW weight decay.')
    parser.add_argument('--scale_learning_rate', type=str, default='False',
                        help="'True'/'False'. When True, scale lr by "
                             'num_nodes * devices_per_node.')

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
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Alias exposed for Facilitator_Dataset device logic.')
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
                        help='Also record training metrics every N epochs '
                             '(captures epoch-averaged values). None disables.')
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
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='Facilitator input/output embedding dim '
                             '(must match Stage 1 z_t/z_p dim).')
    parser.add_argument('--hid_dim', type=int, default=1024,
                        help='Facilitator hidden layer dim.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate inside the Facilitator.')
    parser.add_argument('--loss_type', type=str, default='MSE',
                        choices=['MSE', 'MMD'],
                        help="Facilitator training loss: 'MSE' (point-wise) or "
                             "'MMD' (distribution-matching).")
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

    parser = argparse.ArgumentParser(description='Stage 2: Facilitator training')
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

    args.scale_learning_rate = str_to_bool(args.scale_learning_rate)
    args.wandb = str_to_bool(args.wandb)
    args.save_metrics_history = str_to_bool(args.save_metrics_history)
    args.metrics_history_all_ranks_val_loss = str_to_bool(
        args.metrics_history_all_ranks_val_loss
    )
    args.use_sync_safe_checkpoint = str_to_bool(args.use_sync_safe_checkpoint)
    args.save_benchmark = str_to_bool(args.save_benchmark)
    args.benchmark_skip_first_epoch = str_to_bool(args.benchmark_skip_first_epoch)
    args.benchmark_all_ranks_memory = str_to_bool(args.benchmark_all_ranks_memory)

    args.swissprot_data_path = nonestr_to_none(args.swissprot_data_path)
    args.pfam_data_path = nonestr_to_none(args.pfam_data_path)
    args.resume_from_checkpoint = nonestr_to_none(args.resume_from_checkpoint)
    args.pretrained_weights = nonestr_to_none(args.pretrained_weights)
    args.early_stopping_metric = nonestr_to_none(args.early_stopping_metric)
    args.checkpoint_every_n_steps = nonestr_to_none(args.checkpoint_every_n_steps)
    args.checkpoint_every_n_epochs = nonestr_to_none(args.checkpoint_every_n_epochs)

    if args.run_id is None:
        args.run_id = f"stage2_{args.loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Facilitator_DataModule expects Rama's sentinel strings 'None', not Python None.
    # Re-inject the sentinels for the DataModule's branching logic.
    args.swissprot_data_path_sentinel = args.swissprot_data_path if args.swissprot_data_path is not None else 'None'
    args.pfam_data_path_sentinel = args.pfam_data_path if args.pfam_data_path is not None else 'None'

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


def load_data(args) -> Facilitator_DataModule:
    # Facilitator_DataModule branches on exact string 'None' for missing inputs.
    # Swap the Python None back to 'None' for the constructor.
    shim_args = argparse.Namespace(**vars(args))
    shim_args.swissprot_data_path = shim_args.swissprot_data_path_sentinel
    shim_args.pfam_data_path = shim_args.pfam_data_path_sentinel
    return Facilitator_DataModule(args=shim_args)


def build_pl_model(args) -> PL_Facilitator:
    return PL_Facilitator(args=args)


def load_pretrained_weights(PL_model, checkpoint_path: str):
    logger.info("Loading pretrained Facilitator weights from %s", checkpoint_path)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
        sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
    missing, unexpected = PL_model.model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))
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
    else:
        logger.warning("No best checkpoint path reported; skipping save_model")


def get_final_embeddings(PL_model, final_dataloader, device='cuda', devices=1):
    logger.info("Running predict pass to collect z_t_to_p embeddings...")
    pred_trainer = Trainer(
        max_epochs=1,
        devices=devices,
        accelerator=device,
        enable_progress_bar=True,
        logger=False,
    )
    PL_model.eval()
    PL_model.text_to_protein_joint_embeddings = []
    pred_trainer.predict(PL_model, dataloaders=final_dataloader)
    return PL_model.text_to_protein_joint_embeddings


def save_embeddings(embedding_data, predictions, output_dict_path):
    n_text = len(embedding_data['text_embedding'])
    if n_text != predictions.shape[0]:
        raise AssertionError(
            f"Shape mismatch: text_embedding={n_text}, predictions={predictions.shape[0]}"
        )
    embedding_data['text_to_protein_embedding'] = [pred for pred in predictions]
    os.makedirs(os.path.dirname(os.path.abspath(output_dict_path)), exist_ok=True)
    torch.save(embedding_data, output_dict_path)
    logger.info("Saved Stage 2 embeddings to %s", output_dict_path)


def train_model(args, PL_model, data_module):
    logger.info("Beginning Stage 2 training...")
    output_root = args.output_root
    run_id = args.run_id
    devices_per_node = args.devices_per_node
    num_nodes = args.num_nodes
    acc_grad_batches = args.acc_grad_batches
    epochs = args.epochs
    precision = args.precision
    use_wandb = args.wandb

    if args.scale_learning_rate:
        n = num_nodes * devices_per_node
        logger.info("Scaling LR by %s", n)
        args.lr *= n

    checkpoint_dir = os.path.join(output_root, args.checkpoints_folder, run_id)
    run_dir = os.path.join(output_root, args.runs_folder, run_id)
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

    callbacks = [LearningRateMonitor(logging_interval='step')]
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
        from biom3.Stage3.callbacks import MetricsHistoryCallback
        callbacks.append(MetricsHistoryCallback(
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
        ))

    if args.early_stopping_metric is not None:
        callbacks.append(EarlyStopping(
            monitor=args.early_stopping_metric,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=args.early_stopping_mode,
            verbose=True,
        ))

    if args.device == 'cuda' and (devices_per_node > 1 or num_nodes > 1):
        strategy = 'ddp'
    elif args.device == 'xpu' and devices_per_node == 1 and num_nodes == 1:
        # Aurora Lightning's _choose_strategy() falls back to
        # SingleDeviceStrategy(device="cpu") for XPU because 'xpu' isn't in its
        # CUDA/MPS/GPU branch; that makes trainer.strategy.root_device report
        # CPU and breaks DeviceStatsMonitor. Pin the strategy explicitly.
        strategy = SingleDeviceStrategy(device=torch.device('xpu'))
    else:
        strategy = 'auto'

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
    else:
        trainer_precision = precision

    trainer = Trainer(
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        devices=devices_per_node,
        num_nodes=num_nodes,
        accelerator=args.device,
        strategy=strategy,
        accumulate_grad_batches=acc_grad_batches,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        precision=trainer_precision,
        max_epochs=epochs,
        val_check_interval=args.val_check_interval,
        limit_val_batches=coerce_limit_batches(args.limit_val_batches),
        num_sanity_val_steps=0,
    )

    if args.resume_from_checkpoint is None:
        logger.info("Train from scratch")
        trainer.fit(PL_model, data_module)
    else:
        logger.info("Resume from checkpoint: %s", args.resume_from_checkpoint)
        trainer.fit(PL_model, data_module, ckpt_path=args.resume_from_checkpoint)

    if get_global_rank() == 0:
        print_gpu_initialization()

    save_model(
        args=args,
        checkpoint_path=checkpoint_callbacks[0].dirpath,
        artifacts_path=artifacts_dir,
        PL_model=PL_model,
        trainer=trainer,
    )

    return trainer, artifacts_dir


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
            stage="stage2",
            dataset_probe=lambda: _stage2_dataset_probe(args),
            model_probe=lambda: _stage2_model_probe(args),
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

        data_module = load_data(args)
        PL_model = build_pl_model(args)
        logger.info("Facilitator parameters: %s", sum(p.numel() for p in PL_model.model.parameters()))

        if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
            PL_model = load_pretrained_weights(PL_model, args.pretrained_weights)
        elif args.pretrained_weights is not None:
            logger.warning("Pretrained weights path does not exist: %s", args.pretrained_weights)

        trainer, artifacts_dir = train_model(args, PL_model, data_module)

        # ---- Final predict pass + save transformed embeddings ----
        if get_global_rank() == 0:
            if args.swissprot_data_path_sentinel != 'None' and data_module.all_swiss_dataloader is not None:
                logger.info("Infer SwissProt dataset...")
                preds = get_final_embeddings(
                    PL_model, data_module.all_swiss_dataloader,
                    device=args.device, devices=1,
                )
                if args.output_swissprot_dict_path:
                    save_embeddings(
                        data_module.swissprot_data, preds, args.output_swissprot_dict_path,
                    )

            if args.pfam_data_path_sentinel != 'None' and data_module.all_pfam_dataloader is not None:
                logger.info("Infer Pfam dataset...")
                preds = get_final_embeddings(
                    PL_model, data_module.all_pfam_dataloader,
                    device=args.device, devices=1,
                )
                if args.output_pfam_dict_path:
                    save_embeddings(
                        data_module.pfam_data, preds, args.output_pfam_dict_path,
                    )

        # ---- Manifest + args.json ----
        if get_global_rank() == 0:
            elapsed = datetime.now() - start_time
            args_path = os.path.join(artifacts_dir, "args.json")
            backup_if_exists(args_path)
            with open(args_path, "w") as f:
                json.dump({k: v for k, v in vars(args).items() if not k.startswith("_")},
                          f, indent=2, default=str)

            total_params = sum(p.numel() for p in PL_model.model.parameters())
            outputs = {
                "seed": args.seed,
                "total_params": total_params,
                "batch_size": args.batch_size,
                "effective_lr": args.lr,
                "precision": args.precision,
                "devices_per_node": args.devices_per_node,
                "num_nodes": args.num_nodes,
                "acc_grad_batches": args.acc_grad_batches,
                "epochs": args.epochs,
                "loss_type": args.loss_type,
                "emb_dim": args.emb_dim,
                "hid_dim": args.hid_dim,
            }

            resolved_paths = {
                "checkpoint_dir": os.path.abspath(checkpoint_dir),
                "artifacts_dir": os.path.abspath(artifacts_dir),
            }
            if args.swissprot_data_path is not None:
                resolved_paths["swissprot_data_path"] = os.path.abspath(args.swissprot_data_path)
            if args.pfam_data_path is not None:
                resolved_paths["pfam_data_path"] = os.path.abspath(args.pfam_data_path)
            if args.output_swissprot_dict_path:
                resolved_paths["output_swissprot_dict_path"] = os.path.abspath(args.output_swissprot_dict_path)
            if args.output_pfam_dict_path:
                resolved_paths["output_pfam_dict_path"] = os.path.abspath(args.output_pfam_dict_path)
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


def _stage2_dataset_probe(args):
    """CPU-only data-module probe for the dry-run report."""
    if 'data_module' not in _DRY_RUN_CACHE:
        _DRY_RUN_CACHE['data_module'] = load_data(args=args)
    data_module = _DRY_RUN_CACHE['data_module']
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


def _stage2_model_probe(args):
    """CPU-only model probe for the dry-run report."""
    PL_model = build_pl_model(args=args)
    return PL_model.model


def parse_arguments(args):
    return retrieve_all_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
