"""BioM3 Stage 3: ProteoScribe finetuning on cleaned record / sequence data.

Unlike ``run_PL_training`` (which trains on precomputed z_c embeddings stored in
HDF5), this script finetunes ProteoScribe directly on a JSONL dataset of cleaned
records ``{sequence, fields: {key: raw_description, ...}, sequence_length}``. For
each example a ``record_schema`` composes a caption from the fields (per-key
dropout, optional label-adding, shuffle, concatenate; see
:mod:`biom3.core.dataloaders`), which is then embedded to z_c on-device through a
frozen text->z_c front-end (PenCL text branch + Facilitator) before the standard
ProteoScribe diffusion objective runs. Because the caption is re-composed every
epoch, z_c cannot be precomputed.

This is purely a *finetuning* script: it always loads pretrained ProteoScribe
weights (or resumes from a Lightning checkpoint) and freezes all but a chosen
subset of the transformer. Most of the training infrastructure (trainer setup,
callbacks, checkpoint saving, freezing, arg coercion) is reused from
``run_PL_training``.

Example usage:

    biom3_finetune_stage3 \
        --config_path configs/stage3_training/finetune_generalized_v1.json \
        --run_id my_finetune_run
"""

import os
import sys
import time
import logging
import warnings
import argparse
from datetime import datetime

import numpy as np
import torch

import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.PL_wrapper as PL_mod
import biom3.Stage3.run_PL_training as base
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.Stage3.finetune_embedder import build_text_to_zc_embedder, build_protein_to_zp_embedder
from biom3.core.dataloaders import load_compose_plugins
from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.core.dry_run import run_dry_run
from biom3.core.run_utils import setup_file_logging, teardown_file_logging
from biom3.core.distributed import get_global_rank
from biom3.backend.device import (
    print_gpu_initialization, setup_logger, set_float32_matmul_precision,
    resolve_device, check_devices_per_node,
)

logger = setup_logger(__name__)


def get_finetune_args(parser):
    """Finetuning-specific arguments (data, record schema, embedder)."""
    parser.add_argument('--finetune_data_path', default=None, type=str,
                        help='path to JSONL of cleaned {sequence, fields: {...}, '
                             'sequence_length} records')
    parser.add_argument('--record_schema', default=None, type=str,
                        help='output schema mapping output name -> spec, as a JSON '
                             'object string (or set directly via config). Composes '
                             'the caption via a registered compose function '
                             '(referenced by name) and passes the sequence through.')
    parser.add_argument('--compose_plugins', default=None,
                        help='external .py files (or dotted module names) to import '
                             'before the dataset is built, so their @register_compose '
                             'functions become referenceable by name in the '
                             'record_schema. Keeps dataset-specific caption logic out '
                             'of the biom3 package. JSON list in config, or a single '
                             'path on the CLI.')
    parser.add_argument('--length_field', default='sequence_length', type=str,
                        help='record key holding the precomputed (ungapped) '
                             'sequence length used for length filtering; computed '
                             'from the sequence when absent')
    parser.add_argument('--caption_key', default='caption', type=str,
                        help='schema output key feeding the text tokenizer')
    parser.add_argument('--sequence_output_key', default='sequence', type=str,
                        help='schema output key feeding the protein-sequence encoder')
    parser.add_argument('--lazy_records', default='False', type=str,
                        help='read records lazily (JsonlRecordStore) instead of '
                             'eagerly into memory')

    # Frozen text -> z_c embedding front-end (PenCL text branch + Facilitator)
    parser.add_argument('--stage1_config_path', default=None, type=str,
                        help='PenCL (Stage 1) inference config for the text encoder')
    parser.add_argument('--stage2_config_path', default=None, type=str,
                        help='Facilitator (Stage 2) inference config')
    parser.add_argument('--pencl_weights', default=None, type=str,
                        help='PenCL weights (.bin/.pt/.ckpt); only text branch is used')
    parser.add_argument('--facilitator_weights', default=None, type=str,
                        help='Facilitator weights (.bin/.pt/.ckpt)')

    # LoRA finetuning — an alternative to block-freezing. When enabled the base
    # is fully frozen and low-rank adapters are trained on the attention Q/V
    # projections (+ y_mlp), instead of unfreezing the last N blocks/layers.
    parser.add_argument('--use_lora', default='False', type=str,
                        help='use LoRA adapters instead of block-freezing')
    parser.add_argument('--lora_r', default=16, type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', default=32, type=int,
                        help='LoRA scaling alpha (typically ~2*r)')
    parser.add_argument('--lora_dropout', default=0.05, type=float,
                        help='dropout applied to the LoRA input')
    parser.add_argument('--lora_target_patterns', default='.fn.to_q,.fn.to_v', type=str,
                        help='comma-separated module-name substrings to wrap with LoRA')
    parser.add_argument('--lora_unfreeze_y_mlp', default='True', type=str,
                        help='also train the y_mlp z_c-conditioning MLP')

    # --train_alpha / --eval_alpha / --zp_path are defined in run_PL_training.get_args.
    parser.add_argument('--zp_batch_size', default=64, type=int,
                        help='batch size for the one-off z_p precompute pass')
    return parser


def retrieve_all_args(argv):
    """Build the finetuning arg namespace (CLI > JSON config > defaults)."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_path', '-c', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description='Stage 3: ProteoScribe finetuning')
    parser.add_argument('--config_path', '-c', type=str, default=None,
                        help='Path to JSON config file. Values are overridden by CLI args.')
    base.get_args(parser=parser)
    base.get_model_args(parser=parser)
    mod.add_model_args(parser=parser)
    base.get_path_args(parser=parser)
    base.get_wrapper_args(parser=parser)
    get_finetune_args(parser=parser)

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    raw_argv = list(argv)
    args = parser.parse_args(argv)
    args._argv = raw_argv

    base.apply_arg_type_conversions(args)
    _apply_finetune_arg_conversions(args)
    return args


def _apply_finetune_arg_conversions(args):
    args.lazy_records = base.str_to_bool(args.lazy_records)
    args.finetune_data_path = base.nonestr_to_none(args.finetune_data_path)
    args.split_manifest_path = base.nonestr_to_none(args.split_manifest_path)
    args.pencl_weights = base.nonestr_to_none(args.pencl_weights)
    args.facilitator_weights = base.nonestr_to_none(args.facilitator_weights)
    args.stage1_config_path = base.nonestr_to_none(args.stage1_config_path)
    args.stage2_config_path = base.nonestr_to_none(args.stage2_config_path)

    schema = args.record_schema
    if isinstance(schema, str):
        import json
        schema = json.loads(schema)
    if not schema:
        raise ValueError(
            "--record_schema (a JSON object) is required for finetuning"
        )
    args.record_schema = schema

    # Load any external compose-function plugins now, before the record_schema is
    # resolved against the registry (at dataset construction). Their module-body
    # @register_compose decorators run as an import side effect.
    plugins = args.compose_plugins
    if isinstance(plugins, str):
        plugins = plugins.strip()
        if plugins.startswith('['):
            import json
            plugins = json.loads(plugins)
        elif plugins:
            plugins = [plugins]
        else:
            plugins = []
    args.compose_plugins = plugins or []
    load_compose_plugins(args.compose_plugins)

    # Coerce the finetune subset selectors the same way run_PL_training.main does
    # (-2 sentinel == "unspecified" -> -1 == all).
    if args.finetune_last_n_blocks == -2:
        args.finetune_last_n_blocks = -1
    if args.finetune_last_n_layers == -2:
        args.finetune_last_n_layers = -1

    # LoRA options
    args.use_lora = base.str_to_bool(args.use_lora)
    args.lora_unfreeze_y_mlp = base.str_to_bool(args.lora_unfreeze_y_mlp)
    if isinstance(args.lora_target_patterns, str):
        args.lora_target_patterns = tuple(
            p.strip() for p in args.lora_target_patterns.split(',') if p.strip())
    return args


def parse_arguments(argv):
    return retrieve_all_args(argv)


def load_embedder_configs(args):
    if args.stage1_config_path is None or args.stage2_config_path is None:
        raise ValueError(
            "Both --stage1_config_path and --stage2_config_path are required for "
            "finetuning (they configure the frozen text->z_c embedder)."
        )
    stage1_args = convert_to_namespace(load_json_config(args.stage1_config_path))
    stage2_args = convert_to_namespace(load_json_config(args.stage2_config_path))
    return stage1_args, stage2_args


def load_data(args, stage1_args):
    if args.finetune_data_path is None:
        raise ValueError("--finetune_data_path (JSONL) is required for finetuning.")
    data_module = PL_mod.GeneralizedDataModule(
        jsonl_path=args.finetune_data_path,
        record_schema=args.record_schema,
        text_model_path=stage1_args.text_model_path,
        text_max_length=stage1_args.text_max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        image_size=args.image_size,
        sequence_key=args.sequence_output_key,
        caption_key=args.caption_key,
        length_field=args.length_field,
        lazy=args.lazy_records,
        split_manifest_path=args.split_manifest_path,
        needs_unique_sequences=PL_mod.alpha_spec_uses_zp(args.train_alpha),
    )
    data_module.setup()
    return data_module


def load_model(args, data_module, stage1_args, stage2_args):
    args.traindata_len = (
        len(data_module.train_dataloader()) // args.devices_per_node // args.acc_grad_batches
    )
    logger.info('Length of a training epoch in batch gradient updates: %s',
                args.traindata_len)

    if stage2_args.emb_dim != args.text_emb_dim:
        raise ValueError(
            f"Facilitator output dim ({stage2_args.emb_dim}) != ProteoScribe "
            f"text_emb_dim ({args.text_emb_dim}); z_c would not match the "
            f"conditioning MLP. Align the configs."
        )

    if get_global_rank() == 0:
        print_gpu_initialization()

    # On resume, weights come from the Lightning checkpoint via trainer.fit; on a
    # fresh finetune, load the pretrained ProteoScribe.
    model_fpath = None if args.resume_from_checkpoint is not None else args.pretrained_weights
    model = prepare_model_ProteoScribe(
        config_args=args,
        model_fpath=model_fpath,
        device=args.device,
        strict=True,
        eval=False,
        attempt_correction=True,
        verbosity=2,
    )
    logger.info('ProteoScribe model size: %s',
                sum(p.numel() for p in model.parameters()))

    embedder = build_text_to_zc_embedder(
        stage1_args=stage1_args,
        stage2_args=stage2_args,
        pencl_weights=args.pencl_weights,
        facilitator_weights=args.facilitator_weights,
        device=None,
    )
    zp_lookup = None
    if PL_mod.alpha_spec_uses_zp(args.train_alpha):
        zp_lookup = _precompute_zp_lookup(args, stage1_args, data_module)

    PL_model = PL_mod.PL_ProtARDM_Finetune(
        args=args,
        model=model,
        embedder=embedder,
        zp_lookup=zp_lookup,
        train_alpha=args.train_alpha,
        eval_alpha=args.eval_alpha,
    )
    return PL_model


def _precompute_zp_lookup(args, stage1_args, data_module):
    """Map every unique train/val sequence to its z_p, once.

    Valid because PenCL's protein branch is frozen for the whole run, so z_p
    never changes. ESM-2 is released afterwards — it is only needed for this
    pass and is far larger than the resulting embeddings.
    """
    zp_embedder = build_protein_to_zp_embedder(
        stage1_args=stage1_args,
        pencl_weights=args.pencl_weights,
        device=args.device,
    )
    try:
        seqs = data_module.unique_sequences()
        logger.info('Precomputing z_p for %d unique sequences (batch size %d)',
                    len(seqs), args.zp_batch_size)
        z_p = zp_embedder.embed_sequences(
            seqs, device=args.device, batch_size=args.zp_batch_size,
        )
    finally:
        del zp_embedder
        base.clear_gpu_cache()
    logger.info('z_p lookup built: %d entries, dim %d', z_p.size(0), z_p.size(1))
    return dict(zip(seqs, z_p))


def _finetune_dataset_probe(args):
    """CPU-only data-module probe for the dry-run report.

    Returns ``(train_len, val_len, sample_batch)``. Building the data module
    instantiates the BioBERT tokenizer; if its weights are absent the failure
    is surfaced as a note in the report by run_dry_run.
    """
    stage1_args, _ = load_embedder_configs(args)
    data_module = load_data(args, stage1_args=stage1_args)
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()
    train_len = len(train_dl.dataset) if hasattr(train_dl.dataset, '__len__') else None
    val_len = len(val_dl.dataset) if hasattr(val_dl.dataset, '__len__') else None
    sample = next(iter(train_dl), None)
    return train_len, val_len, sample


def _finetune_model_probe(args):
    """CPU-only ProteoScribe model probe for the dry-run report."""
    args.device = 'cpu'
    return mod.get_model(
        args=args,
        data_shape=(args.image_size, args.image_size),
        num_classes=args.num_classes,
    )


def main(args, ds_config=None):
    base._MAIN_START_MONOTONIC = time.perf_counter()
    base._LAST_TRAINER = None
    base._BACKUP_HISTORY.clear()
    start_time = datetime.now()

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

    if dry_run:
        return run_dry_run(
            args,
            stage="stage3_finetune",
            dataset_probe=lambda: _finetune_dataset_probe(args),
            model_probe=lambda: _finetune_model_probe(args),
        )

    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    logs_dir = os.path.join(run_dir, base._LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, base._ARTIFACTS_SUBDIR)
    checkpoint_dir = os.path.join(args.output_root, args.checkpoints_folder, args.run_id)
    if get_global_rank() == 0:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    log_path, file_handler = setup_file_logging(artifacts_dir)
    try:
        set_float32_matmul_precision(args.float32_matmul_precision)
        base.clear_gpu_cache()

        seed = args.seed
        if seed <= 0:
            seed = np.random.randint(2**32)
            args.seed = seed
        base.set_seed(seed)
        logger.info("Using seed: %s", seed)

        stage1_args, stage2_args = load_embedder_configs(args)
        data_module = load_data(args, stage1_args=stage1_args)
        PL_model = load_model(
            args, data_module=data_module,
            stage1_args=stage1_args, stage2_args=stage2_args,
        )

        # Finetuning script: freeze the base and expose only the requested trainable
        # subset. requires_grad flags are not persisted in Lightning checkpoints, so
        # this is re-applied on resume too. LoRA is an alternative to block-freezing.
        if getattr(args, "use_lora", False):
            from biom3.Stage3.lora import apply_lora_finetuning
            # LoRA and block-freezing are mutually exclusive strategies: under LoRA the
            # whole base is frozen and only adapters (+ y_mlp) train, so the
            # block-freezing selectors do not apply. Warn if they were set to anything
            # other than the "all/unspecified" sentinels so the ignore isn't silent.
            if (args.finetune_last_n_blocks not in (-1, -2)
                    or args.finetune_last_n_layers not in (-1, -2)
                    or args.finetune_output_layers):
                logger.warning(
                    "use_lora=True: ignoring block-freezing args "
                    "(finetune_last_n_blocks=%s, finetune_last_n_layers=%s, "
                    "finetune_output_layers=%s). LoRA freezes the full base and trains "
                    "only the adapters + y_mlp.",
                    args.finetune_last_n_blocks, args.finetune_last_n_layers,
                    args.finetune_output_layers,
                )
            PL_model = apply_lora_finetuning(
                PL_model,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_patterns=args.lora_target_patterns,
                unfreeze_y_mlp=args.lora_unfreeze_y_mlp,
            )
        else:
            PL_model = base.freeze_except_last_n_blocks_and_layers(
                PL_model=PL_model,
                n_blocks=args.finetune_last_n_blocks,
                n_layers=args.finetune_last_n_layers,
                finetune_output_layers=args.finetune_output_layers,
            )

        base._write_build_manifest(
            args=args,
            artifacts_dir=artifacts_dir,
            checkpoint_dir=checkpoint_dir,
            PL_model=PL_model,
            start_time=start_time,
        )

        exit_reason = "completed"
        exception = None
        try:
            base.train_model(
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
                elapsed = time.perf_counter() - base._MAIN_START_MONOTONIC
                if elapsed >= time_limit_seconds:
                    exit_reason = "time_limit_exceeded"
            completed_epochs = (
                base._LAST_TRAINER.current_epoch if base._LAST_TRAINER is not None else None
            )
            completed_steps = (
                base._LAST_TRAINER.global_step if base._LAST_TRAINER is not None else None
            )
            base._write_run_summary(
                artifacts_dir=artifacts_dir,
                start_time=start_time,
                exit_reason=exit_reason,
                exception=exception,
                completed_epochs=completed_epochs,
                completed_steps=completed_steps,
            )
            # LoRA runs emit BOTH the small adapter delta (portable, per-family) and
            # the full merged plain ProteoScribe checkpoint (drop-in for continued
            # training / downstream). Rank 0 only; ZeRO-2 replicates params so
            # PL_model.model is complete. Merge mutates the model in place, so do this
            # after the run summary and guard it so an export hiccup is non-fatal.
            if (getattr(args, "use_lora", False)
                    and exit_reason in ("completed", "time_limit_exceeded")
                    and get_global_rank() == 0):
                try:
                    from biom3.Stage3.lora import export_lora_finetuned
                    export_lora_finetuned(
                        PL_model.model,
                        lora_path=os.path.join(artifacts_dir, "lora_weights.pt"),
                        merged_path=os.path.join(artifacts_dir, "state_dict.merged.pth"),
                        target_patterns=args.lora_target_patterns,
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("LoRA export failed (non-fatal): %s", e)
            if base._MAIN_START_MONOTONIC is not None:
                total = int(time.perf_counter() - base._MAIN_START_MONOTONIC)
                h, rem = divmod(total, 3600)
                m, s = divmod(rem, 60)
                logger.info(
                    "Program exiting. Total elapsed time: %d:%02d:%02d", h, m, s,
                )
    finally:
        teardown_file_logging("biom3", file_handler)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
