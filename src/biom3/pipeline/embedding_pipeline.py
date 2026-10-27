"""
End-to-end embedding pipeline: CSV → Stage 1 → Stage 2 → compiled HDF5.

Python replacement for scripts/embedding_pipeline.sh. Runs PenCL inference,
Facilitator sampling, and HDF5 compilation in sequence, constructing the
intermediate file paths automatically from --output_dir and --prefix.

With --generate the terminal step is ProteoScribe sampling instead of HDF5
compilation, giving CSV → Stage 1 → Stage 2 → Stage 3 generated sequences.
With --skip_facilitator Stage 2 is replaced by the identity map, z_c = z_t, for
weight sets that have no Facilitator.
Running this entrypoint under a launcher (scripts/launchers/*_{single,multi}node.sh)
shards the work: Stage 1 splits batches across ranks and merges on rank 0,
Stage 2 and the HDF5 compile run on rank 0 alone, and the Stage 3 sampler
shards by rank. Run directly, it is an ordinary single-process pipeline.
"""

import argparse
import os
import sys
from argparse import Namespace
from datetime import datetime

import torch

from biom3.backend.device import setup_logger
from biom3.backend.device import DEVICE_CHOICES, resolve_device
from biom3.core.distributed import barrier, is_main_process
from biom3.core.helpers import load_json_config, optional_positive_int
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)
from biom3.core.weight_sets import (
    check_normalize_zc,
    check_text_attention_mask,
    merge_weight_set,
)

logger = setup_logger(__name__)


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="BioM3 Embedding Pipeline (Stage 1 → Stage 2 → HDF5)"
    )
    # Required paths
    parser.add_argument(
        "-i", "--input_data_path", type=str, required=True,
        help="Path to input CSV (sequences + text prompts)"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Directory for all output files"
    )
    parser.add_argument(
        "--weight_set", type=str, default=None,
        help="Path to a weight-set bundle JSON (e.g. configs/weights/run1_base.json) "
             "providing pencl/facilitator weights. Explicit --*_weights override it."
    )
    parser.add_argument(
        "--pencl_weights", type=str, default=None,
        help="Path to PenCL model weights or checkpoint (overrides --weight_set)"
    )
    parser.add_argument(
        "--facilitator_weights", type=str, default=None,
        help="Path to Facilitator model weights or checkpoint (overrides --weight_set)"
    )
    parser.add_argument(
        "--pencl_config", type=str, required=True,
        help="Path to Stage 1 JSON config (stage1_config_PenCL_inference.json)"
    )
    parser.add_argument(
        "--facilitator_config", type=str, default=None,
        help="Path to Stage 2 JSON config (stage2_config_Facilitator_sample.json); "
             "required unless --skip_facilitator is given"
    )
    parser.add_argument(
        "--skip_facilitator", action="store_true", default=False,
        help="Replace Stage 2 with the identity map: z_c is written as a copy "
             "of z_t, and the remaining steps run as usual. Facilitator config "
             "and weights are not needed"
    )
    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Filename prefix for intermediate and final output files, the "
             "run log, and the manifest"
    )

    # Optional overrides
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=list(DEVICE_CHOICES),
        help="Device for inference (default: auto = the detected backend: CUDA, XPU, else CPU)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for Stage 1 PenCL inference (default: 256)"
    )
    parser.add_argument(
        "--stage2_batch_size", type=int, default=65536,
        help="Batch size for Stage 2 Facilitator inference (default: 65536)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of dataloader workers for Stage 1 (default: 0)"
    )
    parser.add_argument(
        "--text_padding", type=str, default="max_padding",
        choices=["max_padding", "dynamic"],
        help="Stage 1 caption padding. 'max_padding' pads to text_max_length, "
             "matching training; 'dynamic' pads to the batch's longest caption, "
             "which makes z_t depend on batch composition unless "
             "--text_attention_mask is given (default: max_padding)"
    )
    parser.add_argument(
        "--text_attention_mask", action="store_true",
        help="Stage 1: pass the caption attention mask to BERT. Set it to match "
             "how the PenCL weights were trained; off (default) for run1_base. "
             "A weight set can record that as "
             "pencl_trained_with_text_attention_mask, and the pipeline warns "
             "when this flag disagrees with it"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable autocast in Stage 1 and run the forward pass in fp32 "
             "(default: autocast on, bf16 on xpu / fp16 on cuda)"
    )
    parser.add_argument(
        "--float32_matmul_precision", type=str, default=None,
        choices=["highest", "high", "medium"],
        help="Stage 1 fp32 matmul precision. Pair 'highest' with --no_amp for a "
             "deterministic fp32 forward pass (default: the config value, 'high')"
    )
    parser.add_argument(
        "--cross_comparison_sample_limit", type=int, default=0,
        help="Samples used for Stage 1's O(n^2) cross-comparison metrics. "
             "0 (default) skips them, -1 uses all, a positive value uses that "
             "many. Print-only; the compiled embeddings are unaffected"
    )
    parser.add_argument(
        "--mmd_sample_limit", type=int, default=1000,
        help="Sample limit for MMD computation in Stage 2 (default: 1000)"
    )
    parser.add_argument(
        "--dataset_key", type=str, default="MMD_data",
        help="HDF5 group name for compiled output (default: MMD_data)"
    )

    # Stage 3 generation (terminal step instead of HDF5 compilation)
    parser.add_argument(
        "--generate", action="store_true", default=False,
        help="Run Stage 3 ProteoScribe sampling after Stage 2 instead of "
             "compiling to HDF5"
    )
    parser.add_argument(
        "--proteoscribe_weights", type=str, default=None,
        help="Path to ProteoScribe weights or checkpoint (overrides --weight_set); "
             "required with --generate"
    )
    parser.add_argument(
        "--proteoscribe_config", type=str, default=None,
        help="Path to Stage 3 JSON config; required with --generate"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Stage 3 sampling seed (default: 0)"
    )
    parser.add_argument(
        "--no_fasta", dest="fasta", action="store_false", default=True,
        help="Do not write generated sequences as FASTA (Stage 3; FASTA is on by default)"
    )
    parser.add_argument(
        "--token_strategy", type=str, default=None,
        help="Stage 3 token strategy (passed through to the sampler)"
    )
    parser.add_argument(
        "--unmasking_order", type=str, default=None,
        help="Stage 3 unmasking order (passed through to the sampler)"
    )
    parser.add_argument(
        "--normalize_zc", action="store_true",
        help="Stage 3: scale each conditioning vector to unit length (passed through "
             "to the sampler; for a ProteoScribe trained with --normalize_zc True)"
    )
    parser.add_argument(
        "--num_replicas", type=optional_positive_int, default=None,
        help="Stage 3 sequences per prompt (passed through to the sampler; "
             "unset defers to the sampler's config-then-default resolution)"
    )
    parsed = parser.parse_args(args)

    if not parsed.skip_facilitator and not parsed.facilitator_config:
        parser.error("--facilitator_config is required unless --skip_facilitator is given")

    weight_keys = ["pencl_weights"]
    if not parsed.skip_facilitator:
        weight_keys.append("facilitator_weights")
    if parsed.generate:
        weight_keys.append("proteoscribe_weights")

    merge_weight_set(parsed, parsed.weight_set, keys=tuple(weight_keys))
    missing = [k for k in weight_keys if not getattr(parsed, k)]
    if missing:
        parser.error(
            "missing weight path(s): " + ", ".join("--" + m for m in missing)
            + " (provide them directly or via --weight_set)"
        )
    if parsed.generate and not parsed.proteoscribe_config:
        parser.error("--proteoscribe_config is required with --generate")
    return parsed


def _build_stage3_argv(args, input_path, output_path):
    argv = [
        "-i", input_path,
        "-c", args.proteoscribe_config,
        "-m", args.proteoscribe_weights,
        "-o", output_path,
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.fasta:
        argv.append("--fasta")
    if args.token_strategy:
        argv += ["--token_strategy", args.token_strategy]
    if args.unmasking_order:
        argv += ["--unmasking_order", args.unmasking_order]
    if args.num_replicas is not None:
        argv += ["--num_replicas", str(args.num_replicas)]
    if args.normalize_zc:
        argv.append("--normalize_zc")
    return argv


def _write_identity_zc(pencl_output, facilitator_output):
    """Stand in for Stage 2 with the identity map: z_c is a copy of z_t."""
    embedding_dataset = torch.load(pencl_output, weights_only=False)
    embedding_dataset["z_c"] = embedding_dataset["z_t"].clone()
    torch.save(embedding_dataset, facilitator_output)


def main(args):
    args.device = resolve_device(args.device)
    from biom3.Stage1.run_PenCL_inference import (
        parse_arguments as parse_stage1_args,
        main as run_stage1,
    )
    from biom3.Stage2.run_Facilitator_sample import (
        parse_arguments as parse_stage2_args,
        main as run_stage2,
    )
    from biom3.data_prep.compile_stage2_data_to_hdf5 import (
        parse_arguments as parse_compile_args,
        main as run_compile,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up dual logging (console + file). Only the main rank owns the file.
    file_handler = None
    if is_main_process():
        log_path, file_handler = setup_file_logging(
            args.output_dir, log_filename=f"{args.prefix}.run.log")
    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(
            "Embedding pipeline (Stage 1 -> %s -> %s)",
            "identity" if args.skip_facilitator else "Stage 2",
            "Stage 3" if args.generate else "HDF5",
        )
        logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
        logger.info("Command:     %s", " ".join(sys.argv))
        logger.info("=" * 60)

        mask_summary, mask_warning = check_text_attention_mask(
            args.weight_set, args.pencl_weights, args.text_attention_mask)
        zc_warning = []
        if args.generate:
            zc_summary, zc_warning = check_normalize_zc(
                args.weight_set, args.proteoscribe_weights, args.normalize_zc)
        if is_main_process():
            logger.info(mask_summary)
            if args.generate:
                logger.info(zc_summary)
            for line in mask_warning + zc_warning:
                logger.warning(line)

        # Load config contents for manifest
        pencl_config_contents = load_json_config(args.pencl_config)
        if not args.skip_facilitator:
            facilitator_config_contents = load_json_config(args.facilitator_config)

        # Intermediate file paths
        pencl_output = os.path.join(args.output_dir, f"{args.prefix}.PenCL_emb.pt")
        facilitator_output = os.path.join(args.output_dir, f"{args.prefix}.Facilitator_emb.pt")
        hdf5_output = os.path.join(args.output_dir, f"{args.prefix}.compiled_emb.hdf5")
        generated_output = os.path.join(args.output_dir, f"{args.prefix}.generated.pt")

        # --- Stage 1: PenCL inference ---
        logger.info("=" * 60)
        logger.info("Stage 1: PenCL inference")
        logger.info("=" * 60)
        stage1_args = parse_stage1_args([
            "-i", args.input_data_path,
            "-c", args.pencl_config,
            "-m", args.pencl_weights,
            "-o", pencl_output,
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--cross_comparison_sample_limit", str(args.cross_comparison_sample_limit),
            "--text_padding", args.text_padding,
        ] + (["--no_amp"] if args.no_amp else [])
          + (["--text_attention_mask"] if args.text_attention_mask else [])
          + (["--float32_matmul_precision", args.float32_matmul_precision]
             if args.float32_matmul_precision else []))
        run_stage1(stage1_args, _setup_logging=False)

        # Stage 1 shards across ranks when launched under a launcher and writes
        # pencl_output on the main rank only; wait for it before reading.
        barrier()

        # --- Stage 2: Facilitator sampling ---
        # Main rank only: the Facilitator is a small MLP, so there is nothing to gain
        # from sharding it, and every rank writing the same file would race.
        if args.skip_facilitator and is_main_process():
            logger.info("=" * 60)
            logger.info("Stage 2 skipped: z_c is a copy of z_t")
            logger.info("=" * 60)
            _write_identity_zc(pencl_output, facilitator_output)
        elif is_main_process():
            logger.info("=" * 60)
            logger.info("Stage 2: Facilitator sampling")
            logger.info("=" * 60)
            stage2_args = parse_stage2_args([
                "-i", pencl_output,
                "-c", args.facilitator_config,
                "-m", args.facilitator_weights,
                "-o", facilitator_output,
                "--device", args.device,
                "--batch_size", str(args.stage2_batch_size),
                "--mmd_sample_limit", str(args.mmd_sample_limit),
            ])
            run_stage2(stage2_args, _setup_logging=False)
        barrier()

        if args.generate:
            # --- Stage 3: ProteoScribe sampling ---
            from biom3.Stage3.run_ProteoScribe_sample import (
                parse_arguments as parse_stage3_args,
                main as run_stage3,
            )

            logger.info("=" * 60)
            logger.info("Stage 3: ProteoScribe sampling")
            logger.info("=" * 60)
            stage3_argv = _build_stage3_argv(args, facilitator_output, generated_output)
            run_stage3(parse_stage3_args(stage3_argv), _setup_logging=False)
            final_output = generated_output
        else:
            # --- Compile to HDF5 --- (main rank only; single small write)
            if is_main_process():
                logger.info("=" * 60)
                logger.info("Compiling Stage 2 output to HDF5")
                logger.info("=" * 60)
                compile_args = parse_compile_args([
                    "-i", facilitator_output,
                    "-o", hdf5_output,
                    "--dataset_key", args.dataset_key,
                ])
                run_compile(compile_args, _setup_logging=False)
            final_output = hdf5_output

        logger.info("=" * 60)
        if not is_main_process():
            return

        logger.info("Pipeline complete. Output: %s", final_output)
        logger.info("=" * 60)
        for line in mask_warning + zc_warning:
            logger.warning(line)

        # Write manifest
        elapsed = datetime.now() - start_time
        outputs = {
            "pencl_output": os.path.abspath(pencl_output),
            "facilitator_output": os.path.abspath(facilitator_output),
        }
        resolved_paths = {
            "input_data_path": os.path.abspath(args.input_data_path),
            "weight_set": os.path.abspath(args.weight_set) if args.weight_set else None,
            "pencl_weights": os.path.abspath(args.pencl_weights),
            "pencl_config": os.path.abspath(args.pencl_config),
        }
        config_contents = {"pencl": pencl_config_contents}
        if not args.skip_facilitator:
            resolved_paths["facilitator_weights"] = os.path.abspath(args.facilitator_weights)
            resolved_paths["facilitator_config"] = os.path.abspath(args.facilitator_config)
            config_contents["facilitator"] = facilitator_config_contents
        if args.generate:
            outputs["generated_output"] = os.path.abspath(generated_output)
            resolved_paths["proteoscribe_weights"] = os.path.abspath(args.proteoscribe_weights)
            resolved_paths["proteoscribe_config"] = os.path.abspath(args.proteoscribe_config)
            config_contents["proteoscribe"] = load_json_config(args.proteoscribe_config)
        else:
            outputs["hdf5_output"] = os.path.abspath(hdf5_output)

        write_manifest(
            args, args.output_dir, start_time, elapsed,
            outputs=outputs,
            resolved_paths=resolved_paths,
            config_contents=config_contents,
            manifest_filename=f"{args.prefix}.build_manifest.json",
        )
        logger.info("Done in %s", elapsed)
    finally:
        teardown_file_logging("biom3", file_handler)
