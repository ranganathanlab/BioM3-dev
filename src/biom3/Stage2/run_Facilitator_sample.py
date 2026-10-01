"""BioM3 Stage 2: Facilitator sampling

Mimics the workflow described at
    https://huggingface.co/niksapraljak1/BioM3#stage-2-facilitator-sampling

Follows execution of Stage 1: run_PenCL_inference

Config file:
    configs/inference/stage2_Facilitator.json  (uses _base_configs composition)

Example usage:

biom3_Facilitator_sample \
    --input_data_path "outputs/pencl_embeddings.pt" \
    --config_path "configs/inference/stage2_Facilitator.json" \
    --model_path "./weights/Facilitator/BioM3_Facilitator_epoch20.bin" \
    --output_data_path "outputs/facilitator_embeddings.pt"

Example usage (CPU, limited MMD computation):

biom3_Facilitator_sample \
    --input_data_path "outputs/pencl_embeddings.pt" \
    --config_path "configs/inference/stage2_Facilitator.json" \
    --model_path "./weights/Facilitator/BioM3_Facilitator_epoch20.bin" \
    --output_data_path "outputs/facilitator_embeddings.pt" \
    --device cpu \
    --mmd_sample_limit 256

"""

import copy
import os
import sys
import argparse
import yaml
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

import biom3.Stage1.model as mod
from biom3.core.io import load_and_prepare_model
from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)
from biom3.backend.device import setup_logger, set_float32_matmul_precision
from biom3.backend.device import DEVICE_CHOICES, resolve_device

logger = setup_logger(__name__)

# Step 0: Argument Parser Function
def parse_arguments(args):
    parser = argparse.ArgumentParser(description="BioM3 Facilitator Model (Stage 2)")
    parser.add_argument('-i', '--input_data_path', type=str, required=True,
                        help="Path to the input embeddings (e.g., PenCL_test_outputs.pt)")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="Path to the JSON configuration file (stage2_config_Facilitator_sample.json)")
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help="Path to the Facilitator model weights (e.g., BioM3_Facilitator_epoch20.bin)")
    parser.add_argument('-o', '--output_data_path', type=str, required=True,
                        help="Path to save the output embeddings (e.g., Facilitator_test_outputs.pt)")
    
    parser.add_argument('--device', type=str, default="auto",
                        choices=list(DEVICE_CHOICES),
                        help="available device; auto = the detected backend (CUDA, XPU, else CPU)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size")
    parser.add_argument("--mmd_sample_limit", type=int, default=-1,
                        help="limit on the number of samples used to compute MMD. If -1, use all")
    parser.add_argument("--float32_matmul_precision", type=str, default=None,
                        choices=["highest", "high", "medium"],
                        help="fp32 matmul precision. 'high' (config default) enables "
                             "TF32 tensor cores; 'highest' forces full fp32 for bitwise "
                             "reproducibility. Overrides the config value when set.")
    return parser.parse_args(args)


# Step 3: Load Pre-trained Model
def prepare_model(config_args, model_path, device) -> nn.Module:
    # Initialize the model graph
    model = mod.Facilitator(
        in_dim=config_args.emb_dim,
        hid_dim=config_args.hid_dim,
        out_dim=config_args.emb_dim,
        dropout=config_args.dropout
    )
    # Load model weights
    model = load_and_prepare_model(
        model, model_path, 
        device=device, 
        strict=True, 
        eval_mode=True,
        attempt_correction=True,
        substitutions={"model.main.": "main."}
    )
    logger.info("Model loaded successfully with weights!")
    return model


# Step 4: Compute MMD Loss
def compute_mmd_loss(x, y, kernel="rbf", sigma=1.0):
    def rbf_kernel(a, b, sigma):
        pairwise_distances = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-pairwise_distances / (2 * sigma ** 2))

    K_xx = rbf_kernel(x, x, sigma)
    K_yy = rbf_kernel(y, y, sigma)
    K_xy = rbf_kernel(x, y, sigma)

    mmd_loss = K_xx.mean() - 2 * K_xy.mean() + K_yy.mean()
    return mmd_loss


def main(args, _setup_logging=True):
    args.device = resolve_device(args.device)

    # ----- Suppress noisy library warnings -----
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")

    # Set up dual logging (console + file)
    outdir = os.path.dirname(os.path.abspath(args.output_data_path))
    os.makedirs(outdir, exist_ok=True)
    file_handler = None
    if _setup_logging:
        log_path, file_handler = setup_file_logging(outdir)
    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Facilitator sampling (Stage 2)")
        logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
        logger.info("Command:     %s", " ".join(sys.argv))
        logger.info("=" * 60)

        # Load configuration
        config_dict = load_json_config(args.config_path)
        raw_config = copy.deepcopy(config_dict)
        config_args = convert_to_namespace(config_dict)

        # fp32 matmul precision (TF32): CLI overrides config, config default is "high".
        set_float32_matmul_precision(
            args.float32_matmul_precision
            or getattr(config_args, "float32_matmul_precision", "high")
        )

        mmd_sample_limit = args.mmd_sample_limit
        device = torch.device(args.device)  # TODO: cuda results in OOM

        # Load model
        model = prepare_model(
            config_args=config_args, 
            model_path=args.model_path,
            device=device,
        )
        model.to(device)

        # Load input embeddings
        embedding_dataset = torch.load(args.input_data_path)

        # Run inference to get facilitated embeddings.
        z_t = embedding_dataset['z_t']
        z_p = embedding_dataset['z_p']
        batch_size = max(1, args.batch_size)
        num_rows = len(z_t)

        z_c_batches = []
        sq_err_zc_zp = []
        sq_err_zt_zp = []
        with torch.no_grad():
            for start in range(0, num_rows, batch_size):
                end = min(start + batch_size, num_rows)
                z_t_batch = z_t[start:end].to(device)
                z_p_batch = z_p[start:end].to(device)
                z_c_batch = model(z_t_batch)
                z_c_batches.append(z_c_batch.cpu())
                # Left on the device; .item() here would sync once per batch.
                sq_err_zc_zp.append((z_c_batch - z_p_batch).pow(2).sum())
                sq_err_zt_zp.append((z_t_batch - z_p_batch).pow(2).sum())

        z_c = torch.cat(z_c_batches) if z_c_batches else z_t[:0]
        embedding_dataset['z_c'] = z_c

        # Compute evaluation metrics
        # 1. MSE between embeddings
        # Summed in fp64 and divided by the element count to match F.mse_loss.
        num_elements = z_t.numel()
        mse_zc_zp = (torch.stack(sq_err_zc_zp).double().sum() / num_elements).item() if num_elements else 0.0
        mse_zt_zp = (torch.stack(sq_err_zt_zp).double().sum() / num_elements).item() if num_elements else 0.0

        # 2. Compute L2 norms for first batch
        batch_idx = 0
        norm_z_t = torch.norm(z_t[batch_idx], p=2).item()
        norm_z_p = torch.norm(z_p[batch_idx], p=2).item()
        norm_z_c = torch.norm(z_c[batch_idx], p=2).item()

        # 3. Compute MMD between embeddings
        k = min(mmd_sample_limit, len(z_t))
        mmd_zc_zp = model.compute_mmd(z_c[0:k].to(device), z_p[0:k].to(device))
        mmd_zp_zt = model.compute_mmd(z_p[0:k].to(device), z_t[0:k].to(device))

        # Print results
        logger.info("\n=== Facilitator Model Output ===")
        logger.info("Shape of z_t (Text Embeddings): %s", z_t.shape)
        logger.info("Shape of z_p (Protein Embeddings): %s", z_p.shape)
        logger.info("Shape of z_c (Facilitated Embeddings): %s", z_c.shape)

        logger.info("\n=== Norm (L2 Magnitude) Results for Batch Index 0 ===")
        logger.info("Norm of z_t (Text Embedding): %.6f", norm_z_t)
        logger.info("Norm of z_p (Protein Embedding): %.6f", norm_z_p)
        logger.info("Norm of z_c (Facilitated Embedding): %.6f", norm_z_c)

        logger.info("\n=== Mean Squared Error (MSE) Results ===")
        logger.info("MSE between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): %.6f", mse_zc_zp)
        logger.info("MSE between Text Embeddings (z_t) and Protein Embeddings (z_p): %.6f", mse_zt_zp)

        logger.info("\n=== Max Mean Discrepancy (MMD) Results ===")
        logger.info("MMD between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): %.6f", mmd_zc_zp)
        logger.info("MMD between Text Embeddings (z_t) and Protein Embeddings (z_p): %.6f", mmd_zp_zt)

        # Save output embeddings
        torch.save(embedding_dataset, args.output_data_path)
        logger.info("Facilitator embeddings saved to %s", args.output_data_path)

        # Write manifest
        elapsed = datetime.now() - start_time
        if _setup_logging:
            write_manifest(
                args, outdir, start_time, elapsed,
                outputs={
                    "num_samples": int(z_t.shape[0]),
                    "embedding_dim": int(z_c.shape[1]),
                    "mse_zc_zp": float(mse_zc_zp),
                    "mse_zt_zp": float(mse_zt_zp),
                    "mmd_zc_zp": float(mmd_zc_zp),
                    "mmd_zp_zt": float(mmd_zp_zt),
                    "output_file": os.path.abspath(args.output_data_path),
                },
                resolved_paths={
                    "input_data_path": os.path.abspath(args.input_data_path),
                    "model_path": os.path.abspath(args.model_path),
                    "json_config": os.path.abspath(args.config_path),
                },
                config_contents=raw_config,
            )
            logger.info("Done in %s", elapsed)
    finally:
        teardown_file_logging("biom3", file_handler)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
