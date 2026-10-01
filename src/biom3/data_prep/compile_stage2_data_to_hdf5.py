"""
Compile Stage 2 Facilitator outputs into HDF5 format for Stage 3 finetuning.

Takes the .pt output of biom3_Facilitator_sample (containing acc_id, sequence,
and z_c keys) and writes an HDF5 file with a named dataset group ready for
the ProteoScribe finetuning pipeline.
"""

import argparse
import os
import sys
from datetime import datetime

import h5py
import numpy as np
import torch

from biom3.backend.device import setup_logger
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)

logger = setup_logger(__name__)


# Maps internal keys to those used in the Facilitator output (.pt file).
INPUT_KEYS = {
    "ACCESSION_ID": "acc_id",
    "SEQUENCE": "sequence",
    "FACILITATOR_EMBEDDING": "z_c",
}

# Maps internal keys to those written into the HDF5 output.
OUTPUT_KEYS = {
    "ACCESSION_ID": "acc_id",
    "SEQUENCE": "sequence",
    "SEQUENCE_LENGTH": "sequence_length",
    "FACILITATOR_EMBEDDING": "text_to_protein_embedding",
}


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Compile Stage 2 Facilitator data to HDF5"
    )
    parser.add_argument(
        "-i", "--input_data_path", type=str, required=True,
        help="Path to Stage 2 Facilitator output (.pt file)"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Path to write the compiled HDF5 file"
    )
    parser.add_argument(
        "--dataset_key", type=str, default="MMD_data",
        help="Group name under which data is stored in the HDF5 file (default: MMD_data)"
    )
    return parser.parse_args(args)


def main(args, _setup_logging=True):
    # Set up dual logging (console + file)
    outdir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(outdir, exist_ok=True)
    file_handler = None
    if _setup_logging:
        log_path, file_handler = setup_file_logging(outdir)
    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Compile Stage 2 data to HDF5")
        logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
        logger.info("Command:     %s", " ".join(sys.argv))
        logger.info("=" * 60)

        input_data = torch.load(args.input_data_path, weights_only=False)

        sequences = input_data[INPUT_KEYS["SEQUENCE"]]
        acc_ids = input_data[INPUT_KEYS["ACCESSION_ID"]]
        text_to_protein_embedding = input_data[INPUT_KEYS["FACILITATOR_EMBEDDING"]].cpu()
        sequence_length = np.array([len(seq) for seq in sequences], dtype=int)

        data_out = {
            "ACCESSION_ID": acc_ids,
            "SEQUENCE": sequences,
            "SEQUENCE_LENGTH": sequence_length,
            "FACILITATOR_EMBEDDING": text_to_protein_embedding,
        }

        with h5py.File(args.output_path, "w") as outfile:
            outfile.create_group(args.dataset_key)
            for k, output_key in OUTPUT_KEYS.items():
                outfile.create_dataset(
                    f"{args.dataset_key}/{output_key}",
                    data=data_out[k],
                )

        logger.info("Compiled HDF5 saved to %s (group: %s, %s samples)",
                    args.output_path, args.dataset_key, len(sequences))

        # Write manifest
        elapsed = datetime.now() - start_time
        if _setup_logging:
            write_manifest(
                args, outdir, start_time, elapsed,
                outputs={
                    "num_samples": len(sequences),
                    "dataset_key": args.dataset_key,
                    "output_file": os.path.abspath(args.output_path),
                },
                resolved_paths={
                    "input_data_path": os.path.abspath(args.input_data_path),
                },
            )
            logger.info("Done in %s", elapsed)
    finally:
        teardown_file_logging("biom3", file_handler)
