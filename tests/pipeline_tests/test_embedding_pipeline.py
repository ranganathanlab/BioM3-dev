"""Tests for entrypoint biom3_embedding_pipeline

Tests script: src/biom3/pipeline/embedding_pipeline.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise

# Stage 2 torch.load calls need this for .pt files containing non-tensor data.
# Normally set by environment.sh; make it explicit here for CI/test contexts.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from tests.conftest import DATDIR, TMPDIR, remove_dir, check_downloads

import h5py
from biom3.pipeline.embedding_pipeline import parse_arguments, main, _build_stage3_argv
from biom3.Stage3.run_ProteoScribe_sample import parse_arguments as parse_stage3_arguments

pytestmark = [pytest.mark.slow]

#####################
##  Configuration  ##
#####################

OUTPUTS_DIR = os.path.join(TMPDIR, "pipeline_outputs")

# Required weights that need to be downloaded to run entrypoint test
REQUIRED_DOWNLOADS = [
    "weights/LLMs/esm2_t33_650M_UR50D.pt",
    "weights/PenCL/BioM3_PenCL_epoch20.bin",
    "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
]


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
    "expect_error_context", [
    does_not_raise(),
])
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_embedding_pipeline(expect_error_context, device):
    import torch
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")

    prefix = "test_pipeline"
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with expect_error_context:
        args = parse_arguments([
            "-i", "None",
            "-o", OUTPUTS_DIR,
            "--pencl_weights", "weights/PenCL/BioM3_PenCL_epoch20.bin",
            "--facilitator_weights", "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
            "--pencl_config", "tests/_data/configs/test_stage1_config_v1.json",
            "--facilitator_config", "tests/_data/configs/test_stage2_config_v1.json",
            "--prefix", prefix,
            "--device", device,
            "--batch_size", "4",
            "--mmd_sample_limit", "5",
        ])
        main(args)

        # Verify all intermediate and final output files exist
        errors = []
        expected_files = [
            f"{prefix}.PenCL_emb.pt",
            f"{prefix}.Facilitator_emb.pt",
            f"{prefix}.compiled_emb.hdf5",
            f"{prefix}.run.log",
            f"{prefix}.build_manifest.json",
        ]
        for fname in expected_files:
            fpath = os.path.join(OUTPUTS_DIR, fname)
            if not os.path.isfile(fpath):
                errors.append(f"Expected output file not found: {fpath}")
        for fname in ["run.log", "build_manifest.json"]:
            if os.path.exists(os.path.join(OUTPUTS_DIR, fname)):
                errors.append(f"Unprefixed {fname} written to the output directory")

        # Verify HDF5 structure
        hdf5_path = os.path.join(OUTPUTS_DIR, f"{prefix}.compiled_emb.hdf5")
        if os.path.isfile(hdf5_path):
            with h5py.File(hdf5_path, "r") as f:
                if "MMD_data" not in f:
                    errors.append("Group 'MMD_data' not in HDF5 file")
                else:
                    for ds in ["acc_id", "sequence", "sequence_length",
                               "text_to_protein_embedding"]:
                        if f"MMD_data/{ds}" not in f:
                            errors.append(f"dataset MMD_data/{ds} not in HDF5")

        remove_dir(OUTPUTS_DIR)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("extra_args, expected", [
    [[], 0],
    [["--cross_comparison_sample_limit", "1000"], 1000],
])
def test_cross_comparison_sample_limit_forwarded(extra_args, expected):
    """The pipeline exposes Stage 1's cross-comparison cap, defaulting to off."""
    args = parse_arguments([
        "-i", "in.csv",
        "-o", OUTPUTS_DIR,
        "--pencl_weights", "weights/PenCL/BioM3_PenCL_epoch20.bin",
        "--facilitator_weights", "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
        "--pencl_config", "configs/inference/stage1_PenCL.json",
        "--facilitator_config", "configs/inference/stage2_Facilitator.json",
        "--prefix", "test",
    ] + extra_args)
    assert args.cross_comparison_sample_limit == expected


@pytest.mark.parametrize("extra_args, expected", [
    [[], None],
    [["--num_replicas", "7"], 7],
    [["--num_replicas", "None"], None],
])
def test_num_replicas_forwarded(extra_args, expected):
    """--num_replicas reaches the Stage 3 sampler only when set."""
    args = parse_arguments([
        "-i", "in.csv",
        "-o", OUTPUTS_DIR,
        "--pencl_weights", "weights/PenCL/BioM3_PenCL_epoch20.bin",
        "--facilitator_weights", "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
        "--pencl_config", "configs/inference/stage1_PenCL.json",
        "--facilitator_config", "configs/inference/stage2_Facilitator.json",
        "--prefix", "test",
        "--generate",
        "--proteoscribe_weights", "weights/ProteoScribe/model.bin",
        "--proteoscribe_config", "configs/inference/stage3_ProteoScribe_sample.json",
    ] + extra_args)
    assert args.num_replicas == expected
    stage3_argv = _build_stage3_argv(args, "facilitator.pt", "generated.pt")
    assert ("--num_replicas" in stage3_argv) == (expected is not None)
    assert parse_stage3_arguments(stage3_argv).num_replicas == expected


def test_num_replicas_rejects_invalid():
    with pytest.raises(SystemExit):
        parse_arguments([
            "-i", "in.csv",
            "-o", OUTPUTS_DIR,
            "--pencl_weights", "weights/PenCL/BioM3_PenCL_epoch20.bin",
            "--facilitator_weights", "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
            "--pencl_config", "configs/inference/stage1_PenCL.json",
            "--facilitator_config", "configs/inference/stage2_Facilitator.json",
            "--prefix", "test",
            "--num_replicas", "0",
        ])
