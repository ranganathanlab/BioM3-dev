"""Tests for entrypoint biom3_PenCL_inference

Tests script: src/biom3/Stage1/run_PenCL_inference.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir, get_args, check_downloads

import torch

from biom3.Stage1.run_PenCL_inference import parse_arguments, main

pytestmark = [pytest.mark.slow]

#####################
##  Configuration  ##
#####################

# Directory containing text files with command line arguments
ARGS_DIR = os.path.join(DATDIR, "entrypoint_args")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs", "stage1_inference")

# Required weights that need to be downloaded to run entrypoint test. Each case
# also needs the model its args file names with -m.
REQUIRED_DOWNLOADS = [
    "weights/LLMs/esm2_t33_650M_UR50D.pt",
]


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize("argstring_fpath, expect_error_context", [
    [f"{ARGS_DIR}/stage1_args_v1.txt", does_not_raise()],
    [f"{ARGS_DIR}/stage1_args_v2.txt", does_not_raise()],
    [f"{ARGS_DIR}/stage1_args_v3.txt", does_not_raise()],
])
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_entrypoint(
        argstring_fpath, expect_error_context, device
    ):
    # Parse the command line string
    argstring = get_args(argstring_fpath)
    model_path = parse_arguments(argstring).model_path
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS + [model_path])
    if issues:
        pytest.skip(reason=skip_reason)
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Run entrypoint, manually adding device to the argstring.
    with expect_error_context:
        args = parse_arguments(argstring)
        args.device = device
        main(args)
        # Verify results can be loaded
        res = torch.load(
            os.path.join(OUTPUTS_DIR, "test_PenCL_embeddings.pt")
        )
        errors = []
        expected_keys = [
            "z_t", "z_p", "text_prompts", "sequence", "acc_id"
        ]
        for k in expected_keys:
            if k not in res:
                msg = f"key {k} not found in results"
                errors.append(msg)
        remove_dir(OUTPUTS_DIR)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("extra_args, expected", [
    [[], 0],
    [["--cross_comparison_sample_limit", "0"], 0],
    [["--cross_comparison_sample_limit", "1000"], 1000],
    [["--cross_comparison_sample_limit", "-1"], -1],
])
def test_cross_comparison_sample_limit_default(extra_args, expected):
    """The O(n^2) cross-comparison metrics are off unless asked for."""
    args = parse_arguments([
        "-i", "None",
        "-c", "configs/inference/stage1_PenCL.json",
        "-m", "weights/PenCL/BioM3_PenCL_epoch20.bin",
        "-o", "out.pt",
    ] + extra_args)
    assert args.cross_comparison_sample_limit == expected


@pytest.mark.parametrize("extra_args, expected", [
    [[], False],
    [["--no_amp"], True],
])
def test_no_amp_flag(extra_args, expected):
    """Autocast is on by default; --no_amp turns it off."""
    args = parse_arguments([
        "-i", "None",
        "-c", "configs/inference/stage1_PenCL.json",
        "-m", "weights/PenCL/BioM3_PenCL_epoch20.bin",
        "-o", "out.pt",
    ] + extra_args)
    assert args.no_amp is expected
