"""Tests for entrypoint biom3_ProteoScribe_sample

Tests script: src/biom3/Stage2/run_ProteoScribe_sample.py

"""

import json
import pytest
import os
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir, get_args, check_downloads

import torch
import numpy as np
import pytorch_lightning as pl
from biom3.Stage3.run_ProteoScribe_sample import (
    parse_arguments, main, load_json_config, convert_to_namespace
)
from biom3.Stage3.io import build_model_ProteoScribe
import biom3.Stage3.PL_wrapper as Stage3_PL_mod

pytestmark = [pytest.mark.slow]

#####################
##  Configuration  ##
#####################

# Directory containing text files with command line arguments
ARGS_DIR = os.path.join(DATDIR, "entrypoint_args")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs", "stage3_sample")
CKPT_DIR = os.path.join(TMPDIR, "checkpoints")
NUM_REPLICAS_DIR = os.path.join(TMPDIR, "outputs", "stage3_sample_num_replicas")

# Required weights that need to be downloaded to run entrypoint test
REQUIRED_DOWNLOADS = [
    "weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin",
]

# Test data paths for checkpoint tests
MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")
TEST_EMBEDDINGS = os.path.join(DATDIR, "embeddings/test_Facilitator_embeddings.pt")


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
    "argstring_fpath, expect_error_context", [
    [f"{ARGS_DIR}/stage3_args_v2.txt", does_not_raise()],
])
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_entrypoint(
        argstring_fpath, expect_error_context, device
    ):
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    # Parse the command line string
    argstring = get_args(argstring_fpath)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Run entrypoint
    with expect_error_context:
        args = parse_arguments(argstring)
        args.device = device
        main(args)
    remove_dir(OUTPUTS_DIR)


@pytest.mark.parametrize(
    "argstring_fpath", [
    f"{ARGS_DIR}/stage3_args_v2.txt",
])
@pytest.mark.parametrize(
    "seed1, seed2", [
        [1, 2],
        [1, 1]
    ]
)
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_reproducibility(
        argstring_fpath, seed1, seed2, device
    ):
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    # Parse the command line string
    argstring = get_args(argstring_fpath)
    # Run 1
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    args = parse_arguments(argstring)
    args.device = device
    args.seed = seed1
    main(args)
    res_dict1 = torch.load(
        os.path.join(OUTPUTS_DIR, "test_ProteoScribe_samples.pt")
    )
    remove_dir(OUTPUTS_DIR)
    # Run 2
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    args = parse_arguments(argstring)
    args.device = device
    args.seed = seed2
    main(args)
    res_dict2 = torch.load(
        os.path.join(OUTPUTS_DIR, "test_ProteoScribe_samples.pt")
    )
    remove_dir(OUTPUTS_DIR)
    # Compare results (prompt-indexed: {prompt_N: [replica_seqs], _metadata: {...}})
    expect_same = seed1 == seed2
    errors = []
    prompt_keys1 = [k for k in res_dict1 if not k.startswith("_")]
    prompt_keys2 = [k for k in res_dict2 if not k.startswith("_")]
    assert len(prompt_keys1) == len(prompt_keys2)
    for key in prompt_keys1:
        replicas1 = res_dict1[key]
        replicas2 = res_dict2[key]
        observed_same = np.all(
            [s1 == s2 for s1, s2 in zip(replicas1, replicas2)]
        )
        if expect_same and not observed_same:
            msg = f"Expected same results for {key} but results differed."
            msg += f"\n  Replicas 1: {replicas1}"
            msg += f"\n  Replicas 2: {replicas2}"
            errors.append(msg)
        elif not expect_same and observed_same:
            msg = f"Expected different results for {key} but results matched."
            msg += f"\n  Replicas 1 == Replicas 2: {replicas1}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


###############################################################################
######################   CHECKPOINT LOADING TESTS   ###########################
###############################################################################

@pytest.fixture
def mini_checkpoint_path():
    """Create a Lightning checkpoint from the mini model test weights."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "mini_checkpoint.ckpt")
    # Load config and build model
    config_dict = load_json_config(MINI_CONFIG)
    config_args = convert_to_namespace(config_dict)
    config_args.device = "cpu"
    model = build_model_ProteoScribe(config_args)
    # Load raw weights into model
    state_dict = torch.load(MINI_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    # Wrap in PL module and save as Lightning checkpoint
    pl_model = Stage3_PL_mod.PL_ProtARDM(args=config_args, model=model)
    trainer = pl.Trainer(max_epochs=0)
    trainer.strategy.connect(pl_model)
    trainer.save_checkpoint(ckpt_path)
    yield ckpt_path
    if os.path.exists(CKPT_DIR):
        remove_dir(CKPT_DIR)


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_entrypoint_load_from_checkpoint(mini_checkpoint_path, device):
    """Test that main() works when given a Lightning .ckpt file."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, "test_ProteoScribe_samples.pt")
    argstring = [
        "-i", TEST_EMBEDDINGS,
        "-c", MINI_CONFIG,
        "-m", mini_checkpoint_path,
        "-o", output_path,
    ]
    args = parse_arguments(argstring)
    args.device = device
    main(args)
    assert os.path.exists(output_path), "Output file was not created"
    result = torch.load(output_path)
    assert isinstance(result, dict), "Output should be a dict"
    assert len(result) > 0, "Output dict should not be empty"
    remove_dir(OUTPUTS_DIR)


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_entrypoint_without_checkpoint_flag(device):
    """Test that main() still works without --load_from_checkpoint (raw weights)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, "test_ProteoScribe_samples.pt")
    argstring = [
        "-i", TEST_EMBEDDINGS,
        "-c", MINI_CONFIG,
        "-m", MINI_WEIGHTS,
        "-o", output_path,
    ]
    args = parse_arguments(argstring)
    args.device = device
    main(args)
    assert os.path.exists(output_path), "Output file was not created"
    result = torch.load(output_path)
    assert isinstance(result, dict), "Output should be a dict"
    assert len(result) > 0, "Output dict should not be empty"
    remove_dir(OUTPUTS_DIR)


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_checkpoint_and_weights_produce_same_output(mini_checkpoint_path, device):
    """Test that loading from checkpoint produces the same output as loading raw weights."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    seed = 42
    # Run with raw state dict weights
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path_raw = os.path.join(OUTPUTS_DIR, "test_raw.pt")
    args_raw = parse_arguments([
        "-i", TEST_EMBEDDINGS,
        "-c", MINI_CONFIG,
        "-m", MINI_WEIGHTS,
        "-o", output_path_raw,
        "--seed", str(seed),
    ])
    args_raw.device = device
    main(args_raw)
    result_raw = torch.load(output_path_raw)
    remove_dir(OUTPUTS_DIR)
    # Run with Lightning checkpoint
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path_ckpt = os.path.join(OUTPUTS_DIR, "test_ckpt.pt")
    args_ckpt = parse_arguments([
        "-i", TEST_EMBEDDINGS,
        "-c", MINI_CONFIG,
        "-m", mini_checkpoint_path,
        "-o", output_path_ckpt,
        "--seed", str(seed),
    ])
    args_ckpt.device = device
    main(args_ckpt)
    result_ckpt = torch.load(output_path_ckpt)
    remove_dir(OUTPUTS_DIR)
    # Compare results (prompt-indexed structure)
    prompt_keys_raw = sorted(k for k in result_raw if not k.startswith("_"))
    prompt_keys_ckpt = sorted(k for k in result_ckpt if not k.startswith("_"))
    assert prompt_keys_raw == prompt_keys_ckpt
    for key in prompt_keys_raw:
        seqs_raw = result_raw[key]
        seqs_ckpt = result_ckpt[key]
        assert seqs_raw == seqs_ckpt, (
            f"{key} mismatch:\n  raw:  {seqs_raw}\n  ckpt: {seqs_ckpt}"
        )


###############################################################################
###########################   NUM REPLICAS TESTS   ############################
###############################################################################

REQUIRED_PATH_ARGS = ["-i", "in.pt", "-c", "config.json", "-m", "model.pth", "-o", "out.pt"]


@pytest.mark.parametrize("cli, expected", [
    [[], None],
    [["--num_replicas", "4"], 4],
    [["--num_replicas", "None"], None],
])
def test_parse_num_replicas(cli, expected):
    args = parse_arguments(REQUIRED_PATH_ARGS + cli)
    assert args.num_replicas == expected


@pytest.mark.parametrize("value", ["0", "-1", "two", "2.5"])
def test_parse_num_replicas_rejects(value):
    with pytest.raises(SystemExit):
        parse_arguments(REQUIRED_PATH_ARGS + ["--num_replicas", value])


OMIT = object()


def _write_mini_config(num_replicas):
    """Write the mini config with num_replicas set, or removed when OMIT."""
    config = load_json_config(MINI_CONFIG)
    config.pop("num_replicas", None)
    if num_replicas is not OMIT:
        config["num_replicas"] = num_replicas
    os.makedirs(NUM_REPLICAS_DIR, exist_ok=True)
    path = os.path.join(NUM_REPLICAS_DIR, "config.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


@pytest.mark.parametrize("cli, config_num_replicas, expected", [
    [["--num_replicas", "3"], 2, 3],
    [["--num_replicas", "None"], 2, 2],
    [[], OMIT, 5],
    [[], None, 5],
    [[], "None", 5],
])
def test_num_replicas_resolution(cli, config_num_replicas, expected):
    config_path = _write_mini_config(config_num_replicas)
    output_path = os.path.join(NUM_REPLICAS_DIR, "samples.pt")
    args = parse_arguments([
        "-i", TEST_EMBEDDINGS,
        "-c", config_path,
        "-m", MINI_WEIGHTS,
        "-o", output_path,
        "--device", "cpu",
    ] + cli)
    main(args)
    result = torch.load(output_path)
    with open(os.path.join(NUM_REPLICAS_DIR, "build_manifest.json")) as f:
        manifest = json.load(f)
    remove_dir(NUM_REPLICAS_DIR)
    assert result["_metadata"]["num_replicas"] == expected
    prompt_keys = [k for k in result if not k.startswith("_")]
    assert prompt_keys
    assert all(len(result[k]) == expected for k in prompt_keys)
    assert manifest["outputs"]["num_replicas"] == expected


def test_num_replicas_invalid_config_value():
    config_path = _write_mini_config(0)
    args = parse_arguments([
        "-i", TEST_EMBEDDINGS,
        "-c", config_path,
        "-m", MINI_WEIGHTS,
        "-o", os.path.join(NUM_REPLICAS_DIR, "samples.pt"),
        "--device", "cpu",
    ])
    try:
        with pytest.raises(ValueError, match="config num_replicas must be a positive integer"):
            main(args)
    finally:
        remove_dir(NUM_REPLICAS_DIR)
