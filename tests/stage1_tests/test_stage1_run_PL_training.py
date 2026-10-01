"""Smoke test for `biom3_train_stage1` (Stage 1 PenCL training).

Runs one epoch of default-mode training on a tiny CSV and asserts that the
expected artifacts land in the run directory.
"""

import json
import os
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from tests.conftest import DATDIR, TMPDIR, remove_dir, get_args, check_downloads

from biom3.Stage1.run_PL_training import parse_arguments, main


ARGS_DIR = os.path.join(DATDIR, "entrypoint_args", "training")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs", "stage1_smoke")

REQUIRED_DOWNLOADS = [
    "weights/LLMs/esm2_t33_650M_UR50D.pt",
    "weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
]


def prefix_paths(args):
    if args.data_path is not None:
        args.data_path = os.path.join(DATDIR, args.data_path)
    if args.pfam_data_path is not None:
        args.pfam_data_path = os.path.join(DATDIR, args.pfam_data_path)
    if args.output_root is not None:
        args.output_root = os.path.join(TMPDIR, args.output_root)
    if args.pfam_splits_dir is not None:
        args.pfam_splits_dir = os.path.join(TMPDIR, args.pfam_splits_dir)
    if args.pretrained_weights is not None:
        args.pretrained_weights = os.path.join(DATDIR, args.pretrained_weights)


@pytest.mark.parametrize(
    "argstring_fpath, expect_error_context, expected_dataset_type", [
        [f"{ARGS_DIR}/stage1_training_args_scratch_v1.txt", does_not_raise(), "default"],
        [f"{ARGS_DIR}/stage1_training_args_pfam_v1.txt", does_not_raise(), "pfam"],
        [f"{ARGS_DIR}/stage1_training_args_pfam_uniformity_v1.txt", does_not_raise(), "pfam"],
    ],
)
@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_stage1_train_from_scratch(argstring_fpath, expect_error_context, expected_dataset_type, device):
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip(reason="device=xpu and xpu not available")

    argstring = get_args(argstring_fpath)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with expect_error_context:
        args = parse_arguments(argstring)
        prefix_paths(args)
        args.device = device
        main(args)

    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    checkpoint_dir = os.path.join(args.output_root, args.checkpoints_folder, args.run_id)

    assert os.path.exists(os.path.join(artifacts_dir, "args.json")), \
        f"args.json missing under {artifacts_dir}"
    assert os.path.exists(os.path.join(artifacts_dir, "build_manifest.json")), \
        f"build_manifest.json missing under {artifacts_dir}"
    assert os.path.exists(os.path.join(checkpoint_dir, "state_dict.best.pth")), \
        f"state_dict.best.pth missing under {checkpoint_dir}"

    with open(os.path.join(artifacts_dir, "args.json")) as f:
        args_dict = json.load(f)
    assert args_dict["dataset_type"] == expected_dataset_type
    assert args_dict["epochs"] == 1

    remove_dir(OUTPUTS_DIR)
