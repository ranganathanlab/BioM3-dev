"""
Common helper functions

"""

import json
import os
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


def _resolve_config_paths(paths: list[str], base_dir: str) -> list[str]:
    """Resolve config paths relative to *base_dir*."""
    resolved = []
    for p in paths:
        if not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        resolved.append(p)
    return resolved


def load_json_config(json_path: str, _visited=None, *,
                     track_provenance: bool = False):
    """Load JSON configuration with optional config composition.

    Two special keys control composition:

    ``_base_configs``
        List of paths loaded **before** the current file.  The current
        file's values override base values.

    ``_overwrite_configs``
        List of paths loaded **after** the current file.  Their values
        override the current file's values.

    Priority (lowest → highest):
        _base_configs  <  current file  <  _overwrite_configs  <  CLI

    Paths are resolved relative to the directory containing the JSON file.
    Both keys are removed from the returned dict.  Circular references
    raise ``ValueError``.

    When ``track_provenance=True`` returns a tuple ``(merged, provenance)``
    where ``provenance[key]`` is the absolute path of the file that
    supplied that key's final value.
    """
    if _visited is None:
        _visited = set()

    real_path = os.path.realpath(json_path)
    if real_path in _visited:
        raise ValueError(f"Circular config reference detected: {json_path}")
    _visited.add(real_path)

    with open(json_path, "r") as f:
        config = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_configs_list = config.pop("_base_configs", None)
    overwrite_configs_list = config.pop("_overwrite_configs", None)

    merged: dict = {}
    provenance: dict = {}

    if base_configs_list:
        for bp in _resolve_config_paths(base_configs_list, base_dir):
            sub_merged, sub_prov = load_json_config(
                bp, _visited=_visited, track_provenance=True,
            )
            merged.update(sub_merged)
            provenance.update(sub_prov)

    merged.update(config)
    for k in config:
        provenance[k] = real_path

    if overwrite_configs_list:
        for op in _resolve_config_paths(overwrite_configs_list, base_dir):
            sub_merged, sub_prov = load_json_config(
                op, _visited=_visited, track_provenance=True,
            )
            merged.update(sub_merged)
            provenance.update(sub_prov)

    if track_provenance:
        return merged, provenance
    return merged


def convert_to_namespace(config_dict: dict) -> Namespace:
    """Recursively convert a dictionary to an argparse Namespace."""
    for key, value in config_dict.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)


def get_num_named_weights(model: nn.Module) -> int:
    """Retrieve total number of model weights associated with named parameters."""
    num = np.sum([v.numel() for k, v in model.named_parameters()])
    return num


def load_state_dict_with_correction_attempt(
        model,
        state_dict, 
        substitutions={},
        verbosity=2,
):
    """Load a state_dict and attempt to infer missing keys."""
        
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.debug("Encountered error loading state_dict: %s", e)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            logger.warning("Missing keys in checkpoint: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected_keys)
        for k in unexpected_keys:
            new_k = k
            for s0, s1 in substitutions.items():
                if s0 in k:
                    new_k = new_k.replace(s0, s1)
            if new_k in missing_keys:
                state_dict[new_k] = state_dict.pop(k)
                logger.info("Replaced state_dict key `%s` with `%s`", k, new_k)
        # Reload model with updated parameter names
        model.load_state_dict(
            state_dict, strict=True
        )


def compare_model_params(
        model1: nn.Module,
        model2: nn.Module,
        verbosity=1,
) -> dict:
    num_weights1 = get_num_named_weights(model1)
    num_weights2 = get_num_named_weights(model2)
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    set1 = set(params1.keys())
    set2 = set(params2.keys())
    intersection = set1.intersection(set2)
    set1_not_set2 = set1 - set2
    set2_not_set1 = set2 - set1
    max_differences = {}
    with torch.no_grad():
        for name in sorted(list(intersection)):
            vals1 = params1[name]
            vals2 = params2[name]
            logger.debug("%s %s", type(vals1), type(vals2))
            if vals1.shape == vals2.shape:
                max_differences[name] = torch.max(torch.abs(vals1 - vals2))
            else:                
                max_differences[name] = np.inf # indicate mismatch in shape
    return {
        "num_weights1": num_weights1,
        "num_weights2": num_weights2,
        "model1_only": sorted(list(set1_not_set2)),
        "model2_only": sorted(list(set2_not_set1)),
        "common_names": sorted(list(intersection)),
        "differences": max_differences,
        "diff_shape_key": np.inf,
    }


def coerce_limit_batches(value):
    """Coerce a --limit_*_batches value into the form PyTorch Lightning expects.

    PL Trainer accepts int (>=1, absolute batch count) or float (in (0,1],
    fraction of the loader). Floats >1.0 are rejected. Argparse stores the
    flag as float (so JSON ints become floats too); this helper restores the
    int when the value is meant as an absolute count.

    Returns None unchanged. Otherwise: int(value) if value > 1 else float(value).
    """
    if value is None:
        return None
    return int(value) if value > 1 else float(value)


def optional_positive_int(value, name="value"):
    """Coerce a positive-int setting that may be unset, from argparse or JSON.

    ``None``, ``''`` and ``'None'`` (any case) mean unset and return ``None``,
    so the caller can fall back to its next source. Otherwise accepts an int,
    an integral float, or an integer string. Usable directly as an argparse
    ``type=``.

    Raises ValueError for any other value, or one below 1.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("", "none"):
            return None
    invalid = ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise invalid
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise invalid from None
    if number < 1:
        raise invalid
    return number
