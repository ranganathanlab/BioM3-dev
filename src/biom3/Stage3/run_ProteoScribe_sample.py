"""BioM3 Stage 3: ProteoScribe sampling

Mimics the workflow described at
    https://huggingface.co/niksapraljak1/BioM3#stage-3-proteoscribe

Config file:
    configs/inference/stage3_ProteoScribe_sample.json  (uses _base_configs composition)

Example usage:

biom3_ProteoScribe_sample \
    --input_path "outputs/facilitator_embeddings.pt" \
    --config_path "configs/inference/stage3_ProteoScribe_sample.json" \
    --model_path "./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin" \
    --output_path "outputs/generated_sequences.pt"

Example usage (reproducible run with fixed seed, CPU):

biom3_ProteoScribe_sample \
    --input_path "outputs/facilitator_embeddings.pt" \
    --config_path "configs/inference/stage3_ProteoScribe_sample.json" \
    --model_path "./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin" \
    --output_path "outputs/generated_sequences.pt" \
    --seed 42 \
    --device cpu

"""

import copy
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
import random
import warnings
import numpy as np
import pandas as pd
import argparse
import tqdm as tqdm

import torch
import torch.nn as nn

import biom3.Stage3.sampling_analysis as Stage3_sample_tools
import biom3.Stage3.animation_tools as Stage3_ani_tools
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.core.helpers import load_json_config, convert_to_namespace, optional_positive_int
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)
from biom3.backend.device import setup_logger, get_backend_name, set_float32_matmul_precision
from biom3.backend.device import DEVICE_CHOICES, resolve_device
from biom3.core.distributed import (
    barrier,
    broadcast_int,
    gather_object_to_main,
    init_distributed_if_launched,
    is_main_process,
)
from biom3.Stage3.inpaint import (
    RUNTIME_TOKENS,
    build_template_state,
    build_sampling_path_row,
    load_inpaint_config,
    resolve_template,
)

logger = setup_logger(__name__)

DEFAULT_NUM_REPLICAS = 5


# Step 0: Argument Parser Function
def parse_arguments(args):
    parser = argparse.ArgumentParser(description="BioM3 Inference Script (Stage 1)")
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help="Path to input embeddings")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="Path to the JSON configuration file (stage3_config_ProteoScribe_sample.json)")
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help="Path to the pre-trained model weights (pytorch_model.bin)")
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="Path to save output embeddings")
    parser.add_argument('--seed', type=int, default=0,
                        help="seed for random number generation")
    parser.add_argument('--device', type=str, default="auto",
                        choices=list(DEVICE_CHOICES),
                        help="available device; auto = the detected backend (CUDA, XPU, else CPU)")
    parser.add_argument('--unmasking_order', type=str, default=None,
                        choices=["random", "confidence", "confidence_no_pad"],
                        help="Position unmasking order: 'random' (default), "
                             "'confidence' (most-confident first), or "
                             "'confidence_no_pad' (most-confident first, "
                             "skipping positions predicted as PAD)")
    parser.add_argument('--token_strategy', type=str, default=None,
                        choices=["sample", "argmax"],
                        help="Token selection: 'sample' (Gumbel-max, default) or 'argmax' (deterministic)")
    parser.add_argument('--num_replicas', type=optional_positive_int, default=None,
                        metavar='N',
                        help="Sequences to generate per prompt. Overrides the "
                             "config's num_replicas; when neither sets it, "
                             f"defaults to {DEFAULT_NUM_REPLICAS}. 'None' means unset.")
    parser.add_argument('--animate_prompts', type=str, nargs='+', default=None,
                        metavar='IDX',
                        help="Prompt indices to animate (e.g. 0 1 2), 'all', or 'none'. "
                             "Animation is disabled by default (omit this flag).")
    parser.add_argument('--animate_replicas', type=str, default='1',
                        metavar='N',
                        help="Replicas to animate: integer i animates range(0, i), "
                             "'all' animates every replica, 'none' disables. Default: 1.")
    parser.add_argument('--animation_dir', type=str, default=None,
                        help="Output directory for GIF animations. "
                             "Default: <output_dir>/animations/")
    parser.add_argument('--animation_style', type=str, default='brightness',
                        choices=['brightness', 'colorbar', 'logo', 'gauge'],
                        help="Probability distribution visualization: "
                             "'brightness' (dim/vivid cells), "
                             "'colorbar' (stacked AA bars above cells), "
                             "'logo' (stacked AA bars with letters), "
                             "'gauge' (bottom-up fill meter showing confidence). "
                             "Requires --store_probabilities for colorbar/logo/gauge.")
    parser.add_argument('--animation_metrics', type=str, nargs='*', default=None,
                        metavar='NAME',
                        help="Per-position metric boxes to add to the animation. "
                             "Currently supported: 'confidence' (derived from "
                             "--store_probabilities). Multiple metrics are stacked.")
    parser.add_argument('--save_animation_frames', action='store_true', default=False,
                        help="Also save the per-step token trajectory for each "
                             "animated (prompt, replica) as a JSON record "
                             "(token vocab + frames[step][position] token "
                             "indices, index 0 == '-' being the still-masked "
                             "state), next to the GIF in --animation_dir. "
                             "The record also carries, per amino-acid position, "
                             "the model's probability for that residue from the "
                             "step it was placed to the last step. Recorded for "
                             "the animated pairs only, so runs without this flag "
                             "do no extra work. Lightweight and faithful to the "
                             "realized path; intended for downstream interactive "
                             "rendering.")
    parser.add_argument('--no_gif', action='store_true', default=False,
                        help="Skip rendering GIF animations. Useful with "
                             "--save_animation_frames when only the trajectory "
                             "data is wanted and the (slow, large) GIF is not.")
    parser.add_argument('--store_probabilities', action='store_true', default=False,
                        help="Store per-step conditional probabilities for each "
                             "(prompt, replica) pair as .npz files. "
                             "Shapes: probs[steps, seq_len, num_classes]. "
                             "Memory-intensive for long sequences or many replicas.")
    parser.add_argument('--fasta', action='store_true', default=False,
                        help="Write one FASTA file per prompt to <output_dir>/fasta/")
    parser.add_argument('--fasta_merge', action='store_true', default=False,
                        help="Also write a single merged FASTA with all sequences "
                             "(requires --fasta)")
    parser.add_argument('--fasta_dir', type=str, default=None,
                        help="Output directory for FASTA files. "
                             "Default: <output_dir>/fasta/")
    parser.add_argument('--pre_unmask', action='store_true', default=False,
                        help="Start diffusion from a partially-unmasked state. "
                             "Requires --pre_unmask_config.")
    parser.add_argument('--pre_unmask_config', type=str, default=None,
                        help="Path to JSON config describing the pre-unmask "
                             "strategy (strategy, fill_with, diffusion_budget). "
                             "Required when --pre_unmask is set.")
    parser.add_argument('--inpaint', action='store_true', default=False,
                        help="Start diffusion from a partially-filled template "
                             "(in-painting); fixed residues are frozen and only "
                             "mask ('-') positions are generated. Requires "
                             "--inpaint_config. Mutually exclusive with --pre_unmask.")
    parser.add_argument('--inpaint_config', type=str, default=None,
                        help="Path to JSON config describing the in-painting "
                             "template(s): keys 'template', 'per_prompt', "
                             "'auto_add_start', 'auto_add_stop'. Required when "
                             "--inpaint is set.")
    parser.add_argument('--float32_matmul_precision', type=str, default=None,
                        choices=["highest", "high", "medium"],
                        help="fp32 matmul precision. 'high' (config default) enables "
                             "TF32 tensor cores; 'highest' forces full fp32 for bitwise "
                             "reproducibility. Overrides the config value when set.")
    parser.add_argument('--alpha', type=float, default=0.0,
                        help="condition generation on a blend of the text and "
                             "sequence embeddings: y = alpha * z_p + (1 - alpha) "
                             "* z_c. 0.0 = text only (default, the established "
                             "behaviour); 1.0 = sequence only. Requires z_p in "
                             "the embedding file when > 0.")
    return parser.parse_args(args)


# Step 3: load model with pretrained weights
def prepare_model(args, config_args) ->nn.Module:
    """
    Prepare the model and PyTorch Lightning Trainer using a flat args object.

    Supports raw state dicts (.bin, .pth, .pt), Lightning checkpoints (.ckpt),
    and sharded DeepSpeed checkpoint directories. Format is auto-detected.
    """
    model = prepare_model_ProteoScribe(
        config_args=config_args,
        model_fpath=args.model_path,
        device=config_args.device,
        strict=True,
        eval=True,
        attempt_correction=True,
    )
    logger.info("Stage 3 model loaded from: %s (loaded on %s)", args.model_path, config_args.device)
    return model


def parse_animate_prompts(values):
    """Parse --animate_prompts raw values. Returns None, 'all', or a list of ints."""
    if values is None:
        return None
    if len(values) == 1 and values[0].lower() == 'none':
        return None
    if len(values) == 1 and values[0].lower() == 'all':
        return 'all'
    return [int(v) for v in values]


def parse_animate_replicas(value):
    """Parse --animate_replicas raw value. Returns None, 'all', or an int."""
    if value is None or value.lower() == 'none':
        return None
    if value.lower() == 'all':
        return 'all'
    return int(value)


def resolve_animate_prompts(parsed, num_prompts):
    """Resolve parsed prompt spec to a set of indices, or None if animation is off.

    Out-of-range indices are dropped with a warning rather than raised, so a
    caller asking to animate "the first N" without knowing the run's exact
    prompt count (as a dispatcher expanding a count into 0..N-1 does) caps to
    what exists instead of failing the job at Stage 3. When nothing valid
    remains, animation is off. Mirrors resolve_animate_replicas' clamping.
    """
    if parsed is None:
        return None
    if parsed == 'all':
        return set(range(num_prompts))
    valid = {i for i in parsed if 0 <= i < num_prompts}
    dropped = [i for i in parsed if not (0 <= i < num_prompts)]
    if dropped:
        logger.warning(
            "--animate_prompts indices out of range dropped: %s (valid range 0–%d)",
            dropped, num_prompts - 1,
        )
    return valid or None


def resolve_animate_replicas(parsed, num_replicas):
    """Resolve parsed replica spec to a set of indices, or None if animation is off."""
    if parsed is None:
        return None
    if parsed == 'all':
        return set(range(num_replicas))
    if parsed > num_replicas:
        logger.warning(
            "--animate_replicas=%d exceeds num_replicas=%d; clamping to %d",
            parsed, num_replicas, num_replicas,
        )
        parsed = num_replicas
    return set(range(parsed))


# Vocabulary entries that are not amino acids: the still-masked state and the
# structural tokens. Positions that end on one of these get no confidence
# series — there is no residue to be confident about, and the PAD tail would
# otherwise dominate the record.
_NON_RESIDUE_TOKENS = frozenset(('-', '<START>', '<END>', '<PAD>'))


def _confidence_series(final_frame, tokens, token_probs):
    """One entry per position: the probability trace of the residue placed there.

    Each entry runs from the step the position was unmasked to the last step,
    so its first value is the probability the residue was drawn with and the
    rest are the model's later readings of a residue it can now see (see
    ``TokenProbRecorder`` — those are out-of-sample). ``None`` for a position
    that does not end on an amino acid, or that was never sampled (an
    in-painting template residue). Rounded to 3 decimals, which is finer than
    the bf16 forward pass resolves.
    """
    series = []
    for pos, placed in enumerate(token_probs.placed_at.tolist()):
        token = tokens[int(final_frame[pos])]
        if placed < 0 or token in _NON_RESIDUE_TOKENS:
            series.append(None)
            continue
        series.append([round(float(v), 3) for v in token_probs.values[placed:, pos]])
    return series


def save_animation_frames(animation_frames, tokens, animation_dir, confidence=None):
    """Write one JSON trajectory record per animated (prompt, replica) pair.

    Each ``prompt_{p}_replica_{r}.json`` carries the token vocabulary and
    ``frames[step][position]`` token indices (index 0 == '-', the still-masked
    state that resolves to a residue as the step count rises). The frames are
    the realized sampled path, faithful in a way an argmax over stored
    probabilities would not be.

    ``confidence`` maps a pair to its ``TokenProbRow``, recorded only when the
    caller asked for it. A pair that has one also gets a ``confidence`` field:
    one entry per position, as described in ``_confidence_series``. Returns the
    paths written, in iteration order.
    """
    os.makedirs(animation_dir, exist_ok=True)
    written = []
    for (p_idx, r_idx), frames in animation_frames.items():
        record = {
            "prompt_index": int(p_idx),
            "replica_index": int(r_idx),
            "tokens": list(tokens),
            "frames": [np.asarray(f).astype(int).tolist() for f in frames],
        }
        token_probs = (confidence or {}).get((p_idx, r_idx))
        if token_probs is not None:
            record["confidence"] = _confidence_series(
                np.asarray(frames[-1]).astype(int), tokens, token_probs,
            )
        json_path = os.path.join(
            animation_dir, f"prompt_{p_idx}_replica_{r_idx}.json")
        with open(json_path, "w") as fh:
            json.dump(record, fh, separators=(",", ":"))
        logger.info("Animation frames saved: %s (%d steps)", json_path, len(frames))
        written.append(json_path)
    return written


_PRE_UNMASK_SUPPORTED_STRATEGIES = ("last_k",)
_PRE_UNMASK_FILL_ALIASES = {
    "PAD": "<PAD>",
    "pad": "<PAD>",
    "<PAD>": "<PAD>",
}


def load_pre_unmask_config(config_path: str) -> dict:
    """Load and validate a pre-unmask config JSON file.

    Required keys: ``strategy``, ``fill_with``, ``diffusion_budget``.
    Unknown keys raise; unsupported enum values raise.
    """
    if config_path is None:
        raise ValueError("--pre_unmask requires --pre_unmask_config")
    with open(config_path) as fh:
        cfg = json.load(fh)
    required = {"strategy", "fill_with", "diffusion_budget"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"pre_unmask_config is missing keys: {sorted(missing)}")
    unknown = set(cfg) - required
    if unknown:
        raise ValueError(f"pre_unmask_config has unknown keys: {sorted(unknown)}")
    if cfg["strategy"] not in _PRE_UNMASK_SUPPORTED_STRATEGIES:
        raise ValueError(
            f"pre_unmask strategy {cfg['strategy']!r} not supported; "
            f"valid: {_PRE_UNMASK_SUPPORTED_STRATEGIES}"
        )
    if not isinstance(cfg["diffusion_budget"], int) or cfg["diffusion_budget"] <= 0:
        raise ValueError(
            f"pre_unmask diffusion_budget must be a positive int, got {cfg['diffusion_budget']!r}"
        )
    return cfg


def _resolve_fill_token_id(fill_with: str, tokens: list) -> int:
    """Map a ``fill_with`` alias (e.g. 'PAD') to its token id in ``tokens``."""
    canonical = _PRE_UNMASK_FILL_ALIASES.get(fill_with)
    if canonical is None:
        raise ValueError(
            f"pre_unmask fill_with={fill_with!r} not supported; "
            f"valid aliases: {sorted(_PRE_UNMASK_FILL_ALIASES)}"
        )
    if canonical not in tokens:
        raise ValueError(
            f"Token {canonical!r} not present in vocabulary; cannot resolve fill id"
        )
    return tokens.index(canonical)


def _derive_sample_seed(base_seed: int, prompt_idx: int, replica_idx: int) -> int:
    """Stable seed for the (prompt_idx, replica_idx) pair.

    Uses ``numpy.random.SeedSequence`` so the same triple yields the same
    output across processes / Python interpreters / world sizes. Plain
    ``hash()`` is unsuitable here because it is salted by PYTHONHASHSEED.
    """
    ss = np.random.SeedSequence(entropy=[int(base_seed), int(prompt_idx), int(replica_idx)])
    return int(ss.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF


def _make_sample_seeds(batch, base_seed: int) -> list:
    """One stable seed per ``(global_idx, prompt_idx, replica_idx)`` triple."""
    return [_derive_sample_seed(base_seed, p_idx, r_idx) for (_, p_idx, r_idx) in batch]


def _pre_revealed_offset(args, diffusion_steps: int) -> int:
    """Number of positions revealed before diffusion starts.

    For default sampling this is 0 (``sequence_length == diffusion_steps``).
    For pre_unmask the PAD tail and for in-painting the frozen template
    residues occupy ``sequence_length - diffusion_steps`` slots that the model
    must count as already-revealed when forming its time index. The diffusion
    clock therefore starts at this offset, not 0.
    """
    return getattr(args, 'sequence_length', diffusion_steps) - diffusion_steps


def _build_initial_mask_state(args, batch_size: int, tokens: list, sample_seeds: list = None):
    """Construct the starting tensor and sampling path for a generation batch.

    Returns ``(extract_digit_samples, sampling_path)`` where:

    - ``extract_digit_samples`` has shape ``(batch_size, sequence_length)`` and
      holds the initial mask state (0 = masked, non-zero = already filled).
    - ``sampling_path`` is a per-batch permutation of ``range(diffusion_steps)``
      that determines the order in which masked positions are unmasked.

    When ``args.pre_unmask`` is False (default), returns an all-zero tensor of
    shape ``(batch_size, args.diffusion_steps)`` — matching the original
    behaviour where ``sequence_length == diffusion_steps``.

    When ``args.pre_unmask`` is True, positions ``[0, D)`` are left masked
    (value 0) and positions ``[D, sequence_length)`` are filled with the
    resolved ``fill_with`` token id. Only the first ``D`` positions are
    unmasked during diffusion, and the sampling-path values are shifted up by
    ``offset = sequence_length - D`` so the model's time index (which equals
    the path step) reflects the PAD tail as already revealed.
    """
    diffusion_steps = args.diffusion_steps
    sequence_length = getattr(args, 'sequence_length', diffusion_steps)

    def _perm(i):
        if sample_seeds is None:
            return torch.randperm(diffusion_steps)
        gen = torch.Generator().manual_seed(int(sample_seeds[i]))
        return torch.randperm(diffusion_steps, generator=gen)

    if not getattr(args, 'pre_unmask', False):
        init = torch.zeros(batch_size, sequence_length, dtype=torch.long)
        sampling_path = torch.stack([_perm(i) for i in range(batch_size)])
        return init, sampling_path

    if diffusion_steps > sequence_length:
        raise ValueError(
            f"pre_unmask diffusion_budget ({diffusion_steps}) must be "
            f"<= sequence_length ({sequence_length})"
        )
    fill_with = getattr(args, 'pre_unmask_fill_with', 'PAD')
    fill_id = _resolve_fill_token_id(fill_with, tokens)
    init = torch.full((batch_size, sequence_length), fill_id, dtype=torch.long)
    init[:, :diffusion_steps] = 0
    # The PAD tail occupies the first (sequence_length - D) slots of the
    # revealed timeline, so the content positions diffuse over path-values
    # [offset, offset + D). Paired with extract_time = offset in the sampler,
    # this keeps the model's time index equal to the true revealed count.
    offset = sequence_length - diffusion_steps
    sampling_path = torch.stack([_perm(i) for i in range(batch_size)]) + offset
    return init, sampling_path


def _resolve_inpaint_args(config_args, config_args_parser):
    """Resolve in-painting CLI/config onto ``config_args``.

    Sets ``config_args.inpaint`` and (when enabled) ``config_args.inpaint_config``.
    Enforces that ``--inpaint`` and ``--pre_unmask`` are mutually exclusive.
    Returns ``config_args`` for convenience.
    """
    config_args.inpaint = bool(getattr(config_args_parser, 'inpaint', False))
    if config_args.inpaint:
        if getattr(config_args, 'pre_unmask', False):
            raise ValueError("--inpaint and --pre_unmask are mutually exclusive")
        config_args.inpaint_config = load_inpaint_config(
            getattr(config_args_parser, 'inpaint_config', None)
        )
        logger.info(
            "In-painting enabled: shared_template=%s per_prompt=%d "
            "auto_add_start=%s auto_add_stop=%s seq_len=%d",
            config_args.inpaint_config.get('template') is not None,
            len(config_args.inpaint_config.get('per_prompt') or {}),
            config_args.inpaint_config['auto_add_start'],
            config_args.inpaint_config['auto_add_stop'],
            getattr(config_args, 'sequence_length', None),
        )
    elif getattr(config_args_parser, 'inpaint_config', None):
        logger.warning("--inpaint_config ignored because --inpaint is not set")
    return config_args


def _generate_inpaint(args, tokens, rank_work_items, base_seed,
                      rank_local_sequences, process_batch):
    """Drive in-painting generation for this rank's ``(global_idx, p, r)`` slice.

    Each prompt's template is parsed into a frozen initial state plus the set of
    mask positions to generate. Work items are bucketed by mask count ``D`` so
    every batch shares a uniform diffusion budget (the samplers run a single
    scalar ``args.diffusion_steps`` budget). Per-item permutations and token
    sampling are driven by the same world-size-invariant
    ``_derive_sample_seed(base_seed, p, r)`` used by the default path, so
    in-painting outputs depend only on (seed, indices) — not world size or
    batch packing. Templates with no masks are emitted verbatim.
    """
    seq_len = getattr(args, 'sequence_length', args.diffusion_steps)
    cfg = args.inpaint_config

    # Templates are per-prompt (identical across replicas); cache by prompt.
    state_cache = {}
    buckets = defaultdict(list)  # D -> list of (global_idx, p, r, state_row, path_row, seed)
    for (i, p_idx, r_idx) in rank_work_items:
        if p_idx not in state_cache:
            template = resolve_template(p_idx, cfg)
            state_cache[p_idx] = build_template_state(
                template, seq_len,
                auto_add_start=cfg.get('auto_add_start', True),
                auto_add_stop=cfg.get('auto_add_stop', True),
                tokens=tokens,
                mask_symbol=cfg.get('mask_symbol', '*'),
                pad_symbol=cfg.get('pad_symbol', '-'),
            )
        state_row, mask_positions = state_cache[p_idx]
        D = int(mask_positions.numel())
        seed = _derive_sample_seed(base_seed, p_idx, r_idx)
        gen = torch.Generator().manual_seed(int(seed))
        path_row = build_sampling_path_row(mask_positions, seq_len, generator=gen)
        buckets[D].append((i, p_idx, r_idx, state_row, path_row, seed))

    # Fully-specified templates (no masks): emit as-is, no diffusion.
    if 0 in buckets:
        zero_items = buckets.pop(0)
        logger.warning(
            "In-painting: %d (prompt, replica) item(s) have no mask positions; "
            "emitting the template unchanged.", len(zero_items),
        )
        for (_i, p_idx, r_idx, state_row, _path, _seed) in zero_items:
            design_sequence = Stage3_ani_tools.convert_num_to_char(tokens, state_row)
            clean_sequence = design_sequence.replace('<START>', '').replace('<END>', '').replace('<PAD>', '')
            rank_local_sequences[(p_idx, r_idx)] = clean_sequence

    total = sum(len(v) for v in buckets.values())
    logger.info(
        "In-painting: %d rank-local item(s) across %d mask-count bucket(s) %s | seq_len=%d",
        total, len(buckets), {d: len(v) for d, v in sorted(buckets.items())}, seq_len,
    )

    bs = args.batch_size_sample
    for D in sorted(buckets):
        items = buckets[D]
        for start in tqdm.trange(0, len(items), bs, desc=f"inpaint D={D}"):
            chunk = items[start:start + bs]
            batch = [(i, p_idx, r_idx) for (i, p_idx, r_idx, _s, _p, _sd) in chunk]
            initial_samples = torch.stack([s for (_i, _p, _r, s, _pp, _sd) in chunk])
            batch_perms = torch.stack([pp for (_i, _p, _r, _s, pp, _sd) in chunk])
            sample_seeds = [sd for (_i, _p, _r, _s, _pp, sd) in chunk]
            process_batch(batch, initial_samples, batch_perms, sample_seeds, D)


# Step 4: Sample sequences from the model
@torch.no_grad()
def batch_stage3_generate_sequences(
        args: any,
        model: nn.Module,
        z_t: torch.Tensor,
        animate_prompts: set = None,
        animate_replicas: set = None,
        store_probabilities: bool = False,
        record_confidence: bool = False,
    ) -> tuple:
    """Generate protein sequences in batches using a denoising model.

    All ``(prompt_idx, replica_idx)`` pairs are flattened into a single work
    list and processed in batches of ``args.batch_size_sample`` so that GPU
    capacity is shared across both prompts and replicas.

    Args:
        args: Configuration object containing model and sampling parameters
            (``num_replicas``, ``batch_size_sample``, ``diffusion_steps``,
            ``unmasking_order``, ``device``, ...).
        model: Pre-trained denoising model.
        z_t: Conditioning embeddings of shape ``[num_prompts, dim]``.  May
            also be passed as a list of tensors, in which case they are
            stacked.
        animate_prompts: Set of prompt indices to capture per-step frames
            for.  When combined with ``animate_replicas``, the intersection
            determines which (prompt, replica) pairs are recorded for GIF
            rendering.  ``None`` disables animation capture.
        animate_replicas: Set of replica indices to capture per-step frames
            for.  See ``animate_prompts``.
        store_probabilities: When True, the per-step conditional
            distributions and final frames are captured for every
            (prompt, replica) pair and returned in ``results``.
        record_confidence: When True, the animated pairs — and only those —
            also record, at every step, the model's probability for the token
            then at each position (see ``TokenProbRecorder``). Off by default,
            so an ordinary run does no extra work.

    Returns:
        dict: rank-local results with keys

          * ``rank_local_sequences``: ``{(p_idx, r_idx): str}`` cleaned
            sequence for every (prompt, replica) pair this rank
            generated. Sparse — full assembly happens in ``main()`` via
            ``_merge_shards`` after the all-ranks gather.
          * ``num_prompts``: number of prompts in the input embedding tensor.
          * ``animation_frames``: ``{(p_idx, r_idx): [per-step token arrays]}``
            for the (prompt, replica) pairs selected via
            ``animate_prompts`` × ``animate_replicas``.  Empty when
            animation is disabled. Stays rank-local — every rank writes
            its own GIFs into the shared output dir (filenames embed
            global ``(p_idx, r_idx)`` so collisions are impossible).
          * ``animation_confidence``: ``{(p_idx, r_idx): TokenProbRow}`` for the
            animated pairs when ``record_confidence=True``. Empty otherwise.
            Rank-local, like ``animation_frames``.
          * ``tokens``: token vocabulary list (index → string).
          * ``stored_probs``: ``{(p_idx, r_idx): np.ndarray [steps, seq_len, num_classes]}``
            of per-step conditional distributions, populated when
            ``store_probabilities=True``.  Empty otherwise. Rank-local.
          * ``stored_final_frames``: ``{(p_idx, r_idx): np.ndarray [seq_len]}``
            of final-step token indices, populated alongside
            ``stored_probs``.  Rank-local.
    """

    # Handle z_t if passed as a list of tensors
    if isinstance(z_t, list) and all(isinstance(item, torch.Tensor) for item in z_t):
        logger.info("z_t is a list of tensors with %s tensors.", len(z_t))
        z_t = torch.stack(z_t)

    # Move model and inputs to the target device (CPU or CUDA)
    model.to(args.device)
    z_t = z_t.to(args.device)

    # Amino acid tokenization including special characters (canonical runtime
    # vocab, shared with the in-painting template parser so token ids agree).
    tokens = list(RUNTIME_TOKENS)

    if '<PAD>' not in tokens:
        raise ValueError("Token vocabulary is missing '<PAD>' — cannot resolve pad_token_id")
    args.pad_token_id = tokens.index('<PAD>')

    # Build flat work list of (prompt_idx, replica_idx) pairs to parallelize
    # across both prompts and replicates. Under torch.distributed each rank
    # then processes only `i % world_size == rank` of the global list, so
    # the global indexing is preserved for per-sample seed derivation.
    rank = int(getattr(args, "_rank", 0))
    world_size = int(getattr(args, "_world_size", 1))
    base_seed = int(getattr(args, "_base_seed", 0))

    num_prompts = z_t.size(0)
    work_items = [
        (p_idx, r_idx)
        for p_idx in range(num_prompts)
        for r_idx in range(args.num_replicas)
    ]
    rank_work_items = [
        (i, p_idx, r_idx)
        for i, (p_idx, r_idx) in enumerate(work_items)
        if i % world_size == rank
    ]

    # Rank-local sparse output: only the (p_idx, r_idx) pairs this rank generates
    rank_local_sequences = {}  # (p_idx, r_idx) -> str

    animate = animate_prompts is not None and animate_replicas is not None
    animation_frames = {}  # (p_idx, r_idx) -> list of numpy arrays, one per diffusion step
    animation_confidence = {}  # (p_idx, r_idx) -> TokenProbRow, when record_confidence
    stored_probs = {}      # (p_idx, r_idx) -> np.ndarray [steps, seq_len, num_classes]
    stored_final_frames = {}  # (p_idx, r_idx) -> np.ndarray [seq_len], final token indices

    unmasking_order = getattr(args, 'unmasking_order', 'random')

    def _process_batch(batch, initial_samples, batch_perms, sample_seeds, diffusion_steps):
        """Run one uniform-budget batch and scatter results into rank-local outputs.

        ``batch`` is a list of ``(global_idx, p_idx, r_idx)`` triples.
        ``diffusion_steps`` is the number of unmasking steps for this batch
        (full sequence length for default/pre_unmask, or the per-bucket mask
        count for in-painting); it is applied to ``args.diffusion_steps`` for
        the sampler call and restored afterwards.
        """
        prev_steps = args.diffusion_steps
        args.diffusion_steps = diffusion_steps
        try:
            batch_prompt_indices = [p_idx for (_, p_idx, _) in batch]
            batched_z_text_sample = z_t[batch_prompt_indices]

            # Pre-revealed positions (PAD tail for pre_unmask, frozen residues
            # for in-painting) count toward the model's time index, so the
            # diffusion clock starts at `offset`, not 0. The sampling path is
            # shifted by the same offset, keeping path-step == time index ==
            # true revealed count.
            offset = _pre_revealed_offset(args, diffusion_steps)
            extract_time = torch.full((len(batch),), offset, dtype=torch.long)

            # Per-step token probabilities, for the animated rows of this batch
            # only. A batch holding none of them records nothing.
            anim_rows = [
                i for i, (_, p_idx, r_idx) in enumerate(batch)
                if animate and p_idx in animate_prompts and r_idx in animate_replicas
            ] if record_confidence else []
            recorder = (
                Stage3_sample_tools.TokenProbRecorder(rows=anim_rows)
                if anim_rows else None
            )

            if unmasking_order in ('confidence', 'confidence_no_pad'):
                mask_realization_list, _, batch_probs = Stage3_sample_tools.batch_generate_denoised_sampled_confidence(
                    args=args,
                    model=model,
                    extract_digit_samples=initial_samples,
                    extract_time=extract_time,
                    extract_digit_label=batched_z_text_sample,
                    store_probabilities=store_probabilities,
                    skip_pad=(unmasking_order == 'confidence_no_pad'),
                    sample_seeds=sample_seeds,
                    token_prob_recorder=recorder,
                )
            else:
                mask_realization_list, _, batch_probs = Stage3_sample_tools.batch_generate_denoised_sampled(
                    args=args,
                    model=model,
                    extract_digit_samples=initial_samples,
                    extract_time=extract_time,
                    extract_digit_label=batched_z_text_sample,
                    sampling_path=batch_perms,
                    store_probabilities=store_probabilities,
                    sample_seeds=sample_seeds,
                    token_prob_recorder=recorder,
                )

            # Unpack results into rank-local sparse dict
            for i, (_, p_idx, r_idx) in enumerate(batch):
                mask_realization = mask_realization_list[-1][i]
                design_sequence = Stage3_ani_tools.convert_num_to_char(tokens, mask_realization[0])
                clean_sequence = design_sequence.replace('<START>', '').replace('<END>', '').replace('<PAD>', '')
                rank_local_sequences[(p_idx, r_idx)] = clean_sequence

                if animate and p_idx in animate_prompts and r_idx in animate_replicas:
                    animation_frames[(p_idx, r_idx)] = [
                        mask_realization_list[step][i][0].copy()
                        for step in range(diffusion_steps)
                    ]
                    if recorder is not None:
                        animation_confidence[(p_idx, r_idx)] = recorder.row(i)

                if batch_probs is not None:
                    # batch_probs shape: [steps, batch, seq_len, num_classes]
                    stored_probs[(p_idx, r_idx)] = batch_probs[:, i]
                    stored_final_frames[(p_idx, r_idx)] = np.asarray(
                        mask_realization[0], dtype=np.int64,
                    )
        finally:
            args.diffusion_steps = prev_steps

    if getattr(args, 'inpaint', False):
        _generate_inpaint(
            args, tokens, rank_work_items, base_seed,
            rank_local_sequences, _process_batch,
        )
    else:
        # Process this rank's slice in batches of uniform budget.
        num_batches = (len(rank_work_items) + args.batch_size_sample - 1) // args.batch_size_sample
        logger.info(
            "Prompts: %d | Reps/prompt: %d | Total reps: %d | Rank reps: %d | "
            "Batch size: %d | Batches: %d (rank=%d/world=%d)",
            num_prompts, args.num_replicas, num_prompts * args.num_replicas,
            len(rank_work_items), args.batch_size_sample, num_batches, rank, world_size,
        )
        for batch_start in tqdm.trange(0, len(rank_work_items), args.batch_size_sample, desc="batch"):
            batch = rank_work_items[batch_start : batch_start + args.batch_size_sample]
            # Per-sample seeds — world-size-invariant: same (base_seed, p_idx, r_idx)
            # always produces the same seed regardless of how the work is split.
            sample_seeds = _make_sample_seeds(batch, base_seed)
            initial_samples, batch_perms = _build_initial_mask_state(
                args, len(batch), tokens, sample_seeds=sample_seeds,
            )
            _process_batch(batch, initial_samples, batch_perms, sample_seeds, args.diffusion_steps)

    results = {
        "animation_frames": animation_frames,
        "animation_confidence": animation_confidence,
        "tokens": tokens,
        "stored_probs": stored_probs,
        "stored_final_frames": stored_final_frames,
        "rank_local_sequences": rank_local_sequences,
        "num_prompts": num_prompts,
    }
    return results


def _merge_shards(shards, num_prompts: int, num_replicas: int) -> dict:
    """Merge rank-local ``{(p_idx, r_idx): str}`` shards into the public dict.

    Verifies coverage so any silent loss surfaces immediately. Returns
    the same prompt-keyed structure produced by single-rank runs (with
    ``_metadata``) — preserving the existing public output schema.
    """
    merged = {(p, r): None for p in range(num_prompts) for r in range(num_replicas)}
    for shard in shards:
        if shard is None:
            continue
        for key, seq in shard.items():
            if merged.get(key) is not None and merged[key] != seq:
                raise RuntimeError(f"Conflicting shard outputs for {key}")
            merged[key] = seq
    missing = [k for k, v in merged.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing (prompt, replica) pairs after gather: {missing}")

    out = {
        f"prompt_{p}": [merged[(p, r)] for r in range(num_replicas)]
        for p in range(num_prompts)
    }
    out["_metadata"] = {
        "format_version": 2,
        "num_prompts": num_prompts,
        "num_replicas": num_replicas,
    }
    return out


def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.
    
    This function ensures deterministic behavior by setting identical random
    seeds for PyTorch, NumPy, and Python's built-in random module based on
    the seed value specified in the arguments.
    
    Args:
        seed: Random seed
        
    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    return

def blend_conditioning(embedding_dataset, alpha):
    """Blend the text and sequence conditions: y = alpha * z_p + (1-alpha) * z_c.

    z_p comes from Stage 1 (PenCL's protein branch) and z_c from Stage 2, both
    already in the same joint space, so a plain convex combination is valid.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {alpha}")
    z_c = embedding_dataset['z_c']
    if alpha == 0.0:
        return z_c
    if 'z_p' not in embedding_dataset:
        raise KeyError(
            f"--alpha={alpha} needs z_p, which is missing from the embedding "
            f"file (keys: {sorted(embedding_dataset)}). Stage 1 emits z_p and "
            f"Stage 2 preserves it, so this file was most likely built by "
            f"something that dropped it."
        )
    z_p = embedding_dataset['z_p'].to(z_c.device)
    if z_p.shape != z_c.shape:
        raise ValueError(
            f"z_p shape {tuple(z_p.shape)} != z_c shape {tuple(z_c.shape)}; "
            f"they must be row-aligned to blend."
        )
    return alpha * z_p + (1.0 - alpha) * z_c

def main(args, _setup_logging=True):
    args.device = resolve_device(args.device)
    # Parse arguments
    config_args_parser = args

    # ----- Suppress noisy library warnings -----
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")

    # Distributed init — no-op when not launched under mpiexec/torchrun.
    rank, local_rank, world_size, resolved_device = init_distributed_if_launched(args.device)
    if world_size > 1:
        args.device = resolved_device

    # Set up dual logging (console + file). Rank 0 only — non-zero ranks
    # already silence themselves via setup_logger but we don't want them
    # racing on the file handler.
    outdir = os.path.dirname(os.path.abspath(args.output_path))
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
    file_handler = None
    if _setup_logging and rank == 0:
        log_path, file_handler = setup_file_logging(outdir)
    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("ProteoScribe sampling (Stage 3)")
        logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
        logger.info("Command:     %s", " ".join(sys.argv))
        if world_size > 1:
            logger.info("Distributed: rank=%d local_rank=%d world_size=%d device=%s",
                        rank, local_rank, world_size, resolved_device)
        logger.info("=" * 60)

        seed = config_args_parser.seed

        if seed <= 0:
            if rank == 0:
                seed = int(np.random.randint(2**32))
            seed = broadcast_int(seed, src=0)

        set_seed(seed)
        logger.info("Seed: %s", seed)

        # Load and convert JSON config
        config_dict = load_json_config(config_args_parser.config_path)
        raw_config = copy.deepcopy(config_dict)
        config_args = convert_to_namespace(config_dict)

        # fp32 matmul precision (TF32): CLI overrides config, config default is "high".
        set_float32_matmul_precision(
            config_args_parser.float32_matmul_precision
            or getattr(config_args, "float32_matmul_precision", "high")
        )

        config_args.device = config_args_parser.device
        config_args._rank = rank
        config_args._world_size = world_size
        config_args._base_seed = int(seed)

        # Merge CLI overrides for sampling parameters (fall back to JSON, then defaults)
        for attr, default in [('unmasking_order', 'random'), ('token_strategy', 'sample')]:
            cli_val = getattr(config_args_parser, attr, None)
            if cli_val is not None:
                setattr(config_args, attr, cli_val)
            elif not hasattr(config_args, attr):
                setattr(config_args, attr, default)

        cli_num_replicas = optional_positive_int(
            getattr(config_args_parser, 'num_replicas', None), name="--num_replicas")
        config_num_replicas = optional_positive_int(
            getattr(config_args, 'num_replicas', None), name="config num_replicas")
        if cli_num_replicas is not None:
            config_args.num_replicas = cli_num_replicas
        elif config_num_replicas is not None:
            config_args.num_replicas = config_num_replicas
        else:
            config_args.num_replicas = DEFAULT_NUM_REPLICAS

        logger.info("Unmasking order: %s", config_args.unmasking_order)
        logger.info("Token strategy: %s", config_args.token_strategy)
        logger.info("Replicas per prompt: %d", config_args.num_replicas)

        # Pre-unmask feature: snapshot the architectural sequence length (current
        # diffusion_steps value from config == seq_len trained on), then if
        # enabled, override diffusion_steps with the budget D.
        config_args.sequence_length = config_args.diffusion_steps
        config_args.pre_unmask = bool(getattr(config_args_parser, 'pre_unmask', False))
        if config_args.pre_unmask:
            pre_unmask_cfg = load_pre_unmask_config(config_args_parser.pre_unmask_config)
            if pre_unmask_cfg["diffusion_budget"] > config_args.sequence_length:
                raise ValueError(
                    f"pre_unmask diffusion_budget ({pre_unmask_cfg['diffusion_budget']}) "
                    f"must be <= sequence_length ({config_args.sequence_length})"
                )
            config_args.diffusion_steps = pre_unmask_cfg["diffusion_budget"]
            config_args.pre_unmask_strategy = pre_unmask_cfg["strategy"]
            config_args.pre_unmask_fill_with = pre_unmask_cfg["fill_with"]
            logger.info(
                "Pre-unmask enabled: strategy=%s fill_with=%s D=%d seq_len=%d",
                config_args.pre_unmask_strategy,
                config_args.pre_unmask_fill_with,
                config_args.diffusion_steps,
                config_args.sequence_length,
            )
        elif getattr(config_args_parser, 'pre_unmask_config', None):
            logger.warning(
                "--pre_unmask_config ignored because --pre_unmask is not set"
            )

        # In-painting feature: start diffusion from a partially-filled template.
        # The diffusion budget is derived per-prompt from the template's mask count
        # (handled in batch_stage3_generate_sequences), so diffusion_steps is left
        # at the architectural sequence length here.
        _resolve_inpaint_args(config_args, config_args_parser)

        # load test dataset
        embedding_dataset = torch.load(config_args_parser.input_path)

        # load model
        model = prepare_model(args=config_args_parser, config_args=config_args)

        # TODO: re-enable torch.compile once Triton supports this GPU's compute
        # capability (sm_121a / CUDA 12.1). See ptxas "gpu-name" error in tests.
        # if get_backend_name() == "cuda":
        #     model = torch.compile(model)
        #     logger.info("Model compiled with torch.compile (inductor)")

        # Resolve animation targets
        alpha = getattr(config_args_parser, 'alpha', 0.0)
        z_c = blend_conditioning(embedding_dataset, alpha)
        logger.info("Conditioning blend: alpha=%.3f (0 = z_c, 1 = z_p)", alpha)

        num_prompts = z_c.size(0) if isinstance(z_c, torch.Tensor) else len(z_c)
        animate_prompts_set = resolve_animate_prompts(
            parse_animate_prompts(config_args_parser.animate_prompts),
            num_prompts,
        )
        animate_replicas_set = resolve_animate_replicas(
            parse_animate_replicas(config_args_parser.animate_replicas),
            config_args.num_replicas,
        )

        # sample sequences
        store_probs = getattr(config_args_parser, 'store_probabilities', False)
        save_frames = getattr(config_args_parser, 'save_animation_frames', False)
        results = batch_stage3_generate_sequences(
                args=config_args,
                model=model,
                z_t=z_c,
                animate_prompts=animate_prompts_set,
                animate_replicas=animate_replicas_set,
                store_probabilities=store_probs,
                record_confidence=save_frames,
        )
        animation_frames = results["animation_frames"]
        tokens = results["tokens"]
        stored_probs = results["stored_probs"]
        stored_final_frames = results["stored_final_frames"]
        rank_local_sequences = results["rank_local_sequences"]

        # Gather rank-local sequence shards to rank 0 and merge into the
        # public prompt-keyed dict. No-op when world_size == 1.
        if world_size > 1:
            shards = gather_object_to_main(rank_local_sequences)
        else:
            shards = [rank_local_sequences]

        design_sequence_dict = None
        if rank == 0:
            design_sequence_dict = _merge_shards(shards, num_prompts, config_args.num_replicas)
            logger.info("design_sequence_dict=%s", design_sequence_dict)

            torch.save(design_sequence_dict, f"{config_args_parser.output_path}")

        # Write FASTA files (rank 0 only)
        if rank == 0 and getattr(config_args_parser, 'fasta', False):
            fasta_dir = config_args_parser.fasta_dir or os.path.join(outdir, "fasta")
            os.makedirs(fasta_dir, exist_ok=True)
            for prompt_key, replicas in design_sequence_dict.items():
                if prompt_key.startswith("_"):
                    continue
                fasta_path = os.path.join(fasta_dir, f"{prompt_key}.fasta")
                with open(fasta_path, 'w') as fh:
                    for r_idx, seq in enumerate(replicas):
                        fh.write(f">{prompt_key}_replica_{r_idx} seed={seed}\n{seq}\n")
            logger.info("FASTA files written to %s", fasta_dir)
            if getattr(config_args_parser, 'fasta_merge', False):
                merged_path = os.path.join(fasta_dir, "all_sequences.fasta")
                with open(merged_path, 'w') as fh:
                    for prompt_key, replicas in design_sequence_dict.items():
                        if prompt_key.startswith("_"):
                            continue
                        for r_idx, seq in enumerate(replicas):
                            fh.write(f">{prompt_key}_replica_{r_idx} seed={seed}\n{seq}\n")
                logger.info("Merged FASTA written to %s", merged_path)

        # Save / render animations for the selected (prompt, replica) pairs.
        if animation_frames:
            animation_dir = config_args_parser.animation_dir or os.path.join(outdir, "animations")
            os.makedirs(animation_dir, exist_ok=True)

            # Per-step token trajectory, for downstream interactive rendering.
            if getattr(config_args_parser, 'save_animation_frames', False):
                save_animation_frames(
                    animation_frames, tokens, animation_dir,
                    confidence=results["animation_confidence"],
                )

            if not getattr(config_args_parser, 'no_gif', False):
                animation_style = getattr(config_args_parser, 'animation_style', 'brightness')
                requested_metrics = getattr(config_args_parser, 'animation_metrics', None) or []
                if (animation_style != 'brightness' or requested_metrics) and not stored_probs:
                    logger.warning("--animation_style=%s / --animation_metrics requires "
                                   "--store_probabilities; falling back to default animation",
                                   animation_style)
                logger.info("Saving %d GIF animation(s) to %s", len(animation_frames), animation_dir)
                for (p_idx, r_idx), frames in animation_frames.items():
                    gif_path = os.path.join(animation_dir, f"prompt_{p_idx}_replica_{r_idx}.gif")
                    frame_probs = stored_probs.get((p_idx, r_idx)) if stored_probs else None

                    # Build metric annotations for this (prompt, replica)
                    metrics = []
                    if frame_probs is not None and requested_metrics:
                        for name in requested_metrics:
                            if name == "confidence":
                                metrics.append(
                                    Stage3_ani_tools.confidence_metric(frame_probs))
                            else:
                                logger.warning("Unknown animation metric %r, skipping", name)

                    Stage3_ani_tools.generate_sequence_animation(
                        frames=frames,
                        tokens=tokens,
                        output_path=gif_path,
                        probs=frame_probs,
                        prob_style=animation_style,
                        metrics=metrics or None,
                        title=f"Prompt {p_idx} \u00b7 Replica {r_idx}",
                    )
                    logger.info("Animation saved: %s", gif_path)

        # Save per-step conditional probabilities
        if stored_probs:
            probs_dir = os.path.join(outdir, "probabilities")
            os.makedirs(probs_dir, exist_ok=True)
            logger.info("Saving probabilities for %d (prompt, replica) pairs to %s",
                         len(stored_probs), probs_dir)
            for (p_idx, r_idx), probs_array in stored_probs.items():
                # probs_array shape: [steps, seq_len, num_classes]
                npz_path = os.path.join(probs_dir, f"prompt_{p_idx}_replica_{r_idx}.npz")
                np.savez_compressed(
                    npz_path,
                    probs=probs_array,
                    tokens=tokens,
                    final_frame=stored_final_frames[(p_idx, r_idx)],
                )
                logger.info("Probabilities saved: %s (shape=%s)", npz_path, probs_array.shape)

        # Write manifest (rank 0 only)
        elapsed = datetime.now() - start_time
        if _setup_logging and rank == 0:
            write_manifest(
                args, outdir, start_time, elapsed,
                outputs={
                    "num_prompts": num_prompts,
                    "num_replicas": config_args.num_replicas,
                    "seed": seed,
                    "unmasking_order": config_args.unmasking_order,
                    "token_strategy": config_args.token_strategy,
                    "total_sequences": num_prompts * config_args.num_replicas,
                    "output_file": os.path.abspath(args.output_path),
                },
                resolved_paths={
                    "input_path": os.path.abspath(args.input_path),
                    "model_path": os.path.abspath(args.model_path),
                    "json_config": os.path.abspath(args.config_path),
                },
                config_contents=raw_config,
            )
            logger.info("Done in %s", elapsed)
    finally:
        teardown_file_logging("biom3", file_handler)

    # Hold non-rank-0 processes until rank-0 finishes I/O so the launcher
    # doesn't tear down the communicator out from under us.
    if world_size > 1:
        barrier()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)    
