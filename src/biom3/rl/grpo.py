"""GRPO single-GPU trainer for BioM3 Stage 3.

Group Relative Policy Optimization: for each prompt, sample K sequences,
compute group-normalized advantages from a per-sequence reward, and update
the policy with PPO-clip + KL against a frozen reference snapshot.

Ported from biom3-grpo-rl/biom3_grpo.py to use the packaged biom3.* API.
"""

import copy
import json
import os
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import biom3.Stage1.preprocess as S1prep
import biom3.Stage3.animation_tools as S3ani
import biom3.Stage3.sampling_analysis as S3sample
from biom3.backend.device import setup_logger
from biom3.rl.diversity import diversity_stats
from biom3.rl.io import (
    load_facilitator_frozen,
    load_pencl_frozen,
    load_proteoscribe_trainable,
)

logger = setup_logger(__name__)


# Token vocabulary — must match the order baked into Stage 3 weights.
TOKENS = [
    "-", "<START>", "A", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V",
    "W", "Y", "<END>", "<PAD>", "X", "U", "Z", "B", "O",
]
TOK2ID = {t: i for i, t in enumerate(TOKENS)}
PAD_ID = TOK2ID["<PAD>"]
START_ID = TOK2ID["<START>"]
END_ID = TOK2ID["<END>"]
NUM_CLASSES = len(TOKENS)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Mask / absorbing-state token id used by the diffusion model. NOT
# the PAD id. transformer_training_helper.mask_realizations writes 0
# at masked positions; sampling_analysis.batch_generate_denoised_sampled
# initializes the rollout with torch.zeros(...) ⇒ all-mask ⇒ id 0.
# Earlier revisions of this file mistakenly used PAD_ID (=23) as the
# fully-masked input to ``_policy_logprobs``; that has been corrected.
MASK_ID = 0


@dataclass
class GRPOConfig:
    output_dir: str = "./grpo_output"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    beta: float = 0.01           # KL coefficient
    eps: float = 0.20            # PPO clip
    num_generations: int = 4     # K
    batch_size: int = 1          # prompts per gradient step
    steps: int = 200
    save_steps: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42

    # Per-step verbose dump (sequences, prompts, per-replica scalar
    # table) appended to {output_dir}/debug.out. Mirrors the GDPO
    # debug_log option for parity in side-by-side analysis.
    debug_log: bool = True


def load_prompts(path: str) -> List[str]:
    """Read prompts from a text file. Supports comments (#) and `name|text`."""
    prompts: List[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                _, text = line.split("|", 1)
                prompts.append(text.strip())
            else:
                prompts.append(line)
    return prompts


class _PromptEncoder:
    """Wraps Stage 1 + Stage 2 to turn a text prompt into a conditioning
    vector ``z_c`` of shape ``(1, emb_dim)``.

    The padding/length conventions in
    ``Stage1.preprocess.TextSeqPairing_Dataset`` (``padding='max_length'``,
    ``max_length = cfg1.text_max_length``) match Stage 1 training — see
    docs/bug_reports/bert_embedding_mismatch.md for why this matters.

    The caption attention mask is passed to BERT only when
    ``cfg1.text_attention_mask`` is set, for PenCL weights trained with it.
    """

    def __init__(self, s1, s2, cfg1: Namespace, device: torch.device):
        self.s1 = s1
        self.s2 = s2
        self.cfg1 = cfg1
        self.device = device
        self.text_attention_mask = bool(getattr(cfg1, "text_attention_mask", False))
        self._dataset = None

    def _build_dataset(self) -> S1prep.TextSeqPairing_Dataset:
        tmp_df = pd.DataFrame({
            self.cfg1.sequence_keyword: ["A" * 50],
            "[final]text_caption": ["placeholder"],
            self.cfg1.id_keyword: ["GRPO_PROMPT"],
        })
        return S1prep.TextSeqPairing_Dataset(args=self.cfg1, df=tmp_df)

    @torch.no_grad()
    def __call__(self, prompt: str) -> torch.Tensor:
        if self._dataset is None:
            self._dataset = self._build_dataset()
        ds = self._dataset
        tmp_df = pd.DataFrame({
            self.cfg1.sequence_keyword: ["A" * 50],
            "[final]text_caption": [prompt],
            self.cfg1.id_keyword: ["GRPO_PROMPT"],
        })
        ds.df = tmp_df
        ds.text_captions_list = tmp_df["[final]text_caption"].tolist()
        ds.protein_sequence_list = tmp_df[self.cfg1.sequence_keyword].tolist()
        ds.accession_id_list = tmp_df[self.cfg1.id_keyword].tolist()

        x_t, x_p = ds[0]
        x_t = x_t.to(self.device)
        x_p = x_p.to(self.device)
        if self.text_attention_mask:
            x_t_mask = ds.caption_tokenizer(batch_captions=[prompt])["attention_mask"]
            out = self.s1(x_t, x_p, compute_masked_logits=False,
                          x_t_mask=x_t_mask.to(self.device))
        else:
            out = self.s1(x_t, x_p, compute_masked_logits=False)
        z_t = out["text_joint_latent"]
        return self.s2(z_t).to(self.device)


def decode_tokens(tok_tensor: torch.Tensor) -> str:
    """Token ID tensor → clean amino-acid string (special tokens stripped)."""
    raw = S3ani.convert_num_to_char(TOKENS, tok_tensor)
    for special in ("<START>", "<END>", "<PAD>", "<MASK>", "<UNK>", "-"):
        raw = raw.replace(special, "")
    return raw


@torch.no_grad()
def diffusion_rollout(
    s3: torch.nn.Module,
    cfg3: Namespace,
    z_c: torch.Tensor,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``K`` sequences from the absorbing-state diffusion model
    conditioned on ``z_c`` (shape ``(1, emb_dim)``).

    Returns ``(K, L)`` int64 token IDs, where ``L = cfg3.diffusion_steps``.
    """
    L = cfg3.diffusion_steps
    z_rep = z_c.repeat(K, 1)
    perms = torch.stack([torch.randperm(L, device=device) for _ in range(K)])

    mask_list, _, _ = S3sample.batch_generate_denoised_sampled(
        args=cfg3,
        model=s3,
        extract_digit_samples=torch.zeros(K, L, device=device),
        extract_time=torch.zeros(K, dtype=torch.long, device=device),
        extract_digit_label=z_rep,
        sampling_path=perms,
    )
    final = mask_list[-1]   # (K, 1, L)
    out = torch.empty(K, L, dtype=torch.long, device=device)
    for k in range(K):
        out[k] = torch.as_tensor(final[k, 0], dtype=torch.long, device=device)
    return out


def _policy_logprobs(
    s3: torch.nn.Module,
    ids: torch.Tensor,             # (BK, L)
    z_c_rep: torch.Tensor,         # (BK, emb_dim)
    mask_id: int = MASK_ID,
) -> torch.Tensor:
    """Per-token log-probs at the chosen ids under ``s3`` with input fully
    masked at ``t = 0``. Returns shape ``(BK, L)``.

    The fill token must be the diffusion model's MASK / absorbing-state
    token (id 0), NOT PAD (id 23). Earlier revisions of this file
    incorrectly used PAD; the model treats PAD as a real protein-tail
    token, not as "predict me", so the resulting log-probs were off-
    distribution. ``mask_id`` defaults to ``MASK_ID`` for safety; the
    parameter is kept (rather than hardcoded) so callers can opt into a
    different fill in tests.
    """
    BK, L = ids.shape
    x_masked = torch.full_like(ids, mask_id)
    t_steps = torch.zeros(BK, dtype=torch.long, device=ids.device)
    logits = s3(x_masked, t_steps, z_c_rep).float().permute(0, 2, 1)  # (BK, L, V)
    lp = F.log_softmax(logits, dim=-1)
    return lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def _write_debug_step(
    debug_path: str,
    step: int,
    batch_prompts: List[str],
    ids_all: torch.Tensor,           # (BK, L)
    seqs: List[str],                 # decoded, special-stripped
    rewards_raw,                     # list[float] or array, len BK
    adv: torch.Tensor,               # (BK,)
    log_ratio_per_replica: torch.Tensor,  # (BK,) — token-mean log-ratio per sequence
    ratio_per_replica: torch.Tensor,      # (BK,)
    log_ratio_max_per_replica: torch.Tensor,  # (BK,) — token-max |log-ratio|
    valid_lengths: torch.Tensor,     # (BK,) — count of non-PAD tokens
    components_per_replica,          # dict[str, list[float]] | None
    K: int,
    per_replica_diversity: List[float],   # length BK, in [0,1]
) -> None:
    """Append one stanza per training step to GRPO's ``debug.out``.

    GRPO is token-level (no SDMC corruptions) so this is a leaner
    cousin of the GDPO debug dump — no per-corruption mask
    visualization. The all-MASK input fed to ``_policy_logprobs`` is
    constant across steps (token id 0 at every position) so we don't
    log it; what varies are the rolled-out sequences and the per-token
    log-prob ratios, which we summarize per replica.
    """
    BK, L = ids_all.shape
    rewards_list = [float(x) for x in rewards_raw]
    adv_list = adv.detach().cpu().tolist()
    lr_list = log_ratio_per_replica.detach().cpu().tolist()
    r_list = ratio_per_replica.detach().cpu().tolist()
    lr_max_list = log_ratio_max_per_replica.detach().cpu().tolist()
    seqlen_list = [int(x) for x in valid_lengths.detach().cpu().tolist()]

    bar = "=" * 100
    lines: List[str] = []
    lines.append(bar)
    lines.append(f"step={step:>5d}   B*K={BK}   L={L}   algo=GRPO (diffu-GRPO, single-step all-MASK)")
    lines.append(bar)
    lines.append("prompts:")
    for i, p in enumerate(batch_prompts):
        lines.append(f"  [{i}] {p}")
    lines.append("")

    lines.append("per-replica:")
    header = (
        f"  {'k':>3}  {'p':>3}  {'len':>4}  {'reward':>10}  {'advantage':>10}  "
        f"{'diversity':>9}  "
        f"{'log_ratio':>10}  {'ratio':>8}  {'|lr|max':>9}"
    )
    lines.append(header)
    for i in range(BK):
        p_idx = i // K
        k_idx = i % K
        div_i = float(per_replica_diversity[i]) if i < len(per_replica_diversity) else 0.0
        lines.append(
            f"  {k_idx:>3d}  {p_idx:>3d}  {seqlen_list[i]:>4d}  "
            f"{rewards_list[i]:>10.4f}  {adv_list[i]:>+10.4f}  "
            f"{div_i:>9.4f}  "
            f"{lr_list[i]:>+10.4f}  {r_list[i]:>8.4f}  {lr_max_list[i]:>9.4f}"
        )
    if components_per_replica:
        lines.append("")
        lines.append("components per replica:")
        for name, vals in components_per_replica.items():
            vals_s = ", ".join(f"{float(v):.3f}" for v in vals)
            lines.append(f"  {name}: [{vals_s}]")

    lines.append("")
    lines.append("generated sequences (decoded, special tokens stripped):")
    for i in range(BK):
        p_idx = i // K
        k_idx = i % K
        lines.append(f"  [k={k_idx} p={p_idx}] len={len(seqs[i])} {seqs[i]}")

    lines.append("")
    lines.append("raw token-id sequences (full L, before decoding; '_' = MASK):")
    for i in range(BK):
        ids_str = "".join(TOKENS[int(t)] if int(t) != MASK_ID else "_"
                          for t in ids_all[i].tolist())
        lines.append(f"  [k={i % K} p={i // K}] {ids_str}")

    lines.append("")
    with open(debug_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def grpo_train(
    grpo_cfg: GRPOConfig,
    cfg1: Namespace,
    cfg2: Namespace,
    cfg3: Namespace,
    prompts: List[str],
    reward_fn,
    device: torch.device,
    stage1_weights: Optional[str] = None,
    stage2_weights: Optional[str] = None,
    stage3_init_weights: Optional[str] = None,
):
    cfg3.device = str(device)

    torch.manual_seed(grpo_cfg.seed)
    np.random.seed(grpo_cfg.seed)

    logger.info("Loading Stage 1 (PenCL)...")
    s1 = load_pencl_frozen(cfg1, stage1_weights, device=str(device))
    logger.info("Loading Stage 2 (Facilitator)...")
    s2 = load_facilitator_frozen(cfg2, stage2_weights, device=str(device))
    logger.info("Loading Stage 3 (ProteoScribe, trainable)...")
    s3 = load_proteoscribe_trainable(cfg3, stage3_init_weights, device=str(device))
    s3.train()

    logger.info("Snapshotting frozen reference policy...")
    ref_s3 = copy.deepcopy(s3).eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)

    encode_prompt = _PromptEncoder(s1, s2, cfg1, device)

    optimizer = torch.optim.AdamW(
        s3.parameters(),
        lr=grpo_cfg.learning_rate,
        weight_decay=grpo_cfg.weight_decay,
    )

    os.makedirs(grpo_cfg.output_dir, exist_ok=True)
    log_rows = []
    log_path = os.path.join(grpo_cfg.output_dir, "train_log.json")
    debug_path = os.path.join(grpo_cfg.output_dir, "debug.out")
    if grpo_cfg.debug_log and os.path.exists(debug_path):
        # Truncate any previous run's debug log so we don't accumulate
        # across re-runs of the same output_dir.
        open(debug_path, "w").close()
    L = cfg3.diffusion_steps
    eps = grpo_cfg.eps
    beta = grpo_cfg.beta
    B = grpo_cfg.batch_size
    K = grpo_cfg.num_generations
    BK = B * K

    # Diagnostics: snapshot initial trainable params for a cumulative
    # weight-drift norm across all of training. We clone once up-front and
    # again per-step (pre-update) for the per-step delta norm. Cloning
    # ~86M fp32 params is ~340 MB per snapshot — fine on a single tile.
    trainable_params = [p for p in s3.parameters() if p.requires_grad]
    initial_params = [p.detach().clone() for p in trainable_params]
    reward_sum = 0.0

    logger.info(
        "GRPO start: steps=%d K=%d batch=%d beta=%.4f eps=%.2f | %d trainable params",
        grpo_cfg.steps, K, B, beta, eps,
        sum(p.numel() for p in trainable_params),
    )

    # Snapshot the algorithm config as the first row of the log so that
    # downstream analysis can recover what was actually run. Mirrors the
    # GDPO _meta row schema where applicable.
    log_rows.append({
        "_meta": True,
        "algo": "grpo",
        "diffu_grpo_style": True,            # single-step all-MASK token-level estimator
        "num_generations": int(K),
        "batch_size": int(B),
        "beta": float(beta),
        "eps": float(eps),
        "diffusion_steps": int(L),
        "advantage_normalize": True,         # GRPO's classic (R-μ)/σ
    })
    # Flush the meta-only log so that crashes during weight loading or
    # the very first step still leave a parseable train_log.json behind.
    from biom3.rl.plotting import write_train_log_atomic
    write_train_log_atomic(log_path, log_rows)

    for step in range(1, grpo_cfg.steps + 1):
        t0 = time.time()
        # Per-step stage timer. Logs "[t] <stage> <s>" so we can pinpoint
        # hangs between visible log lines (notably the .cpu() sync after
        # the diffusion tqdm finishes, and ESMFold lazy-load on step 1).
        _stamp_t = time.perf_counter()

        def _stamp(label):
            nonlocal _stamp_t
            now = time.perf_counter()
            logger.info("  [t] %-12s %5.2fs", label, now - _stamp_t)
            _stamp_t = now

        batch_prompts = [
            prompts[np.random.randint(len(prompts))] for _ in range(B)
        ]
        logger.info("step=%d batch_prompts:", step)
        for i, p in enumerate(batch_prompts):
            logger.info("  [%d] %s", i, p)

        s3.eval()
        with torch.no_grad():
            z_cs = torch.cat([encode_prompt(p) for p in batch_prompts], dim=0)
            z_cs_rep = z_cs.repeat_interleave(K, dim=0)             # (BK, emb)
            _stamp("encode_z_c")

            ids_per_prompt = [
                diffusion_rollout(s3, cfg3, z_cs[i:i + 1], K, device)
                for i in range(B)
            ]
            ids_all = torch.cat(ids_per_prompt, dim=0)              # (BK, L)
            _stamp("rollout")

            seqs = [decode_tokens(ids_all[i]) for i in range(BK)]
            logger.info("step=%d generated sequences (B*K=%d):", step, BK)
            for i, seq in enumerate(seqs):
                p_idx = i // K
                logger.info(
                    "  [prompt %d / replica %d] len=%d %s",
                    p_idx, i % K, len(seq), seq,
                )
            _stamp("decode")

            # Within-batch sequence diversity. Same diagnostic as GDPO;
            # surfaced unconditionally so mode collapse is visible from
            # train_log.json without instrumentation downstream.
            div_stats = diversity_stats(seqs, group_size=K)
            _stamp("diversity")

            lp_ref_tok = _policy_logprobs(ref_s3, ids_all, z_cs_rep, MASK_ID)
            _stamp("ref_logprobs")

        rewards_raw = reward_fn(seqs)
        _stamp("reward")
        R = torch.tensor(rewards_raw, dtype=torch.float32, device=device)
        Rg = R.view(B, K)
        adv = (
            (Rg - Rg.mean(dim=-1, keepdim=True))
            / (Rg.std(dim=-1, keepdim=True).clamp(min=1e-8))
        ).view(BK)

        s3.train()
        lp_new_tok = _policy_logprobs(s3, ids_all, z_cs_rep, MASK_ID)

        valid = (ids_all != PAD_ID).float()
        log_ratio = lp_new_tok - lp_ref_tok.detach()
        ratio = torch.exp(log_ratio)
        adv_2d = adv.unsqueeze(1).expand_as(valid)
        pg1 = -adv_2d * ratio
        pg2 = -adv_2d * ratio.clamp(1 - eps, 1 + eps)
        pg_loss = (torch.max(pg1, pg2) * valid).sum() / (valid.sum() + 1e-8)

        kl_tok = (
            torch.exp(lp_ref_tok.detach() - lp_new_tok)
            - (lp_ref_tok.detach() - lp_new_tok)
            - 1.0
        )
        kl_loss = (kl_tok * valid).sum() / (valid.sum() + 1e-8)

        loss = pg_loss + beta * kl_loss
        _stamp("new_lp+loss")

        # Snapshot pre-update params for the per-step weight-delta norm.
        pre_step_params = [p.detach().clone() for p in trainable_params]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s3.parameters(), grpo_cfg.max_grad_norm)
        optimizer.step()
        _stamp("backward+opt")

        # Frobenius norm of (θ_after - θ_before) for this step, and the
        # cumulative drift (θ_after - θ_initial). Both are diagnostic only.
        with torch.no_grad():
            dw_step = torch.sqrt(sum(
                ((p - pre) ** 2).sum() for p, pre in zip(trainable_params, pre_step_params)
            )).item()
            dw_total = torch.sqrt(sum(
                ((p - init) ** 2).sum() for p, init in zip(trainable_params, initial_params)
            )).item()
        del pre_step_params
        _stamp("dw_norms")

        dt = time.time() - t0
        clip_frac = ((ratio - 1.0).abs() > eps).float().mean().item()
        mean_reward = R.mean().item()
        reward_sum += mean_reward
        reward_avg = reward_sum / step
        avg_len = float(np.mean([len(s) for s in seqs]))
        num_replicas = len(seqs)
        all_lengths = [len(s) for s in seqs]

        # Token-level log-ratio diagnostics. For GRPO the ratio is per
        # (replica, position); summarize over the valid (non-PAD)
        # positions of each replica for both the trainer log and the
        # debug.out per-replica table.
        with torch.no_grad():
            valid_count = valid.sum(dim=1).clamp(min=1.0)
            log_ratio_per_replica = (log_ratio.detach() * valid).sum(dim=1) / valid_count
            ratio_per_replica = (ratio.detach() * valid).sum(dim=1) / valid_count
            log_ratio_max_per_replica = (log_ratio.detach().abs() * valid).max(dim=1).values
            log_ratio_tok_mean = float(((log_ratio.detach() * valid).sum() / valid.sum().clamp(min=1.0)).item())
            log_ratio_tok_max_abs = float(log_ratio.detach().abs().max().item())
            ratio_tok_mean = float(((ratio.detach() * valid).sum() / valid.sum().clamp(min=1.0)).item())
            valid_lengths = valid.sum(dim=1)

        logger.info(
            "step=%4d | reward=%5.2f (avg=%5.2f) | loss=%.4f pg=%.4f kl=%.4f clip=%.2f "
            "| lr_tok=%.4f (|max|=%.4f) "
            "| div=%.3f (worst_pair=%.3f, uniq=%d) "
            "| dw=%.2e (tot=%.2e) | len=%.2f | %.1fs | replicas=%d | all_lengths=%s",
            step, mean_reward, reward_avg,
            loss.item(), pg_loss.item(), kl_loss.item(), clip_frac,
            log_ratio_tok_mean, log_ratio_tok_max_abs,
            float(div_stats["diversity_mean"]),
            float(div_stats["diversity_min_pair"]),
            int(div_stats["unique_count"]),
            dw_step, dw_total, avg_len, dt, num_replicas, str(all_lengths)
        )
        row = {
            "step": step,
            "reward": round(mean_reward, 3),
            "reward_avg": round(reward_avg, 3),
            # Raw per-replica reward values for downstream plotting
            # (length BK = batch_size * num_generations).
            "rewards_per_replica": [float(x) for x in rewards_raw],
            "loss": round(loss.item(), 5),
            "pg": round(pg_loss.item(), 5),
            "kl": round(kl_loss.item(), 5),
            "clip_frac": round(clip_frac, 4),
            # Token-level log-ratio diagnostics (GRPO is token-level so
            # these are aggregated over (BK, L) valid positions). Mirrors
            # GDPO's sequence-level log_ratio_seq fields in spirit.
            "log_ratio_tok": round(log_ratio_tok_mean, 4),
            "log_ratio_tok_max_abs": round(log_ratio_tok_max_abs, 4),
            "ratio_tok": round(ratio_tok_mean, 4),
            "dw_step": dw_step,
            "dw_total": dw_total,
            "avg_len": round(avg_len, 1),
            "diversity_mean": round(float(div_stats["diversity_mean"]), 4),
            "diversity_min_pair": round(float(div_stats["diversity_min_pair"]), 4),
            "unique_count": int(div_stats["unique_count"]),
            "per_replica_diversity": [float(x) for x in div_stats["per_replica_diversity"]],
        }
        # CompositeReward (and any reward exposing the same hook) reports
        # per-component values. Surface the per-step mean of each so
        # downstream analysis can see which objective is moving the policy.
        components_per_replica = None
        last_components = getattr(reward_fn, "last_components", None)
        if callable(last_components):
            comps = last_components()
            if comps:
                row["components"] = {
                    name: round(float(np.mean(vals)), 3) for name, vals in comps.items()
                }
                components_per_replica = {
                    name: [float(x) for x in vals] for name, vals in comps.items()
                }
                row["components_per_replica"] = components_per_replica
                logger.info(
                    "  components: %s",
                    ", ".join(f"{n}={row['components'][n]:.3f}" for n in row["components"]),
                )
        log_rows.append(row)
        # Atomic per-step flush so a job hitting walltime or crashing
        # leaves behind a fully-parseable train_log.json with every
        # step recorded up to that point.
        try:
            write_train_log_atomic(log_path, log_rows)
        except Exception as e:  # pragma: no cover
            logger.warning("train_log.json flush failed (non-fatal): %s", e)

        if grpo_cfg.debug_log:
            try:
                _write_debug_step(
                    debug_path=debug_path,
                    step=step,
                    batch_prompts=batch_prompts,
                    ids_all=ids_all,
                    seqs=seqs,
                    rewards_raw=rewards_raw,
                    adv=adv,
                    log_ratio_per_replica=log_ratio_per_replica,
                    ratio_per_replica=ratio_per_replica,
                    log_ratio_max_per_replica=log_ratio_max_per_replica,
                    valid_lengths=valid_lengths,
                    components_per_replica=components_per_replica,
                    K=K,
                    per_replica_diversity=div_stats["per_replica_diversity"],
                )
            except Exception as e:  # pragma: no cover
                logger.warning("debug.out write failed (non-fatal): %s", e)

        if step % grpo_cfg.save_steps == 0:
            ckpt_path = os.path.join(grpo_cfg.output_dir, f"step{step}.pt")
            torch.save({"step": step, "model_state": s3.state_dict()}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

    final_path = os.path.join(grpo_cfg.output_dir, "final.pt")
    torch.save({"step": grpo_cfg.steps, "model_state": s3.state_dict()}, final_path)
    # train_log.json was already flushed after every step; this final
    # rewrite is a no-op safety net in case any exception path skipped
    # the in-loop flush.
    write_train_log_atomic(log_path, log_rows)
    logger.info("GRPO done. Final checkpoint: %s", final_path)

    # Best-effort end-of-run diagnostic plots.
    try:
        from biom3.rl.plotting import plot_train_log
        plot_train_log(log_path, grpo_cfg.output_dir, algo="grpo")
    except Exception as e:  # pragma: no cover
        logger.warning("plot_train_log failed (non-fatal): %s", e)

    return log_rows
