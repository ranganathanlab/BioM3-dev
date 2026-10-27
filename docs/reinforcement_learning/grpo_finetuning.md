# GRPO fine-tuning of Stage 3

GRPO (Group Relative Policy Optimization) RL fine-tuning of the Stage 3
ProteoScribe diffusion model. Single-GPU only in this revision; multi-GPU
is deferred (Phase 4 in
[docs/.claude_prompts/PROMPT_grpo_integration.md](../.claude_prompts/PROMPT_grpo_integration.md)).

## What it does

For each text prompt, sample K candidate protein sequences from the
diffusion policy, score each with ESMFold mean pLDDT, and update the
policy with PPO-clip + Schulman $k_3$ KL against a frozen reference
snapshot, using group-normalized advantages.

The math is laid out in
[PROMPT_grpo_integration.md](../.claude_prompts/PROMPT_grpo_integration.md);
see "Equations".

## Layout

```
src/biom3/rl/
  __init__.py
  __main__.py            # biom3_grpo_train entry-point wrapper
  grpo.py                # GRPOConfig, grpo_train, sampling + log-prob helpers
  rewards/               # reward package (see below)
    __init__.py          #   re-exports Reward, build_reward, and all reward classes
    base.py              #   Reward Protocol, VALID_AA
    registry.py          #   build_reward
    stub.py esmfold.py aa_fraction.py surrogate.py tsv_lookup.py
    composite.py diversity.py
  io.py                  # frozen Stage 1/2 loaders + trainable Stage 3
  rollout.py             # sequence rollout helpers
  run_grpo_train.py      # argparse + load_json_config composition

configs/grpo/
  _base_grpo.json        # base config (no weights set)
  example_grpo.json      # 100-step example overlay
  prompts/example_prompts.txt

scripts/grpo_train_singlenode.sh
jobs/aurora/_template_grpo_singlenode.pbs

tests/rl_tests/test_grpo_smoke.py
```

## Prerequisites

- Stage 1 (PenCL) weights — for the frozen text encoder.
- Stage 2 (Facilitator) weights — for the frozen text→protein bridge.
- Stage 3 (ProteoScribe) init weights — any of: raw `.pt`/`.bin`,
  Lightning `.ckpt`, or a DeepSpeed sharded directory. Loaded through
  `biom3.Stage3.io.prepare_model_ProteoScribe`, which handles all three.
- ESMFold (`facebook/esmfold_v1`) — only if `reward = "esmfold_plddt"`.
  Install the optional extras:

  ```bash
  pip install -e '.[grpo]'
  ```

  Use `reward = "stub"` (the deterministic length+diversity reward in
  `biom3.rl.rewards.StubReward`) if you want to drive the loop without
  ESMFold (smoke testing, debugging).

## Configuration

GRPO uses the standard `core.helpers.load_json_config` composition.
`configs/grpo/_base_grpo.json` wires Stage 1/2/3 inference configs as
defaults; experiment configs overlay it via `_base_configs`. Precedence
(low → high) is `_base_configs` < current file < `_overwrite_configs` < CLI.

Required fields (must end up resolved by the time the trainer starts). The three weight
paths are required: a missing path, or a file that does not populate its model, stops
the run.

| Field | What |
|---|---|
| `stage1_config` | Path to PenCL JSON config |
| `stage2_config` | Path to Facilitator JSON config |
| `stage3_config` | Path to ProteoScribe JSON config |
| `stage1_weights` | PenCL `.bin`/`.pt` |
| `stage2_weights` | Facilitator `.bin`/`.pt` |
| `stage3_init_weights` | ProteoScribe init (any supported format) |
| `weight_set` | Optional weight-set JSON. Fills the three weights above when they are not given (and, when passed as `--weight_set` on the command line, replaces ones set in the JSON config), and its record of how PenCL was trained is checked against `text_attention_mask`. RL does not scale `z_c` to unit length, so it warns when the set records ProteoScribe as trained that way |
| `text_attention_mask` | Pass the caption attention mask to BERT when embedding prompts (default `false`). Set it to match how the PenCL weights were trained; a mismatch with the weight set logs a loud warning |
| `prompts_path` | UTF-8 text file, one prompt per line |

GRPO hyperparameters (defaults shown):

| Field | Default | Notes |
|---|---|---|
| `steps` | 200 | total gradient updates |
| `num_generations` | 4 | $K$ — sequences per prompt |
| `batch_size` | 1 | prompts per gradient update |
| `learning_rate` | 1e-5 | AdamW |
| `weight_decay` | 1e-6 | AdamW |
| `beta` | 0.01 | KL coefficient $\beta$ |
| `eps` | 0.20 | PPO clip $\epsilon$ |
| `max_grad_norm` | 1.0 | grad clip |
| `save_steps` | 50 | checkpoint cadence |
| `seed` | 42 | torch + numpy |
| `reward` | `esmfold_plddt` | `esmfold_plddt` \| `stub` \| `tsv_lookup` (see Reward catalog) |

## Reward catalog

A reward is any callable taking `List[str]` of decoded amino-acid
sequences and returning `List[float]` scalars (one per sequence). The
``Reward`` ``Protocol`` lives in [src/biom3/rl/rewards/base.py](../../src/biom3/rl/rewards/base.py).

| Name | Class | What |
|---|---|---|
| `esmfold_plddt` | `ESMFoldReward` | Mean pLDDT in [0, 100] from `facebook/esmfold_v1`. Lazy-loaded; needs `[grpo]` extras. |
| `stub` | `StubReward` | Deterministic length + diversity score in [0, 100]. CPU-friendly; for smoke tests. |
| `tsv_lookup` | `TsvLookupReward` | Look up a per-sequence scalar from a TSV/CSV file (closed-world ranking only — see below). |
| `aa_fraction` | `AAFractionReward` | Closed-form synthetic ground truth used by the surrogate-in-the-loop demo. Peaks at a target fraction of a chosen AA, falls off linearly. |
| `surrogate` | `SurrogateReward` | Wraps a fitted regressor (`predictor.predict(features)`) + a `Featurizer`. Constructed in code rather than via `build_reward`; load the joblib + sidecar produced by `train_grpo_surrogate.py`. |

A reward MAY also expose `last_components() -> Dict[str, List[float]]`
returning the per-component values from its most recent call. The
trainer logs the per-step mean of each component into `train_log.json`
under `"components"` and the run log when present.

### `tsv_lookup` — using experimental data

For experimental measurements (stability ΔΔG, binding affinity, fitness
assay readouts) keyed by sequence:

```
configs/grpo/your_run.json:
  "reward": "tsv_lookup"

configs/grpo/your_run.json (or CLI):
  --tsv_lookup_path  path/to/measurements.tsv
  --tsv_lookup_key   sequence
  --tsv_lookup_value ddG
```

(Currently `build_reward("tsv_lookup", ...)` accepts the constructor
kwargs directly. CLI plumbing for the kwargs above is straightforward
to add when needed; for now wire them via a custom config with
`_overwrite_configs`.)

**TSV format:** UTF-8 with a header row.

```
sequence    ddG
MNGTEGPNFY...    -2.4
MNGTEGPNFL...    -1.7
...
```

`key_column` (default `sequence`) names the sequence column,
`value_column` (default `value`) names the scalar column,
`delimiter` defaults to `\t`. Sequences are cleaned to canonical AAs
(20-letter alphabet) before lookup unless `clean=False`.

**Misses** (sequence not in the TSV) are unavoidable when the policy
generates novel sequences — that's the point of generation. Two
strategies:

- `miss_strategy="penalty"` (default), with `miss_value` (default 0.0):
  return a constant on miss. Simple, but if *every* sequence in a
  group misses, the within-group standard deviation is 0 and the
  group-normalized advantage is undefined (the loop currently uses
  `clamp(min=1e-8)`, so the gradient just collapses to ~0). Fine for
  closed-world tasks (ranking N known variants); not what you want
  for generative exploration.
- `miss_strategy="skip"`: returns `NaN` on miss. Reserved for a future
  iteration of `grpo_train` that filters NaN groups. Today this
  behaves as a `NaN` penalty (still a fixed value).

For novel-sequence work, the right answer is the **surrogate-in-the-loop
workflow** described below — train a regressor on the TSV, use the
regressor as the reward.

### `CompositeReward` — multi-objective

Construct in code (not via `build_reward`); pass the assembled
`CompositeReward` to `grpo_train` as `reward_fn`:

```python
from biom3.rl.rewards import CompositeReward, ESMFoldReward, TsvLookupReward

reward_fn = CompositeReward(
    {
        "plddt":    (ESMFoldReward(device=device), 0.5),
        "stability": (TsvLookupReward("ddG.tsv", value_column="ddG"), 1.0),
    },
    reduction="weighted_sum",   # or "product"
)
```

`weighted_sum` is appropriate when components are on similar scales
(or you've normalized them). `product` is "all of these must be high"
— a near-zero on any component nukes the score, useful when one
objective is a hard constraint.

**Scale alignment matters.** pLDDT is [0, 100], a ΔΔG might be
[-5, +5] kcal/mol, an activity flag is 0/1. Either rescale per
component before composing, or pick weights that compensate. The
`"components"` key in `train_log.json` (mean per step) lets you see
which objective is actually moving the policy — if one component's
contribution dwarfs the others, your weights are wrong.

## Surrogate-in-the-loop workflow

For real wet-lab use, the policy generates *novel* sequences that won't
appear in any TSV. The architecturally correct approach is a
surrogate-regressor reward:

```
lab measurements (TSV: sequence, scalar)
    → train surrogate s_φ(sequence) → predicted scalar              (reward proxy)
    → GRPO fine-tune the policy with s_φ as the reward
    → sample new candidate sequences
    → send top-K to the lab → new measurements → retrain, repeat
```

The pieces you need are already in the repo:

- [`scripts/make_grpo_synthetic_eval.py`](../../scripts/make_grpo_synthetic_eval.py) — for development, builds a synthetic 2k-row TSV from `data/datasets/SH3/FINAL_SH3_all_dataset_with_prompts.csv` with `functional_score = AAFractionReward + small noise`.
- [`scripts/train_grpo_surrogate.py`](../../scripts/train_grpo_surrogate.py) — fits an sklearn regressor (Ridge or MLP) on a TSV of (sequence, scalar) using a chosen featurizer (one-hot or ESM-2 mean-pool). Writes a joblib + a small JSON sidecar describing the featurizer config.
- [`scripts/eval_grpo_checkpoint.py`](../../scripts/eval_grpo_checkpoint.py) — samples N sequences per prompt from a Stage 3 checkpoint and scores with a configurable reward (always also computes the `aa_fraction` ground truth, so a fair before/after comparison is always available).
- [`SurrogateReward(predictor, featurizer)`](../../src/biom3/rl/rewards/surrogate.py) — wraps a fitted regressor + featurizer as a per-sequence reward. Constructed in code; reload via:

  ```python
  import joblib, json
  from biom3.rl.featurizers import build_featurizer
  from biom3.rl.rewards import SurrogateReward

  predictor  = joblib.load("outputs/grpo/surrogate.joblib")
  cfg        = json.load(open("outputs/grpo/surrogate.joblib.config.json"))
  featurizer = build_featurizer(cfg["featurizer"], **cfg["featurizer_kwargs"])
  reward_fn  = SurrogateReward(predictor=predictor, featurizer=featurizer)
  ```

### Featurizers

| Name | Class | What | Output dim |
|---|---|---|---|
| `onehot` | `OneHotFeaturizer(max_length=256)` | Pad/truncate + one-hot over the 20 canonical AAs. CPU-only, no model deps. | `max_length × 20` |
| `esm2` | `ESM2MeanPoolFeaturizer(model_path, layer=33, ...)` | Mean-pool the per-residue ESM-2 embedding at the chosen layer. Lazy-loaded; same `weights/LLMs/esm2_t33_650M_UR50D.pt` Stage 1 already uses. | `1280` (650M, layer 33) |

### End-to-end synthetic recipe

```bash
# 1. Build the synthetic TSV (CPU, ~30 s)
python scripts/make_grpo_synthetic_eval.py \
    --input data/datasets/SH3/FINAL_SH3_all_dataset_with_prompts.csv \
    --output_dir data/datasets/SH3/ \
    --n 2000 --n_test 200 --seed 42

# 2. Train surrogate (one-hot + Ridge ≈ 10 s on CPU; ESM-2 + Ridge ≈ 2-5 min on Aurora)
python scripts/train_grpo_surrogate.py \
    --train_tsv data/datasets/SH3/synthetic_train.tsv \
    --test_tsv  data/datasets/SH3/synthetic_test.tsv \
    --featurizer onehot --model ridge \
    --output_path outputs/grpo/surrogate_aa.joblib

# Expect: test R² > 0.7 on the synthetic ground truth.

# 3. Evaluate the BASELINE Stage 3 weights against the GROUND TRUTH reward
python scripts/eval_grpo_checkpoint.py \
    --config_path  configs/grpo/example_grpo.json \
    --checkpoint   ./weights/ProteoScribe/ProteoScribe_SH3_epoch52.ckpt \
    --prompts_tsv  data/datasets/SH3/synthetic_test.tsv \
    --n_per_prompt 4 \
    --reward       aa_fraction \
    --output_dir   outputs/grpo/eval

# 4. GRPO fine-tune with the SURROGATE as reward (construct SurrogateReward in code,
#    pass directly to grpo_train; CLI plumbing for surrogate_path through
#    biom3_grpo_train is a small follow-up if you want it).

# 5. Re-run step 3 against the GRPO-fine-tuned checkpoint. Diff summary.json
#    "gt_mean" — post-GRPO should be strictly higher.
```

### What the synthetic demo proves

The ground truth (`AAFractionReward`) is closed-form. The surrogate is
a regressor fit on noisy samples of that ground truth. We train GRPO
*using the surrogate as the reward*, then evaluate the resulting policy
*against the ground truth*. If post-fine-tune ground-truth mean >
baseline ground-truth mean, the surrogate is transmitting useful signal
and GRPO is acting on it correctly. That's the gold-standard validation
of the loop, no lab data required.

In real use, replace step 1 with a TSV produced by your wet-lab assay,
and skip the ground-truth comparison (you don't have one). The rest of
the recipe is identical.

## Prompt file format

Plain UTF-8, one prompt per line. Empty lines and lines starting with
`#` (after strip) are ignored. The `name|text` form is accepted; only
the `text` part is used. See `configs/grpo/prompts/example_prompts.txt`.

## Running locally (single GPU)

```bash
source environment.sh
./scripts/grpo_train_singlenode.sh \
    configs/grpo/example_grpo.json \
    my_grpo_run_001 \
    xpu \
    --stage1_weights weights/PenCL/BioM3_PenCL_epoch20.bin \
    --stage2_weights weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --stage3_init_weights weights/ProteoScribe/state_dict.best.pth
```

`xpu` for Aurora; use `cuda` on Polaris/Spark and `cpu` only for very
small smoke runs (the diffusion rollout is the dominant cost).

On Aurora, pin to a specific tile:

```bash
ZE_AFFINITY_MASK=0 ./scripts/grpo_train_singlenode.sh ...
```

CLI args override JSON config; JSON config overrides argparse defaults.

## Submitting on Aurora (PBS)

Copy the template, replace the placeholders, submit:

```bash
cp jobs/aurora/_template_grpo_singlenode.pbs jobs/aurora/job_grpo_<run>.pbs
# Edit <JOB_NAME>, <QUEUE>, <CONFIG_NAME>, <STAGE3_WEIGHTS_PATH>.
qsub jobs/aurora/job_grpo_<run>.pbs
```

The template requests `select=1`, pins to tile 0 via `ZE_AFFINITY_MASK`,
and tees output to `logs/${run_id}.${PBS_JOBID}.o`.

## Output layout

```
{output_root}/{run_id}/
├── step50.pt            # {"step": 50, "model_state": ...}
├── step100.pt
├── ...
├── final.pt
└── train_log.json       # one row per step: reward, loss, pg, kl, clip_frac, avg_len
```

Resulting `final.pt` is a raw Stage 3 state-dict and feeds straight back
into `biom3.Stage3.io.prepare_model_ProteoScribe(model_fpath=...)` for
downstream sampling/evaluation via `biom3_ProteoScribe_sample`.

## Testing

```bash
# CPU smoke (no weights, no ESMFold, no GPU):
PYTHONPATH=src pytest tests/rl_tests/ -v
```

Covers token vocabulary constants, prompt parsing, decoding, stub
reward range, sampler 3-tuple unpacking, policy log-prob shape, and an
end-to-end inner GRPO update on the mini Stage 3 fixture
(`tests/_data/models/stage3/weights/minimodel1_ds128_weights1.pth`).

## Watch out for

- **Editable install drift.** `biom3.__file__` may resolve to a sibling
  worktree. If `import biom3.rl` fails after a clean install, either
  `PYTHONPATH=src` for the run, or `pip install -e .` from this tree
  (note: re-installing affects sibling worktrees too).
- **BERT padding.** Stage 1 must use `padding="max_length"`,
  `max_length=512`. The `_PromptEncoder` in
  [src/biom3/rl/grpo.py](../../src/biom3/rl/grpo.py) routes through
  `Stage1.preprocess.TextSeqPairing_Dataset` which already does this —
  don't bypass it.
- **`strict=False` on weight load.** Inherited from the donor and the
  rest of the codebase. Missing/extra keys will be silently ignored;
  if you change ProteoScribe architecture, check the loader's missing
  keys list before trusting a run.
- **Reference model memory.** GRPO snapshots a frozen ProteoScribe via
  `copy.deepcopy`, doubling Stage 3 GPU memory. Fine for ~86M params on
  a single Aurora tile / H100; revisit if the model grows.
- **Single-GPU only.** No DDP, no DeepSpeed in the trainer. Don't try
  to launch via `mpiexec` — `grpo_train_singlenode.sh` deliberately
  doesn't.
