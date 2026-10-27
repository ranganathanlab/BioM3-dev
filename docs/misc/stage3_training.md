# Stage 3 Training (ProteoScribe)

Stage 3 trains a conditional diffusion transformer (ProteoScribe) that generates protein sequences from text-conditioned embeddings. It is the third and final stage in the BioM3 pipeline: PenCL (Stage 1) produces joint text/protein embeddings, the Facilitator (Stage 2) maps text embeddings into protein space, and ProteoScribe (Stage 3) uses those conditioned embeddings to drive an autoregressive diffusion model over amino acid sequences.

Training uses PyTorch Lightning with DeepSpeed Stage 2 as the distributed strategy. The entry point is:

```
biom3_train_stage3 --config_path <path_to_json>
```

> **CLI argument reference:** see [CLI_reference.md#biom3_train_stage3--stage-3-proteoscribe-training-and-finetuning](../CLI_reference.md#biom3_train_stage3--stage-3-proteoscribe-training-and-finetuning) for the full argument table. This document focuses on workflows, output layout, and per-machine submission.

---

## Pretraining Workflows

### From scratch (primary data only)

The standard workflow. Uses a single primary HDF5 dataset (typically SwissProt embeddings from Stage 2) with epoch-based training.

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id my_run_001 \
    --epochs 100
```

When `training_strategy` is `auto` (the default) and no `secondary_data_paths` are set, training runs in `primary_only` mode. The Trainer uses `max_epochs` and validates after each epoch.

### Phase 2 with secondary data

After pretraining on the primary dataset, you can continue training on a combined primary+secondary dataset (e.g., adding Pfam). This switches to step-based training.

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_phase2.json \
    --run_id my_run_phase2 \
    --resume_from_checkpoint /path/to/checkpoints/run_id/last.ckpt \
    --start_secondary True \
    --secondary_data_paths ./data/pfam_embeddings.hdf5 \
    --max_steps 3000000
```

When `start_secondary` is set, the model weights are loaded from `--resume_from_checkpoint` and training begins fresh on the combined dataset. The Trainer uses `max_steps`, `val_check_interval`, and `limit_val_batches` instead of epoch counting.

### Resuming from checkpoint

To resume a run that was interrupted (same dataset, same optimizer state):

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id my_run_001 \
    --resume_from_checkpoint /path/to/checkpoints/my_run_001/last.ckpt
```

This calls `trainer.fit(..., ckpt_path=...)` which restores the full training state (model weights, optimizer, scheduler, epoch/step counters).

---

## Finetuning

Finetuning freezes most of the model and trains only selected layers on a new dataset. Requires pretrained weights.

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_example.json \
    --run_id finetune_001 \
    --finetune True \
    --pretrained_weights /path/to/state_dict.best.pth \
    --finetune_last_n_blocks 4 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True
```

| Flag | Description |
|---|---|
| `--finetune` | Enable finetuning mode |
| `--pretrained_weights` | Path to `.bin` or `.pth` weight file to load before finetuning |
| `--finetune_last_n_blocks` | Number of last transformer blocks to unfreeze (`-1` = all, `0` = none) |
| `--finetune_last_n_layers` | Number of last transformer layers to unfreeze (`-1` = all, `0` = none) |
| `--finetune_output_layers` | Whether to unfreeze the transformer output layers (norm and out) |

---

## Configuration

### Precedence

Configuration values are resolved in this order (highest priority first):

1. **CLI arguments** passed directly on the command line
2. **JSON config** values loaded from `--config_path`
3. **Argparse defaults** defined in code

In practice: put stable hyperparameters in JSON, override per-job settings (epochs, run_id, device count) via CLI or job templates.

### Config composition

Training configs support two special keys for layered composition:

- **`_base_configs`** — list of paths loaded *before* the current file. The current file's values override base values.
- **`_overwrite_configs`** — list of paths loaded *after* the current file. Their values override the current file's values.

Priority (lowest → highest):

```
_base_configs  <  current file  <  _overwrite_configs  <  CLI args
```

Paths resolve relative to the directory containing the JSON file. Both keys are stripped from the result. Circular references raise `ValueError`.

#### Base config layout

```
configs/stage3_training/
├── models/
│   ├── _base_ProteoScribe_1block.json      # 1-block transformer (paper architecture)
│   └── _base_ProteoScribe_16blocks.json    # 16-block transformer
├── machines/
│   ├── _aurora.json                  # xpu, 12 devices
│   ├── _polaris.json                 # cuda, 4 devices
│   └── _spark.json                   # cuda, 1 device
├── pretrain_scratch_v2.json          # uses _base_configs → models/
└── ...
```

#### Example: model base + machine overwrite

```json
{
  "description": "Pretrain 16-block on Polaris",
  "tags": [],
  "notes": [],
  "_base_configs": ["./models/_base_ProteoScribe_16blocks.json"],
  "_overwrite_configs": ["./machines/_polaris.json"],

  "device": "cpu",
  "devices_per_node": 1,
  "lr": 1e-4,
  ...
}
```

Here `device` and `devices_per_node` in the file body serve as local-testing defaults, but the Polaris machine config overrides them with `cuda` / `4`. CLI args still win over everything.

### Choosing `limit_val_batches` and `limit_train_batches`

Both flags accept two value conventions, decided at runtime by the magnitude:

- **`> 1`** → absolute batch count (e.g. `200` = 200 batches per validation pass). Wall time is predictable across dataset sizes.
- **`(0, 1]`** → fraction of the val/train set (e.g. `0.05` = 5%). Wall time scales with dataset size.

**Recommended recipes:**

| Scenario | `valid_size` | `limit_val_batches` | `limit_train_batches` | Why |
|---|---|---|---|---|
| Small dataset (~10K rows), many epochs (100s) | `0.2` | `1.0` (full val set) | `None` (full train set) | Validation is fast even on the full set; full coverage gives stable loss-curve interpretation across epochs. |
| Large dataset (10M+ rows), few epochs (10s) | `0.2` | `200` (absolute) | `None` (full train set) | Fractional caps balloon validation time; absolute count keeps each val pass to seconds while still giving a statistically meaningful sample (~3200 sequences at `batch_size=16`). |
| Step-based training (`combine` strategy) | `0.2` | `200` (absolute) | `None` | Combined with `val_check_interval`, an absolute val cap makes each validation event a fixed cost. |
| Quick smoke test / CI | -- | `3` (absolute) | `3` (absolute) | Bound the entire run to a few batches; matches the pattern used in `configs/benchmark/`. |

**Argparser defaults today:** `--limit_val_batches=200` (Stage 3) and `=1.0` (Stage 1); `--limit_train_batches=None` (both). Existing production configs in `configs/stage3_training/*.json` explicitly set `limit_val_batches: 0.05` (a fraction). To switch a config to the absolute-count convention, change the value to e.g. `200` (no quotes — JSON int) and the runtime will treat it as a batch count.

### Key config parameters

The primary config example is `configs/stage3_training/pretrain_scratch_v2.json`. The table below lists the most important parameters.

| Parameter | Default | Description |
|---|---|---|
| `primary_data_path` | -- | Path to primary HDF5 training dataset |
| `secondary_data_paths` | `null` | List of secondary HDF5 dataset paths |
| `start_secondary` | `false` | Phase transition: load primary weights, train on combined data |
| `epochs` | `1` | Number of epochs (used in `primary_only` mode) |
| `max_steps` | `100000` | Max training steps (used in `combine` mode) |
| `batch_size` | `16` | Mini-batch size per device |
| `lr` | `3e-4` | Base learning rate |
| `scale_learning_rate` | `true` | Multiply LR by `num_nodes * devices_per_node` |
| `normalize_zc` | `false` | Scale the conditioning vector to unit length before the model sees it, after any `z_p` blend. Generation must then use `--normalize_zc` too, or the model is conditioned on vectors of a length it was not trained on |
| `loss_positions` | `"all"` | Which unsampled positions drive the gradients: `"all"`, or `"non_pad"` for the sequence without its padding. See [Loss terms and padding](#loss-terms-and-padding) |
| `scheduler_gamma` | `null` | LR scheduler (`"coswarmup"` or a float gamma for StepLR). `coswarmup` warms up over the first epoch and decays over `epochs`, counting the optimizer steps one rank takes per epoch on `num_nodes * devices_per_node` ranks |
| `warmup_steps` | `500` | LR warmup steps (for cosine warmup scheduler) |
| `weight_decay` | `1e-6` | AdamW weight decay |
| `precision` | `"no"` | Training precision (`"bf16"`, `"fp16"`, `"32"`, `"no"`) |
| `diffusion_steps` | `256` | Number of diffusion timesteps |
| `transformer_blocks` | `1` | Number of transformer blocks |
| `transformer_depth` | `16` | Depth per transformer block |
| `transformer_dim` | `512` | Transformer hidden dimension |
| `transformer_heads` | `16` | Number of attention heads |
| `val_check_interval` | `10000` | Steps between validation (step-based mode) |
| `limit_val_batches` | `200` | Cap validation batches per check. Values >1 are an absolute batch count; values in (0,1] are a fraction. See "Choosing limit_val_batches and limit_train_batches" below. |
| `output_root` | `null` | Base directory for all training outputs |
| `checkpoint_monitors` | `[{"metric": "val_loss", "mode": "min"}]` | List of metrics to monitor for checkpointing |
| `seed` | `0` | Random seed |

---

## Distributed strategy

Stage 3's Lightning trainer strategy is chosen via `--distributed_strategy`:

- `deepspeed_zero2` (default): DeepSpeed ZeRO Stage 2 with CPU offload of optimizer states and parameters. Memory-efficient for large models — optimizer states are sharded across the world size, gradients are sharded, params stay replicated. Writes a sharded checkpoint *directory* (one file per shard) which `save_model` automatically converts to a single fp32 `state_dict.pth` at the end of training.
- `ddp`: plain DDP with `static_graph=True` and `gradient_as_bucket_view=True`. Full replication — every rank holds full params, grads, and optimizer state. Lower comm overhead per step but higher per-rank memory. Writes a single-file `last.ckpt` which `save_model` materializes into `state_dict.pth` via a copy (no ZeRO conversion needed).

Pick `ddp` when (a) the model is small enough that ZeRO-2's CPU offload latency is the bottleneck, (b) you're chasing an Aurora/xccl hang and want plain-DDP semantics for triage, or (c) downstream tooling needs a single-file `.ckpt`.

Both strategies produce identical `state_dict.best.pth` artifacts in `artifacts_dir`, so downstream sampling and evaluation paths are unaffected by the choice.

This flag is distinct from `--training_strategy`, which selects `primary_only` vs `combine` data mixing.

## Per-Machine Instructions

Each machine has a job template in `jobs/` that sets machine-specific constants (device count, modules, filesystem mounts) and delegates to the shared training scripts in `scripts/`. The templates construct a `run_id` automatically from the config name, node count, device count, epoch count, and a timestamp.

### Polaris (NVIDIA GPUs)

**Template:** `jobs/polaris/_template_stage3_pretrain_from_scratch.pbs`

Edit the template placeholders, then submit:

```bash
cd /path/to/BioM3-dev

# Edit the template:
#   <JOB_NAME>    — PBS job name
#   <NUM_NODES>   — number of nodes (in both the PBS header and the variable)
#   <QUEUE>       — PBS queue (e.g., debug, preemptable, prod)
#   <CONFIG_NAME> — name of your JSON config (without .json extension)

qsub jobs/polaris/_template_stage3_pretrain_from_scratch.pbs
```

Key constants: `num_devices=4` (A100s per node), `device=cuda`.

Uses `scripts/stage3_train_multinode.sh` which launches via `mpiexec`.

See [setup_polaris.md](../setup/setup_polaris.md) for environment setup.

### Aurora (Intel GPUs)

**Template:** `jobs/aurora/_template_stage3_pretrain_from_scratch.pbs`

Same editing pattern as Polaris. Key differences:

- `num_devices=12` (Intel GPUs per node)
- `device=xpu`
- `filesystems=home:flare`
- Sources `environment.sh` before launching (sets `ONEAPI_DEVICE_SELECTOR`, etc.)

```bash
qsub jobs/aurora/_template_stage3_pretrain_from_scratch.pbs
```

See [setup_aurora.md](../setup/setup_aurora.md) for environment setup (including the custom `lightning` fork).

### DGX Spark (single NVIDIA GPU)

**Template:** `jobs/spark/_template_stage3_pretrain_from_scratch.sh`

Single-node, single-GPU. Uses `scripts/stage3_train_singlenode.sh` instead of the multinode wrapper.

```bash
cd /path/to/BioM3-dev

# Edit <CONFIG_NAME>, then run:
bash jobs/spark/_template_stage3_pretrain_from_scratch.sh
```

Key constants: `num_nodes=1`, `num_devices=1`, `device=cuda`.

See [setup_spark.md](../setup/setup_spark.md) for environment setup.

---

## Output Directory Structure

All outputs are written under `--output_root` (default in config: `./outputs/Stage3/pretraining`):

```
{output_root}/
├── checkpoints/
│   └── {run_id}/
│       ├── epoch=0-step=1000.ckpt/     # DeepSpeed sharded checkpoint dir
│       ├── epoch=1-step=2000.ckpt/     # DeepSpeed sharded checkpoint dir
│       ├── last.ckpt -> epoch=1-...    # Symlink to latest checkpoint
│       ├── single_model.best.pth       # Converted fp32 Lightning checkpoint (best)
│       ├── single_model.last.pth       # Converted fp32 Lightning checkpoint (last)
│       ├── state_dict.best.pth         # Unwrapped nn.Module state_dict (best)
│       ├── state_dict.last.pth         # Unwrapped nn.Module state_dict (last)
│       ├── state_dict.pth -> state_dict.best.pth  # Default symlink
│       ├── state_dict_ema.best.pth     # EMA model state_dict (if available)
│       └── params.csv                  # Total parameter count
└── runs/
    └── {run_id}/
        ├── logs/
        │   ├── lightning_logs/         # TensorBoard event files
        │   └── wandb/                  # W&B run data (if enabled)
        └── artifacts/
            ├── args.json               # Serialized training arguments
            ├── build_manifest.json      # Data/config provenance
            ├── run.log                  # Python logger output
            ├── state_dict.best.pth     # Copy of best weights for easy access
            ├── checkpoint_summary.json # Summary of all checkpoint monitors
            └── metrics_history.pt      # Per-step train + per-epoch val metrics (numpy arrays)
```

The `run_id` is typically constructed by job templates as:

```
{config_name}_n{num_nodes}_d{num_devices}_e{epochs}_V{YYYYMMDD_HHMMSS}
```

---

## Checkpoints

### DeepSpeed conversion

Lightning + DeepSpeed saves checkpoints as directories containing sharded ZeRO state. After training completes, the `save_model` function (on rank 0 only) converts these into usable formats:

1. **`convert_zero_checkpoint_to_fp32_state_dict`** merges shards into a single fp32 Lightning checkpoint (`single_model.*.pth`)
2. The Lightning checkpoint is loaded, the PL wrapper is unwrapped, and the raw `nn.Module` state_dict is saved (`state_dict.*.pth`)
3. If the model has an EMA copy, it is also saved (`state_dict_ema.*.pth`)

### Best vs last

- The primary `ModelCheckpoint` tracks `val_loss` (or the first entry in `checkpoint_monitors`) with `save_top_k=2` and `save_last="link"`
- After training, both the best and last checkpoints are converted
- If best and last point to the same checkpoint, symlinks are created instead of duplicating
- `state_dict.pth` symlinks to `state_dict.best.pth` by default

### Multi-metric checkpoint monitors

You can track additional metrics by setting `checkpoint_monitors` in your JSON config:

```json
{
  "checkpoint_monitors": [
    {"metric": "val_loss", "mode": "min"},
    {"metric": "val_current_hard_acc", "mode": "max"}
  ]
}
```

The first entry is the primary monitor (top-2, with `last.ckpt` symlink). Additional entries each keep the single best checkpoint and produce a separate `state_dict.best_{metric}.pth` in both the checkpoints directory and artifacts.

### Periodic saving

In addition to metric-based saves:

- `checkpoint_every_n_steps` -- save at fixed step intervals
- `checkpoint_every_n_epochs` -- save at fixed epoch intervals
- For step-based training (`combine` strategy), periodic step saving defaults to `log_every_n_steps` if not explicitly set

---

## Metrics

### Logged metrics

All metrics are logged with `on_step=True, on_epoch=True`. The `{stage}` prefix is either `train` or `val`.

| Metric | Description |
|---|---|
| `{stage}_loss` | The loss that drives the gradients: `{stage}_loss_all` or `{stage}_loss_non_pad`, as `loss_positions` selects |
| `{stage}_loss_all` | Cross-entropy over every unsampled position, padding included |
| `{stage}_loss_non_pad` | The same over unsampled positions whose true token is not padding |
| `{stage}_loss_pad` | The same over unsampled positions whose true token is padding |
| `{stage}_prev_hard_acc` | Hard accuracy on previously unmasked positions |
| `{stage}_prev_soft_acc` | Soft accuracy on previously unmasked positions |
| `{stage}_fut_hard_acc` | Hard accuracy on future (still masked) positions |
| `{stage}_fut_soft_acc` | Soft accuracy on future positions |
| `{stage}_current_hard_acc` | Hard accuracy on the currently unmasked position |
| `{stage}_current_soft_acc` | Soft accuracy on the currently unmasked position |
| `{stage}_current_ppl` | Perplexity at the currently unmasked position |
| `{stage}_prev_ppl` | Perplexity over previously unmasked positions |
| `{stage}_fut_ppl` | Perplexity over future positions |
| `{stage}_pos_entropy` | Positional entropy of predictions |
| `{stage}_gpu_memory_usage` | GPU memory allocated (bytes) |
| `train_grad_norm` | Global L2 gradient norm (train only, step-level) |
| `lr-AdamW` | Current learning rate (from `LearningRateMonitor`) |

### Loss terms and padding

Every sequence is padded at its tail with pad tokens up to the model's fixed length, so
most positions of a typical protein are padding. The loss is computed three ways on every
step and all three are logged: over all unsampled positions, over the non-pad ones (the
`<START>`, residue and `<END>` tokens) and over the pad ones.

The diffusion time and the unmasking order are drawn over the whole fixed-length window, so
some of the positions to predict fall in the padding, and how many fall on the sequence
depends on its length and on the draw. `loss_non_pad` therefore divides each sequence's
summed negative log-probability by the number of non-pad positions that sequence actually
had to predict, not by anything derived from the diffusion time, and averages over the
sequences that had any; `loss_pad` does the same for pad positions. A model that charges
the same for every token reports that cost whatever the sequence length. `loss_all` keeps
its original normalisation, the number of unsampled positions in the window plus one.

Padding is a property of the target, not of the input. The mask token marks the positions
the model has not been shown; a pad that has already been sampled is context and is in none
of the three terms.

`loss_positions` chooses which term drives the gradients, and that term is also what
`{stage}_loss` reports, so `val_loss` and best-checkpoint selection follow it. Values of
`val_loss` are therefore not comparable between runs with different `loss_positions`; use
the explicit terms for that.

Generation decodes a sequence by dropping the special tokens from all positions, so the
length of a generated protein is set by where the model emits pad tokens. With
`loss_positions: "non_pad"` nothing in the gradients asks the model to predict padding. In
the one comparison run so far, a finetune of `run1_base` on 1,024 Swiss-Prot proteins (832
steps, one seed, everything else equal), `loss_pad` rose about tenfold with `"non_pad"` and
did not move with `"all"`, and sequences generated for a 72-residue protein's prompt
averaged 326 residues against 81. That is one short run, not a settled result; `loss_pad`
is logged so the effect can be watched in any run.

The non-pad term is also about as many times larger than `loss_all` as the model's length
is longer than the sequence (4.6 times in that test, in the gradient norm too), so with
`"non_pad"` the same learning rate acts correspondingly larger.

With the default `"all"`, `loss_non_pad` is still logged at every step and recorded in the
metrics history, so the sequence-only loss can be followed over training. It is not used
for checkpoint selection unless it is added to `checkpoint_monitors`.

### Normalising the conditioning vector

`normalize_zc` (default `false`) divides the conditioning vector by its length just before
the model sees it, in training and validation. It applies to the vector that conditions the
model, so after any `z_p` blend, and it works the same whether `z_c` is read from an HDF5
file or computed on the device.

A model trained this way has to be generated from the same way: pass `--normalize_zc` to
`biom3_ProteoScribe_sample`, or to `biom3_embedding_pipeline --generate`, which hands it on.
The weights themselves do not say how they were trained, so a weight set can record it:

```json
{
    "proteoscribe_weights": "./weights/ProteoScribe/my_model.bin",
    "proteoscribe_trained_with_normalized_zc": true
}
```

The flag always decides what a run does. When it disagrees with the record, the run logs a
`Z_C NORMALISATION DOES NOT MATCH THE WEIGHTS` warning at the start and again at the end.
The check runs in `biom3_train_stage3` and `biom3_finetune_stage3` (against the weights the
run starts from), in `biom3_ProteoScribe_sample` (given `--weight_set`), and in the
pipeline's `--generate`. It needs a weight set that names the ProteoScribe file in use;
without one nothing is checked. A model you train with `normalize_zc` is a new set of
weights, so it is covered only once a weight set names it and records `true`.

Which weights a run starts from follows the usual precedence. A path given on the command
line wins. `--weight_set` on the command line comes next and replaces a `pretrained_weights`
set in the JSON config; the run log says so. A weight set named inside the config only fills
paths the config leaves unset.

RL post-training and the multidomain scripts do not normalise. RL warns when its weight set
says the ProteoScribe weights were trained with normalised vectors.

### sync_dist behavior

Validation metrics use `sync_dist=True` to aggregate across all ranks. Training metrics use `sync_dist=False` (each rank logs independently) to avoid communication overhead, except for `{stage}_gpu_memory_usage` which always syncs.

### MetricsHistoryCallback

When `save_metrics_history` is `true` (default), the `MetricsHistoryCallback` saves a single `metrics_history.pt` file to the artifacts directory. The file contains a dict with two keys:

```python
data = torch.load("artifacts/metrics_history.pt")
data["train"]  # dict of numpy arrays: {"global_step": array, "epoch": array, "train_loss": array, ...}
data["val"]    # dict of numpy arrays: {"global_step": array, "epoch": array, "val_loss": array, "loss_gap": array, ...}
```

Each metric is stored as a 1-D numpy array indexed by recording event. The `loss_gap` field in validation records is `val_loss - train_loss_epoch` when both are available.

Config keys:

- `metrics_history_ranks` -- which ranks save (default: `[0]`)
- `metrics_history_every_n_steps` -- recording frequency for training metrics (default: `1`)

---

## Early Stopping

Disabled by default (`early_stopping_metric: null`). To enable:

```json
{
  "early_stopping_metric": "val_loss",
  "early_stopping_patience": 10,
  "early_stopping_min_delta": 0.0,
  "early_stopping_mode": "min"
}
```

| Parameter | Description |
|---|---|
| `early_stopping_metric` | Metric to monitor (set to `null` to disable) |
| `early_stopping_patience` | Number of validation checks with no improvement before stopping |
| `early_stopping_min_delta` | Minimum change to count as an improvement |
| `early_stopping_mode` | `"min"` or `"max"` |

---

## Troubleshooting

### OOM errors

- Reduce `batch_size`. This is the per-device batch size.
- Increase `acc_grad_batches` to maintain effective batch size while reducing memory per step.
- Set `num_workers: 0` (already the default). Workers with `num_workers > 0` can duplicate GPU memory on CUDA.
- Reduce `transformer_blocks` or `transformer_depth` to shrink the model.
- Use `precision: "bf16"` if not already enabled.

### NCCL / communication timeouts

- Ensure all nodes can reach each other. On PBS systems, `$PBS_NODEFILE` must be correct.
- Set `NCCL_DEBUG=INFO` to diagnose connectivity issues.
- The multinode script uses `mpiexec --envall` to forward environment variables -- make sure `WANDB_API_KEY` and other required variables are exported before job submission.

### DeepSpeed configuration

The default DeepSpeed config uses ZeRO Stage 2 (set via `strategy='deepspeed_stage_2'` in the Trainer). The inline `ds_config` dict in `run_PL_training.py` configures ZeRO Stage 1 with CPU offloading, but it is currently not passed to the strategy (the Trainer string shorthand is used instead). To customize DeepSpeed behavior, pass a `ds_config` dict to `train_model()`.

### `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`

Both training scripts (`stage3_train_singlenode.sh` and `stage3_train_multinode.sh`) export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true`. This is required because some checkpoints contain non-tensor objects that fail with PyTorch's `weights_only=True` default.

### W&B authentication

Set `WANDB_API_KEY` in your shell profile (e.g., `.bashrc`) before submitting jobs. The job templates pass `$WANDB_API_KEY` to the training scripts, which export it.

### Step-based vs epoch-based confusion

If your run ends immediately or doesn't validate, check `training_strategy`. When `auto` (the default):
- No `secondary_data_paths` --> `primary_only` (epoch-based, uses `epochs`)
- `secondary_data_paths` provided --> `combine` (step-based, uses `max_steps`, `val_check_interval`)

If you set `start_secondary: true` but forget to provide `--secondary_data_paths`, data loading will fail.
