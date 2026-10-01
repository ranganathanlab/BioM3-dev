# BioM3 CLI Reference

Reference for the core inference, training, and analysis entrypoints declared in [pyproject.toml](../pyproject.toml). Each section gives the synopsis, required and optional arguments, the canonical config file (where applicable), and a representative example. For per-stage prose context (output layouts, metrics, per-machine job submission), see the deeper docs linked from each section.

Entrypoints covered elsewhere:

| Entrypoints | Where |
| ----------- | ----- |
| `biom3_build_dataset`, `biom3_build_taxid_index`, `biom3_csv_to_parquet`, `biom3_build_source_*`, `biom3_build_annotation_cache`, `biom3_build_pfam_subsets` | [dbio_examples.md](dbio/dbio_examples.md) |
| `biom3_grpo_train`, `biom3_gdpo_train`, `biom3_dpo_train` | [reinforcement_learning/](reinforcement_learning/) |
| `biom3_benchmark_*`, `biom3_plot_benchmark`, `biom3_compile_hdf5`, `biom3_cluster_split`, `biom3_stratified_cluster_split`, `biom3_app` | not documented — invoke with `--help` for current options |

> All entrypoints accept `--help`. The tables below mirror the source-of-truth `add_argument` declarations; if behavior diverges, the source wins.

> **Config composition**: every entrypoint that accepts `--config_path` loads JSON via `core.helpers.load_json_config`, which honors two special keys: `_base_configs` (loaded before the current file — current file overrides) and `_overwrite_configs` (loaded after — they override). CLI args override everything. See [stage3_training.md#config-composition](misc/stage3_training.md#config-composition).

---

## The `biom3` command

`biom3 <command> [<args>]` runs the entrypoints below under shorter names. Each command takes exactly the same arguments as the `biom3_*` entrypoint it runs, so the argument tables in this document apply unchanged. The `biom3_*` entrypoints remain available.

| Command | Runs | Listed by |
| ------- | ---- | --------- |
| `biom3 embed` | `biom3_embedding_pipeline` | `biom3 --help` |
| `biom3 generate` | `biom3_ProteoScribe_sample` | `biom3 --help` |
| `biom3 finetune` | `biom3_train_stage3 --finetune True` | `biom3 --help` |
| `biom3 finetune-generalized` | `biom3_finetune_stage3` | `biom3 --help` |
| `biom3 train stage1` | `biom3_train_stage1` | `biom3 --help` |
| `biom3 train stage2` | `biom3_train_stage2` | `biom3 --help` |
| `biom3 train stage3` | `biom3_train_stage3` | `biom3 --help` |
| `biom3 fetch weights` | `biom3_fetch_weights` | `biom3 --help` |
| `biom3 fetch dataset` | `biom3_fetch_dataset` | `biom3 --help` |
| `biom3 stage1 infer` | `biom3_PenCL_inference` | `biom3 --help --all` (advanced) |
| `biom3 stage2 sample` | `biom3_Facilitator_sample` | `biom3 --help --all` (advanced) |
| `biom3 manifold fit` | `biom3_fit_manifold` | `biom3 --help --all` (advanced) |
| `biom3 manifold score` | `biom3_score_manifold` | `biom3 --help --all` (advanced) |
| `biom3 rl grpo` | `biom3_grpo_train` | `biom3 --help --all` (experimental) |
| `biom3 rl gdpo` | `biom3_gdpo_train` | `biom3 --help --all` (experimental) |
| `biom3 rl dpo` | `biom3_dpo_train` | `biom3 --help --all` (experimental) |
| `biom3 multidomain finetune` | `biom3_finetune_multidomain` | `biom3 --help --all` (experimental) |
| `biom3 multidomain sample` | `biom3_sample_multidomain` | `biom3 --help --all` (experimental) |

```bash
biom3 --help                 # the main commands
biom3 --help --all           # every command, including advanced and experimental
biom3 train --help           # the commands in a group
biom3 train stage3 --help    # one command's arguments
biom3 --version
```

Notes:

- `biom3 finetune` sets `--finetune True` itself. Passing `--finetune` to it is an error; use `biom3 train stage3` to train without finetuning.
- Commands in every group are runnable. The group only decides whether `biom3 --help` lists the command without `--all`.
- Not every entrypoint has a `biom3` command. Dataset building (`biom3_build_*`), `biom3_app`, and the benchmarks are available only under their `biom3_*` names.
- An existing editable install needs `pip install -e .` rerun once to gain the `biom3` command. Container images built before the command was added do not have it; use the `biom3_*` names there.
- The commands are declared in [src/biom3/cli/registry.py](../src/biom3/cli/registry.py). Exposing another entrypoint is one entry there, provided its module defines `parse_arguments(argv)` and `main(args)`.

---

## Inference Entrypoints

### `biom3_PenCL_inference` — Stage 1 PenCL inference

Produces joint protein/text embeddings (`z_p`, `z_t`) from a CSV of (sequence, prompt) pairs.

**Source:** [src/biom3/Stage1/run_PenCL_inference.py](../src/biom3/Stage1/run_PenCL_inference.py)
**Config:** `configs/inference/stage1_PenCL.json`
**Deeper doc:** none (this reference + module docstring are authoritative).

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_data_path` | str | Path to input CSV (sequences + prompts). Pass `None` to use the built-in test dataset. |
| `-c`, `--config_path` | str | Path to JSON config (e.g. `configs/inference/stage1_PenCL.json`). |
| `-m`, `--model_path` | str | Path to pretrained PenCL weights (`.bin`) or Lightning checkpoint (`.ckpt`). |
| `-o`, `--output_path` | str | Path to write output embeddings (`.pt`). |

#### Optional arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--device` | str | `auto` | One of `auto`, `cpu`, `cuda`, `xpu`. `auto` = the detected backend: CUDA, then XPU, else CPU. |
| `--batch_size` | int | 32 | Inference batch size. |
| `--num_workers` | int | 0 | DataLoader worker count. |
| `--load_from_checkpoint` | flag | False | Force loading `model_path` as a Lightning `.ckpt` (otherwise inferred from extension). |
| `--no_amp` | flag | False | Disable autocast and run the forward pass in fp32. Autocast (bf16 on xpu, fp16 on cuda) is on by default. bf16 rounding depends on tensor shape, so results vary slightly with batch size; pair `--no_amp` with `--float32_matmul_precision highest` when comparing runs. |
| `--cross_comparison_sample_limit` | int | 0 | Samples used for the O(n²) cross-comparison metrics (dot-product probabilities, homology matrix). `0` = skip entirely (default), `-1` = all, positive = that many. Each metric allocates an n×n fp32 matrix (~25 GB at n=80k), so `-1` is only safe on small datasets. **Print-only** — saved embeddings are unaffected. |

#### Example

```bash
biom3_PenCL_inference \
    --input_data_path data/my_proteins.csv \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt \
    --batch_size 64 \
    --cross_comparison_sample_limit 1000
```

---

### `biom3_Facilitator_sample` — Stage 2 Facilitator sampling

Maps Stage 1 text embeddings (`z_t`) into the protein-embedding space (`z_c`). Consumes the `.pt` output of `biom3_PenCL_inference`.

**Source:** [src/biom3/Stage2/run_Facilitator_sample.py](../src/biom3/Stage2/run_Facilitator_sample.py)
**Config:** `configs/inference/stage2_Facilitator.json`

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_data_path` | str | Path to Stage 1 output embeddings (`.pt`). |
| `-c`, `--config_path` | str | Path to JSON config. |
| `-m`, `--model_path` | str | Path to Facilitator weights (`.bin`). |
| `-o`, `--output_data_path` | str | Path to write Stage 2 output embeddings (`.pt`). |

#### Optional arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--device` | str | `auto` | One of `auto`, `cpu`, `cuda`, `xpu`. `auto` = the detected backend: CUDA, then XPU, else CPU. |
| `--mmd_sample_limit` | int | -1 | Cap samples for MMD computation. `-1` = all. **Print-only** — saved `z_c` embeddings are unaffected. |

#### Example

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt \
    --mmd_sample_limit 256
```

---

### `biom3_ProteoScribe_sample` — Stage 3 sequence generation

Generates protein sequences from facilitated embeddings via diffusion sampling. Consumes the `.pt` output of `biom3_Facilitator_sample`.

**Source:** [src/biom3/Stage3/run_ProteoScribe_sample.py](../src/biom3/Stage3/run_ProteoScribe_sample.py)
**Config:** `configs/inference/stage3_ProteoScribe_sample.json`
**Deeper docs:** [sequence_generation_animation.md](misc/sequence_generation_animation.md) for `--animate_*` and `--animation_*` flags.

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_path` | str | Path to Stage 2 output embeddings (`.pt`). |
| `-c`, `--config_path` | str | Path to JSON config. |
| `-m`, `--model_path` | str | Path to pretrained ProteoScribe weights (`pytorch_model.bin`). |
| `-o`, `--output_path` | str | Path to write generated sequences (`.pt`). |

#### Optional arguments — sampling

| Arg | Type | Default | Description |
|---|---|---|---|
| `--seed` | int | 0 | RNG seed. |
| `--device` | str | `auto` | One of `auto`, `cpu`, `cuda`, `xpu`. `auto` = the detected backend: CUDA, then XPU, else CPU. |
| `--unmasking_order` | str | None | One of `random`, `confidence`, `confidence_no_pad`. Defaults to `random`. |
| `--token_strategy` | str | None | One of `sample` (Gumbel-max, default) or `argmax` (deterministic). |
| `--num_replicas` | int | None | Sequences to generate per prompt. Overrides the config's `num_replicas`; when neither sets it, 5. `None` means unset. |
| `--pre_unmask` | flag | False | Start diffusion from a partially-unmasked state. Requires `--pre_unmask_config`. |
| `--pre_unmask_config` | str | None | Path to JSON describing the pre-unmask strategy. |
| `--alpha` | float | 0.0 | Weight on `z_p` when conditioning: `y = alpha * z_p + (1 - alpha) * z_c`. `0` (default) is text-only. Anything above 0 requires `z_p` in `--input_path`, which Stage 1 emits and Stage 2 preserves. Only meaningful for a model trained with a matching blend — see [Conditioning blend](#conditioning-blend-alpha). |

#### Optional arguments — output

| Arg | Type | Default | Description |
|---|---|---|---|
| `--fasta` | flag | False | Write one FASTA file per prompt to `<output_dir>/fasta/`. |
| `--fasta_merge` | flag | False | Also write a single merged FASTA (requires `--fasta`). |
| `--fasta_dir` | str | None | Output directory for FASTA files. |
| `--store_probabilities` | flag | False | Save per-step probabilities as `.npz`. Memory-intensive. |

#### Optional arguments — animation

See [sequence_generation_animation.md](misc/sequence_generation_animation.md) for full details.

| Arg | Type | Default | Description |
|---|---|---|---|
| `--animate_prompts` | str (n+) | None | Prompt indices to animate (e.g. `0 1 2`), `all`, or `none`. |
| `--animate_replicas` | str | `1` | Replicas to animate: integer `i` = `range(0, i)`, `all`, or `none`. |
| `--animation_dir` | str | None | Output directory for GIFs. Default: `<output_dir>/animations/`. |
| `--animation_style` | str | `brightness` | One of `brightness`, `colorbar`, `logo`, `gauge`. |
| `--animation_metrics` | str (n*) | None | Per-position metric overlays (e.g. `confidence`). |

#### Example

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --seed 42 \
    --fasta
```

---

### `biom3_embedding_pipeline` — End-to-end embedding pipeline

Runs `biom3_PenCL_inference` → `biom3_Facilitator_sample` → HDF5 compilation in sequence. Intermediate paths are constructed automatically from `--output_dir` and `--prefix`.

**Source:** [src/biom3/pipeline/embedding_pipeline.py](../src/biom3/pipeline/embedding_pipeline.py)
**Config:** delegates to two configs (one per stage).
**Deeper doc:** [embedding_pipeline.md](misc/embedding_pipeline.md).

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_data_path` | str | Path to input CSV (sequences + prompts). |
| `-o`, `--output_dir` | str | Directory for all output files. |
| `--pencl_weights` | str | Path to PenCL weights or checkpoint. |
| `--facilitator_weights` | str | Path to Facilitator weights or checkpoint. |
| `--pencl_config` | str | Path to Stage 1 JSON config. |
| `--facilitator_config` | str | Path to Stage 2 JSON config. |
| `--prefix` | str | Filename prefix for intermediate and final output files, the run log (`<prefix>.run.log`), and the manifest (`<prefix>.build_manifest.json`). |

#### Optional arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--device` | str | `auto` | One of `auto`, `cpu`, `cuda`, `xpu`. `auto` = the detected backend: CUDA, then XPU, else CPU. |
| `--batch_size` | int | 256 | Stage 1 batch size. |
| `--num_workers` | int | 0 | Stage 1 DataLoader worker count. |
| `--no_amp` | flag | False | Forwarded to Stage 1: run the forward pass in fp32 instead of autocast. |
| `--float32_matmul_precision` | str | config (`high`) | Forwarded to Stage 1. Pair `highest` with `--no_amp` for a deterministic fp32 forward pass. |
| `--cross_comparison_sample_limit` | int | 0 | Forwarded to Stage 1. `0` = skip the O(n²) cross-comparison metrics (default), `-1` = all, positive = that many. **Print-only**. |
| `--mmd_sample_limit` | int | 1000 | Stage 2 MMD sample cap. |
| `--dataset_key` | str | `MMD_data` | HDF5 group name for the compiled output. |

#### Optional arguments — Stage 3 generation

With `--generate`, the terminal step is [`biom3_ProteoScribe_sample`](#biom3_proteoscribe_sample--stage-3-sequence-generation) instead of HDF5 compilation.

| Arg | Type | Default | Description |
|---|---|---|---|
| `--generate` | flag | False | Run Stage 3 sampling after Stage 2 instead of compiling to HDF5. |
| `--proteoscribe_weights` | str | None | ProteoScribe weights or checkpoint (overrides `--weight_set`). Required with `--generate`. |
| `--proteoscribe_config` | str | None | Stage 3 JSON config. Required with `--generate`. |
| `--seed` | int | 0 | Stage 3 sampling seed. `0` or less picks one at random. |
| `--no_fasta` | flag | False | Skip the FASTA output, which is on by default here. |
| `--token_strategy` | str | None | Forwarded to the sampler. |
| `--unmasking_order` | str | None | Forwarded to the sampler. |
| `--num_replicas` | int | None | Forwarded to the sampler. Unset defers to the Stage 3 config, else 5. |

---

## Training Entrypoints

All three training entrypoints share a similar argparser shape (assembled from `get_args`, `get_model_args`, `get_path_args`, `get_wrapper_args`) and read CLI < JSON < argparse defaults. Stringified booleans (e.g. `--wandb True`, `--scale_learning_rate False`) are converted via `str_to_bool` inside the script.

### `biom3_train_stage1` — Stage 1 PenCL training

Trains the joint protein/text encoder (PenCL) from a CSV dataset.

**Source:** [src/biom3/Stage1/run_PL_training.py](../src/biom3/Stage1/run_PL_training.py)
**Config dir:** `configs/stage1_training/`
**Deeper doc:** none yet — see [stage3_training.md](misc/stage3_training.md) for the analogous training conventions (output layout, checkpoint formats, metric history).

#### Required arguments

None positional. `--config_path` is technically optional but expected in practice.

#### Key arguments

The argparser declares ~50 flags. Highlights below; run `biom3_train_stage1 --help` for the complete list.

| Arg | Type | Default | Description |
|---|---|---|---|
| `--config_path`, `-c` | str | None | Path to JSON config. CLI overrides JSON. |
| `--run_id` | str | None | Unique identifier for this run; drives output directory naming. |
| `--data_path` | str | None | Path to Swiss-Prot CSV. |
| `--pfam_data_path` | str | `'None'` | Path to Pfam CSV (required when `dataset_type=pfam`). |
| `--dataset_type` | str | `default` | One of `default`, `masked`, `pfam`, `pfam_ablated`. |
| `--device` | str | `auto` | One of `auto`, `cpu`, `cuda`, `xpu`. `auto` = the detected GPU backend (CUDA, then XPU); it never falls back to CPU, so pass `cpu` to train on CPU. The run also stops early if `--devices_per_node` exceeds the devices this process can see. |
| `--devices_per_node` | int | 1 | GPUs (CUDA) or tiles (XPU) per node. (Deprecated alias: `--gpu_devices` — still accepted, emits a warning.) |
| `--num_nodes` | int | 1 | Nodes participating in training. |
| `--batch_size` | int | 8 | Per-device mini-batch size. |
| `--epochs` | int | 20 | Training epochs. |
| `--valid_size` | float | 0.1 | Train/val split fraction (0.1 = 90/10). |
| `--limit_val_batches` | float | 1.0 | See [Choosing limit_val_batches](misc/stage3_training.md#choosing-limit_val_batches-and-limit_train_batches). |
| `--limit_train_batches` | float | None | Same convention as above. |
| `--head_lr` / `--protein_encoder_lr` / `--text_encoder_lr` | float | varies | Per-component learning rates. |
| `--scale_learning_rate` | str | `'False'` | `'True'`/`'False'`. Scale LR by world size. |
| `--precision` | str | `'32'` | One of `'32'`, `'16'`, `'bf16'`, `'bf16-mixed'`. |
| `--uniformity_weight` | float | 0.0 | Weight of the Wang & Isola (2020) uniformity term on L2-normalized joint embeddings (pfam dataset types). 0 disables it. Logged as `{train,valid}_loss_unif_{protein,text}`. |
| `--uniformity_t` | float | 2.0 | Scale `t` of the uniformity term's Gaussian potential; its high-dimensional floor is about `-2t`. |
| `--uniformity_on` | str | `both` | `protein`, `text`, or `both` (mean of the two). |
| `--log_uniformity` | str | `'False'` | `'True'`/`'False'`. Measure and log the uniformity term without training on it, so a weight-0 control reports the same metric. Implied when the weight is > 0. |
| `--mask_same_sequence` | str | `'False'` | `'True'`/`'False'`. Drop candidates whose protein sequence is identical to the anchor's from the contrastive denominator. Swiss-Prot carries ~8.9 caption variants per accession, so ~5.6% of rows have such a false negative in the pooled batch. |
| `--mask_same_family` | str | `'False'` | `'True'`/`'False'`. Drop candidates drawn on the same Pfam family as the anchor. At M = 49,152 a row has ~25 of these, against the single `(i, i±N)` homolog the index rule already covers; for L_PFC they are the items it exists to pull together. Both flags default off so Run 1 stays reproducible. |
| `--resume_from_checkpoint` | str | `'None'` | Path to a Lightning `.ckpt` to resume from. |
| `--pretrained_weights` | str | `'None'` | Path to a raw weights file (no optimizer state). |
| `--wandb` | str | `'False'` | `'True'`/`'False'`. Enable wandb logging (requires `WANDB_API_KEY`). |
| `--output_root` | str | `./outputs/Stage1/pretraining` | Base output directory. |

#### Example (Aurora job submission via wrapper)

```bash
# In a PBS template
./scripts/stage1_train_singlenode.sh \
    configs/stage1_training/pretrain_pfam_v1.json \
    12 xpu my_run_001 \
    --epochs 20 --resume_from_checkpoint None --wandb ${use_wandb}
```

---

### `biom3_train_stage2` — Stage 2 Facilitator training

Trains the Facilitator that aligns text embeddings (`z_t`) to protein embeddings (`z_p`). Consumes Stage 1 output `.pt` dicts.

**Source:** [src/biom3/Stage2/run_PL_training.py](../src/biom3/Stage2/run_PL_training.py)
**Config dir:** `configs/stage2_training/`

#### Required arguments

None positional.

#### Key arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--config_path`, `-c` | str | None | Path to JSON config. CLI overrides JSON. |
| `--run_id` | str | None | Unique identifier for this run. |
| `--swissprot_data_path` | str | `'None'` | Path to Stage 1 SwissProt embeddings `.pt` dict. |
| `--pfam_data_path` | str | `'None'` | Path to Stage 1 Pfam embeddings `.pt` dict. |
| `--output_swissprot_dict_path` | str | None | Where to save Stage 2 SwissProt embeddings dict. |
| `--output_pfam_dict_path` | str | None | Where to save Stage 2 Pfam embeddings dict. |
| `--device` | str | `auto` | One of `auto`, `cpu`, `cuda`, `xpu`. `auto` = the detected GPU backend (CUDA, then XPU); it never falls back to CPU, so pass `cpu` to train on CPU. The run also stops early if `--devices_per_node` exceeds the devices this process can see. |
| `--devices_per_node` | int | 1 | GPUs/tiles per node. (Deprecated alias: `--gpu_devices`.) |
| `--num_nodes` | int | 1 | Nodes. |
| `--batch_size` | int | 32 | Per-device batch size. |
| `--epochs` | int | 20 | Training epochs. |
| `--valid_size` | float | 0.2 | Train/val split (0.2 = 80/20). |
| `--limit_val_batches` | float | 1.0 | See [Choosing limit_val_batches](misc/stage3_training.md#choosing-limit_val_batches-and-limit_train_batches). |
| `--lr` | float | 1e-3 | Base learning rate. |
| `--loss_type` | str | `MSE` | One of `MSE` (point-wise) or `MMD` (distribution-matching). |
| `--emb_dim` / `--hid_dim` | int | 512 / 1024 | Facilitator I/O and hidden dims. |
| `--wandb` | str | `'False'` | Enable wandb (requires `WANDB_API_KEY`). |

Run `biom3_train_stage2 --help` for the complete list.

---

### `biom3_train_stage3` — Stage 3 ProteoScribe training and finetuning

Trains the conditional diffusion transformer that generates protein sequences. Supports pretraining from scratch, secondary-data continuation, and selective-layer finetuning.

**Source:** [src/biom3/Stage3/run_PL_training.py](../src/biom3/Stage3/run_PL_training.py)
**Config dir:** `configs/stage3_training/`
**Deeper doc:** [stage3_training.md](misc/stage3_training.md) — output layout, metric definitions, checkpointing details, finetuning recipes, per-machine job templates.

#### Key arguments

The argparser is the largest in the project (70+ flags across `get_args`, `get_model_args`, `get_path_args`, `get_wrapper_args`). The most commonly-edited flags:

| Arg | Type | Default | Description |
|---|---|---|---|
| `--config_path` | str | None | Path to JSON config. |
| `--run_id` | str | None | Unique identifier for this run. |
| `--primary_data_path` | str | `'None'` | Path to primary HDF5 training dataset. |
| `--secondary_data_paths` | str (n+) | None | One or more secondary HDF5 dataset paths. |
| `--training_strategy` | str | `auto` | One of `auto`, `primary_only`, `combine`. |
| `--start_secondary` | str | `'False'` | `'True'`/`'False'`. Phase transition: load primary weights, then train on combined data. |
| `--train_alpha` | str | `zc` | Conditioning blend during training. See [Conditioning blend](#conditioning-blend-alpha). |
| `--eval_alpha` | str | `spread` | Conditioning blend for validation batches. |
| `--zp_path` | str | None | Facilitator `.pt` holding `z_p` row-aligned with `--primary_data_path`. Required when `--train_alpha` puts weight on `z_p`. |
| `--epochs` | int | 1 | Used in `primary_only` mode. |
| `--max_steps` | int | 100000 | Used in `combine` mode. |
| `--val_check_interval` | int | 10000 | Steps between validations (step-based mode). |
| `--limit_val_batches` | float | 200 | Cap val batches per check. See [Choosing limit_val_batches](misc/stage3_training.md#choosing-limit_val_batches-and-limit_train_batches). |
| `--limit_train_batches` | float | None | Same convention as above. |
| `--batch_size` | int | 16 | Per-device batch size. |
| `--lr` | float | 3e-4 | Base learning rate. |
| `--scale_learning_rate` | str | `'True'` | Scale LR by world size. |
| `--precision` | str | `no` | One of `no`, `fp16`, `bf16`, `32`. |
| `--device` | str | `auto` | One of `auto`, `cpu`, `cuda`, `xpu`. `auto` = the detected GPU backend (CUDA, then XPU); it never falls back to CPU, so pass `cpu` to train on CPU. The run also stops early if `--devices_per_node` exceeds the devices this process can see. |
| `--devices_per_node` | int | 1 | GPUs/tiles per node. (Deprecated alias: `--gpu_devices`.) |
| `--num_nodes` | int | 1 | Nodes. |
| `--distributed_strategy` | str | `deepspeed_zero2` | One of `deepspeed_zero2` (DeepSpeed ZeRO-2 + CPU offload, sharded checkpoint dir) or `ddp` (plain DDP with `static_graph=True`, single-file checkpoint). Distinct from `--training_strategy` which selects `primary_only` vs `combine` *data* mixing. |
| `--resume_from_checkpoint` | str | `'None'` | Path to a Lightning `.ckpt` to resume from. |
| `--pretrained_weights` | str | `'None'` | Path to raw weights to load before training. |
| `--finetune` | str | `'False'` | `'True'`/`'False'`. Enable finetuning mode. |
| `--finetune_last_n_blocks` | int | -2 | -1 = all, 0 = none, N = last N blocks. |
| `--finetune_last_n_layers` | int | -2 | Same convention as blocks. |
| `--finetune_output_layers` | str | `"True"` | Whether to unfreeze the transformer output layers. |
| `--checkpoint_every_n_steps` / `--checkpoint_every_n_epochs` | int | None | Periodic snapshot cadence (orthogonal to best-metric saves). |
| `--checkpoint_monitors` | JSON | None | List of `{metric, mode}` dicts for multi-metric checkpointing. |
| `--early_stopping_metric` | str | None | Metric to monitor (`'val_loss'`, etc.). None disables. |
| `--save_metrics_history` | str | `'True'` | Save MetricsHistoryCallback JSONL. |
| `--wandb` | str | `'False'` | Enable wandb (requires `WANDB_API_KEY`). |
| `--output_root` | str | None | Base output directory. |

Run `biom3_train_stage3 --help` for the full list, including model-architecture flags (`--diffusion_steps`, `--transformer_blocks`, etc.) and benchmark flags (`--save_benchmark`, `--benchmark_per_step`).

#### Conditioning blend (alpha)

ProteoScribe is conditioned on `z_c`, the text embedding from Stage 2. The blend instead conditions on

```
y = alpha * z_p + (1 - alpha) * z_c
```

where `z_p` is PenCL's protein-branch embedding of the sequence itself. Both live in the same joint space, so the convex combination is meaningful. `alpha` is always the weight on `z_p`: `alpha=0` is the default text-only behaviour, `alpha=1` conditions on the sequence alone. Training across a range of alpha teaches the model to accept either kind of conditioning at generation time, which `biom3_ProteoScribe_sample --alpha` then selects.

`--train_alpha` accepts:

| Value | Meaning |
|---|---|
| `zc` (default) | Text only — no `z_p` is loaded and nothing changes from a run without the flag. |
| `zp` | Sequence only. |
| `blend` | Per-example schedule: `alpha=1` w.p. .25, `alpha=0` w.p. .25, `U(0,1)` otherwise. |
| a number in `[0, 1]` | A constant blend. `0.5` means exactly 0.5, not the schedule. |

`--eval_alpha` takes the same values except `blend`, plus `spread` (the default): each validation example draws its own alpha deterministically from a hash of its sequence, so the metric covers the whole operating range while staying identical across epochs, DDP ranks, and batchings. Best-checkpoint selection then reflects the range rather than a single point. The `blend` schedule is rejected here because a resampled validation alpha makes val loss incomparable epoch to epoch.

The two entrypoints obtain `z_p` differently:

- `biom3_train_stage3` reads it from `--zp_path`, the Stage 2 Facilitator `.pt` that `biom3_compile_hdf5` built `--primary_data_path` from. Rows must correspond one-to-one; the row count and a sequence fingerprint are both checked at setup, so a `.pt` from a different Facilitator run fails loudly instead of pairing unrelated proteins. Blending requires a single HDF5 — secondary sources have no `z_p` of their own.
- `biom3_finetune_stage3` needs no `--zp_path`: it precomputes `z_p` for every unique train/val sequence through PenCL's frozen protein branch (`--pencl_weights`, batched by `--zp_batch_size`) and releases ESM-2 afterwards.

#### Example: pretrain from scratch

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id my_run_001 \
    --epochs 100
```

#### Example: finetune

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --run_id finetune_001 \
    --finetune True \
    --pretrained_weights /path/to/state_dict.best.pth \
    --finetune_last_n_blocks 4 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True
```

See [stage3_training.md](misc/stage3_training.md) for resumption, secondary-data continuation, and per-machine submission examples.

---

### `biom3_finetune_stage3` — Stage 3 finetuning on cleaned records

`biom3.Stage3.__main__:run_stage3_finetuning` → `src/biom3/Stage3/run_ProteoScribe_finetuning.py`

Finetunes ProteoScribe on a **JSONL dataset of cleaned records** rather than on precomputed `z_c` in HDF5 (which is what `biom3_train_stage3` consumes). For each record, a `--record_schema` composes a caption from the record's fields, which is embedded to `z_c` on-device through a frozen text→`z_c` front-end (PenCL text branch + Facilitator). The caption is re-composed every epoch, so `z_c` cannot be precomputed — hence the separate entrypoint.

This entrypoint is always finetuning: it loads pretrained ProteoScribe weights or resumes from a Lightning checkpoint, and freezes all but a chosen subset of the transformer. Trainer setup, callbacks, checkpointing, freezing, and arg coercion are reused from `run_PL_training`, so the shared arguments in [`biom3_train_stage3`](#biom3_train_stage3--stage-3-proteoscribe-training-and-finetuning) apply here too — including `--dry_run` and the wandb handling described below.

#### Key arguments — data and captions

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--finetune_data_path` | `None` | JSONL dataset of `{sequence, fields, sequence_length}` records |
| `--record_schema` | `None` | Schema composing a caption from record fields (per-key dropout, label-adding, shuffle, concatenate) |
| `--compose_plugins` | `None` | Extra caption-composition plugins (see `biom3.core.dataloaders`) |
| `--length_field` | `sequence_length` | Record key holding the sequence length |
| `--caption_key` | `caption` | Output key for the composed caption |
| `--sequence_output_key` | `sequence` | Record key holding the sequence |
| `--lazy_records` | `False` | Stream records instead of loading the dataset into memory |

#### Key arguments — frozen text→z_c front-end

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--stage1_config_path` | `None` | PenCL config for the frozen text encoder |
| `--stage2_config_path` | `None` | Facilitator config |
| `--pencl_weights` | `None` | PenCL weights |
| `--facilitator_weights` | `None` | Facilitator weights |
| `--zp_batch_size` | `64` | Batch size for `z_p` precomputation |

#### Key arguments — LoRA and conditioning

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--use_lora` | `False` | Enable LoRA instead of block/layer unfreezing |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--lora_target_patterns` | `.fn.to_q,.fn.to_v` | Module-name patterns to wrap with LoRA |
| `--lora_unfreeze_y_mlp` | `True` | Also unfreeze the conditioning MLP |
| `--train_alpha` | `zc` | Conditioning blend at train time. Shared with `biom3_train_stage3` — see [Conditioning blend](#conditioning-blend-alpha). |
| `--eval_alpha` | `spread` | Conditioning blend for validation batches. |
| `--zp_batch_size` | `64` | Batch size for the one-off `z_p` precompute (needs `--pencl_weights`). |

#### Example

```bash
biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_v1.json \
    --run_id my_finetune_run
```

With LoRA, using the prepared config:

```bash
biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_lora_v1.json \
    --run_id my_lora_run \
    --use_lora True \
    --lora_r 16
```

---

## Analysis Entrypoints

### `biom3_fit_manifold` — fit a latent manifold to a reference set

Fits a manifold to a reference cloud of embeddings and writes it as a single `.npz`, tagged with the method that produced it. Under the default method (`gaussian_shrinkage`: centroid + Ledoit-Wolf precision) fitting is the expensive half — a `D × D` shrinkage inverse — so do it once per reference set and score many query sets against the stored result.

**Source:** [src/biom3/geometry/run_fit_manifold.py](../src/biom3/geometry/run_fit_manifold.py)
**Deeper doc:** [misc/manifold_distance.md](misc/manifold_distance.md)

| Arg | Type | Default | Description |
|---|---|---|---|
| `--reference` | str | *required* | Path to an `(M, D)` reference matrix (`.npy` or `.npz`). `M ≥ 8` under the default method. |
| `--reference_key` | str | `None` | Array name to read when `--reference` is a multi-array `.npz`. |
| `--method` | str | `gaussian_shrinkage` | Manifold fitting method; choices come from the method registry. |
| `--label` | str | `""` | Free-form provenance string stored on the fitted manifold. |
| `-o`, `--output` | str | *required* | Path to write the fitted manifold (`.npz`; the extension is appended if absent). |

### `biom3_score_manifold` — score embeddings against a fitted manifold

Scores each query against a stored manifold; lower means closer to it. The method is read from the manifold file. Under the default method the score is a Mahalanobis distance in units of reference-cloud standard deviations.

**Source:** [src/biom3/geometry/run_score_manifold.py](../src/biom3/geometry/run_score_manifold.py)
**Deeper doc:** [misc/manifold_distance.md](misc/manifold_distance.md)

| Arg | Type | Default | Description |
|---|---|---|---|
| `--manifold` | str | *required* | Path to a manifold written by `biom3_fit_manifold`. |
| `--queries` | str | *required* | Path to an `(N, D)` query matrix (`.npy` or `.npz`). |
| `--queries_key` | str | `None` | Array name to read when `--queries` is a multi-array `.npz`. |
| `--ids` | str | `None` | Text file of query identifiers, one per line, in row order. |
| `--no_norm_check` | flag | off | Skip the query-vs-reference mean-norm sanity check (only for methods that perform one). |
| `-o`, `--output` | str | *required* | Output path: CSV (`id`, `score`, and `band` for methods that carry a reference band), or the raw `(N,)` array if it ends in `.npy`. |

```bash
biom3_fit_manifold \
    --reference zp_naturals.npy \
    --label "PenCL z_p run1_trackC_step187000 / Sho1 naturals" \
    -o manifold_sho1.npz

biom3_score_manifold \
    --manifold manifold_sho1.npz \
    --queries zp_designs.npy \
    --ids design_ids.txt \
    -o scores.csv
```

---

## Pre-flight dry-run (`--dry_run`)

All three training entrypoints (`biom3_train_stage{1,2,3}`) accept a `--dry_run` flag that parses everything (CLI + `--config_path` JSON + `_base_configs`/`_overwrite_configs` composition + argparse defaults), prints a side-effect-free pre-flight report, and exits without training.

The report has four sections:

1. **Effective configuration** — every arg with its source: `CLI`, `JSON: <path>` (the specific file in the composition chain), or `default`.
2. **Output paths the run would create** — `run_dir`, `logs_dir`, `artifacts_dir`, `checkpoint_dir`, `args.json`, `build_manifest.json`, `run.log` (resolved to absolute paths; nothing is created on disk).
3. **Distributed / batch math** — `num_nodes`, `devices_per_node`, `world_size`, `micro_batch_size`, `acc_grad_batches`, `effective_batch_size`, `train_dataset_len`, `val_dataset_len`, `batches_per_epoch_per_rank`, `steps_per_epoch`, plus stage-relevant fields (`epochs` or `max_steps`/`val_check_interval`).
4. **Memory estimate** — strategy-aware: full-replication (`ddp` / `single_device`) or DeepSpeed ZeRO-2 sharded (Stage 3 default). Reports `total_params`, `per_rank_param_gb`, `per_rank_grad_gb`, `per_rank_optimizer_gb`, and the `per_minibatch_input_gb` for one micro-batch on device. Activation memory is **not** included (requires a forward pass).

| Flag | Type | Description |
|---|---|---|
| `--dry_run True\|False` | str-bool | Enable the dry-run preview. Default `False`. |
| `--dry_run_output False\|True\|<path>` | str | Where to write `dry_run_report.json`. `False` (default): stdout only. `True`: write to `<artifacts_dir>/dry_run_report.json`. Any other string: treated as a filepath. |

```bash
# stdout only
biom3_train_stage3 --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id smoke_001 --dry_run True

# also persist the JSON report alongside the run's artifacts
biom3_train_stage3 ... --dry_run True --dry_run_output True

# write to an arbitrary path
biom3_train_stage3 ... --dry_run True --dry_run_output ./preflight.json
```

---

## Wandb handling (training entrypoints)

All three training entrypoints accept `--wandb True|False` (stringified bool). When invoked through the HPC wrappers ([scripts/stage{1,2,3}_train_{single,multi}node.sh](../scripts/)), the resolution rules in [scripts/_wandb_resolve.sh](../scripts/_wandb_resolve.sh) apply:

| Input | Result |
|---|---|
| `--wandb False` | wandb OFF (always honored) |
| `--wandb True` + `WANDB_API_KEY` set | wandb ON |
| `--wandb True` + no `WANDB_API_KEY` | wrapper errors out before exec |
| no `--wandb` + `WANDB_API_KEY` set | defaults ON |
| no `--wandb` + no `WANDB_API_KEY` | defaults OFF (warns) |

In job templates, edit the `use_wandb=True` variable near `epochs=` to flip wandb per job.

---

## Inspecting argparser output

The most reliable, always-current reference is the entrypoint's own help:

```bash
biom3_PenCL_inference --help
biom3_Facilitator_sample --help
biom3_ProteoScribe_sample --help
biom3_embedding_pipeline --help
biom3_train_stage1 --help
biom3_train_stage2 --help
biom3_train_stage3 --help
biom3_finetune_stage3 --help
```

If any table in this document drifts from `--help` output, the argparser is the source of truth.
