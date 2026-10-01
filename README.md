# BioM3 development

A working project developing and investigating the [BioM3 framework](https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1).

## Installation and setup

The `BioM3-dev` repo is available to clone from GitHub.

```bash
git clone https://github.com/ranganathanlab/BioM3-dev.git && cd BioM3-dev
```

### pip install

Install the core package:

```bash
# Editable (development) install
pip install -e .

# From GitHub (latest release on main)
pip install 'biom3 @ git+https://github.com/ranganathanlab/BioM3-dev.git'

# For a reproducible build, pin to a tag from https://github.com/ranganathanlab/BioM3-dev/tags
# pip install 'biom3 @ git+https://github.com/ranganathanlab/BioM3-dev.git@v0.1.0aN'
```

### Environment setup

**Important:** Before running tests or scripts, source the `environment.sh` file to set required
environment variables. The environment variables needed may differ across machines — see the
Usage section in each machine's setup doc for details.

```bash
source environment.sh
```

**Note:** *Some tests require pretrained weights that are too large to commit to git. These tests
are skipped automatically when the weights are absent. To run the full test suite, populate the
`weights/` directory using the shared weights sync script — see
[docs/setup/setup_shared_weights.md](./docs/setup/setup_shared_weights.md) for machine-specific paths, the
list of required files, and setup instructions.*

For installation and setup instructions on the following machines, refer to the setup instructions located in the `docs/` folder.

| Machine | Instructions |
| ------- | ------------ |
| Polaris (ALCF) | [setup_polaris.md](./docs/setup/setup_polaris.md) |
| Polaris (ALCF), container | [setup_polaris_container.md](./docs/setup/setup_polaris_container.md) |
| Aurora (ALCF) | [setup_aurora.md](./docs/setup/setup_aurora.md) |
| Aurora (ALCF), container | [setup_aurora_container.md](./docs/setup/setup_aurora_container.md) |
| DGX Spark | [setup_spark.md](./docs/setup/setup_spark.md) |
| Docker | [setup_docker.md](./docs/setup/setup_docker.md) |

To run BioM3 from the published container image without cloning this repository at all —
install Docker, fetch the weights, embed and generate — see
[docs/setup/user_quickstart.md](./docs/setup/user_quickstart.md).

## Usage

After the pip installation, a number of entrypoints should be available from the command line. These include scripts to run Stages 1, 2, and 3 in inference mode, training entrypoints for all three stages, and Stage 3 finetuning.

> **CLI reference:** see [docs/CLI_reference.md](./docs/CLI_reference.md) for the full per-entrypoint argument tables. The walkthroughs below show common invocations; the reference covers the complete argument surface.

### End-to-end inference pipeline

The three inference stages form a sequential pipeline. The output of each stage feeds into the next:

```txt
Input text prompts / protein sequences
        │
        ▼
biom3_PenCL_inference         → outputs/pencl_embeddings.pt       (z_t, z_p)
        │
        ▼
biom3_Facilitator_sample      → outputs/facilitator_embeddings.pt  (z_t, z_p, z_c)
        │
        ▼
biom3_ProteoScribe_sample     → outputs/generated_sequences.pt
```

### Configuration files

Each inference entrypoint takes a JSON config file from `configs/inference/` (e.g. `configs/inference/stage1_PenCL.json`) that controls model hyperparameters and paths to backbone LLM weights.
Training uses JSON configs too, one directory per stage: `configs/stage1_training/`, `configs/stage2_training/`, and `configs/stage3_training/`. Each directory carries `models/` (shared architecture bases) and `machines/` (per-machine device settings), composed via the `_base_configs` / `_overwrite_configs` keys. The wrapper scripts in `scripts/` pass these configs through to the entrypoints; CLI arguments override JSON values, which override argparse defaults.

### Stage 1 (inference)

Run PenCL inference from the entrypoint `biom3_PenCL_inference`, which accesses the script `src/biom3/Stage1/run_PenCL_inference.py`.
This stage encodes protein sequences and text descriptions into a shared latent space using ESM2 (protein) and BiomedBERT (text) encoders.

**Preparation:** Edit `configs/inference/stage1_PenCL.json` and set `seq_model_path` and `text_model_path` to the paths of the downloaded LLM weights.

```json
"seq_model_path": "/path/to/weights/LLMs/esm2_t33_650M_UR50D.pt",
"text_model_path": "/path/to/weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
```

**Arguments:** see [docs/CLI_reference.md#biom3_pencl_inference--stage-1-pencl-inference](./docs/CLI_reference.md#biom3_pencl_inference--stage-1-pencl-inference).

#### Example: using the built-in test dataset with raw weights

```bash
biom3_PenCL_inference \
    --input_data_path None \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt
```

#### Example: using a custom CSV input, larger batch size, and CPU

```bash
biom3_PenCL_inference \
    --input_data_path data/my_proteins.csv \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt \
    --device cpu \
    --batch_size 16 \
    --num_workers 4
```

#### Example: loading from a Lightning checkpoint

```bash
biom3_PenCL_inference \
    --input_data_path None \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.ckpt \
    --output_path outputs/pencl_embeddings.pt
```

The `.ckpt` extension is detected automatically; alternatively, use `--load_from_checkpoint` to force checkpoint loading regardless of extension.

### Stage 2 (inference)

Run Facilitator sampling from the entrypoint `biom3_Facilitator_sample`, which accesses the script `src/biom3/Stage2/run_Facilitator_sample.py`.
This stage maps text embeddings (`z_t`) into the protein embedding distribution (`z_c`) using a learned MMD-based alignment model.

**Arguments:** see [docs/CLI_reference.md#biom3_facilitator_sample--stage-2-facilitator-sampling](./docs/CLI_reference.md#biom3_facilitator_sample--stage-2-facilitator-sampling).

#### Example: standard usage following Stage 1

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt
```

#### Example: limiting MMD computation to 256 samples (useful for large datasets)

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt \
    --device cpu \
    --mmd_sample_limit 256
```

### Stage 3

#### Pretraining

The entrypoint `biom3_train_stage3` (script `src/biom3/Stage3/run_PL_training.py`) handles both pretraining and finetuning of ProteoScribe.
Training configuration is specified via a JSON config file passed with `--config_path`, with per-job overrides (device, number of nodes, run ID, etc.) passed as CLI arguments.
CLI arguments override JSON values, which override argparse defaults.

Example JSON configs are in `configs/stage3_training/`.
Configs support layered composition: `_base_configs` for shared model architectures (`configs/stage3_training/models/`) and `_overwrite_configs` for per-machine settings (`configs/stage3_training/machines/`). See [docs/misc/stage3_training.md](./docs/misc/stage3_training.md) for details.
Wrapper scripts `scripts/stage3_train_multinode.sh` (multi-node via `mpiexec`) and `scripts/stage3_train_singlenode.sh` (single-node) handle environment setup and launch the entrypoint.
HPC job templates in `jobs/{polaris,aurora,spark}/` demonstrate how to use these wrappers.

```bash
# Direct usage
biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id my_run_v1 \
    --epochs 10

# Via multinode wrapper (from an HPC job template). Wandb is auto-resolved
# from the WANDB_API_KEY env var; pass `--wandb True|False` to override.
./scripts/stage3_train_multinode.sh \
    configs/stage3_training/pretrain_scratch_v2.json \
    2 4 auto my_run_v1 \
    --epochs 10 --wandb True
```

#### Finetuning

There are two finetuning paths, differing in what the model is finetuned *on*.

**On precomputed `z_c` embeddings (HDF5)** — use `biom3_train_stage3` with `--finetune True`, plus flags for the pretrained weights and which transformer blocks/layers to unfreeze.
Example configs: `configs/stage3_training/finetune_v1.json`, `finetune_v2.json`.

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --run_id finetune_v1 \
    --pretrained_weights ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1
```

**On cleaned records (JSONL)** — use `biom3_finetune_stage3` (script `src/biom3/Stage3/run_ProteoScribe_finetuning.py`).
This finetunes directly on a JSONL dataset of `{sequence, fields, sequence_length}` records. A `--record_schema` composes the caption from the record's fields each epoch (per-key dropout, shuffling, label-adding), which is then embedded to `z_c` on-device through a frozen text→`z_c` front-end (PenCL text branch + Facilitator). Because the caption is re-composed every epoch, `z_c` cannot be precomputed — that is the reason for the separate entrypoint.
It always loads pretrained ProteoScribe weights (or resumes from a Lightning checkpoint) and supports LoRA via `--use_lora True`.
Example configs: `configs/stage3_training/finetune_generalized_v1.json`, `finetune_generalized_lora_v1.json`.

```bash
biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_v1.json \
    --run_id my_finetune_run
```

#### Inference (Generation)

Run ProteoScribe sampling from the entrypoint `biom3_ProteoScribe_sample`, which accesses the script `src/biom3/Stage3/run_ProteoScribe_sample.py`.
This stage generates protein sequences from the facilitated text embeddings (`z_c`) produced by Stage 2, using the conditional diffusion transformer.

**Arguments:** see [docs/CLI_reference.md#biom3_proteoscribe_sample--stage-3-sequence-generation](./docs/CLI_reference.md#biom3_proteoscribe_sample--stage-3-sequence-generation). Animation, FASTA, and pre-unmask flags are documented there as well.

> **Note:** The number of sequences per prompt is set by `--num_replicas`, which overrides `num_replicas` in the JSON config (default 5 when neither sets it). Batch size and diffusion steps are set by `batch_size_sample` and `diffusion_steps` in the JSON config.

#### Example: standard usage following Stage 2

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt
```

#### Example: reproducible run with a fixed seed

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --seed 42
```

#### Example: animate the denoising process for selected prompts

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --animate_prompts 0 1 2 \
    --animate_replicas 2
```

GIFs are written to `outputs/animations/prompt_<P>_replica_<R>.gif`. See [docs/misc/sequence_generation_animation.md](./docs/misc/sequence_generation_animation.md) for details.

## Contributing

Contributions from both internal collaborators and external contributors are welcome. New work is branched off `dev` and merged back into `dev` via pull request — `main` is reserved for tagged releases.

See [docs/contributing.md](./docs/contributing.md) for the full workflow: forking and cloning, creating a personal branch from `dev`, commit conventions, and opening a pull request.

## References

[1] Natural Language Prompts Guide the Design of Novel Functional Protein Sequences. Nikša Praljak, Hugh Yeh, Miranda Moore, Michael Socolich, Rama Ranganathan, Andrew L. Ferguson. bioRxiv 2024.11.11.622734; doi: [10.1101/2024.11.11.622734](https://doi.org/10.1101/2024.11.11.622734)
