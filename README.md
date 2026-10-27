# BioM3 development

A working project developing and investigating the [BioM3 framework](https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1), which generates protein sequences from natural language prompts in three stages: PenCL (Stage 1), Facilitator (Stage 2), and ProteoScribe (Stage 3).

## Documentation

| Topic | Doc |
| ----- | ---- |
| Setup and usage | [User guide](./docs/USER_GUIDE.md) |
| Containerized workflows | [Container quickstart](./docs/setup/user_quickstart.md) |
| Command Line Reference | [CLI reference](./docs/CLI_reference.md) |
| Developer topics | [Dev guide](./docs/DEV_GUIDE.md) |

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

# With the Streamlit web app's dependencies
pip install -e '.[app]'

# From GitHub (latest release on main)
pip install 'biom3 @ git+https://github.com/ranganathanlab/BioM3-dev.git'

# For a reproducible build, pin to a tag from https://github.com/ranganathanlab/BioM3-dev/tags
# pip install 'biom3 @ git+https://github.com/ranganathanlab/BioM3-dev.git@v0.1.0aN'
```

### Environment setup

Before running tests or scripts, source the `environment.sh` file to set required environment variables.

```bash
source environment.sh
```

The setup differs across machines. Refer to the instructions for yours:

| Machine | Instructions |
| ------- | ------------ |
| Polaris (ALCF) | [setup_polaris.md](./docs/setup/setup_polaris.md) |
| Polaris (ALCF), container | [setup_polaris_container.md](./docs/setup/setup_polaris_container.md) |
| Aurora (ALCF) | [setup_aurora.md](./docs/setup/setup_aurora.md) |
| Aurora (ALCF), container | [setup_aurora_container.md](./docs/setup/setup_aurora_container.md) |
| DGX Spark | [setup_spark.md](./docs/setup/setup_spark.md) |
| Docker | [setup_docker.md](./docs/setup/setup_docker.md) |

## Usage

After installation, the `biom3` command and the `biom3_*` entrypoints are available from the command line. The [user guide](./docs/USER_GUIDE.md) walks through fetching weights, embedding prompts, generating sequences, and finetuning ProteoScribe. The [CLI reference](./docs/CLI_reference.md) lists every entrypoint and its arguments.

For topics the user guide does not cover:

| Topic | Where |
| ----- | ----- |
| The `biom3` command and what each subcommand runs | [CLI reference](./docs/CLI_reference.md#the-biom3-command) |
| Running Stage 1, 2, or 3 individually | [CLI reference](./docs/CLI_reference.md#inference-entrypoints) |
| Stage 3 training from scratch, config composition, and job templates | [stage3_training.md](./docs/misc/stage3_training.md) |
| Finetuning on JSONL records with `biom3_finetune_stage3` | [CLI reference](./docs/CLI_reference.md#biom3_finetune_stage3--stage-3-finetuning-on-cleaned-records) |
| Animating the generation process | [sequence_generation_animation.md](./docs/misc/sequence_generation_animation.md) |
| Shared weights on Polaris, Aurora, and DGX Spark | [setup_shared_weights.md](./docs/setup/setup_shared_weights.md) |

## Contributing

Contributions from both internal collaborators and external contributors are welcome. New work is branched off `dev` and merged back into `dev` via pull request — `main` is reserved for tagged releases.

See [docs/contributing.md](./docs/contributing.md) for the full workflow: forking and cloning, creating a personal branch from `dev`, commit conventions, and opening a pull request.

## References

[1] Natural Language Prompts Guide the Design of Novel Functional Protein Sequences. Nikša Praljak, Hugh Yeh, Miranda Moore, Michael Socolich, Rama Ranganathan, Andrew L. Ferguson. bioRxiv 2024.11.11.622734; doi: [10.1101/2024.11.11.622734](https://doi.org/10.1101/2024.11.11.622734)
