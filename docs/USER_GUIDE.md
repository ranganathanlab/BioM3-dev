# BioM3 User Guide

## Contents

- [About](#about)
- [Quickstart instructions](#quickstart-instructions)
  - [Case 1: Using BioM3 through a repo checkout](#case-1-using-biom3-through-a-repo-checkout)
    - [Embedding and generation workflow](#embedding-and-generation-workflow)
  - [Case 2: Using BioM3 via Docker](#case-2-using-biom3-via-docker)
    - [Embedding and generation workflow](#embedding-and-generation-workflow-1)
  - [Case 3: Using BioM3 on HPC environments through Apptainer](#case-3-using-biom3-on-hpc-environments-through-apptainer)
    - [Embedding and generation workflow](#embedding-and-generation-workflow-2)
- [Detailed Usage Instructions](#detailed-usage-instructions)
  - [Finetuning ProteoScribe](#finetuning-proteoscribe)
    - [On a single CUDA device, from a code checkout](#on-a-single-cuda-device-from-a-code-checkout)
    - [On a single CUDA device, from a Docker image](#on-a-single-cuda-device-from-a-docker-image)
    - [On Aurora, from a code checkout](#on-aurora-from-a-code-checkout)
    - [On Aurora, using an Apptainer image](#on-aurora-using-an-apptainer-image)

## About

BioM3 is a multimodal biological model that is capable of generating functional protein sequences from text prompts.
It works in three stages, each a distinct module: PenCL (Stage 1), Facilitator (Stage 2), and ProteoScribe (Stage 3).
The method is described in [Natural Language Prompts Guide the Design of Novel Functional
Protein Sequences](https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1).

This guide covers basic setup and usage of BioM3.

## Quickstart instructions

Setup instructions vary depending on the particular use case and computing environment. For basic workflows, we recommend using the published BioM3 container images via Docker. Alternatively, one can clone the repository, create a working conda environment, and run BioM3 commands with the installed package. Finally, for use of BioM3 on HPC environments such as Midway (UChicago) and Aurora (ALCF) we provide additional instructions for the use of Apptainer in place of Docker.

### Case 1: Using BioM3 through a repo checkout

Start by cloning the repository. 

```bash
git clone https://github.com/ranganathanlab/BioM3-dev.git && cd BioM3-dev
```

We recommend the use of conda for managing python environments, however, venv may be required in some cases, such as on HPC clusters like Aurora.
A conda environment named `biom3-env` stored under `venvs/` is the convention used here.
Refer to the per-machine setup guides under `docs/setup/` for machine-specific installation commands.

Once you have installed `biom3` into your `biom3-env` environment, verify that the installation is working by running a short suite of tests. Note that you must first source the `environment.sh` file.

```bash
source environment.sh
python -m pytest tests --quick
```

Next, fetch the current set of model weights with the following command.

```bash
biom3_fetch_weights run1_base -o weights
```

The result is the following expected layout:

```
weights/
├── Facilitator/run1_base_facilitator.bin
├── LLMs/
│   ├── BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/
│   └── esm2_t33_650M_UR50D.pt
├── PenCL/run1_base_pencl.bin
└── ProteoScribe/run1_base_proteoscribe.bin
```

This command may take some time, as it downloads multiple gigabytes of model weights.
This will populate the `weights` directory with the necessary weights for each module.
With these weights, one can use BioM3 to embed protein sequences and text prompts, and then generate new sequences from those embeddings.

#### Embedding and generation workflow

To begin, create a csv file containing the text prompts and sequences that you want to embed. The csv should contain the header: `primary_Accession,protein_sequence,[final]text_caption`. You may use any value for the ID/accession field. Take care to wrap text captions in double quotes, in case the caption includes commas.

Note that the embeddings of sequence and text are independent processes. However, BioM3 currently requires both fields to be given. If you wish to only embed a sequence, or only embed a text caption, you may enter a dummy string (e.g. `"AAAA"` or `"Dummy text"` in the corresponding entry. Presently, an empty string input will raise an error. In the case of a dummy input, an embedding of the nonsense string will still be produced, and one should take care not to make use of those embeddings in downstream tasks.

```bash
cat > data/prompts.csv <<'EOF'
primary_Accession,protein_sequence,[final]text_caption
P69222,MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR,"PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm."
my_prompt_1,"AAAA","PROTEIN NAME: SH3 domain protein. FUNCTION: Small adaptor module that binds proline-rich motifs and mediates protein-protein interactions in signal transduction. SUBCELLULAR LOCATION: Cytoplasm."
EOF
```

We will now apply the embedding half of BioM3 (Stages 1 and 2) to transform the input text and captions into vector representations in a joint embedding space. We specify the use of the "Run 1" weights that we fetched above with the `--weight_set` argument, which points to a configuration file specifying the particular set of model weights to use for each stage. Both stages also require a configuration file specifying the particular hyperparameters of each module. Prebuilt configuration files are included in the checkout, and referred to in the command below.

```bash
biom3_embedding_pipeline \
    -i data/prompts.csv \
    -o outputs/demo --prefix example \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json
```

The embedding pipeline first transforms the input text captions into a vector $z_t$ and the protein sequence into a vector $z_p$ (Stage 1). Then, the Facilitator module further refines the text representation $z_t$ into a "refined" text embedding $z_c$ (Stage 2). The result of the command above is a directory `outputs/demo` containing the following:

```text
outputs/demo/
├── example.PenCL_emb.pt          # Stage 1 output: z_t and z_p
├── example.Facilitator_emb.pt    # Stage 2 output: z_t, z_p, and z_c
├── example.compiled_emb.hdf5     # the same embeddings packaged into hdf5 format
├── example.build_manifest.json   # the exact arguments, weights, and configs used
└── example.run.log               # the run's console output
```

The filenames come from `--prefix`, and everything populates under the `--output_dir`. Note that the Stage 2 file is a superset of the Stage 1 file; the Facilitator adds `z_c` to the dictionary it was given.

After running the embedding half of BioM3, one can then use the refined text embeddings, $z_c$, to condition the Stage 3 module ProteoScribe and generate novel sequences. The command below takes as input the refined embeddings, and generates a specified number of sequences (`--num_replicas` argument; default 5) for each individual embedding (i.e. text caption). Here again, we specify the weight file to use as well as a configuration file.

```bash
biom3_ProteoScribe_sample \
    -i outputs/demo/example.Facilitator_emb.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/demo/generation/generated.pt --fasta --num_replicas 5
```

Results populate under `outputs/demo/generation`. The results file `generated.pt` stores the generated sequences by prompt. Results can be loaded and viewed in an interactive python session as follows:

```python
import torch
results = torch.load("outputs/demo/generation/generated.pt")
prompt0_sequences = results["prompt_0"]
for i, s in enumerate(prompt0_sequences):
    print(f"Replicate {i}:", s)
```

By including the `--fasta` argument, per-prompt fasta files are produced under the `fasta` subdirectory.

### Case 2: Using BioM3 via Docker

Docker images can be thought of as self-contained software packages. One simply needs to "pull" the image from a public registry and use Docker to run it, as what is called a "container." The advantage of this approach is that one doesn't need to create a brand new conda or virtual environment. The full set of dependencies is specified in the image and are downloaded the first time the container is started.

To use Docker, one must first install it on their machine. Instructions can be found and followed online.

<!-- TODO: Eventually need to change this to ranganathanlaba -->
BioM3 Docker images are published at `ghcr.io/natural-machine/biom3`. There are four variants:

| Variant | Use case | Architectures |
| ------- | --- | ------------- |
| `cuda` | Workstations with NVIDIA GPUs | amd64, arm64 |
| `cpu` | Inference without a GPU.  | amd64, arm64 |
| `xpu` | ALCF Aurora, single node | amd64 |
| `xpu-oneapi` | ALCF Aurora, multi-node | amd64 |

Tags are patterned `<variant>-<commit>`, where the commit specifies the particular state of the BioM3 code used to create the image. As new versions of BioM3 are released, these images will be updated. To pull an image onto your workstation, specify a particular tag and run:

```bash
docker pull ghcr.io/natural-machine/biom3:cuda-2bf065b
```

The image carries the code, standard configuration files under `configs/`, and test
fixtures under `tests/`. It does not carry weights or data. Start by creating these directories.

```bash
cd my-project
mkdir -p weights data outputs
```

Now, fetch the current set of model weights with the following. We first point to the image that we previously pulled. Docker runs the specified biom3 command inside of this container and stops the container when the command finished (`--rm`). Docker typically runs as root, so we specify our user profile explicitly (`-u "$(id -u):$(id -g)"`). Finally, we mount the local weights directory that we created above, which Docker will have access to with the path to the right of the colon (`"$PWD/weights:/app/weights"`).

```bash
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cuda-2bf065b
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/weights:/app/weights" \
    $BIOM3_IMAGE biom3_fetch_weights run1_base -o /app/weights

```

This command may take some time, as it downloads multiple gigabytes of model weights.
This will populate the `weights` directory with the necessary weights for each module.
With these weights, one can use BioM3 to embed protein sequences and text prompts, and then generate new sequences from those embeddings.

#### Embedding and generation workflow

The general workflow to embed protein sequences and captions, and to then generate novel proteins from those embeddings, is detailed above under [Case 1: Embedding and generation workflow](#embedding-and-generation-workflow). Every `biom3` command in that section runs unchanged inside the container — only the `docker run` prefix and the mounted paths differ.

Refer to the previous section for conceptual details. The equivalent commands, run using Docker, are provided below.

The mount points matter here. The container's working directory is `/app`, and both the weight-set bundle and the `biom3` arguments use paths relative to it, so `weights/`, `data/` and `outputs/` have to land at `/app/weights`, `/app/data` and `/app/outputs` for those relative paths to resolve. Results appear under `outputs/demo/`, owned by the user rather than by root because of `-u`. Drop `--gpus all` when using the `cpu` image.

```bash
# Name the image once, so the commands below can refer to it.
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cuda-2bf065b

# Make sure directories exist before mounting them, otherwise Docker creates missing ones as root.
mkdir -p data outputs

# Write data/prompts.csv on the host, exactly as in Case 1 above.
cat > data/prompts.csv <<'EOF'
primary_Accession,protein_sequence,[final]text_caption
P69222,MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR,"PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm."
my_prompt_1,"AAAA","PROTEIN NAME: SH3 domain protein. FUNCTION: Small adaptor module that binds proline-rich motifs and mediates protein-protein interactions in signal transduction. SUBCELLULAR LOCATION: Cytoplasm."
EOF

# Stages 1 and 2: embed the captions and sequences into z_t, z_p, and z_c.
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/data:/app/data:ro" \
    -v "$PWD/outputs:/app/outputs" \
    $BIOM3_IMAGE \
    biom3_embedding_pipeline \
        -i data/prompts.csv \
        -o outputs/demo --prefix example \
        --weight_set configs/weights/run1_base.json \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json

# Stage 3: generate novel sequences conditioned on the refined embeddings z_c.
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/outputs:/app/outputs" \
    $BIOM3_IMAGE \
    biom3_ProteoScribe_sample \
        -i outputs/demo/example.Facilitator_emb.pt \
        -c configs/inference/stage3_ProteoScribe_sample.json \
        -m weights/ProteoScribe/run1_base_proteoscribe.bin \
        -o outputs/demo/generation/generated.pt --fasta --num_replicas 5
```

### Case 3: Using BioM3 on HPC environments through Apptainer

Many HPC environments do not permit `docker run` because it runs as root. They instead use Apptainer (formerly Singularity). The idea is largely the same as using Docker images, and we start by converting a Docker image into an apptainer compatible `.sif` file.

In this guide, we provide instructions specific to running BioM3 on ALCF's supercomputer Aurora, through Apptainer. 
Start by creating a .sif file from a specific BioM3 docker image. For working on Aurora, we use the `xpu-oneapi` image version. 

Full instructions and troubleshooting hints for building on both Aurora and Polaris are included in the files

- [setup/setup_polaris_container.md](./setup/setup_polaris_container.md)
- [setup/setup_aurora_container.md](./setup/setup_aurora_container.md)

"Bare-metal" installs (i.e. running from code installed into a local environment) are covered in [setup/setup_polaris.md](./setup/setup_polaris.md) and [setup/setup_aurora.md](./setup/setup_aurora.md).

For the following workflow, a clone of the BioM3 repo is required. Clone the repo.

```bash
git clone https://github.com/ranganathanlab/BioM3-dev.git && cd BioM3-dev
```

Next, we create a .sif image from a Docker image. 

```bash
# On a login node of Aurora
module load apptainer
export APPTAINER_CACHEDIR=/flare/NLDesignProtein/$USER/.apptainer/cache
export APPTAINER_TMPDIR=/tmp/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build biom3_xpu-oneapi-2bf065b.sif docker://ghcr.io/natural-machine/biom3:xpu-oneapi-2bf065b
```

This command should produce a file `biom3_xpu-oneapi-2bf065b.sif` under the current directory. Whereas the Docker commands above point to a specified Docker image, the following commands instead point to the .sif image.
Importantly, running `apptainer exec` on Aurora requires that one be on a compute node. As such, request an interactive job or submit a pbs job to run the following. Also note that one must load apptainer.

```bash
# On a compute node of Aurora
cd /path/to/BioM3-dev
module load apptainer

export BIOM3_IMAGE="biom3_xpu-oneapi-2bf065b.sif"
mkdir -p weights
apptainer exec --bind "$PWD/weights:/app/weights" \
    "$BIOM3_IMAGE" biom3_fetch_weights run1_base -o /app/weights
```

Three differences from the Docker form: Apptainer already runs as you rather than as root,
so there is no `-u` and no ownership problem to correct. `--bind host:container` replaces
`-v`, with the same `/app/weights` target, since the bundle's paths still resolve against
the image's `/app` working directory. And `exec` runs the command directly instead of
through the image entrypoint. Create `weights/` yourself beforehand: Apptainer will not
create a missing bind source for you.

This command may take some time, as it downloads multiple gigabytes of model weights.
This will populate the `weights` directory with the necessary weights for each module.
With these weights, one can use BioM3 to embed protein sequences and text prompts, and then generate new sequences from those embeddings.

#### Embedding and generation workflow

The general workflow to embed protein sequences and captions, and to then generate novel proteins from those embeddings, is detailed above under [Case 1: Embedding and generation workflow](#embedding-and-generation-workflow). Every `biom3` command in that section runs unchanged inside the container — only the `apptainer exec` prefix and the bound paths differ.

Refer to that section for conceptual details. The equivalent commands, run using apptainer, are provided below. As with the weights fetch, these commands must be run from a compute node.

```bash
# On a compute node of Aurora, from the root of the checkout.

cd /path/to/BioM3-dev

module load apptainer
export BIOM3_IMAGE="$PWD/biom3_xpu-oneapi-2bf065b.sif"

# The wrapper mounts data/ and outputs/ by name, so they have to exist.
mkdir -p data outputs

# Write data/prompts.csv on the host, exactly as in Case 1 above.
cat > data/prompts.csv <<'EOF'
primary_Accession,protein_sequence,[final]text_caption
P69222,MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR,"PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm."
my_prompt_1,"AAAA","PROTEIN NAME: SH3 domain protein. FUNCTION: Small adaptor module that binds proline-rich motifs and mediates protein-protein interactions in signal transduction. SUBCELLULAR LOCATION: Cytoplasm."
EOF

# Stages 1 and 2: embed the captions and sequences into z_t, z_p, and z_c.
scripts/aurora/apptainer_run.sh biom3_embedding_pipeline \
    -i data/prompts.csv \
    -o outputs/demo --prefix example \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json

# Stage 3: generate novel sequences conditioned on the refined embeddings z_c.
scripts/aurora/apptainer_run.sh biom3_ProteoScribe_sample \
    -i outputs/demo/example.Facilitator_emb.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/demo/generation/generated.pt --fasta  --num_replicas 5
```

## Detailed Usage Instructions

### Finetuning ProteoScribe

The Stage 3 module, ProteoScribe, is an order-agnostic autoregressive diffusion model (ARDM) that can be conditioned on a text prompt and used to generate protein sequences. The base model weights of ProteoScribe (included in the `run1_base` bundle) were tuned through a training process in which the ProteoScribe model saw sequence-text pairs across a broad set of protein families. In order to generate high-quality sequences specific to a particular protein family, we find that it is necessary to finetune ProteoScribe on a per-family basis.

The finetuning process begins with the curation and construction of a suitable finetuning dataset. One should first assemble a collection of protein sequence-caption pairs, for example from the SwissProt or Pfam databases. Create a csv file with the header `primary_Accession,protein_sequence,[final]text_caption`. The accession field is arbitrary, but should contain string values. The sequence and text columns should contain the protein sequences and text captions, respectively, wrapped in double quotes in case of commas.

As a worked example, this section uses a small published dataset of green fluorescent protein (GFP) sequences: 219 sequence-caption pairs from the Pfam family PF01353. Fetch it into `data/` with

```bash
biom3_fetch_dataset gfp_demo -o data
```

This writes `data/gfp_sample_dataset.csv`, along with `data/gfp_sample_dataset.NOTICE.md`. To finetune on your own family instead, substitute your csv file in the commands below.

Start by running the embedding pipeline (Stages 1 and 2) with the family csv as input, as documented above, following the specific instructions for your particular use case. In the commands below, we assume that outputs are directed to `outputs/ft_embeddings/`:

```bash
biom3_embedding_pipeline \
    -i data/gfp_sample_dataset.csv \
    -o outputs/ft_embeddings --prefix gfp_demo \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json
```

This should populate the `outputs/ft_embeddings/` directory with a number of .pt files, as well as a compiled .hdf5 file, `gfp_demo.compiled_emb.hdf5`. This file will serve as the direct input for finetuning. The generic command to run the finetuning entrypoint is shown below, with the essential arguments described. Variations of this command may be used on different machines, detailed below.

The finetuning entrypoint is run via the `biom3_train_stage3` command, with the argument `--finetune True`.

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --finetune True \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True \
    --primary_data_path outputs/ft_embeddings/gfp_demo.compiled_emb.hdf5 \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root outputs/gfp_ft_results \
    --run_id gfp_ft001 \
    --device cuda --num_nodes 1 --devices_per_node 1 \
    --distributed_strategy ddp \
    --epochs 3 --batch_size 16 --wandb False
```

**Description of arguments:**

* `--config_path`: The training configuration file. This file may contain default values for the arguments below, which are overwritten by arguments passed through the command line.
* `--finetune`: Must be `True` to run the finetuning path.
* `--finetune_last_n_blocks`: How many of the final transformer blocks to unfreeze. `-1` unfreezes all blocks, `0` none.
* `--finetune_last_n_layers`: How many layers within each unfrozen block to train. `-1` unfreezes all layers of those blocks, `0` none.
* `--finetune_output_layers`: Whether to also unfreeze the final norm and output layer. Together these three arguments set how much of the model adapts; the run log reports the resulting trainable and frozen parameter counts.
* `--primary_data_path`: Path to the hdf5 file containing embedded data from Stage 2.
* `--pretrained_weights`: Initial ProteoScribe weights to finetune from.
* `--output_root`: The output directory into which all finetuning results will populate.
* `--run_id`: A short name to identify the run.
* `--device`: `cuda`, `xpu`, `cpu`, or `auto`, depending on available hardware.
* `--num_nodes`: The number of nodes to use.
* `--devices_per_node`: The number of individual devices per node (e.g. 12 on Aurora nodes, if utilizing all tiles in a flat hierarchy).
* `--distributed_strategy`: `ddp` or `deepspeed_zero2`. `ddp` writes a single plain .ckpt and is the simpler choice on one device or where nodes do not share a filesystem. `deepspeed_zero2` (default) shards optimizer state and offloads to CPU, which is used when the model or optimizer state is large; its checkpoints are directories that get converted to state_dict.best.pth at the end.
* `--epochs`: The number of epochs to finetune for.
* `--batch_size`: Batch size.
* `--wandb`: `True` or `False`. Set to True to track the run in weights&biases, provided you have an API key. Otherwise set to False. When enabled, a `wandb/` directory appears under `runs/<run_id>/logs/` alongside `lightning_logs/`.

A finetuning run should result in a populated output directory with the structure shown below. Specifying the `output_root` as `outputs/gfp_ft_results` creates the directory (if it doesn't already exist) as well as subdirectories for checkpoints and individual run results. Both subdirectories are keyed on `--run_id`, so a single `output_root` can hold many runs without them colliding, and the bulky checkpoints stay separate from the small logs and artifacts. Continuing a run to further epochs is done with `--resume_from_checkpoint`, pointing at a specific `.ckpt` under `checkpoints/<run_id>/`.

```txt
outputs/gfp_ft_results/
├── checkpoints/gfp_ft001/
│   └── epoch=2-step=33.ckpt
└── runs/gfp_ft001/
    ├── logs/
    │   └── lightning_logs/
    └── artifacts/
        ├── state_dict.best.pth
        ├── args.json
        ├── build_manifest.json
        ├── checkpoint_summary.json
        ├── dataset_splits.pt
        ├── metrics_history.pt
        ├── run_summary.json
        └── run.log
```

**Description of outputs:**

* `checkpoints/<run_id>/`: Checkpoints saved during the given run.
* `runs/<run_id>/logs`: Lightning logs generated during the given run.
* `runs/<run_id>/artifacts`: Artifacts produced over the course of finetuning, and on completion. These artifacts include
  * `state_dict.best.pth`: Single file containing the optimal model weights based on validation loss.
  * `args.json`: A full list of passed arguments, either through the command line or the config file.
  * `build_manifest.json`: Provenance tracking manifest.
  * `checkpoint_summary.json`: Descriptive summary of checkpoints.
  * `dataset_splits.pt`: Provenance tracker; Datapoint indices corresponding to the training, validation and test splits.
  * `metrics_history.pt`: Pytorch data file containing the history of tracked metrics over the course of the run.
  * `run_summary.json`: How the run ended, including the exit reason and any exception.
  * `run.log`: Full run log.

The sections below detail the finetuning command to be used in each of the following cases:

* Finetuning on a machine with a single CUDA GPU, from a repo checkout
* Finetuning on a machine with a single CUDA GPU, using docker
* Finetuning on a machine with multiple CUDA GPUs
* Finetuning on Aurora using a single node and device
* Finetuning on Aurora using multiple nodes and 12 devices per node

#### On a single CUDA device, from a code checkout

Running on a machine with a cuda device and a cloned copy of the BioM3-dev repo is strightforward. Ensure that the config file, weights, and input data are present. One important prerequisite is to source the `environment.sh` file prior to running the command.

```bash
source environment.sh
biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --finetune True \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True \
    --primary_data_path outputs/ft_embeddings/gfp_demo.compiled_emb.hdf5 \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root outputs/gfp_ft_results \
    --run_id gfp_ft001 \
    --device cuda --num_nodes 1 --devices_per_node 1 \
    --distributed_strategy ddp \
    --epochs 3 --batch_size 16 --wandb False
```

#### On a single CUDA device, from a Docker image

Follow the steps detailed above in the Quickstart instructions to ensure Docker is installed on your machine and that you have pulled down a BioM3 image with a `cuda` tag.

Fetch the weights as described in the Quickstart. Then fetch the GFP dataset and embed it, using the same commands as above run through the container:

```bash
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cuda-2bf065b
mkdir -p weights data outputs

docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/data:/app/data" \
    $BIOM3_IMAGE biom3_fetch_dataset gfp_demo -o /app/data

docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/data:/app/data:ro" \
    -v "$PWD/outputs:/app/outputs" \
    $BIOM3_IMAGE \
    biom3_embedding_pipeline \
        -i data/gfp_sample_dataset.csv \
        -o outputs/ft_embeddings --prefix gfp_demo \
        --weight_set configs/weights/run1_base.json \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json
```

Then run the finetuning command as above, with the appropriate docker additions:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/outputs:/app/outputs" \
    $BIOM3_IMAGE \
    biom3_train_stage3 \
        --config_path configs/stage3_training/finetune_v1.json \
        --finetune True \
        --finetune_last_n_blocks 1 \
        --finetune_last_n_layers -1 \
        --finetune_output_layers True \
        --primary_data_path outputs/ft_embeddings/gfp_demo.compiled_emb.hdf5 \
        --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
        --output_root outputs/gfp_ft_results \
        --run_id gfp_ft001 \
        --device cuda --num_nodes 1 --devices_per_node 1 \
        --distributed_strategy ddp \
        --epochs 3 --batch_size 16 --wandb False
```

#### On Aurora, from a code checkout

Clone the repository on Aurora and follow the instructions described above in the Quickstart instructions to create a virtual python environment. In order to run the commands below on real data, we assume that the demonstration GFP dataset has already been fetched with `biom3_fetch_dataset gfp_demo -o data`. On Aurora, this may require first installing oras. This can be done as follows:

```bash
# One time on an Aurora head node to install oras
ORAS_VERSION=1.3.3
curl -LO "https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz"
mkdir -p ~/.local/bin
tar -xzf "oras_${ORAS_VERSION}_linux_amd64.tar.gz" -C ~/.local/bin oras
rm "oras_${ORAS_VERSION}_linux_amd64.tar.gz"
export PATH="$HOME/.local/bin:$PATH"    # add to ~/.bashrc or ~/.zshrc to keep it
oras version
```

With oras installed, demo datasets can be fetched with

```bash
biom3_fetch_dataset gfp_demo -o data
```

Next, embed the dataset (Stages 1 and 2). The embedding runs on a single device, so run it once, on a compute node, before either of the finetuning cases below.

```bash
# On an Aurora compute node
module load frameworks
source venvs/biom3-env/bin/activate
source environment.sh

biom3_embedding_pipeline \
    -i data/gfp_sample_dataset.csv \
    -o outputs/ft_embeddings --prefix gfp_demo \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json
```

**Single node, 1 device per node**

On a compute node, either interactively or through a PBS script, run the following from the root of the checkout. `scripts/stage3_train_singlenode.sh` takes the config, the number of devices, the device type and the run ID as positional arguments, fills in `--num_nodes 1` and `--devices_per_node`, and launches the run through `mpiexec`.

```bash
# On an Aurora compute node
module load frameworks
source venvs/biom3-env/bin/activate
source environment.sh

scripts/stage3_train_singlenode.sh \
    configs/stage3_training/finetune_v1.json 1 xpu gfp_ft001 \
    --finetune True \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True \
    --primary_data_path outputs/ft_embeddings/gfp_demo.compiled_emb.hdf5 \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root outputs/gfp_ft_results \
    --distributed_strategy ddp \
    --epochs 3 --batch_size 16 --wandb False
```

**Multinode, 12 devices per node**

Save the following as a PBS script, e.g. `gfp_ft.pbs`, set `select` to the number of nodes, and submit it from the root of the checkout with `qsub gfp_ft.pbs`. `scripts/stage3_train_multinode.sh` takes the number of nodes and the devices per node in addition to the arguments above. With the dataset split across many devices, each device holds a single validation batch, so `--limit_val_batches 1.0` is required.

```bash
#!/bin/bash -l
#PBS -A <project>
#PBS -N gfp_ft
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q <queue>
#PBS -j oe

cd ${PBS_O_WORKDIR}
module load frameworks
source venvs/biom3-env/bin/activate
source environment.sh

NUM_NODES=$(wc -l < ${PBS_NODEFILE})

scripts/stage3_train_multinode.sh \
    configs/stage3_training/finetune_v1.json ${NUM_NODES} 12 xpu gfp_ft001 \
    --finetune True \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True \
    --primary_data_path outputs/ft_embeddings/gfp_demo.compiled_emb.hdf5 \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root outputs/gfp_ft_results \
    --limit_val_batches 1.0 \
    --distributed_strategy ddp \
    --epochs 3 --batch_size 16 --wandb False
```

#### On Aurora, using an Apptainer image

Follow the steps detailed above in the Quickstart instructions to convert an appropriate docker image into an Apptainer image, or check with your PI to see if a shared Apptainer image already exists.

With the dataset fetched as described above, embed it (Stages 1 and 2) once, on a compute node, before either of the finetuning cases below. The embedding runs as a single process, so it uses `scripts/aurora/apptainer_run.sh`, as in the Quickstart instructions.

```bash
# On an Aurora compute node, from the root of the checkout
module load apptainer
export BIOM3_IMAGE="$PWD/biom3_xpu-oneapi-2bf065b.sif"

scripts/aurora/apptainer_run.sh biom3_embedding_pipeline \
    -i data/gfp_sample_dataset.csv \
    -o outputs/ft_embeddings --prefix gfp_demo \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json
```

**Single node, 1 device per node**

On a compute node, either interactively or through a PBS script, run the following command from the root of the checkout. Do not `module load frameworks`: the image carries its own software stack. `scripts/aurora/apptainer_mpi_run.sh` starts one container per rank under the host's `mpiexec`, so `biom3_train_stage3` is called directly.

```bash
module load apptainer
export BIOM3_IMAGE="$PWD/biom3_xpu-oneapi-2bf065b.sif"

NGPU_PER_NODE=1 NGPU_TOTAL=1 BIOM3_RANK_SOURCE=mpi \
scripts/aurora/apptainer_mpi_run.sh biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --finetune True \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True \
    --primary_data_path outputs/ft_embeddings/gfp_demo.compiled_emb.hdf5 \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root outputs/gfp_ft_results \
    --run_id gfp_ft001 \
    --device xpu --num_nodes 1 --devices_per_node 1 \
    --distributed_strategy ddp \
    --epochs 3 --batch_size 16 --wandb False
```

**Multinode, 12 devices per node**

Through a PBS script, request N nodes and run the following command. Submit it from the root of the checkout. `BIOM3_FABRIC_DIR` and `BIOM3_FI_PROVIDER` bind the host's libfabric into each container so that communication between nodes uses Aurora's Slingshot network; confirm the libfabric path with `ls -d /opt/cray/libfabric/*/lib64`.

```bash
#!/bin/bash -l
#PBS -A <project>
#PBS -N gfp_ft
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q <queue>
#PBS -j oe

cd ${PBS_O_WORKDIR}
module load apptainer
export BIOM3_IMAGE="$PWD/biom3_xpu-oneapi-2bf065b.sif"

NUM_NODES=$(wc -l < ${PBS_NODEFILE})
export NGPU_PER_NODE=12 NGPU_TOTAL=$((NUM_NODES * 12))
export BIOM3_RANK_SOURCE=mpi
export BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 BIOM3_FI_PROVIDER=cxi

scripts/aurora/apptainer_mpi_run.sh biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --finetune True \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True \
    --primary_data_path outputs/ft_embeddings/gfp_demo.compiled_emb.hdf5 \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root outputs/gfp_ft_results \
    --run_id gfp_ft001 \
    --device xpu --num_nodes ${NUM_NODES} --devices_per_node 12 \
    --limit_val_batches 1.0 \
    --distributed_strategy ddp \
    --epochs 3 --batch_size 16 --wandb False
```
