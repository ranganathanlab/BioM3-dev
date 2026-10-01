# BioM3 quickstart (container, no source checkout)

How to go from a bare machine to generated protein sequences using only the published
BioM3 container image. You do not need to clone the repository, install Python, or create
a conda environment. Everything BioM3 needs — the code, the configuration files and the
example inputs — is already inside the image; you supply a directory for the model weights,
one for your inputs and one for the results.

Contents:

1. [What you need](#1-what-you-need)
2. [Install Docker](#2-install-docker)
3. [Pick an image](#3-pick-an-image)
4. [Create a project directory](#4-create-a-project-directory)
5. [Fetch the model weights](#5-fetch-the-model-weights)
6. [Prepare your input file](#6-prepare-your-input-file)
7. [Embed and generate in one command](#7-embed-and-generate-in-one-command)
8. [Embedding and generating as separate steps](#8-embedding-and-generating-as-separate-steps)
9. [Reading the results](#9-reading-the-results)
10. [Troubleshooting](#10-troubleshooting)

## 1. What you need

- A computer running Linux, macOS or Windows.
- Docker (installed in step 2).
- Free disk space: about 15 GB with the GPU image, about 8 GB with the CPU image. Most of
  that is the 6.4 GB of model weights.
- An NVIDIA GPU is optional. Without one, use the CPU image; everything still works, only
  more slowly.

No BioM3 account, login or access token is needed. Both the image and the weights are
published publicly.

## 2. Install Docker

| Your machine | Install |
| ------------ | ------- |
| Windows, macOS | Docker Desktop, from <https://docs.docker.com/desktop/> |
| Linux | Docker Engine, from <https://docs.docker.com/engine/install/> |

Check that it works:

```bash
docker run --rm hello-world
```

You should see "Hello from Docker!". If instead you get a permission error on Linux, your
user is not yet in the `docker` group — follow Docker's post-installation steps, or put
`sudo` in front of every `docker` command in this guide.

### If you have an NVIDIA GPU

Install the NVIDIA Container Toolkit as well, from
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>.
It is what lets a container see the GPU. Check it:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If that prints a table of GPUs, you are set. If it errors, skip it and use the CPU image —
nothing else in this guide changes except the image name and the `--gpus all` flag.

Apple Silicon Macs have no NVIDIA GPU. Use the CPU image.

## 3. Pick an image

| Hardware | Image | Download |
| -------- | ----- | -------- |
| NVIDIA GPU | `ghcr.io/natural-machine/biom3:cuda-779859b` | 4.5–5.4 GB |
| No GPU (incl. Apple Silicon) | `ghcr.io/natural-machine/biom3:cpu-779859b` | 0.5–0.6 GB |

Both are built for Intel/AMD (`amd64`) and ARM (`arm64`); Docker picks the right one
automatically. Pull your image now so the later steps do not stall on a download:

```bash
docker pull ghcr.io/natural-machine/biom3:cuda-779859b
```

The `779859b` part is the source version this image was built from. Pinning it means your
results stay reproducible; a newer guide will name a newer tag.

The rest of this guide writes `ghcr.io/natural-machine/biom3:cuda-779859b`. If you are on
the CPU image, substitute `cpu-779859b` everywhere **and drop the `--gpus all` flag**.

## 4. Create a project directory

BioM3 reads and writes through three directories that you own. Create them:

```bash
mkdir -p ~/biom3-project/weights ~/biom3-project/data ~/biom3-project/outputs
cd ~/biom3-project
```

| Directory | Holds | The container sees it as |
| --------- | ----- | ------------------------ |
| `weights/` | the downloaded model weights | `/app/weights` (read-only) |
| `data/` | your input files | `/app/data` (read-only) |
| `outputs/` | everything BioM3 produces | `/app/outputs` (writable) |

Run every command below from `~/biom3-project`, because they refer to these directories
as `$PWD/weights` and so on.

## 5. Fetch the model weights

The weights are not in the image — they are 6.4 GB, and which set you want depends on what
you are doing. This guide uses `run1_base`, the standard published set. The image carries a
command that downloads it:

```bash
docker run --rm -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/weights" \
    ghcr.io/natural-machine/biom3:cuda-779859b \
    biom3_fetch_weights run1_base -o /weights
```

`/weights` here is just a scratch mount point for this one setup step, not part of BioM3's
layout — the later commands mount the same host directory at `/app/weights` instead, and
read-only. Do not shorten this to `-o ./weights`: that resolves to the image's own
`/app/weights`, which is root-owned, so the fetch either fails outright or (if you drop
`-u`) writes 6.4 GB into the container that `--rm` then throws away.

This takes a while — it is 6.4 GB. It prints each file as it lands. If it is interrupted,
run it again: files already on disk are checked against the published checksum and skipped
when they match, so a second run resumes rather than restarts. A partly written file is
never left in place under its real name, so it is simply re-fetched.

If a file exists but does **not** match the published checksum, the command stops and names
it rather than overwriting your copy. Add `--force` to replace it.

Afterwards `weights/` looks like this:

```
weights/
├── Facilitator/run1_base_facilitator.bin          4 MB
├── LLMs/
│   ├── BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/   441 MB
│   ├── esm2_t33_650M_UR50D.pt                     2.6 GB
│   ├── esm2_t33_650M_UR50D-contact-regression.pt
│   └── ESM2_LICENSE.md
├── PenCL/run1_base_pencl.bin                      3.0 GB
└── ProteoScribe/run1_base_proteoscribe.bin        345 MB
```

Add `--dry_run` to the command to see what it would download without downloading it.

## 6. Prepare your input file

BioM3 takes a CSV file with one row per prompt. It must carry these three columns; any
others are ignored:

| Column | What it is |
| ------ | ---------- |
| `primary_Accession` | Any identifier you choose for the row. It is carried into the embedding files. |
| `protein_sequence` | A protein sequence in single-letter amino acid codes. |
| `[final]text_caption` | The natural-language prompt describing the protein you want. |

The square brackets in `[final]text_caption` are part of the column name.

Save your file as `data/prompts.csv`. A two-row example:

```csv
primary_Accession,protein_sequence,[final]text_caption
P69222,MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR,"PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. SUBCELLULAR LOCATION: Cytoplasm."
```

Notes on the prompt text:

- Wrap it in double quotes, since captions contain commas.
- The models were trained on captions written as `KEY: value` sections — `PROTEIN NAME:`,
  `FUNCTION:`, `SUBUNIT:`, `SUBCELLULAR LOCATION:` and similar. Prompts in that form are
  closest to what the model has seen; free-form prose works but is further from training.
- Captions are cut off after 512 word-pieces, so put what matters first.

The `protein_sequence` column is required even when your interest is purely in generating
new sequences from text. BioM3 encodes it alongside the caption and reports how far the
two embeddings sit apart, which is a useful sanity check; it does not constrain what gets
generated. Use a representative sequence of the family you are describing.

If you would rather start from a working file than write one, the image ships an example:

```bash
docker run --rm ghcr.io/natural-machine/biom3:cuda-779859b \
    cat tests/_data/stage1_inputs/sample_text_seqs1.csv > data/prompts.csv
```

## 7. Embed and generate in one command

This runs all three stages: your caption and sequence become embeddings (Stage 1), the
caption embedding is aligned into protein space (Stage 2), and new sequences are generated
from it (Stage 3).

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/data:/app/data:ro" \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/natural-machine/biom3:cuda-779859b \
    biom3_embedding_pipeline \
        -i data/prompts.csv \
        -o outputs/demo --prefix demo \
        --weight_set configs/weights/run1_base.json \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json \
        --generate \
        --proteoscribe_config configs/inference/stage3_ProteoScribe_sample.json
```

What the parts mean:

| Part | Why it is there |
| ---- | --------------- |
| `--rm` | delete the container when it exits; your files are on the mounts, not in it |
| `--gpus all` | give the container the GPU — omit on the CPU image |
| `-u "$(id -u):$(id -g)"` | write output files as you rather than as root (Linux; see [troubleshooting](#10-troubleshooting)) |
| `-v host:container` | attach one of your directories at the path BioM3 expects |
| `-i data/prompts.csv` | your input, as the container sees it |
| `-o outputs/demo --prefix demo` | where results go and what to name them |
| `--weight_set configs/weights/run1_base.json` | which weights to load, by name |
| `--*_config configs/inference/...` | the model definitions, already inside the image |

The `configs/...` paths are files inside the image, not on your machine — that is why you
do not have to supply them.

Five sequences are generated per prompt by default.

By default each run uses a fresh random seed, so running it twice gives different
sequences. Add `--seed 42` (any positive number) to make a run reproducible; a seed of 0 or less
means "pick one at random". The seed actually used is recorded in each FASTA header and in
`run.log` either way, so a run you liked can be reproduced after the fact.

## 8. Embedding and generating as separate steps

Useful when you want to embed a set once and then generate from it repeatedly with
different settings. Embedding is the expensive half, and this way you pay it once.

First embed, by dropping `--generate` and its config from step 7:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/data:/app/data:ro" \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/natural-machine/biom3:cuda-779859b \
    biom3_embedding_pipeline \
        -i data/prompts.csv \
        -o outputs/embeds --prefix run1 \
        --weight_set configs/weights/run1_base.json \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json
```

Then generate from the embeddings it wrote. Note that `data/` is not mounted this time —
the input is now an embedding file under `outputs/`:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" \
    -v "$PWD/outputs:/app/outputs" \
    ghcr.io/natural-machine/biom3:cuda-779859b \
    biom3_ProteoScribe_sample \
        -i outputs/embeds/run1.Facilitator_emb.pt \
        -c configs/inference/stage3_ProteoScribe_sample.json \
        -m weights/ProteoScribe/run1_base_proteoscribe.bin \
        -o outputs/gen_seed42/generated.pt --fasta --seed 42
```

That writes `outputs/gen_seed42/generated.pt` and `outputs/gen_seed42/fasta/prompt_*.fasta`.
The FASTA directory is always created beside the `-o` file, so give each generation run its
own output directory rather than reusing one — otherwise the second run overwrites the
first one's FASTA files.

Re-run that second command with a different `--seed` and a different `-o` directory to get
another sample from the same prompts, without recomputing the embeddings.

## 9. Reading the results

After step 7, `outputs/demo/` contains:

| File | What it is |
| ---- | ---------- |
| `fasta/prompt_0.fasta`, `fasta/prompt_1.fasta`, … | the generated sequences — the thing you came for, one file per input row |
| `demo.generated.pt` | the same sequences plus the tensors behind them |
| `demo.Facilitator_emb.pt` | the embeddings sequences were generated from (`z_c`) |
| `demo.PenCL_emb.pt` | the Stage 1 caption and sequence embeddings (`z_t`, `z_p`) |
| `run.log` | the full log of the run |
| `build_manifest.json` | the exact versions and settings the run used |

Look at the sequences:

```bash
head -6 outputs/demo/fasta/prompt_0.fasta
```

```
>prompt_0_replica_0 seed=3805928775
MAREDVLEVPGTVLELLPNAMFRVKLENGHEIVAHTSGRIEKHFIRILTGDRVKVELSPYDLTKGRITYRYK
>prompt_0_replica_1 seed=3805928775
MPKEEKMELEGIIEEVLPNARFRVEIENGHQIVAHISGKMRRYHIRILPGDRVKVELSPYDLNRGRIIYRHLSKRNNHPPAN
```

`prompt_0` is the first row of your CSV, `prompt_1` the second, and so on — the files are
numbered by row order, not by your `primary_Accession` values.

Without `--generate` (step 8) you get `run1.compiled_emb.hdf5` in place of the `fasta/`
directory and `demo.generated.pt` — the embeddings packaged for training, rather than
sequences.

`run.log` reports how far the caption embedding landed from the real protein's embedding,
as an MSE between `z_c` and `z_p`. A small value means the prompt placed the model near
the protein family you described.

## 10. Troubleshooting

**`docker: permission denied while trying to connect to the Docker daemon`** (Linux) — your
user is not in the `docker` group. Either add it (and log out and back in) or prefix each
command with `sudo`.

**Output files are owned by `root` and you cannot delete them** — you left out
`-u "$(id -u):$(id -g)"`. Add it. On macOS and Windows, Docker Desktop handles ownership
for you and you should **omit** that flag; it is a Linux measure.

**`could not select device driver "" with capabilities: [[gpu]]`** — `--gpus all` was
passed but the NVIDIA Container Toolkit is not installed or not working. Either fix it
(step 2) or drop the flag and use the CPU image.

**`FileNotFoundError: weights/LLMs/esm2_t33_650M_UR50D.pt`** — the weights were not
fetched, or `weights/` was not mounted. Re-run step 5 and check the `-v "$PWD/weights:...`
line is present.

**It is slow** — Stage 1 runs a 650M-parameter protein language model and Stage 3 generates
by iterative denoising, so both take real time. For scale: on one NVIDIA GB10, two prompts
took 2 minutes 44 seconds end to end (step 7), of which 2 minutes 30 seconds was Stage 3
generation. The same two prompts embedded on CPU alone (step 8, first command) took 25
seconds. Generation is the expensive part, and it grows with the number of prompts times
the five replicas each. Start with two or three prompts.

**Windows PowerShell** — the commands above are written for a Linux or macOS shell. In
PowerShell, replace `$PWD` with `${PWD}`, drop `-u "$(id -u):$(id -g)"`, and use a backtick
(`` ` ``) instead of a backslash to continue a line. Or run them unchanged inside WSL2.

**Typing the same long command repeatedly** — set a shortcut for your session:

```bash
alias biom3='docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD/weights:/app/weights:ro" -v "$PWD/data:/app/data:ro" \
    -v "$PWD/outputs:/app/outputs" ghcr.io/natural-machine/biom3:cuda-779859b'
```

Then `biom3 biom3_embedding_pipeline -i data/prompts.csv ...`. The alias is forgotten when
you close the terminal, and it captures `$PWD` at the time each command runs, so keep
working from `~/biom3-project`.
