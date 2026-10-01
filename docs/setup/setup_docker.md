# Setup: Docker

For running BioM3 in a container on any Docker host with an NVIDIA GPU — a
workstation or a cloud GPU instance (AWS, Mithril) — for training (all stages),
finetuning, and generation. Unlike the other targets there is no conda/venv to create:
you build (or pull `ghcr.io/natural-machine/biom3:cuda-dev`) one CUDA image and run it.

> ALCF HPC uses Apptainer instead: see [`setup_polaris_container.md`](./setup_polaris_container.md)
> and [`setup_aurora_container.md`](./setup_aurora_container.md).

The full reference (build options, all run recipes, the GHCR weights bundle, the
streamlit app, multi-arch) lives in [`docker/README.md`](../../docker/README.md). Quick start:

## Prerequisites

- Docker with BuildKit/buildx.
- For GPU runs: an NVIDIA GPU host with the **NVIDIA Container Toolkit** (`--gpus all`).

## Build

One image, all uses (train/finetune/generate + the app). Built per architecture:

```bash
cd /path/to/BioM3-dev
docker/build.sh                          # native arch → tags biom3:cuda
# or pin: docker/build.sh --platform linux/amd64   (AWS/Mithril x86)
#         docker/build.sh --platform linux/arm64   (Grace / GH200)
```

Verify the build (no GPU needed):

```bash
docker run --rm biom3:cuda python -c "import biom3, torch; print(torch.__version__)"
docker run --rm biom3:cuda pytest tests/ --quick
```

## Supplying weights and data

Not baked into the image, and getting them onto the host is up to you. `docker/run.sh`
mounts the conventional layout (`weights/` ro, `data/` ro, `outputs/` rw). Populate
`weights/` per [`setup_shared_weights.md`](./setup_shared_weights.md). If `weights/` or
`data/` holds symlinks to elsewhere on the host, mount those locations too with
`BIOM3_BIND_EXTRA`; see [`docker/README.md`](../../docker/README.md), which also covers
the published GHCR weights bundle.

## Usage

`docker/run.sh <command…>` runs on a GPU host with the standard mounts, forwards
`WANDB_API_KEY` / `NGPU`, and runs as the calling user so files written to `outputs/`
are owned by you (`BIOM3_AS_ROOT=1` runs as root instead). The image sets
`BIOM3_MACHINE=container`, which the training wrappers use to pick their launcher.

```bash
# Train Stage 3 (single GPU)
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 auto run001 --epochs 1

# Train on 4 GPUs (torchrun spawns one rank per GPU inside the container)
NGPU=4 docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 4 auto run001 --epochs 5

# Generate (Stage 1 → 2 → 3): see docker/README.md for the full three-command pipeline
docker/run.sh biom3_PenCL_inference --help
```

Single-node multi-GPU runs through `torchrun`; multi-node across instances uses
`scripts/launchers/container_multinode.sh` — see [`docker/README.md`](../../docker/README.md).
