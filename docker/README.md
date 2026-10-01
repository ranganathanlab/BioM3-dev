# BioM3 Docker images

Recipes for running BioM3 — **training (all stages), finetuning, and generation** — in a
container on any Docker host with a GPU: a workstation, or a cloud GPU instance (AWS,
Mithril).

**One image, all uses.** A single CUDA image holds the full BioM3 install; you pick what
to run at `docker run` time. Built per architecture:

- **x86_64 NVIDIA** — AWS `p3/p4/p5`, `g4/g5/g6`; Mithril H100/H200.
- **ARM64 NVIDIA** — Grace / GH200.

> **ALCF HPC uses Apptainer, not `docker run`.** This CUDA image itself *does* run
> on **Polaris** (NVIDIA) unchanged — converted to a `.sif` and launched with
> `--nv`; see [`setup_polaris_container.md`](../docs/setup/setup_polaris_container.md).
> **Aurora** is Intel/oneAPI and needs its own images — `Dockerfile.xpu` for
> single node (run with `scripts/aurora/apptainer_run.sh`) and
> `Dockerfile.xpu-oneapi` for multi-node (run with
> `scripts/aurora/apptainer_mpi_run.sh`, which launches one container per rank
> under the host `mpiexec`). See
> [`setup_aurora_container.md`](../docs/setup/setup_aurora_container.md). For the
> bare-metal HPC installs see [`setup_polaris.md`](../docs/setup/setup_polaris.md)
> and [`setup_aurora.md`](../docs/setup/setup_aurora.md).

| File | Purpose |
| ---- | ------- |
| `Dockerfile.cuda` | The NVIDIA image: `nvidia/cuda:12.9-base` → py3.12 → torch 2.8 (cu129) → BioM3 (`pip install -e .[app]`). Two-stage: the toolchain lives in a throwaway builder, and the runtime starts from `base` rather than `devel` because the torch wheel already ships every CUDA library it opens. |
| `Dockerfile.cpu` | The slim CPU-only **inference** image: `ubuntu:24.04` → py3.12 → torch 2.8 (cpu) → BioM3, from `requirements/container-cpu.txt`. Embedding, manifold fitting/scoring and Stage 3 sampling. **Not a training image** (no wandb/tensorboard/mpi4py) and no streamlit `app` extra. |
| `Dockerfile.xpu` | The Intel XPU variant, for Aurora single-node. Ubuntu + pip wheels. **amd64 only** — there are no arm64 Intel GPU wheels. |
| `Dockerfile.xpu-oneapi` | Second Aurora variant, built on `intel/oneapi-hpckit` so the container's Intel MPI matches the host launcher's. Exists because multi-node collectives never complete in the `xpu` image. **amd64 only.** |
| [`../.dockerignore`](../.dockerignore) | One ignore file for every variant (the build context is the repo root). Trims the context and keeps gitignored local files such as `configs/jobs/local.env` out of the image. |
| `entrypoint.sh` | Optionally pulls a published GHCR weights bundle, then exec's your command. |
| `build.sh` | `docker buildx` wrapper (variant, platform, tag, push) and the GHCR publish path (`--release`). |
| `push.sh` | Publishes an already-built **single-arch** image under the GHCR tags. For the xpu variant; see [Publishing](#publishing-to-ghcr). |
| `run.sh` | `docker run` wrapper: standard mounts, extra mounts (`BIOM3_BIND_EXTRA`), runs as the calling user, env passthrough. |
| `docker-compose.yml` | Optional services for the streamlit `app` (port 8501) and a `shell`. |

---

## Build

Requires Docker with BuildKit/buildx. The cuda image is ~8.6 GB on disk, the cpu
image ~1.9 GB; the first cuda build downloads the ~3 GB torch cu129 wheel.
Subsequent builds reuse the buildx layer cache.

```bash
# Native architecture (Apple Silicon → arm64; x86 Linux/Mac → amd64):
docker/build.sh                      # tags biom3:cuda

# Explicit platform:
docker/build.sh --platform linux/amd64
docker/build.sh --platform linux/arm64

# The slim CPU-only inference image:
docker/build.sh --variant cpu           # tags biom3:cpu

# The Intel XPU variants (amd64 only):
docker/build.sh --variant xpu           # single-node, validated
docker/build.sh --variant xpu-oneapi    # oneAPI base, for multi-node
```

`--load` is single-platform only, so a multi-platform build must push — see
[Publishing](#publishing-to-ghcr).

**Network note:** the torch wheel comes from `download.pytorch.org`. On a VPN with
upstream DNS filtering (e.g. Tailscale MagicDNS) the download can stall on `Failed to
resolve download.pytorch.org` — disconnect the VPN for the build.

---

## Publishing to GHCR

The public cuda image is a **multi-arch manifest list** covering `linux/amd64` (cloud
instances) and `linux/arm64` (DGX Spark), so both pull the same tag. Build every
architecture in one pass, from **one** host of either architecture. The destination
repo is always given with `--repo`; the scripts have no default:

```bash
REPO=ghcr.io/<org>/biom3
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
docker/build.sh --variant cuda --release --repo "$REPO"
docker buildx imagetools inspect "$REPO:cuda-dev"
```

`--release` pushes `cuda-<sha>` (immutable) and `cuda-dev` (moving; what `cloud/*.yaml`
pull), defaults `--platform` to every architecture the variant supports, and refuses a
dirty tree so the sha tag matches the commit.

The build host needs a **`docker-container`** buildx builder (the stock `default` driver
cannot build multi-platform) and **binfmt/QEMU** for the non-native architecture:

```bash
docker buildx create --name multiarch --driver docker-container --use
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx inspect            # Platforms: must list both architectures
```

`Dockerfile.cuda` has no architecture conditionals — the `nvidia/cuda` base is a
manifest list and the cu129 index serves aarch64 wheels — so multi-arch is purely a
build-orchestration concern.

[`push.sh`](push.sh) publishes an already-built image under the same tags
(`docker/push.sh --variant xpu --repo "$REPO"`), but a local image is **single-arch**.
It is the path for the amd64-only `xpu` variant; for cuda it refuses to overwrite a
multi-arch `-dev` tag (`--force-dev` overrides).

**Publishing makes the baked `src/`, `scripts/`, `tests/`, and `configs/` world-readable.**
The full runbook — token creation, package visibility, anonymous-pull verification — is in
[`../cloud/README.md`](../cloud/README.md#publishing-the-image-ghcr--runbook).

---

## Getting weights and data into the container

BioM3 needs pretrained **weights** (ESM-2 + BioBERT + per-stage checkpoints) and, for
training, **datasets**. Neither is baked into the image. Getting them onto the host is up
to you; the image assumes nothing about where they come from. Two ways to hand them to
the container:

### 1. Bind-mount from the host (default)

`docker/run.sh` mounts the conventional layout automatically:

| Host (default) | Container | Mode | Contents |
| --- | --- | --- | --- |
| `./weights` | `/app/weights` | ro | `LLMs/`, `PenCL/`, `Facilitator/`, `ProteoScribe/` |
| `./data` | `/app/data` | ro | training datasets (CSV / HDF5) |
| `./outputs` | `/app/outputs` | **rw** | checkpoints, logs, generated sequences |
| `./outputs/tests_tmp` | `/app/tests/_tmp` | **rw** | the test suite's scratch (`BIOM3_TESTS_TMP` overrides) |
| `./configs` | `/app/configs` | ro | *(optional)* overrides the configs baked into the image |

Override the host dirs with `BIOM3_WEIGHTS_DIR`, `BIOM3_DATA_DIR`, `BIOM3_OUTPUTS_DIR`,
`BIOM3_CONFIGS_DIR`. The Apptainer wrappers for Polaris and Aurora read the same
settings with the same defaults, from one shared definition
([`scripts/_container_mounts.sh`](../scripts/_container_mounts.sh)). The weights layout mirrors
[`docs/setup/setup_shared_weights.md`](../docs/setup/setup_shared_weights.md).

**Symlinked weights or data.** A symlink inside a mounted directory is resolved *inside*
the container, so if `weights/` or `data/` holds absolute links to files elsewhere on the
host, those locations must be mounted too, at the same path. List them in
`BIOM3_BIND_EXTRA` (comma-separated; each is mounted read-only at the same path). To see
where your links point:

```bash
find weights data -type l -exec readlink {} + | cut -d/ -f1-4 | sort | uniq -c
```

then, for example:

```bash
BIOM3_BIND_EXTRA=/shared/biom3-data,/shared/models docker/run.sh <command>
```

Without them the links dangle inside the container, and the failure surfaces as a
missing file or, for a local Hugging Face model directory, as a malformed repo id.

### 2. The published GHCR weights bundle

`BIOM3_WEIGHTS_BUNDLE=run1_base` makes the entrypoint `oras pull`
`ghcr.io/natural-machine/biom3-weights:run1_base` into `/app/weights` when the container
starts; no credentials are needed. The pull writes into the image's own `/app/weights`,
so the container must run as root (`BIOM3_AS_ROOT=1` with `run.sh`). On a host you reuse,
fetch the bundle once instead and mount it: see
[`docs/setup/weights_bundle.md`](../docs/setup/weights_bundle.md).

---

## Run

`docker/run.sh <command...>` assembles `docker run --gpus all` with the mounts above and
forwards `WANDB_API_KEY`, `NGPU` and the weights-bundle variables. It runs the container
as the calling user, so files written to `outputs/` are owned by you; set
`BIOM3_AS_ROOT=1` to run as root instead. Examples below use it.

Requires the **NVIDIA Container Toolkit** on the host (`--gpus all`).

### Generation (Stage 1 → 2 → 3)

Single-process; no launcher needed. Run the three stages in sequence:

```bash
docker/run.sh biom3_PenCL_inference \
    --input_data_path data/my_proteins.csv \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt

docker/run.sh biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt

docker/run.sh biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt --fasta
```

(Use `--device cpu` for a cheap smoke test on tiny inputs without a GPU.)

### Training (Stages 1, 2, 3)

The existing wrapper scripts work unchanged in the container. Inside, `BIOM3_MACHINE` is
`container`, so they dispatch to `scripts/launchers/container_singlenode.sh`, which uses
**`torchrun`** for multi-GPU (see [How it works](#how-it-works-inside-the-container)).

Wrapper signature: `scripts/stageN_train_singlenode.sh CONFIG_PATH NGPU DEVICE RUN_ID [--overrides…]`

Pass `auto` for `DEVICE`: it picks the image's GPU backend, so the same command runs on
CUDA and on Aurora's XPUs. `NGPU` is always explicit, because how many ranks to run per
node is a layout choice; a request for more devices than the container can see stops
the run with an error saying so.

```bash
# Stage 3 pretrain from scratch, single GPU:
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 auto run001 --epochs 1

# Stage 3, 4 GPUs (NGPU must match the wrapper's NGPU arg; torchrun spawns 4 ranks):
BIOM3_GPUS=all NGPU=4 docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 4 auto run001 --epochs 5

# Stage 1 (PenCL) and Stage 2 (Facilitator):
docker/run.sh scripts/stage1_train_singlenode.sh \
    configs/stage1_training/pretrain_scratch_v1.json 1 auto s1run001
docker/run.sh scripts/stage2_train_singlenode.sh \
    configs/stage2_training/pretrain_scratch_v1.json 1 auto s2run001
```

`WANDB_API_KEY` (forwarded by `run.sh` when set) enables Weights & Biases logging
automatically; otherwise it defaults off. Pass `--wandb True|False` to force it.

### Finetuning (Stage 3)

Finetuning is the Stage 3 trainer with `--finetune` flags + base weights:

```bash
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/finetune_v1.json 1 auto ft001 \
    --finetune True \
    --pretrained_weights weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --finetune_last_n_blocks 1 --finetune_last_n_layers 1 \
    --primary_data_path data/my_finetune_set.hdf5 --epochs 10
```

### Resume (spot preemption)

Checkpoints land in the mounted `outputs/` and survive container exit. After a
preemption, resume from `last.ckpt`:

```bash
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 auto run001 \
    --epochs 5 --resume_from_checkpoint outputs/<...>/checkpoints/run001/last.ckpt
```

### Web app

```bash
docker compose -f docker/docker-compose.yml up app    # http://localhost:8501
```

### Interactive shell

```bash
docker/run.sh bash
# or: docker compose -f docker/docker-compose.yml run --rm shell
```

---

## How it works inside the container

- The cuda and cpu images set `ENV BIOM3_MACHINE=container`, which the training wrappers
  read to choose `scripts/launchers/container_singlenode.sh`.
- **Single-node multi-GPU uses `torchrun`, not `mpiexec`/PBS.**
  `scripts/launchers/container_singlenode.sh`: 1 GPU → `exec` directly; N GPUs →
  `torchrun --standalone --nproc-per-node=N`, which sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE`
  /`MASTER_ADDR`/`MASTER_PORT`. BioM3 already reads these
  ([`core/_dist_env.py`](../src/biom3/core/_dist_env.py)) and PyTorch Lightning
  auto-detects the torchelastic environment (it does not re-spawn).
- OpenMPI is in the image only so `mpi4py` (a pinned dependency) builds and imports;
  it is not used to launch. The runtime stage carries just `openmpi-bin`; the
  headers live in the builder stage.
- `cuda-nvcc` is in the runtime stage and is **required**, not optional: `import
  deepspeed` probes `$CUDA_HOME/bin/nvcc -V` from its op-builder compatibility scan
  and raises `FileNotFoundError` without it, which would break every Stage 3 entry
  point. The rest of the CUDA toolkit is absent — the torch wheel ships its own
  cudart/cuBLAS/cuDNN/NCCL/cuFFT/cuSPARSE/cuSOLVER/cuRAND/nvrtc in
  `site-packages/torch/lib` and resolves them through its RUNPATH, and triton ships
  its own `ptxas`.

## Quick local sanity check (no GPU required)

```bash
docker/build.sh
docker run --rm biom3:cuda python -c "import biom3, torch; print(torch.__version__)"
docker run --rm biom3:cuda pytest tests/ --quick
```

For the cpu variant, exclude the one test module that needs `streamlit` (the `app`
extra is not installed there):

```bash
docker/build.sh --variant cpu
docker run --rm biom3:cpu python -c "import biom3, torch; print(torch.__version__)"
docker run --rm biom3:cpu pytest tests/ --quick \
    --ignore=tests/viz_tests/test_data_browser.py
```

## Multi-node training (SkyPilot)

Finetuning, pretraining, and generation run across multiple instances via
[`scripts/launchers/container_multinode.sh`](../scripts/launchers/container_multinode.sh),
the multi-node analog of `container_singlenode.sh`. SkyPilot launches the task once per
**node** and sets `SKYPILOT_NODE_RANK` / `SKYPILOT_NUM_NODES` / `SKYPILOT_NODE_IPS`; the
launcher translates those into a **`torchrun` static rendezvous**
(`--nnodes --node-rank --master-addr --nproc-per-node`), which sets
`RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`GROUP_RANK` — the same env BioM3
([`core/_dist_env.py`](../src/biom3/core/_dist_env.py)) and Lightning already read. No
Python changes.

- **Enable it:** launch [`cloud/run.mithril.yaml`](../cloud/run.mithril.yaml) with
  `--num-nodes N` and `--gpus` set to the **per-node** GPU count, and run the multi-node
  wrapper in `CMD`: `bash scripts/stage3_train_multinode.sh <config> N <gpus-per-node> auto
  <run_id> --distributed_strategy ddp`. It dispatches to `container_multinode.sh` because
  the image sets `BIOM3_MACHINE=container`. See
  [cloud/README.md](../cloud/README.md#multi-node-training).
- **Checkpointing = DDP, not DeepSpeed.** Cloud instances share **no filesystem**, so
  DeepSpeed ZeRO's per-rank optimizer shards would scatter across nodes' local disks and
  can't be consolidated. Pass `--distributed_strategy ddp`: a single `.ckpt` is then
  written entirely by **global rank 0**, on the head node (`SKYPILOT_NODE_RANK==0`), whose
  `/app/outputs` is the one to copy off. Revisit DeepSpeed multi-node only with a shared FS.
- **NCCL:** the launcher auto-detects the private-net (`10.x`) interface for
  `NCCL_SOCKET_IFNAME` and disables InfiniBand (`NCCL_IB_DISABLE=1`); the task uses
  `--net=host --ipc=host`. Debug a first run with `NCCL_DEBUG=INFO`.
- **Data:** every node needs its own identical copy of the training data. Prefer a
  precompiled HDF5 over on-the-fly CSV embedding, so every rank builds identical
  `DistributedSampler` shards.
- **Generation** parallelizes only Stage-3 sampling (the sampler is rank-aware; only rank
  0 writes). Stages 1–2 run per node and Facilitator sampling is stochastic — verify
  determinism with a fixed `SEED` before trusting multi-node output.

The ALCF path (`scripts/launchers/{aurora,polaris}_multinode.sh`, mpiexec/PBS) remains the
reference for the HPC clusters.
