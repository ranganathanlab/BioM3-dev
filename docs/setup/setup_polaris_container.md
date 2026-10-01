# Running BioM3 on Polaris via Apptainer (CUDA container)

This is the containerized path for Polaris, as an alternative to the bare-metal
`module load conda` install in [setup_polaris.md](./setup_polaris.md).

**Polaris reuses the same CUDA image as the cloud** — Polaris is NVIDIA (A100),
so there is no Polaris-specific Dockerfile. The image built by
[docker/build.sh](../../docker/build.sh) (`Dockerfile.cuda`, also used for
AWS/Mithril) runs on Polaris unchanged; only the *run* wrapper differs.

**Scope: single node** (4 A100s). Multi-node NCCL over Polaris's fabric needs the
host MPI bound in and is not wired up here — see [setup_polaris.md](./setup_polaris.md)
for the bare-metal multi-node path.

## Prerequisites

- The public CUDA image on GHCR: `ghcr.io/natural-machine/biom3:cuda-dev`. It is a
  multi-arch manifest list, so Polaris pulls the `linux/amd64` half automatically. If
  it is not already pushed, publish it from any Docker host (either architecture):
  `docker/build.sh --variant cuda --release --repo ghcr.io/<org>/biom3` — see
  [docker/README.md](../../docker/README.md#publishing-to-ghcr) for the build-host
  requirements.
- No GHCR login needed for the pull (the image is public).

## How the same image "just works" on Polaris

Everything Polaris-specific is applied at run time by
[scripts/polaris/apptainer_run.sh](../../scripts/polaris/apptainer_run.sh), not
baked into an image:

- `apptainer exec --nv` exposes Polaris's GPUs + NVIDIA driver (the CUDA analog of
  the XPU path's `/dev/dri` bind).
- `--bind /grand` for data/weights on Lustre.
- **`unset BIOM3_MACHINE`** before sourcing `environment.sh`. The cloud image bakes
  `BIOM3_MACHINE=container` (for AWS/Mithril); unsetting it lets `environment.sh`
  fingerprint `/grand` and select the `polaris` profile instead.

## Workflow

### 1. Convert to a .sif (on a Polaris COMPUTE node)

Per [Containers on Polaris](https://docs.alcf.anl.gov/polaris/containers/containers/),
container build/run is supported **on compute nodes, not login nodes**. Grab an
interactive node, load Apptainer, and set the ALCF proxy so the `docker://` pull
can reach the internet:

```bash
qsub -I -A NLDesignProtein -l select=1 -l filesystems=home:grand -l walltime=01:00:00 -q debug
# on the compute node:
module load apptainer                         # or the site's current apptainer module
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

apptainer build --fakeroot biom3_cuda.sif docker://ghcr.io/natural-machine/biom3:cuda-dev
```

Offline alternative (no registry): `docker save biom3:cuda -o biom3_cuda.tar` on the
build host, `scp` it over, then
`apptainer build --fakeroot biom3_cuda.sif docker-archive://biom3_cuda.tar`.

### 2. Smoke-test the GPUs

```bash
scripts/polaris/apptainer_run.sh python -c \
  "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
# expect: cuda True 4
```

### 3. Run a stage (single node)

```bash
BIOM3_WEIGHTS_DIR=/grand/NLDesignProtein/sharepoint/BioM3-data-share/weights \
BIOM3_DATA_DIR=/grand/NLDesignProtein/sharepoint/BioM3-data-share/data \
scripts/polaris/apptainer_run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 4 auto run001 --epochs 1
```

Host dirs bind onto `/app/{weights,data,outputs}`; `outputs/` is writable, weights
and data are read-only. See the script header for all `BIOM3_*` knobs.

## Troubleshooting

- **`torch.cuda.is_available()` is False.** The image ships a CUDA 12.9 (cu129)
  torch build; it needs a Polaris NVIDIA driver new enough to run the CUDA 12.x
  runtime. Check the host driver with `nvidia-smi` on the compute node. If it is
  too old, rebuild the cloud image against a CUDA minor that matches Polaris's
  driver (adjust the torch `--index-url` in [docker/Dockerfile.cuda](../../docker/Dockerfile.cuda),
  e.g. `whl/cu126`) and republish.
- **`docker://` pull hangs / DNS failure during build.** The ALCF proxy vars are
  not set — export the four `*_proxy` variables above on the compute node.
- **Build fails without `--fakeroot`.** Polaris unprivileged builds require it;
  keep `--fakeroot` on the `apptainer build` line.
- **`OSError: [Errno 30] Read-only file system` (e.g. running the test suite).**
  The `.sif` is read-only, and some code writes into the image tree
  (`/app/tests/_tmp`, `.pytest_cache`). `apptainer_run.sh` passes
  `--writable-tmpfs` (an ephemeral RAM-backed overlay) to absorb these; if you
  invoke `apptainer exec` by hand, add `--writable-tmpfs` yourself. The test suite
  writes more than that overlay holds, so `apptainer_run.sh` also mounts
  `<outputs>/tests_tmp` at `/app/tests/_tmp` (override with `BIOM3_TESTS_TMP`); by
  hand, add `--bind <dir>:/app/tests/_tmp`.
