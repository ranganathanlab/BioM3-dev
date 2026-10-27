# Running BioM3 on Aurora via Apptainer (Intel XPU container)

This is the containerized path for Aurora, as an alternative to the bare-metal
`module load frameworks` install in [setup_aurora.md](./setup_aurora.md).

There are **two** Aurora images, and which one you want depends on node count:

- **Single node** — [docker/Dockerfile.xpu](../../docker/Dockerfile.xpu), run
  with [apptainer_run.sh](../../scripts/aurora/apptainer_run.sh). Validated: the
  full test suite passes and 12-tile training runs within ~8% of bare metal.
  Most of this document describes this path.
- **Multi-node** — [docker/Dockerfile.xpu-oneapi](../../docker/Dockerfile.xpu-oneapi),
  run with [apptainer_mpi_run.sh](../../scripts/aurora/apptainer_mpi_run.sh).
  Runs across nodes over Slingshot/CXI and scales — two nodes at roughly twice a
  single node's throughput — provided the host's Cray libfabric is bound in.
  See [Multi-node](#multi-node).

The multi-node image was rebuilt and re-validated, on one node and across nodes,
after Aurora moved to `frameworks/2026.1.0` in October 2026. The single-node
image has not been re-validated on that stack.

## Why a separate image from the CUDA one

Aurora's GPUs are Intel Data Center GPU Max (Ponte Vecchio), driven by oneAPI /
Level-Zero — there is no CUDA, so the `biom3:cuda` image would run there only on
CPU. The Aurora images install Intel's `+xpu` torch wheels instead, with native
`torch.xpu` and the `xccl` distributed backend:

- `Dockerfile.xpu`: `torch==2.8.0+xpu`, plus `intel-extension-for-pytorch==2.8.10+xpu`,
  kept because it is part of the stack this image was validated with. The
  `addison-nm/lightning` fork no longer requires it.
- `Dockerfile.xpu-oneapi`: `torch==2.14.1+xpu` on oneAPI 2026.1, the version under
  Aurora's `frameworks/2026.1.0`, and no IPEX.

Aurora's `module load frameworks` runs a source-built torch `2.13.0a0` that a
container cannot reproduce; the oneapi image matches the oneAPI version beneath it.
See [PyTorch on Aurora](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/).

The image has to follow the host. When Aurora went from `frameworks/2025.3.1` to
`2026.1.0`, which also moved Cray libfabric from 1.22.0 to 2.3.1, the oneAPI 2025.3
image stopped completing collectives over CXI: DDP's parameter check failed on
some ranks at start-up. Rebuilding on oneAPI 2026.1 fixed it. `module -t list`
on a compute node shows the versions to match.

## Prerequisites

- A Docker host with buildx to build + push the images (Aurora nodes have no
  Docker). The images are amd64-only, since Intel GPU torch wheels are x86_64-only:
  an x86_64 host builds them natively, and an arm64 host builds them under QEMU
  emulation (setup in [docker/README.md](../../docker/README.md#publishing-to-ghcr)).
- GHCR push access for the one-time publish (see [cloud/README.md](../../cloud/README.md)
  and [docker/push.sh](../../docker/push.sh)). The published image is public, so
  the Aurora-side pull needs no login.

## Workflow

### 1. Build + push the XPU image (off Aurora)

```bash
REPO=ghcr.io/<org>/biom3
docker/build.sh --variant xpu --release --repo "$REPO"   # -> $REPO:xpu-dev (+ :xpu-<sha>)
```

`--release` builds and pushes in one pass; `--repo` is required. Unlike the cuda
variant it stays **amd64-only**, so there is no manifest list to assemble. To publish
an image you have already built locally, `docker/push.sh --variant xpu --repo "$REPO"`
pushes the same two tags without rebuilding.

### 2. Convert to a .sif (Aurora login node)

Apptainer ([Containers on Aurora](https://docs.alcf.anl.gov/aurora/containers/containers/))
converts the Docker image to a `.sif`. The two scratch locations want opposite
things, so set them separately:

- `APPTAINER_CACHEDIR` on `/flare`, where it persists across login nodes. It
  holds the downloaded layer blobs, so a warm cache makes a rebuild of a new tag
  pull nothing.
- `APPTAINER_TMPDIR` on node-local disk. Unpacking layers is millions of small
  file creates, chowns and `setxattr`s — the access pattern Lustre is worst at.
  Only the finished `.sif` needs to land on `/flare`.

```bash
export APPTAINER_CACHEDIR=/flare/NLDesignProtein/$USER/.apptainer/cache
export APPTAINER_TMPDIR=/tmp/$USER/apptainer-tmp     # check `df -h /tmp` for room
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

apptainer build /flare/NLDesignProtein/$USER/biom3_xpu.sif \
    docker://ghcr.io/natural-machine/biom3:xpu-dev
```

Prefer the immutable `<variant>-<sha>` tag over `-dev` when validating a specific
build: it pins what you tested, and it cannot collide with a cached pull of an
older image under a moving tag.

Offline alternative (no registry): `docker save biom3:xpu -o biom3_xpu.tar` on the
build host, `scp` it over, then
`apptainer build biom3_xpu.sif docker-archive://biom3_xpu.tar`.

### 3. Smoke-test the GPUs (interactive, on a compute node)

Grab an interactive node, point the wrapper at the image you built, then check that
torch sees the 12 tiles:

```bash
export BIOM3_IMAGE=/flare/NLDesignProtein/$USER/biom3_xpu.sif
scripts/aurora/apptainer_run.sh python -c \
  "import torch; print('xpu', torch.xpu.is_available(), torch.xpu.device_count())"
# expect: xpu True 12
```

`apptainer_run.sh` binds
`/flare`, sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (so each tile is its own device,
matching `num_devices=12` in the PBS templates), and sources `environment.sh`
inside the container — which fingerprints `/flare` to select the `aurora` profile
and apply the oneCCL/`xccl`/NUMEXPR settings.

### 4. Run the test suite

```bash
scripts/aurora/apptainer_run.sh pytest tests/ --include_requires_gpu
```

The tests write their scratch into the image at `/app/tests/_tmp`. The wrapper mounts
`<outputs>/tests_tmp` there (override with `BIOM3_TESTS_TMP`), because the
`--writable-tmpfs` overlay that absorbs other writes is too small for the suite.
Everything a run writes therefore lands under the outputs directory; point
`BIOM3_OUTPUTS_DIR` at a scratch location to keep test files out of your real
`outputs/`.

### 5. Run a stage (single node)

```bash
scripts/aurora/apptainer_run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 12 auto run001 --epochs 1
```

The command is the same as on a CUDA machine except for the device count (`12`),
which you always state: it is a layout choice, not something the wrapper infers.
`auto` picks the XPU backend, and the run stops early if 12 devices are not visible.

The wrappers mount the checkout's `weights/` and `data/` by default, with the same
settings and defaults as `docker/run.sh` (see
[docker/README.md](../../docker/README.md#getting-weights-and-data-into-the-container)).
Set `BIOM3_WEIGHTS_DIR` / `BIOM3_DATA_DIR` to use other directories; they are
mount sources, so they must exist on the node. Binding the shared copy directly
needs its real layout — `BioM3-data-share/data/weights`, not
`BioM3-data-share/weights` — and a path that does not exist fails every rank at
container creation with `mount source ... doesn't exist`, before any Python runs.

The repo's `weights/` and `data/` entries are absolute symlinks into
`/lus/flare/projects/...`, so **both** launchers bind `/lus` as well as `/flare`
(on Aurora `/flare` *is* `/lus/flare/projects`). Binding only `/flare` leaves
every one of those links dangling, and the symptom appears far from the cause —
`transformers` reports a local model directory as a malformed Hub repo id
(`HFValidationError`) rather than "no such file". Verify with
`apptainer_run.sh ls -lL /app/weights/LLMs/`.

Host dirs bind onto `/app/{weights,data,outputs}`; `outputs/` is writable, weights
and data are read-only. See the script header for all `BIOM3_*` knobs.

## Collectives: what works and what doesn't

oneCCL inside the container needs several settings that bare metal gets for free.
All of them are handled by [apptainer_run.sh](../../scripts/aurora/apptainer_run.sh);
they are recorded here because the failure modes are opaque.

| Need | Why | Failure if missing |
| --- | --- | --- |
| `libze_loader.so` symlink | oneCCL `dlopen`s the unversioned name, which only the `-dev` package ships. torch is unaffected — it links `.so.1` directly. | `could not open the library: libze_loader.so`, then `ze_data was not initialized` on every collective |
| `FI_PROVIDER=tcp` | A shell with `module load frameworks` exports `cxi,tcp;ofi_rxm`; apptainer forwards it, and the container's libfabric has no cxi provider. | `fi_getinfo error: ret -61, providers 0` → `failed to initialize ATL` |
| `CCL_PROCESS_LAUNCHER=torchrun` | The host sets `pmix`, but no PMIx server is reachable in the container. `torchrun` reads `LOCAL_RANK`/`LOCAL_WORLD_SIZE`. | `PMIx_Init failed: PMIX_ERR_UNREACH` → `local_idx >= 0 && local_idx < local_count failed` |
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` | In the container the Level Zero fabric query reports no Xe Link between some tiles, so oneCCL treats the node as PCIe-connected and disables its device-to-device `topo` algorithm. Aurora's stacks are Xe Link connected, so the wrapper skips the check. | `topology recognition shows PCIe connection between devices`, repeated per rank, and slower collectives: 12-tile Stage 3 training ran at 0.38 it/s with the check versus 0.48 it/s without |
| `CCL_ATL_TRANSPORT=ofi` | Under torchrun oneCCL finds no MPI launcher and falls back to `ofi` anyway. | `did not find MPI-launcher specific variables, switch to ATL/OFI`, once per rank (harmless) |

**Single node** works with the above. GPU-to-GPU transfers use Level-Zero IPC
rather than the fabric, so the tcp provider carries only out-of-band traffic.

## Multi-node

Multi-node works, over Slingshot/CXI, and scales: two nodes run roughly twice a
single node's throughput. It needs the host's Cray libfabric bound in — see
[CXI](#cxi) — without which it falls back to `tcp` and ends up *slower* than one
node.

It needs a different image and a different launcher from the single-node path:

| | Single node | Multi-node |
| --- | --- | --- |
| Image | [Dockerfile.xpu](../../docker/Dockerfile.xpu) | [Dockerfile.xpu-oneapi](../../docker/Dockerfile.xpu-oneapi) |
| Test suite | 1134 passed / 162 skipped (before `frameworks/2026.1.0`) | 1520 passed / 2 failed / 95 skipped (`xpu-oneapi-abd9941`) |
| Launcher | [apptainer_run.sh](../../scripts/aurora/apptainer_run.sh) — one container, torchrun spawns ranks | [apptainer_mpi_run.sh](../../scripts/aurora/apptainer_mpi_run.sh) — host mpiexec spawns one container per rank |
| Rank source | PALS env vars translated to `RANK`/`LOCAL_RANK` | `MPIEnvironment` via mpi4py (`BIOM3_RANK_SOURCE=mpi`) |

The two failures in `xpu-oneapi-abd9941` are the Stage 1 checkpoint-loading
cases, which named a file the published test weights do not include. They now
load `run1_base_pencl.ckpt`, which the bundle does ship, so later images should
not show them.

The oneapi image exists because the Ubuntu-based one cannot do this: its mpi4py
is built against OpenMPI, so under Aurora's Intel MPI it never bootstraps,
Lightning falls back to a local environment, and every rank reports global rank
0. Intel's oneAPI base image (`intel/oneapi`) supplies an Intel MPI that matches
the host launcher.

```bash
# 2 nodes, 24 tiles. Run this from the shell `qsub -I` gives you: the wrapper
# reads $PBS_NODEFILE, which PBS sets only there. Do not `module load frameworks`
# — the container carries its own stack, and the module only exports host values
# the wrapper must override.
module load apptainer
SIF=/flare/NLDesignProtein/$USER/biom3_xpu-oneapi.sif    # the oneapi .sif you built
ls -d /opt/cray/libfabric/*/lib64                         # confirm BIOM3_FABRIC_DIR below

NGPU_PER_NODE=12 NGPU_TOTAL=24 BIOM3_RANK_SOURCE=mpi \
BIOM3_FABRIC_DIR=/opt/cray/libfabric/2.3.1/lib64 BIOM3_FI_PROVIDER=cxi \
BIOM3_IMAGE="$SIF" \
scripts/aurora/apptainer_mpi_run.sh \
    biom3_train_stage3 --config_path configs/stage3_training/pretrain_scratch_v1.json \
    --device auto --devices_per_node 12 --num_nodes 2 --run_id mn001 --epochs 2
```

There is no progress bar on this path: under `mpiexec` each rank's stdout is a pipe,
and `--progress_bar auto` shows the bar only on a terminal. Follow the run in
TensorBoard or W&B, or from the per-epoch validation lines; `--progress_bar True`
forces the bar, though the launcher may forward it in bursts.

One additional setting this path needs, handled by the wrapper:
`CCL_ZE_IPC_EXCHANGE=sockets`. Each rank is its own container with its own PID
namespace, so oneCCL's default `pidfd` handle exchange is denied
(`pidfd_getfd failed: ... Operation not permitted`). Under `apptainer_run.sh`
every rank is a torchrun child of one container, so it never arises.

### Measured throughput

Stage 3 pretraining, `pretrain_scratch_v1.json`, 86.2M params, batch 32/rank:

| Config | it/s | samples/s | read at step |
| --- | --- | --- | --- |
| bare metal, 12 ranks, 1 node | 0.62 | ~238 | 16 |
| container, 12 ranks, 1 node | 0.57 | ~219 | 7 |
| container, 24 ranks, 2 nodes, tcp | 0.17 | ~131 | 334 |
| container, 24 ranks, 2 nodes, **cxi** | **0.64** | **~492** | 499 |

Single-node containers cost roughly 8% against bare metal. Two nodes over CXI is
3.8x the tcp figure and about twice a single node; the single-node container
number was read very early, so treat the scaling ratio as approximate until both
are measured at the same step.

These figures are from the oneAPI 2025.3 image with Cray libfabric 1.22.0 and
have not been re-measured on the current stack.

### CXI

Aurora's CXI provider is in **HPE's Cray libfabric**, not the image and not
Intel MPI's bundled libfabric — that one ships efa/psm3/rxm/shm/tcp/verbs and no
cxi, under `/opt/aurora` as well as in the image. Point `BIOM3_FABRIC_DIR` at
the directory holding Cray's `libfabric.so.1`:

```
BIOM3_FABRIC_DIR=/opt/cray/libfabric/2.3.1/lib64 BIOM3_FI_PROVIDER=cxi
```

The version in that path follows Aurora's software stack (it was `1.22.0` until
October 2026), so confirm it on a compute node with
`ls -d /opt/cray/libfabric/*/lib64`. If the directory does not exist the wrapper
stops at once and names `BIOM3_FABRIC_DIR`.

The wrapper then binds it at `/hostfabric`, prepends it to `LD_LIBRARY_PATH`,
`LD_PRELOAD`s it, and sets `I_MPI_OFI_LIBRARY_INTERNAL=0`. All four are needed:
`LD_PRELOAD` because pip's oneCCL ships its own libfabric at
`/opt/venv/lib/libfabric.so.1` and finds it through RPATH, which
`LD_LIBRARY_PATH` cannot override; `I_MPI_OFI_LIBRARY_INTERNAL=0` because Intel
MPI otherwise prefers its own. Cray's libfabric needs `libcxi.so.1`, which is in
`/usr/lib64` and so arrives via the existing `/hostevent` bind.

`CCL_ATL_TRANSPORT` must be `mpi` here, which is the wrapper's default. On `ofi`
oneCCL opens libfabric providers itself and fails on cxi with
`fi_getinfo error: ret -61, providers 0`, even with Cray's library loaded and
`fi_info -p cxi` listing every domain from inside the container — it requests
capabilities the provider will not grant. Riding the Intel MPI that already works
over cxi avoids the question. (ALCF's recipe sets `ofi` because its container has
no usable MPI; that constraint does not apply to this image.)

See [oneCCL on Aurora](https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/)
and the container recipe in `_misc/sample_script.sh`.

## Troubleshooting

- **`torch.xpu.device_count()` is 0.** See which Level-Zero libraries the process
  loaded (`grep libze_ /proc/self/maps` from inside it). The loader
  (`libze_loader.so.1`) and the GPU driver (`libze_intel_gpu.so.1`) have to come
  from the same place: the image's loader with the host's driver finds no
  devices. That is what happened under `apptainer_mpi_run.sh` with the oneAPI
  2026.1 image, because the host's `/usr/lib64`, bound at `/hostevent`, sat on
  `LD_LIBRARY_PATH` ahead of the image's own library directory; the wrapper now
  puts the image's first. For the same reason, binding the host's driver into the
  image is not a dependable fallback. If the image's own driver sees no devices
  under a plain `apptainer exec`, it does not match Aurora's kernel driver and
  the image needs a different `libze-intel-gpu1`. Confirm the device nodes are
  visible with `clinfo` inside the container.
- **A one-rank run with `--distributed_strategy ddp` hangs at the first training
  step.** Seen with the oneAPI 2026.1 image built before the Stage 3 fix
  (`xpu-oneapi-fccf427`): a one-member process group left the main thread spinning
  in the Level-Zero driver. Stage 3 now uses a single-device strategy for one
  rank, so images from `xpu-oneapi-abd9941` on are not affected.
- **Dependency-conflict warning during build.** Expected — the `addison-nm/lightning`
  fork vs pyproject's pin — and harmless, same as the bare-metal Aurora install.
- **`OSError: [Errno 30] Read-only file system` (e.g. running the test suite).**
  The `.sif` is read-only, and some code writes into the image tree
  (`/app/tests/_tmp`, `.pytest_cache`). `apptainer_run.sh` passes
  `--writable-tmpfs` (an ephemeral RAM-backed overlay) to absorb these; if you
  invoke `apptainer exec` by hand, add `--writable-tmpfs` yourself.
- **`OSError: [Errno 28] No space left on device` in the test suite.** The
  `--writable-tmpfs` overlay is small. `apptainer_run.sh` mounts `<outputs>/tests_tmp`
  at `/app/tests/_tmp` for this; if you invoke `apptainer exec` by hand, bind a host
  dir there yourself (`--bind <dir>:/app/tests/_tmp`).
- **`Failed to create user namespace` on `apptainer exec`.** You are on a login
  node. Building the `.sif` works there, but running it needs a compute node.
