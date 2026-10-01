# Container interface design: run BioM3 from the image alone

Status: draft for review. Nothing here is implemented.

## Goal

Someone who has only a published image — Docker, or a `.sif` converted from it — should
be able to:

1. Point the run at **their own** weights, data, configs and outputs directories.
2. Type the **same `biom3_*` command** whether it is prefixed by `docker run`,
   `apptainer exec`, or `mpiexec … apptainer exec`.
3. Not have to know any Aurora-specific environment variables.

No repo clone, no host-side wrapper scripts from this repository.

## Non-goals

- Removing the host's own launcher. Multi-node on an HPC site needs that site's `mpiexec`
  and its PBS hostfile; no image can supply them.
- Removing bind mounts or GPU flags. Those are the user's to choose and are accepted as
  part of the runtime prefix.
- Deleting `docker/run.sh` or the Apptainer wrappers. They stay as conveniences, but stop
  being the only supported path.

## What already works

The image bakes `src/`, `scripts/`, `tests/`, `configs/` and `environment.sh` with
`WORKDIR /app`, so nothing the entry points need at runtime is missing. Every shipped
config uses paths relative to the working directory (`./weights/…`, `./data/…`,
`./outputs/…`), so **fixed container paths with arbitrary host paths** already works:

```bash
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -v /my/weights:/app/weights:ro \
  -v /my/data:/app/data:ro \
  -v /my/outputs:/app/outputs \
  ghcr.io/natural-machine/biom3:cuda-dev \
  biom3_ProteoScribe_sample -i … -o outputs/gen.pt --fasta
```

Single-process runs, and anything launched by an external `mpiexec`, already meet goal 2.

## What blocks it

### 1. A user configs mount shadows the shipped base configs

Verified by running it:

```
$ docker run --rm -v ./myconfigs:/app/configs:ro <image> \
    python -c "from biom3.core.helpers import load_json_config; load_json_config('/app/configs/pretrain_scratch_v1.json')"
FileNotFoundError: '/app/configs/./models/_base_ProteoScribe_1block.json'
```

`_base_configs` resolves relative to the config file's own directory, so replacing
`/app/configs` removes the `models/` and `machines/` bases every shipped config composes
from.

### 2. Single-node multi-GPU requires naming a launcher script

Today: `NGPU=4 … bash scripts/launchers/container_singlenode.sh biom3_train_stage3 …`.
That is a different command from the single-GPU one, so goal 2 fails for this one case.

### 3. The Aurora environment lives in host wrappers

`scripts/aurora/apptainer_{run,mpi_run}.sh` set roughly a dozen `CCL_*`, `ZE_*`,
`FI_*` and `I_MPI_*` variables. Without the repo, the user would have to supply them by
hand, which violates goal 3.

## Why the Aurora variables cannot simply be baked in

They are not machine constants. Sorting them by what they are actually a function of:

| Function of | Variables | Bakeable |
| --- | --- | --- |
| Machine + image | `NUMEXPR_MAX_THREADS`, `TMPDIR`, `CCL_ATL_SYNC_COLL`, `CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD`, `CCL_ROOT` | Yes |
| Launch topology | `CCL_ATL_TRANSPORT` (`ofi` vs `mpi`), `CCL_ZE_IPC_EXCHANGE`, `CCL_PROCESS_LAUNCHER` | No |
| Entry-point rank layout | `ZE_FLAT_DEVICE_HIERARCHY`, `ZE_AFFINITY_MASK`, `CCL_WORKER_AFFINITY` | No |
| Host paths | `LD_PRELOAD`, `FI_PROVIDER_PATH`, `I_MPI_PMI_LIBRARY` | No |

Only the first row is invariant. Two concrete counter-examples for the rest:

- `apptainer_run.sh` sets `CCL_ATL_TRANSPORT=ofi`; `apptainer_mpi_run.sh` sets `mpi`. Same
  machine, different launch topology, different value — with a comment at
  [`apptainer_mpi_run.sh:147`](../../scripts/aurora/apptainer_mpi_run.sh#L147) explaining
  that `ofi` fails outright on the one-container-per-rank path.
- The container wrappers force `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (12 tiles), but
  [`run_gdpo_smoke_multixpu.sh`](../../scripts/run_gdpo_smoke_multixpu.sh) never sets it
  and states it expects `torch.xpu.device_count() == 6`, i.e. the COMPOSITE default.
  Baking `FLAT` would hand that path 12 tiles where it expects 6 GPUs.

`ZE_AFFINITY_MASK` is not inert either: [`backend/xpu.py:26`](../../src/biom3/backend/xpu.py#L26)
resolves to `xpu:0` whenever it is set, and [`backend/device.py:80`](../../src/biom3/backend/device.py#L80)
skips the device-count check on it.

## The approach: profiles in the image, one variable to select

Keep the whole matrix inside the image, versioned with the code, and let the caller name
their run shape once:

```bash
BIOM3_PROFILE=aurora-tile-mpi
```

The user never types `CCL_ZE_IPC_EXCHANGE` and never reasons about FLAT versus COMPOSITE.
Each profile is a combination that has been validated on hardware, rather than a guess
assembled per run. The profile name is an **explicit** signal, chosen by whoever also
chose `mpiexec` versus plain `docker run` — it is not inferred. (Inference was considered
and rejected: `is_launched()` returns false under `srun` and true for a stray
`BIOM3_WORLD_SIZE`, so it cannot carry control flow. See the note at the end.)

### Initial layout

`environment.sh` keeps its current job — common settings, then a machine branch — and
gains a profile layer applied after the machine branch.

```
environment.sh
  ├── common            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
  ├── machine branch    fingerprint /flare → aurora, /grand → polaris, …
  │                     invariant facts only (row 1 of the table above)
  └── profile layer     BIOM3_PROFILE → topology + layout variables
                        (defaults to the machine's plain single-process profile)
```

Proposed profile set, drawn from what the wrappers and PBS files currently do. **The
values are transcribed from existing code; the set itself needs your confirmation** — see
open questions.

| Profile | Run shape | Sets |
| --- | --- | --- |
| `aurora-tile-torchrun` | 1 node, 1 rank/tile, torchrun inside one container | `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `CCL_ATL_TRANSPORT=ofi`, `CCL_PROCESS_LAUNCHER=torchrun`, `CCL_WORKER_AFFINITY=<12-slot mask>` |
| `aurora-tile-mpi` | N nodes, 1 rank/tile, one container per rank | as above but `CCL_ATL_TRANSPORT=mpi`, `CCL_ZE_IPC_EXCHANGE=sockets`, `I_MPI_PMI_LIBRARY=/hostlib/libpmix.so.2` |
| `aurora-node-mpi` | RL/GDPO, 1 rank/node owning all local tiles | as `aurora-tile-mpi`, plus unset `ZE_AFFINITY_MASK` and unset `CCL_WORKER_AFFINITY` |
| `aurora-single-tile` | single-process RL smoke | `ZE_AFFINITY_MASK=0` |
| `container` | CUDA, torchrun or single process | nothing beyond the common settings |

Invariants that stay in the machine branch, not the profiles: `NUMEXPR_MAX_THREADS=64`,
`TMPDIR=/tmp`, `CCL_ATL_SYNC_COLL=1`, `CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=10000`,
`CCL_ROOT=/opt/venv`.

Every profile variable keeps its existing `BIOM3_*` override, so a one-off experiment does
not need a new profile.

### Applying it without a wrapper

`environment.sh` must run for every `biom3_*` invocation, however the container was
entered. A Docker `ENTRYPOINT` covers `docker run` but is bypassed by `apptainer exec`,
which is how both Aurora wrappers invoke the image. So the hook belongs in Python, in one
shared place the console scripts import first:

- Source `environment.sh` in a subshell, capture the resulting environment, apply the diff
  to `os.environ`. `environment.sh` stays the single source of truth — nothing is ported,
  so nothing drifts.
- It sets no `LD_*` variables, so this works without re-exec. (Confirmed: `grep -E
  'LD_LIBRARY_PATH|LD_PRELOAD' environment.sh` is empty.) Variables that must precede
  library initialisation are applied before `torch` is imported.
- Guard with a sentinel so torchrun children do not redo it.

### Two supporting changes

1. **Drop `ENV BIOM3_MACHINE=container` from `Dockerfile.cuda` and `Dockerfile.cpu`.**
   `Dockerfile.xpu` and `Dockerfile.xpu-oneapi` already omit it deliberately so the
   `/flare` fingerprint wins inside the container. With cuda and cpu matching, `/grand` →
   `polaris` works too, which removes the reason `scripts/polaris/apptainer_run.sh` has to
   `unset BIOM3_MACHINE`, and dissolves open item 2 of the validation matrix rather than
   patching it.
2. **`BIOM3_NPROC=N` re-execs the entry point under `torchrun --standalone
   --nproc-per-node=N`.** Explicit, so it cannot double-spawn under `mpiexec`, where it is
   simply unset. This closes blocker 2.

### The mount contract

| Container path | Holds | Mode |
| --- | --- | --- |
| `/app/weights` | user weights | ro |
| `/app/data` | user data | ro |
| `/app/outputs` | run outputs | rw |
| `/configs` | **user** configs, passed as `--config_path /configs/x.json` | ro |

`/app/configs` stays as shipped and is never mounted over. A user config may reference the
shipped bases by absolute path (`/app/configs/stage3_training/models/_base_*.json`);
`_resolve_config_paths` honors absolute paths
([`core/helpers.py:22`](../../src/biom3/core/helpers.py#L22)).

### What the user still supplies

- bind mounts, per the table above
- GPU flags: `--gpus all` (Docker CUDA), `--nv` (Apptainer CUDA), `--device /dev/dri` plus
  render/video groups (Docker XPU)
- `BIOM3_PROFILE`, and `BIOM3_NPROC` for single-node multi-GPU
- multi-node only: the site `mpiexec`, the PBS hostfile, and the host PMIx and Cray
  libfabric binds with the `LD_PRELOAD`/`FI_PROVIDER_PATH` they require

### Target commands

```bash
# Docker, 4 CUDA GPUs
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -v /my/weights:/app/weights:ro -v /my/data:/app/data:ro \
  -v /my/configs:/configs:ro    -v /my/outputs:/app/outputs \
  -e BIOM3_NPROC=4 <image> \
  biom3_train_stage3 --config_path /configs/mine.json --devices_per_node 4 --run_id r1

# Apptainer, Aurora, 1 node, 12 tiles
BIOM3_PROFILE=aurora-tile-torchrun BIOM3_NPROC=12 \
apptainer exec --writable-tmpfs \
  --bind /my/weights:/app/weights:ro,/my/data:/app/data:ro,/my/configs:/configs:ro,/my/outputs:/app/outputs \
  biom3_xpu.sif \
  biom3_train_stage3 --config_path /configs/mine.json --devices_per_node 12 --run_id r1

# Apptainer under the site mpiexec, Aurora, 2 nodes
BIOM3_PROFILE=aurora-tile-mpi \
mpiexec -n 24 --ppn 12 --hostfile "$PBS_NODEFILE" \
  apptainer exec --writable-tmpfs --bind <mounts>,<pmix>,<libfabric> biom3_xpu-oneapi.sif \
  biom3_train_stage3 --config_path /configs/mine.json --devices_per_node 12 --num_nodes 2 --run_id r1
```

The `biom3_train_stage3` line is identical in all three.

## Open questions

1. **Is the profile set right?** The table is transcribed from the wrappers and PBS files,
   which means the *set* is inferred from scattered code. Two prior attempts to infer
   intent that way were wrong this session, so this needs confirming rather than trusting.
2. **Does RL want FLAT or COMPOSITE?** `aurora_multinode_rl.sh` unsets `ZE_AFFINITY_MASK`
   "so each rank sees all tiles", which under FLAT is 12, while
   `run_gdpo_smoke_multixpu.sh` expects 6. These may be two different profiles rather than
   one.
3. **Does `apptainer exec` really bypass the Docker `ENTRYPOINT`?** The Python-hook
   decision rests on this. Apptainer is not installed on the Spark, so it was not verified
   — one command on Aurora settles it.
4. **Should `/configs` be a blessed mount point**, or should any absolute `--config_path`
   be enough with no convention at all?

## Suggested sequence

Validation belongs on Aurora, so the ordering minimises round trips there.

1. On Aurora, answer open question 3 (`apptainer exec` vs `ENTRYPOINT`) and dump the
   environment of one working `aurora-tile-mpi` run as the reference to reproduce.
2. Add the profile layer to `environment.sh` with the profile set confirmed from question
   1, and the Python hook. No Dockerfile change yet, so the existing wrappers keep working
   and can be diffed against the new path.
3. Check equivalence: for each profile, the environment produced by the wrapper and by
   `BIOM3_PROFILE=…` should match. This is a `diff` of two `env` dumps, not a training
   run, so it is cheap and does not need a GPU allocation.
4. Only then drop `ENV BIOM3_MACHINE=container` and add `BIOM3_NPROC`, rebuild, and re-run
   the affected rows of [`container_validation.md`](../setup/container_validation.md).

Steps 1 and 3 are the ones that need Aurora; 2 and 4 are local.

## Note on why the topology is not inferred

An earlier version of this design keyed the launcher decision on
`biom3.core._dist_env.is_launched()`. Measured behavior, with a reverted tree:

| Environment | `is_launched()` | Correct? |
| --- | --- | --- |
| plain `docker run` | False | yes |
| `srun` (SLURM), 8 tasks | False | **no** — would spawn torchrun inside all 8 tasks |
| `mpiexec`/PALS | True | yes |
| torchrun | True | yes |

`_dist_env` contains no `SLURM_*` variables. SLURM is not in use today, so this is not an
active bug, but it shows the predicate is a property of which launchers happen to be
listed rather than of reality — which is why the profile name is explicit instead.
