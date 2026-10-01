#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/aurora/apptainer_run.sh
#
# Run a command inside the BioM3 Intel-XPU .sif on Aurora, single node. Binds
# /flare, sources environment.sh inside the container (so it auto-detects
# `aurora` and applies the oneCCL/xccl/NUMEXPR settings), then runs whatever
# command you pass.
#
# This is the container replacement for the bare-metal `module load frameworks +
# source venv + source environment.sh` prelude. SINGLE NODE only — multi-node
# runs use apptainer_mpi_run.sh (host mpiexec, one container per rank; see
# docs/setup/setup_aurora_container.md).
#
# USAGE:
#   scripts/aurora/apptainer_run.sh <command...>
#
# EXAMPLES:
#   # interactive shell (GPUs visible; try `python -c "import torch;print(torch.xpu.device_count())"`)
#   scripts/aurora/apptainer_run.sh bash
#
#   # single-node Stage 3 training on 12 tiles
#   scripts/aurora/apptainer_run.sh scripts/stage3_train_singlenode.sh \
#       configs/stage3_training/pretrain_scratch_v1.json 12 auto run001 --epochs 1
#
# ENV (all optional):
#   BIOM3_IMAGE        path to the .sif (default: ./biom3_xpu.sif; the older
#                      BIOM3_SIF is still read)
#   BIOM3_WEIGHTS_DIR, BIOM3_DATA_DIR, BIOM3_OUTPUTS_DIR, BIOM3_TESTS_TMP,
#   BIOM3_CONFIGS_DIR, BIOM3_BIND_EXTRA
#                      mounts, shared with docker/run.sh: see
#                      scripts/_container_mounts.sh
#   BIOM3_FI_PROVIDER  libfabric provider (default tcp; cxi needs host binds)
#   BIOM3_CCL_LAUNCHER CCL_PROCESS_LAUNCHER (default torchrun; hydra|pmix|none)
#   BIOM3_CCL_ATL_TRANSPORT  CCL_ATL_TRANSPORT (default ofi)
#   BIOM3_CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK
#                      CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK (default 0; 1
#                      restores oneCCL's own topology detection)
#   WANDB_API_KEY      forwarded into the container if set
#
#=============================================================================
set -euo pipefail

[[ $# -ge 1 ]] || { echo "USAGE: $0 <command...>   (see --help header)" >&2; exit 1; }
[[ "$1" == "-h" || "$1" == "--help" ]] && { sed -n '3,/^#====/p' "$0"; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../_container_mounts.sh"

SIF="${BIOM3_IMAGE:-${BIOM3_SIF:-./biom3_xpu.sif}}"
[[ -f "${SIF}" ]] || { echo "ERROR: sif '${SIF}' not found. Build it (on a login node):" >&2
    echo "         apptainer build biom3_xpu.sif docker://ghcr.io/natural-machine/biom3:xpu-dev" >&2
    echo "       see docs/setup/setup_aurora_container.md, or set BIOM3_IMAGE." >&2; exit 1; }

command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found." >&2; exit 1; }

# --- Binds ---------------------------------------------------------------
# /flare   : ALCF Lustre; also what environment.sh fingerprints to pick `aurora`.
#
# Do NOT bind /dev/dri. Apptainer mounts /dev by default; adding /dev/dri as a
# user bind remounts it `nodev`, so the GPU character devices become unusable
# and torch.xpu.device_count() returns 0 (clinfo -l also comes back empty).
# The shared mounts include /app/tests/_tmp, which matters here: the test
# suite writes its scratch inside the image, and the --writable-tmpfs overlay
# is too small for it ("No space left on device").
biom3_container_mounts || exit 1
BINDS=("${MOUNTS[@]}")
# /lus alongside /flare: on Aurora /flare IS /lus/flare/projects, and the
# weights/ and data/ trees are symlinks whose targets are spelled /lus/...
# Binding only /flare leaves every one of them dangling inside the container --
# which surfaces far from the cause, e.g. transformers reporting a local model
# directory as a malformed Hub repo id.
[[ -d /flare ]] && BINDS+=("/flare")
[[ -d /lus ]] && BINDS+=("/lus")

BIND_ARG="$(IFS=,; echo "${BINDS[*]}")"

# --- Env into the container ----------------------------------------------
# ZE_FLAT_DEVICE_HIERARCHY=FLAT exposes each of Aurora's 12 tiles as its own
# device (matches num_devices=12 in the PBS templates). On bare metal the
# frameworks module sets this; the container must set it itself.
ENVS=(--env "ZE_FLAT_DEVICE_HIERARCHY=FLAT")

# oneCCL settings that MUST differ from bare metal, because a shell with
# `module load frameworks` exports values that are wrong inside the container
# and apptainer forwards them:
#
#   FI_PROVIDER — the host asks for `cxi,tcp;ofi_rxm`, but the container's
#     libfabric has no cxi provider, so libfabric matches nothing and oneCCL
#     fails with "fi_getinfo error: ret -61, providers 0". tcp is correct for
#     single node (GPU transfers go over Level-Zero IPC, not the fabric).
#     Multi-node over CXI needs the host libfabric bound in; see the runbook.
#
#   CCL_PROCESS_LAUNCHER — the host sets `pmix`, but no PMIx server is
#     reachable in the container, so oneCCL cannot resolve local rank/size.
#     `torchrun` reads LOCAL_RANK/LOCAL_WORLD_SIZE, which covers both a single
#     process and a multi-rank launch.
#   CCL_ROOT — the host points at /opt/aurora/<ver>/oneapi/ccl/latest, which
#     does not exist in the container, so oneCCL cannot find its SPIR-V kernels
#     ("failed to load file containing oneCCL SPIR-V kernels"). The pip-installed
#     oneCCL keeps them under /opt/venv/lib/ccl/kernels.
ENVS+=(--env "FI_PROVIDER=${BIOM3_FI_PROVIDER:-tcp}")
ENVS+=(--env "CCL_PROCESS_LAUNCHER=${BIOM3_CCL_LAUNCHER:-torchrun}")
ENVS+=(--env "CCL_ROOT=/opt/venv")

#   CCL_ATL_TRANSPORT=ofi — under torchrun oneCCL finds no MPI launcher and
#     falls back to ofi anyway, printing a warning per rank; start there.
#
#   CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 — inside the container the Level
#     Zero fabric query reports no Xe Link between some tiles, so oneCCL warns
#     "topology recognition shows PCIe connection" and disables its topo
#     (device-to-device) algorithm for intra-node collectives. Aurora's stacks
#     are Xe Link connected, so skip the check and let oneCCL assume the links.
ENVS+=(--env "CCL_ATL_TRANSPORT=${BIOM3_CCL_ATL_TRANSPORT:-ofi}")
ENVS+=(--env "CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=${BIOM3_CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK:-0}")

# Spawn ranks with torchrun (launchers/container_singlenode.sh), not the host's
# mpiexec: the container has no access to PBS's hostfile, so Hydra fails with
# "unable to find an RMK and the node list". BIOM3_MACHINE stays `aurora`, so
# the Aurora oneCCL/NUMEXPR settings still apply.
ENVS+=(--env "BIOM3_LAUNCHER=${BIOM3_LAUNCHER:-container}")

[[ -n "${WANDB_API_KEY:-}" ]] && ENVS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

# The lightning fork imports pkg_resources, whose deprecation notice every rank
# would print. Only when the caller has no PYTHONWARNINGS of their own: apptainer
# splits --env values on commas, so the two cannot be combined.
[[ -z "${PYTHONWARNINGS:-}" ]] && \
    ENVS+=(--env "PYTHONWARNINGS=ignore:pkg_resources is deprecated as an API")

# Apptainer warns about every host variable that an --env above replaces. The
# host values are wrong inside the container, so drop them from this process.
for e in "${ENVS[@]}"; do [[ "${e}" == --env ]] || unset "${e%%=*}"; done

# `exec` (not `run`) so we bypass the image entrypoint and instead source
# environment.sh ourselves — that is what applies the Aurora oneCCL/xccl vars.
# The passed command runs from /app with "$@" preserved.
set -- bash -lc 'cd /app && source environment.sh >&2 && exec "$@"' _ "$@"

# `--writable-tmpfs`: ephemeral RAM-backed overlay so incidental writes to the
# read-only image (caches, tests/_tmp) succeed; real outputs go to /app/outputs.
# `--quiet` drops apptainer's INFO lines (e.g. "gocryptfs not found"); its
# warnings and errors still print.
echo "+ apptainer exec --writable-tmpfs --bind ${BIND_ARG} ${SIF} <cmd>" >&2
exec apptainer --quiet exec --writable-tmpfs --bind "${BIND_ARG}" "${ENVS[@]}" "${SIF}" "$@"
