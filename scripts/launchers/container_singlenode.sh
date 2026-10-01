#!/usr/bin/env bash
#=============================================================================
#
# FILE: container_singlenode.sh
#
# USAGE: container_singlenode.sh ENTRYPOINT [args...]
#
# DESCRIPTION: Single-node launcher inside a BioM3 Docker container, on any
#   host (a workstation or a cloud GPU instance; BIOM3_MACHINE=container).
#   Required env: NGPU (number of GPUs visible in the container).
#
#   Unlike the ALCF HPC launchers (Aurora/Polaris), there is no PBS, no
#   mpiexec, and no CPU/GPU binding to manage inside a single container.
#   We use torch's native elastic launcher:
#     - NGPU == 1: exec the entry point directly (single process; PyTorch
#       Lightning runs with the LightningEnvironment, no rendezvous).
#     - NGPU  > 1: torchrun spawns one process per GPU and sets
#       RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
#       biom3.core._dist_env already reads these, and PyTorch Lightning
#       auto-detects the torchelastic environment (it will not re-spawn).
#
#   --standalone picks a free MASTER_PORT on 127.0.0.1, so concurrent
#   containers on the same host can't collide.
#
#   Multi-node (across instances) is container_multinode.sh.
#
#=============================================================================
set -euo pipefail

NGPU="${NGPU:?NGPU env var required}"

if [ "${NGPU}" -le 1 ]; then
    exec "$@"
fi

# torchrun launches a Python *script file*, not a console-script name on
# PATH. The biom3 entry points (e.g. biom3_train_stage3) are generated
# console scripts, so resolve the first token to its file path; torchrun
# then runs it under `python`. Fall back to the token verbatim if it's
# already a path.
entry="$1"; shift
entry_path="$(command -v "${entry}" || true)"
[ -n "${entry_path}" ] || entry_path="${entry}"

exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NGPU}" \
    "${entry_path}" \
    "$@"
