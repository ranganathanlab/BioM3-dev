#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage3_train_singlenode.sh
#
# USAGE: stage3_train_singlenode.sh CONFIG_PATH NGPU DEVICE \
#        RUN_ID [additional --key value overrides]
#
#   DEVICE: cuda | xpu | cpu | auto (auto = the detected GPU backend; never CPU)
#
# DESCRIPTION: Single-node wrapper for Stage 3 training (pretraining and
#   finetuning). Dispatches to the machine-specific launcher under
#   scripts/launchers/${BIOM3_LAUNCHER:-${BIOM3_MACHINE}}_singlenode.sh, which knows how to
#   spawn ranks (mpiexec on Aurora/Polaris, bare exec on Spark).
#
#   The JSON config provides model/training hyperparameters; per-job
#   overrides (epochs, resume, finetune flags, etc.) are passed via "$@".
#   Wandb logging is resolved by scripts/_wandb_resolve.sh: explicit
#   --wandb True|False in the args wins; otherwise it defaults to True if
#   WANDB_API_KEY is set, else False. --wandb True without WANDB_API_KEY
#   errors out.
#
#   Requires: source environment.sh first so BIOM3_MACHINE is set.
#
#=============================================================================
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 CONFIG_PATH NGPU DEVICE RUN_ID [--key value ...]"
    echo "Wandb: pass --wandb True|False to override; defaults to True iff WANDB_API_KEY is set."
    exit 1
fi

config_path=$1
NGPU=$2
device=$3
run_id=$4
shift 4

echo "Single-node: NGPU=${NGPU} (${device})"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

# Resolve machine-specific launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve wandb (sets `wandb_resolved`; errors if --wandb True without API key)
source "${SCRIPT_DIR}/_wandb_resolve.sh" "$@"
MACHINE="${BIOM3_MACHINE:?BIOM3_MACHINE not set; source environment.sh first}"

# BIOM3_LAUNCHER decouples "how ranks are spawned" from "whose settings apply".
# Under Apptainer on Aurora, BIOM3_MACHINE is `aurora` (so the oneCCL/xccl vars
# get applied) but ranks must be spawned by torchrun, not the host's mpiexec:
# the container cannot read PBS's hostfile. apptainer_run.sh sets this to
# `container`.
LAUNCHER="${SCRIPT_DIR}/launchers/${BIOM3_LAUNCHER:-${MACHINE}}_singlenode.sh"

if [ ! -x "${LAUNCHER}" ]; then
    echo "ERROR: no launcher for ${MACHINE} singlenode at ${LAUNCHER}"
    exit 1
fi

# Export args the launcher reads from env
export NGPU

exec "${LAUNCHER}" \
    biom3_train_stage3 \
        --config_path "${config_path}" \
        --run_id "${run_id}" \
        --device "${device}" \
        --num_nodes 1 \
        --devices_per_node "${NGPU}" \
        ${wandb_resolved} \
        "$@"
