#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage3_finetune_multinode.sh
#
# USAGE: stage3_finetune_multinode.sh CONFIG_PATH NUM_NODES NGPU_PER_NODE \
#        DEVICE RUN_ID [additional --key value overrides]
#
#   DEVICE: cuda | xpu | cpu | auto (auto = the detected GPU backend; never CPU)
#
# DESCRIPTION: Multi-node wrapper for Stage 3 GENERALIZED finetuning
#   (biom3_finetune_stage3 / run_ProteoScribe_finetuning). Unlike
#   stage3_train_multinode.sh (which drives biom3_train_stage3 on precomputed
#   z_c HDF5 data), this dispatches the JSONL record_schema finetuning path:
#   captions are composed on the fly and embedded to z_c on-device via the
#   frozen PenCL text branch + Facilitator.
#
#   The JSON config provides model/training hyperparameters, the record_schema,
#   the finetuning data path, and the embedder configs/weights; per-job
#   overrides (epochs, resume, finetune flags, etc.) are passed via "$@".
#   Wandb logging is resolved by scripts/_wandb_resolve.sh.
#
#   Requires: source environment.sh first so BIOM3_MACHINE is set.
#   PBS_NODEFILE must be set by PBS at submission time.
#
#=============================================================================
set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 CONFIG_PATH NUM_NODES NGPU_PER_NODE DEVICE RUN_ID [--key value ...]"
    echo "Wandb: pass --wandb True|False to override; defaults to True iff WANDB_API_KEY is set."
    exit 1
fi

config_path=$1
NUM_NODES=$2
NGPU_PER_NODE=$3
device=$4
run_id=$5
shift 5

NGPU_TOTAL="$((NUM_NODES * NGPU_PER_NODE))"

echo "NUM_NODES: ${NUM_NODES}, NGPU_PER_NODE: ${NGPU_PER_NODE}, NGPU_TOTAL: ${NGPU_TOTAL} (${device})"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

# Resolve machine-specific launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve wandb (sets `wandb_resolved`; errors if --wandb True without API key)
source "${SCRIPT_DIR}/_wandb_resolve.sh" "$@"
MACHINE="${BIOM3_MACHINE:?BIOM3_MACHINE not set; source environment.sh first}"
LAUNCHER="${SCRIPT_DIR}/launchers/${BIOM3_LAUNCHER:-${MACHINE}}_multinode.sh"

if [ ! -x "${LAUNCHER}" ]; then
    echo "ERROR: no launcher for ${MACHINE} multinode at ${LAUNCHER}"
    exit 1
fi

# Export args the launcher reads from env
export NGPU_PER_NODE NGPU_TOTAL

exec "${LAUNCHER}" \
    biom3_finetune_stage3 \
        --config_path "${config_path}" \
        --run_id "${run_id}" \
        --device "${device}" \
        --num_nodes "${NUM_NODES}" \
        --devices_per_node "${NGPU_PER_NODE}" \
        ${wandb_resolved} \
        "$@"
