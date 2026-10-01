#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/run.sh
#
# Convenience wrapper for `docker run`: mounts the conventional BioM3
# directories, runs as the calling user (so files written to mounts are owned
# by you), forwards the useful env vars, then passes through whatever command
# you give it.
#
# USAGE:
#   docker/run.sh <command...>
#
# EXAMPLES:
#   # interactive shell
#   docker/run.sh bash
#
#   # single-GPU Stage 3 training
#   docker/run.sh scripts/stage3_train_singlenode.sh \
#       configs/stage3_training/pretrain_scratch_v1.json 1 cuda run001 --epochs 1
#
#   # 4-GPU training (NGPU drives torchrun inside the container)
#   BIOM3_GPUS=all NGPU=4 docker/run.sh scripts/stage3_train_singlenode.sh \
#       configs/stage3_training/pretrain_scratch_v1.json 4 cuda run001 --epochs 1
#
#   # generation
#   docker/run.sh biom3_ProteoScribe_sample --input_path outputs/facilitator.pt \
#       --config_path configs/inference/stage3_ProteoScribe_sample.json \
#       --model_path weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
#       --output_path outputs/generated.pt --device cuda
#
# SYMLINKED WEIGHTS/DATA: a symlink inside a mounted directory is resolved
# inside the container, so wherever an absolute link points must be mounted
# too. Put those host directories in BIOM3_BIND_EXTRA (see docker/README.md).
#
# ENV (all optional):
#   BIOM3_IMAGE        image tag (default: biom3:cuda)
#   BIOM3_DEVICE_KIND  cuda | xpu | cpu (default: inferred from the image tag) —
#                      selects --gpus (cuda), --device /dev/dri (xpu), or no
#                      device flags at all (cpu)
#   BIOM3_GPUS         value for --gpus on cuda (default: all; "none" omits it)
#   BIOM3_WEIGHTS_DIR, BIOM3_DATA_DIR, BIOM3_OUTPUTS_DIR, BIOM3_TESTS_TMP,
#   BIOM3_CONFIGS_DIR, BIOM3_BIND_EXTRA
#                      mounts, shared with the Apptainer wrappers: see
#                      scripts/_container_mounts.sh. Host weights are not
#                      mounted when BIOM3_WEIGHTS_BUNDLE is set.
#   BIOM3_AS_ROOT      1 = run as root in the container instead of as the
#                      calling user
#   Forwarded if set:  WANDB_API_KEY, NGPU and the GHCR weights-bundle vars
#                      (BIOM3_WEIGHTS_BUNDLE, BIOM3_WEIGHTS_BUNDLE_REPO,
#                      BIOM3_SYNC_MODE, GHCR_TOKEN, GHCR_USER).
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../scripts/_container_mounts.sh"

IMAGE="${BIOM3_IMAGE:-biom3:cuda}"
GPUS="${BIOM3_GPUS:-all}"

# Device kind: explicit override, else infer from the image tag (":xpu" -> xpu).
if [[ -n "${BIOM3_DEVICE_KIND:-}" ]]; then
    DEVICE_KIND="${BIOM3_DEVICE_KIND}"
elif [[ "${IMAGE}" == *:xpu || "${IMAGE}" == *:xpu-* ]]; then
    DEVICE_KIND="xpu"
elif [[ "${IMAGE}" == *:cpu || "${IMAGE}" == *:cpu-* ]]; then
    DEVICE_KIND="cpu"
else
    DEVICE_KIND="cuda"
fi

ARGS=(run --rm)
[[ -t 0 && -t 1 ]] && ARGS+=(-it)
[[ "${BIOM3_AS_ROOT:-0}" == "1" ]] || ARGS+=(--user "$(id -u):$(id -g)")
if [[ "${DEVICE_KIND}" == "xpu" ]]; then
    # Intel GPU: expose the DRI render nodes + render/video group membership.
    ARGS+=(--device /dev/dri)
    for grp in render video; do
        gid="$(getent group "${grp}" 2>/dev/null | cut -d: -f3)"
        [[ -n "${gid}" ]] && ARGS+=(--group-add "${gid}")
    done
elif [[ "${DEVICE_KIND}" != "cpu" && "${GPUS}" != "none" ]]; then
    ARGS+=(--gpus "${GPUS}")
fi

# With a weights bundle, the entrypoint pulls into the container's own
# /app/weights, which a host mount would shadow.
if [[ -n "${BIOM3_WEIGHTS_BUNDLE:-}" ]]; then
    biom3_container_mounts --no-weights || exit 1
else
    biom3_container_mounts || exit 1
fi
for m in "${MOUNTS[@]}"; do ARGS+=(-v "${m}"); done

# Forward env vars that are set in the caller's environment. `-e NAME` copies
# the value from this process, keeping secrets off the docker command line.
for v in WANDB_API_KEY NGPU \
         BIOM3_WEIGHTS_BUNDLE BIOM3_WEIGHTS_BUNDLE_REPO BIOM3_SYNC_MODE \
         GHCR_TOKEN GHCR_USER; do
    [[ -n "${!v:-}" ]] && ARGS+=(-e "${v}")
done

exec docker "${ARGS[@]}" "${IMAGE}" "$@"
