#!/usr/bin/env bash
#=============================================================================
#
# FILE: entrypoint.sh  (shared by every BioM3 image)
#
# Optionally pulls a published GHCR weights bundle into /app/weights, then
# exec's the requested command (an interactive shell, a biom3_* CLI, or a
# scripts/*_train_*.sh wrapper).
#
# Weights, data and outputs are otherwise up to the caller: bind-mount them
# (see docker/run.sh, docker/README.md), or fetch them as part of the command.
#
# OPTIONAL GHCR weights bundle (no credentials for a public bundle):
#   BIOM3_WEIGHTS_BUNDLE    e.g. run1_base -> `oras pull` the published bundle
#                           into /app/weights. The bundle's weights/ tree uses
#                           the same layout and filenames as configs/weights/,
#                           so `--weight_set configs/weights/<name>.json` then
#                           resolves. Public bundles need no login; a private
#                           one needs an oras login (or GHCR_TOKEN, used here
#                           to authenticate).
#   BIOM3_WEIGHTS_BUNDLE_REPO  override the default bundle repo.
#   BIOM3_SYNC_MODE         auto (default) | always | never
#                             auto   = skip if /app/weights already has files
#                             always = pull even if it does
#                             never  = skip entirely
#
#=============================================================================
set -euo pipefail

# BIOM3_MACHINE and TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD come from the image's ENV.
# The xpu images leave BIOM3_MACHINE unset; their Apptainer wrappers source
# environment.sh, which detects `aurora`.

SYNC_MODE="${BIOM3_SYNC_MODE:-auto}"

# GHCR weights bundle. `oras pull` lays the artifact out as <dir>/weights/... and
# <dir>/configs/..., so pulling to a staging dir and moving the weights subtree up
# gives /app/weights the layout configs/weights/*.json expects. The bundle's own
# config fragments are provenance, not inputs — the image's configs/inference/
# already carries the matching architecture.
_pull_bundle() {
    local tag="$1" dest=/app/weights
    [[ -z "${tag}" ]] && return 0
    if [[ "${SYNC_MODE}" == "never" ]]; then
        echo "[entrypoint] BIOM3_SYNC_MODE=never; skipping weights bundle." >&2
        return 0
    fi
    if [[ "${SYNC_MODE}" == "auto" \
          && -n "$(find "${dest}" -mindepth 1 -type f -print -quit 2>/dev/null)" ]]; then
        echo "[entrypoint] ${dest} already populated; skipping bundle pull (auto)." >&2
        return 0
    fi
    command -v oras >/dev/null 2>&1 || {
        echo "[entrypoint] ERROR: BIOM3_WEIGHTS_BUNDLE set but oras is not installed." >&2
        return 1
    }
    mkdir -p "${dest}" 2>/dev/null || true
    [[ -w "${dest}" ]] || {
        echo "[entrypoint] ERROR: ${dest} is not writable by uid $(id -u), so the bundle" \
             "cannot be pulled into it. Run the container as root, or fetch the bundle" \
             "on the host (scripts/weights_bundle/fetch_bundle.sh) and mount it." >&2
        return 1
    }
    local repo="${BIOM3_WEIGHTS_BUNDLE_REPO:-ghcr.io/natural-machine/biom3-weights}"
    local stage=/tmp/biom3-bundle
    if [[ -n "${GHCR_TOKEN:-}" ]]; then
        echo "${GHCR_TOKEN}" | oras login ghcr.io -u "${GHCR_USER:-x}" --password-stdin
    fi
    echo "[entrypoint] weights: oras pull '${repo}:${tag}' -> '${dest}'" >&2
    rm -rf "${stage}"; mkdir -p "${stage}"
    oras pull "${repo}:${tag}" -o "${stage}"
    [[ -d "${stage}/weights" ]] || {
        echo "[entrypoint] ERROR: ${repo}:${tag} has no weights/ tree." >&2
        return 1
    }
    cp -a "${stage}/weights/." "${dest}/"
    rm -rf "${stage}"
}

_pull_bundle "${BIOM3_WEIGHTS_BUNDLE:-}"

exec "$@"
