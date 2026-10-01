#!/usr/bin/env bash
# Launch a BioM3 cloud/*.yaml on Mithril via the bundled sky CLI (`mithril sky
# launch`), with a UNIQUE cluster name.
#
# The image is PUBLIC on GHCR (ghcr.io/natural-machine/biom3), so there is no registry
# login or token — Mithril pulls it anonymously. Local launch defaults (e.g. IMAGE or
# FORWARD_ENV) can live in the gitignored configs/jobs/local.env, auto-loaded below via
# --env-file when present (copy configs/jobs/local.env.example to create it).
#
# `mithril sky launch` supports --secret (redacted) and fills `secrets: KEY: null`,
# whereas the `mithril launch` wrapper does not; pass your own credentials that way.
# A unique cluster name prevents Mithril-side "ResourcesUnavailableError" failures.
#
# Usage: scripts/cloud/mithril_launch.sh <task.yaml> [cluster-prefix] [extra sky args...]
#   scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-run --env CMD="nvidia-smi"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK="${1:?usage: mithril_launch.sh <task.yaml> [cluster-prefix] [extra sky args...]}"
PREFIX="${2:-biom3}"
shift || true
shift 2>/dev/null || true

# Auto-load gitignored local launch defaults when present.
LOCAL_ENV="${REPO_ROOT}/configs/jobs/local.env"
ENVFILE=()
[ -f "${LOCAL_ENV}" ] && ENVFILE=(--env-file "${LOCAL_ENV}")

CLUSTER="${PREFIX}-$(date +%y%m%d-%H%M%S)"
echo "[launch] cluster=${CLUSTER}  task=${TASK}"

exec mithril sky launch "${TASK}" -c "${CLUSTER}" -y --down \
  ${ENVFILE[@]+"${ENVFILE[@]}"} \
  "$@"
