#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/push.sh
#
# Tag + push an ALREADY-BUILT local BioM3 image to the registry repo given by
# --repo, with git-derived version tags, so cloud instances pull it instead of
# rebuilding. The image is published PUBLIC, so cloud/*.yaml pull it anonymously
# (no registry token).
#
# Pushes TWO tags per call:
#   <variant>-<shortsha>   immutable, tied to the exact commit (reproducible)
#   <variant>-dev          moving pointer for the dev line (what cloud/*.yaml track)
#
# SINGLE-ARCH ONLY: a local image holds one architecture, so this publishes the
# architecture you built on. That suits the amd64-only xpu variant. For the cuda
# variant, which ships as a multi-arch manifest list, publish with
#   docker/build.sh --variant cuda --release --repo <repo>
# which builds every architecture in one pass and pushes both tags itself. This
# script REFUSES to overwrite a multi-arch <variant>-dev (see --force-dev), since
# doing so would strip an architecture off the tag cloud jobs pull.
#
# Prereqs: docker; a local biom3:<variant> image (docker/build.sh --variant ...); and a
# GHCR login on the PUSH side (pulling a public image needs no login):
#   echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
# GHCR_TOKEN = a classic PAT with write:packages (+ read:packages), SSO-authorized for
# the org. See cloud/README.md for the full one-time runbook.
#
# USAGE:
#   docker/push.sh --repo R [--variant cuda|cpu|xpu|xpu-oneapi] [--local-tag T]
#                  [--allow-dirty] [--force-dev]
#
#   --repo R       registry repo WITHOUT the tag, e.g. ghcr.io/<org>/biom3.
#                  Required; there is no default.
#   --variant V    cuda | cpu | xpu | xpu-oneapi (default cuda) -> pushes
#                  biom3:<V> as <V>-<sha> + <V>-dev
#   --local-tag T  local image to push (default: biom3:<variant>)
#   --allow-dirty  push from a dirty tree; the sha tag gets a -dirty suffix
#   --force-dev    replace a multi-arch <variant>-dev with this single-arch image
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="cuda"
REPO=""
LOCAL_TAG=""
ALLOW_DIRTY=0
FORCE_DEV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)     VARIANT="$2"; shift 2 ;;
        --repo)        REPO="$2"; shift 2 ;;
        --local-tag)   LOCAL_TAG="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --force-dev)   FORCE_DEV=1; shift ;;
        -h|--help)     sed -n '3,39p' "$0"; exit 0 ;;
        *)             echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${REPO}" ]]; then
    echo "ERROR: --repo is required (e.g. --repo ghcr.io/<org>/biom3)." >&2
    exit 1
fi

LOCAL_TAG="${LOCAL_TAG:-biom3:${VARIANT}}"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found." >&2; exit 1; }

# Version from git: <shortsha>, guarded so the tag is truthful (the image is built
# from the working tree, so a dirty tree means the tag would not match the commit).
SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
    if [[ "${ALLOW_DIRTY}" -eq 1 ]]; then
        SHA="${SHA}-dirty"
        echo "WARNING: working tree is dirty; tagging ${VARIANT}-${SHA} (NOT reproducible)." >&2
    else
        echo "ERROR: working tree is dirty. Commit first so ${VARIANT}-<sha> is truthful," >&2
        echo "       or pass --allow-dirty to tag ${VARIANT}-${SHA}." >&2
        exit 1
    fi
fi

VERSION_TAG="${REPO}:${VARIANT}-${SHA}"
MOVING_TAG="${REPO}:${VARIANT}-dev"

# A local image is single-arch. If the remote moving tag is a manifest list,
# pushing over it drops every architecture but this one from the tag cloud jobs
# pull. Attestation entries count as platforms too, so >1 means "manifest list".
REMOTE_PLATFORMS="$(docker buildx imagetools inspect "${MOVING_TAG}" 2>/dev/null \
    | grep -c '^ *Platform:' || true)"
if [[ "${REMOTE_PLATFORMS}" -gt 1 && "${FORCE_DEV}" -eq 0 ]]; then
    echo "ERROR: ${MOVING_TAG} is a multi-arch manifest list; pushing this" >&2
    echo "       single-arch image over it would strip the other architecture(s)." >&2
    echo "       Publish multi-arch instead:" >&2
    echo "         docker/build.sh --variant ${VARIANT} --release --repo ${REPO}" >&2
    echo "       Or pass --force-dev to replace it deliberately." >&2
    exit 1
fi

echo "+ docker tag ${LOCAL_TAG} -> ${VERSION_TAG} , ${MOVING_TAG}" >&2
docker tag "${LOCAL_TAG}" "${VERSION_TAG}"
docker tag "${LOCAL_TAG}" "${MOVING_TAG}"

echo "+ docker push ${VERSION_TAG}" >&2
docker push "${VERSION_TAG}"
echo "+ docker push ${MOVING_TAG}" >&2
docker push "${MOVING_TAG}"

echo "Pushed:" >&2
echo "  ${VERSION_TAG}   (immutable, single-arch)" >&2
echo "  ${MOVING_TAG}    (moving dev pointer — what cloud/*.yaml reference)" >&2
