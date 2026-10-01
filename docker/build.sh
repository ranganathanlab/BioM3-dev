#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/build.sh
#
# Build a BioM3 device image with the repo root as the build context. Works
# from any directory.
#
# PUBLISHING: --release builds every architecture in one pass and pushes the
# conventional tags to the registry repo given by --repo. This is how the public
# image is published:
#   docker/build.sh --variant cuda --release --repo ghcr.io/<org>/biom3
#     -> ghcr.io/<org>/biom3:cuda-dev      (moving; what cloud/*.yaml track)
#     -> ghcr.io/<org>/biom3:cuda-<sha>    (immutable, per commit)
# Both tags are one multi-arch manifest list, so amd64 and arm64 hosts pull the
# same tag. Cross-building the non-native architecture needs QEMU/binfmt in the
# buildx builder (see docker/README.md).
#
# USAGE:
#   docker/build.sh [--variant V] [--platform P] [--tag T] [--push]
#                   [--release --repo R [--allow-dirty]] [-- <buildx args>]
#
#   --variant V    cuda | cpu | xpu | xpu-oneapi (default: cuda). Selects
#                  docker/Dockerfile.<V> and defaults the tag to biom3:<V>.
#                  xpu and xpu-oneapi are amd64-only. cpu is the slim CPU-only
#                  inference image (no training).
#   --platform P   linux/amd64 | linux/arm64 | linux/amd64,linux/arm64
#                  (default: builder's native platform, or every architecture
#                  the variant supports under --release). Cross-arch builds
#                  need QEMU/binfmt registered in the buildx builder.
#   --tag T        image tag (default: biom3:<variant>). Not valid with --release.
#   --push         push to the registry instead of loading locally
#                  (required for multi-platform builds — buildx --load is
#                  single-platform only)
#   --release      tag <repo>:<variant>-dev + <repo>:<variant>-<shortsha> and
#                  push them. Implies --push. Refuses a dirty tree so the sha
#                  tag matches the commit.
#   --repo R       registry repo WITHOUT a tag, e.g. ghcr.io/<org>/biom3.
#                  Required with --release; there is no default.
#   --allow-dirty  allow --release from a dirty tree; the sha tag gets a
#                  -dirty suffix
#
# NETWORK NOTE: the ~3 GB torch wheel comes from download.pytorch.org (cuda) or
# Intel's index (xpu). If you're on a VPN with upstream DNS filtering (e.g.
# Tailscale MagicDNS) and the wheel download stalls on "Failed to resolve",
# disconnect the VPN for the build.
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="cuda"
PLATFORM=""
TAG=""
PUSH=0
RELEASE=0
REPO=""
ALLOW_DIRTY=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)     VARIANT="$2"; shift 2 ;;
        --platform)    PLATFORM="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --push)        PUSH=1; shift ;;
        --release)     RELEASE=1; shift ;;
        --repo)        REPO="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --)            shift; EXTRA=("$@"); break ;;
        -h|--help)     sed -n '3,46p' "$0"; exit 0 ;;
        *)             echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "${VARIANT}" in
    cuda|cpu|xpu|xpu-oneapi) ;;
    *) echo "ERROR: --variant must be cuda, cpu, xpu or xpu-oneapi (got '${VARIANT}')." >&2; exit 1 ;;
esac

DOCKERFILE="${SCRIPT_DIR}/Dockerfile.${VARIANT}"
[[ -f "${DOCKERFILE}" ]] || { echo "ERROR: ${DOCKERFILE} not found." >&2; exit 1; }

# Every architecture the variant is buildable for. Intel GPU wheels are
# amd64-only, so xpu never gets arm64.
if [[ "${VARIANT}" == xpu* ]]; then
    ALL_PLATFORMS="linux/amd64"
else
    ALL_PLATFORMS="linux/amd64,linux/arm64"
fi

if [[ "${VARIANT}" == xpu* && "${PLATFORM}" == *arm64* ]]; then
    echo "ERROR: the ${VARIANT} variant is amd64-only (no arm64 Intel GPU wheels)." >&2
    exit 1
fi

TAGS=()
if [[ "${RELEASE}" -eq 1 ]]; then
    if [[ -n "${TAG}" ]]; then
        echo "ERROR: --tag and --release are mutually exclusive (--release derives" >&2
        echo "       its tags from --repo and the git sha)." >&2
        exit 1
    fi
    if [[ -z "${REPO}" ]]; then
        echo "ERROR: --release needs --repo (e.g. --repo ghcr.io/<org>/biom3)." >&2
        exit 1
    fi

    # The image is built from the working tree, so a dirty tree means the sha
    # tag would not match the commit.
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

    PUSH=1
    PLATFORM="${PLATFORM:-${ALL_PLATFORMS}}"
    TAGS=("${REPO}:${VARIANT}-dev" "${REPO}:${VARIANT}-${SHA}")
else
    TAGS=("${TAG:-biom3:${VARIANT}}")
fi

ARGS=(buildx build -f "${DOCKERFILE}")
for t in "${TAGS[@]}"; do ARGS+=(-t "${t}"); done

[[ -n "${PLATFORM}" ]] && ARGS+=(--platform "${PLATFORM}")

if [[ "${PUSH}" -eq 1 ]]; then
    ARGS+=(--push)
elif [[ "${PLATFORM}" == *,* ]]; then
    echo "ERROR: multi-platform build requires --push (buildx --load is single-platform)." >&2
    exit 1
else
    ARGS+=(--load)
fi

# Append passthrough args only if any (empty-array expansion under `set -u`
# errors on macOS's bash 3.2).
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    ARGS+=("${EXTRA[@]}")
fi
ARGS+=("${REPO_ROOT}")

echo "+ docker ${ARGS[*]}" >&2
exec docker "${ARGS[@]}"
