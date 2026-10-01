#!/usr/bin/env bash
# Push a built bundle directory to an OCI registry as an artifact.
#
#   push_bundle.sh <bundle_dir> <tag> --repo R [--kind weights|dataset] [--dry-run]
#
# Ships every file under <bundle_dir> under one tag. --repo is required, e.g.
# ghcr.io/<org>/biom3-weights or ghcr.io/<org>/biom3-datasets. --kind sets the
# artifact type (default weights). No git awareness — the tag is whatever you
# pass. oras pulls it back with the same relative layout and verifies each
# blob's digest on the way. Auth: oras reuses your existing ~/.docker/config.json
# ghcr.io login (only run `oras login` if a push 401s).
set -euo pipefail

REPO=""
KIND="weights"
DRY_RUN=0
BUNDLE_DIR=""
TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)     REPO="$2"; shift 2 ;;
        --kind)     KIND="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '2,11p' "$0"; exit 0 ;;
        -*)         echo "unknown arg: $1" >&2; exit 1 ;;
        *)          if [[ -z "$BUNDLE_DIR" ]]; then BUNDLE_DIR="$1"; else TAG="$1"; fi; shift ;;
    esac
done

[[ -n "$BUNDLE_DIR" && -n "$TAG" && -n "$REPO" ]] || {
    echo "usage: push_bundle.sh <bundle_dir> <tag> --repo R [--kind weights|dataset] [--dry-run]" >&2
    exit 1
}
case "$KIND" in
    weights) ARTIFACT_TYPE="application/vnd.biom3.weights.bundle.v1"
             FETCH_CMD="biom3_fetch_weights ${TAG} -o <dir> --registry ${REPO}" ;;
    dataset) ARTIFACT_TYPE="application/vnd.biom3.dataset.bundle.v1"
             FETCH_CMD="biom3_fetch_dataset ${TAG} -o <dir> --registry ${REPO}" ;;
    *)       echo "ERROR: --kind must be weights or dataset (got '$KIND')." >&2; exit 1 ;;
esac
[[ -f "$BUNDLE_DIR/MANIFEST.json" ]] || {
    echo "ERROR: $BUNDLE_DIR has no MANIFEST.json — not a built bundle." >&2
    exit 1
}

cd "$BUNDLE_DIR"
FILES=()
while IFS= read -r rel; do FILES+=("$rel"); done < <(find . -type f -printf '%P\n' | sort)

REF="${REPO}:${TAG}"
TOTAL="$(du -sh "$BUNDLE_DIR" | cut -f1)"
echo "push ${#FILES[@]} files (${TOTAL}) -> ${REF}  [${ARTIFACT_TYPE}]" >&2

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  %s\n' "${FILES[@]}" >&2
    exit 0
fi

command -v oras >/dev/null 2>&1 || {
    echo "ERROR: oras not found. Install from https://oras.land/docs/installation" >&2
    exit 1
}
oras push --artifact-type "$ARTIFACT_TYPE" "$REF" "${FILES[@]}"
echo "pushed ${REF}" >&2
echo "fetch: ${FETCH_CMD}" >&2
