# BioM3 Developer Guide

## About

This document details aspects of the BioM3 project that are relevant to developers and advanced users.

## Basic development concepts

The `BioM3-dev` repo is maintained under the [Ranganathan Lab GitHub](https://github.com/ranganathanlab/BioM3-dev).
There are two primary branches, `main` and `dev`. These are both protected, and changes to them should be made via pull requests.

## BioM3 weights and datasets

### Publishing BioM3 weights

### Publishing BioM3 datasets

```bash
cd /path/to/BioM3-dev
REPO=ghcr.io/<org>/biom3  # e.g. ghcr.io/ranganathanlab/biom3

python scripts/weights_bundle/build_bundle.py scripts/weights_bundle/bundle_specs/gfp_demo.json -o ~/biom3-bundles
scripts/weights_bundle/push_bundle.sh ~/biom3-bundles/biom3-datasets-gfp_demo gfp_demo --repo $REPO --kind dataset
```

## Containerizing BioM3

### Creating and publishing BioM3 Docker images

First, clone the `BioM3-dev` repository. A Docker image can be created from the source code and published to the appropriate GitHub organization. The commands are as follows, using build scripts shipped with the BioM3 repo. The destination registry repo is passed explicitly with `--repo`.

```bash
cd /path/to/BioM3-dev
REPO=ghcr.io/<org>/biom3  # e.g. ghcr.io/ranganathanlab/biom3

# multi-arch
docker/build.sh --variant cuda --release --repo "$REPO"
docker/build.sh --variant cpu  --release --repo "$REPO"

# amd64-only (no arm64 Intel GPU wheels)
docker/build.sh --variant xpu         --release --repo "$REPO"
docker/build.sh --variant xpu-oneapi  --release --repo "$REPO"
```

To publish an image that is already built locally (single architecture), push it under the same tags without rebuilding:

```bash
docker/push.sh --variant xpu --repo "$REPO"
```

### Creating Apptainer images for HPC environments

<!-- This section, in particular the code below, should be in agreement with the corresponding section of the User Guide. -->

For details, refer to the appropriate section of the [User Guide](./USER_GUIDE.md#case-3-using-biom3-on-hpc-environments-through-apptainer).

```bash
# On a login node of Aurora

REPO=docker://ghcr.io/<org>/biom3  # e.g. docker://ghcr.io/ranganathanlab/biom3
TAG=<tag>  # e.g. xpu-oneapi-abd9941

DEST=/path/to/biom3_${TAG}.sif  # Location to store the apptainer image

module load apptainer
export APPTAINER_CACHEDIR=/flare/NLDesignProtein/$USER/.apptainer/cache
export APPTAINER_TMPDIR=/tmp/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build ${DEST} "${REPO}:${TAG}"
```

## Working on ALCF machines
