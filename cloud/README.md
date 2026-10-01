# BioM3 cloud jobs (`cloud/`)

`run.mithril.yaml` provisions a GPU on Mithril, pulls the public GHCR image
`ghcr.io/natural-machine/biom3:cuda-dev`, and runs whatever `CMD` you give it. The job
lives in `CMD`; the yaml only describes the machine. Nothing about where data lives is
baked into the image.

The image itself: [`../docker/README.md`](../docker/README.md).

## Launch

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml <name-prefix> \
  --config mithril.limit_price=<max $/hr> \
  --env CMD="<command>" \
  2>&1 | tee run.log
```

`mithril_launch.sh` picks a unique cluster name (Mithril retains bid names, so reuse
fails with a misleading `ResourcesUnavailableError`) and auto-loads
`configs/jobs/local.env` when present. The launch streams the job output — the trailing
`tee` is what keeps a local copy.

Add `--num-nodes N` for multi-node, `--gpus A100:4` for a different GPU count.

## Inputs and outputs

**Weights.** `--env BIOM3_WEIGHTS_BUNDLE=run1_base` makes the container pull the
published GHCR weight set into `/app/weights` before `CMD` runs, with no credentials.
The bundle's filenames match `configs/weights/run1_base.json`, so the job command just
passes `--weight_set configs/weights/run1_base.json`. See
[weights_bundle.md](../docs/setup/weights_bundle.md).

**Other inputs.** The image carries the test data under `tests/_data/`. Anything else is
yours to fetch, with whatever tool your storage needs, at the start of `CMD`, e.g. into
`/app/data`.

**Outputs.** Everything a job writes belongs under `/app/outputs` in the container,
which is `~/biom3/outputs` on each node:

- Training writes under `./outputs/<stage>/...`, and every training config's
  `output_root` points there. Each run gets `runs/<run_id>/` (logs, artifacts) and
  `checkpoints/<run_id>/`.
- Inference entry points write where you tell them, so pass paths under `/app/outputs`
  (e.g. `-o /app/outputs/gen1`).

The cluster is torn down when `CMD` ends (`mithril_launch.sh` passes `--down`), so
copying results elsewhere is part of the job. Append your own upload of `/app/outputs`
to `CMD`, installing its tool first if the image lacks it:

```bash
--env CMD="<your job> && <install your upload tool> && <upload /app/outputs>"
```

Its credentials reach the container by name: add each variable to `FORWARD_ENV`, and
supply the value with `--env NAME=...`, or as a secret (declared under `secrets:` in the
yaml and passed with `--secret NAME`, which reads it from your shell and keeps it out of
the dashboard). On multi-node DDP runs only the head node (`SKYPILOT_NODE_RANK=0`) writes
checkpoints, so guard the upload with `[ "$SKYPILOT_NODE_RANK" = 0 ]`.

## Examples

### test

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-test \
  --config mithril.limit_price=6.00 \
  --env CMD="pytest tests/ --include_requires_gpu" \
  2>&1 | tee test.log
```

Tests that need pretrained weights skip unless those weights are in `/app/weights`.
`--env CMD="pytest tests/ --quick"` is the short version.

### pretrain — Stage 3 from scratch

From-scratch needs no pretrained weights. This uses the test HDF5 baked into the image:

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-pt \
  --config mithril.limit_price=6.00 \
  --env CMD="bash scripts/stage3_train_singlenode.sh \
      configs/stage3_training/pretrain_scratch_v1.json 1 auto pt1 \
      --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
      --epochs 1 --limit_val_batches 1.0" \
  2>&1 | tee pretrain.log
```

Positional args are `CONFIG_PATH NGPU DEVICE RUN_ID`; everything after is forwarded to
`biom3_train_stage3`. For a real dataset, fetch it at the start of `CMD` and point
`--primary_data_path` at it.

### finetune — Stage 3

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-ft \
  --config mithril.limit_price=6.00 \
  --env BIOM3_WEIGHTS_BUNDLE=run1_base \
  --env CMD="bash scripts/stage3_train_singlenode.sh \
      configs/stage3_training/finetune_v1.json 1 auto ft1 \
      --finetune True \
      --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
      --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
      --epochs 1 --limit_val_batches 1.0" \
  2>&1 | tee finetune.log
```

Starting from a CSV instead of a precompiled HDF5, chain the embedding pipeline first
(the `run1_base` bundle also carries the PenCL and Facilitator weights it needs):

```bash
  --env CMD="biom3_embedding_pipeline -i /app/data/my.csv -o /app/outputs/emb --prefix ds1 \
        --weight_set configs/weights/run1_base.json \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json \
      && bash scripts/stage3_train_singlenode.sh \
        configs/stage3_training/finetune_v1.json 1 auto ft1 \
        --finetune True --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
        --primary_data_path /app/outputs/emb/ds1.compiled_emb.hdf5 --epochs 1"
```

### generate — Stage 1 → 2 → 3

Weights come from the published GHCR bundle, so this needs no credentials — the
bundle's filenames are exactly what `configs/weights/run1_base.json` names:

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-gen \
  --config mithril.limit_price=6.00 \
  --env BIOM3_WEIGHTS_BUNDLE=run1_base \
  --env CMD="biom3_embedding_pipeline --generate \
      -i tests/_data/stage1_inputs/sample_text_seqs1.csv \
      -o /app/outputs/gen1 --prefix gen1 \
      --weight_set configs/weights/run1_base.json \
      --pencl_config configs/inference/stage1_PenCL.json \
      --facilitator_config configs/inference/stage2_Facilitator.json \
      --proteoscribe_config configs/inference/stage3_ProteoScribe_sample.json" \
  2>&1 | tee generate.log
```

The prompts CSV needs the columns the PenCL config expects (`protein_sequence` +
`primary_Accession`). FASTA is written by default; `--no_fasta` disables it.

Multi-GPU or multi-node sampling: run the same command under a launcher, e.g.
`NGPU=4 bash scripts/launchers/container_singlenode.sh biom3_embedding_pipeline --generate …`.
Stages 1-2 are deterministic, so every rank computes identical embeddings; the Stage 3
sampler shards by rank and only rank 0 writes.

### multi-node training

`--num-nodes N` runs the task on N instances; `--gpus` is the per-node count. Run the
multi-node wrapper with DDP, since the instances share no filesystem (see
[docker/README.md](../docker/README.md#multi-node-training-skypilot)):

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-mn \
  --num-nodes 2 --gpus A100:8 \
  --config mithril.limit_price=<max $/hr> \
  --env CMD="bash scripts/stage3_train_multinode.sh \
      configs/stage3_training/pretrain_scratch_v1.json 2 8 auto mn1 \
      --distributed_strategy ddp --epochs 1" \
  2>&1 | tee multinode.log
```

---

## Publishing the image (GHCR) — runbook

The image is published **public** at **`ghcr.io/natural-machine/biom3`**, tagged
`cuda-<sha>` (immutable, per commit) and `cuda-dev` (moving; what `run.mithril.yaml`
tracks). Both are **multi-arch manifest lists** covering `linux/amd64` (cloud
instances) and `linux/arm64` (DGX Spark), so the same tag runs on either.

Why GHCR + public:
- **Cost**: Mithril instances are ephemeral, so *every launch pulls the whole image*
  (~5.4 GB compressed for amd64, ~4.4 GB for arm64). GitHub Packages is **free for
  public packages**.
- **Simplicity**: a public image needs **no pull authentication**, so the launch path
  carries no registry token at all.

> **Before you publish**: the image bakes `src/`, `scripts/`, `tests/` (incl. the test
> HDF5s) and `configs/`. Publishing it **makes all of that world-readable** — confirm
> that is intended.

### One-time: create a token and publish

```bash
# 1. Create a classic PAT (push side only).
#    GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
#    → Generate new token (classic) → scopes: write:packages, read:packages
#    → If the org enforces SAML SSO: click "Configure SSO" → Authorize for natural-machine
export GHCR_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# 2. Log in to GHCR (needed to PUSH; pulling a public image needs no login).
echo "$GHCR_TOKEN" | docker login ghcr.io -u addison-nm --password-stdin

# 3. Build and publish both architectures (see the next section for host requirements).
#    The repo is always explicit; it must match IMAGE in run.mithril.yaml.
REPO=ghcr.io/<org>/biom3
docker/build.sh --variant cuda --release --repo "$REPO"

# 4. Make the package PUBLIC — the first push creates it PRIVATE by default.
#    Web: https://github.com/orgs/natural-machine/packages → biom3
#         → Package settings → Danger Zone → Change visibility → Public

# 5. Verify an ANONYMOUS pull — this is exactly what a Mithril instance does.
docker logout ghcr.io
docker pull "$REPO:cuda-dev"
```

### On every image change — one cross-build from any host

The image bakes `src/ scripts/ tests/ configs/`, so any change to them — or to
`docker/entrypoint.sh` — needs a rebuild and repush before it reaches a cloud run.

Both architectures are built in a single pass, from **one** host of either
architecture, and pushed as a manifest list:

```bash
REPO=ghcr.io/<org>/biom3
echo "$GHCR_TOKEN" | docker login ghcr.io -u addison-nm --password-stdin
docker/build.sh --variant cuda --release --repo "$REPO"
docker buildx imagetools inspect "$REPO:cuda-dev"
```

`--release` defaults `--platform` to every architecture the variant supports
(`linux/amd64,linux/arm64` for cuda; amd64 only for xpu), derives the tags from
`--repo` (required, no default) and the git sha, and implies `--push`:

- `cuda-<sha>` — immutable, per commit
- `cuda-dev` — moving; what `run.mithril.yaml` pulls

It refuses a dirty tree so `cuda-<sha>` matches the commit; `--allow-dirty` overrides
and suffixes the sha with `-dirty`. `--tag` is not valid alongside it.

The `inspect` should list `linux/amd64` and `linux/arm64`. Two extra `unknown/unknown`
entries are buildx provenance attestations — expected, and ignored by `docker pull`.

**Requirements on the build host:**

- A **`docker-container`** buildx builder. The stock `default` (docker driver) cannot
  build multi-platform: `docker buildx create --name <n> --driver docker-container --use`.
- **binfmt/QEMU** registered for the non-native architecture:
  `docker run --privileged --rm tonistiigi/binfmt --install all`. Check with
  `docker buildx inspect` — the `Platforms:` line must list both.

Emulating the non-native architecture is slower than a native build, but not
prohibitively so for this image, and buildx caches layers across runs.

#### Single-arch publishing (`push.sh`)

[`docker/push.sh`](../docker/push.sh) pushes a locally-built image under the same two
tags (`docker/push.sh --variant xpu --repo "$REPO"`), but a local image holds only one
architecture. It exists for the **amd64-only xpu variant**. For cuda it refuses to
overwrite a multi-arch `cuda-dev` — doing so would silently strip an architecture off
the tag cloud jobs pull. `--force-dev`
overrides that if you mean it.

## Publishing the weights bundle (GHCR)

Model **weights** ship separately from the image, as an OCI **artifact** (not a runnable
image) at **`ghcr.io/natural-machine/biom3-weights`**. A bundle pairs a set of weight files
(Stages 1–3 plus the ESM-2 and BiomedBERT backbones) with the architecture-only configs
that describe them — ~6.44 GB for `run1_base`. Consumers pull it with **`oras`**, not
`docker`. What goes in a bundle is declared by a spec under
`scripts/weights_bundle/bundle_specs/<name>.json`; the pull / link / run side is in
[../docs/setup/weights_bundle.md](../docs/setup/weights_bundle.md).

The **tag is whatever you pass** to `push_bundle.sh` — there is no git coupling and no
enforced sha. Use a plain name like `run1_base`, or add a suffix of your own for provenance.

Same PAT as the image (`write:packages`, from the one-time step above), but log in with
`oras`:

```bash
# oras is a single static binary — https://oras.land/docs/installation
echo "$GHCR_TOKEN" | oras login ghcr.io -u addison-nm --password-stdin
```

Build from a spec on a machine that has the referenced weights present in `weights/` (e.g.
DGX Spark, where `weights/` is linked from the share), then push under a tag. The tools have
no git awareness — no clean-tree or matching-commit requirement.

```bash
# 1. Build — copies/flattens the files the spec lists, writes a checksummed MANIFEST.json.
python scripts/weights_bundle/build_bundle.py \
    scripts/weights_bundle/bundle_specs/run1_base.json -o ~/biom3-bundles

# 2. Push under a tag. --dry-run prints the file list without uploading.
scripts/weights_bundle/push_bundle.sh ~/biom3-bundles/biom3-weights-run1_base run1_base --repo ghcr.io/<org>/biom3-weights --dry-run
scripts/weights_bundle/push_bundle.sh ~/biom3-bundles/biom3-weights-run1_base run1_base --repo ghcr.io/<org>/biom3-weights

# 3. Make the package PUBLIC — the first push creates it private, same as the image:
#    https://github.com/orgs/natural-machine/packages → biom3-weights
#         → Package settings → Change visibility → Public

# 4. Verify an anonymous pull into a scratch dir (fetch pulls + checksums; it never
#    touches your checkout):
oras logout ghcr.io
scripts/weights_bundle/fetch_bundle.sh /tmp/biom3-weights-check --tag run1_base
```

**Upload constraint.** GHCR times out a single blob upload at **10 minutes**, and `oras`
has no resumable upload. The largest blob (PenCL, ~3 GB) needs a sustained **~41 Mbps**
upstream or it fails at the very end. Blobs are content-addressed, so re-pushing a bundle
that reuses the same file re-uploads only what changed. Weights are architecture-independent,
so there is **no** multi-arch merge step (unlike the image).

---

## Launching from a separate machine

You can drive Mithril launches from any host — e.g. an EC2 instance hosting a web app.
The launching machine is a **thin client**: no GPU, no Docker, no repo, no weights.

### It needs

1. **`uv`** (or pip) — to install the CLI.
2. **`mithril-client`** — `uv tool install -U mithril-client` (provides `mithril` + the
   bundled `sky`).
3. **Mithril auth**: `~/.config/mithril/config.yaml` with `api_key` + `project_id`, or the
   `MITHRIL_API_KEY` / `MITHRIL_PROJECT` env vars.
4. **`cloud/run.mithril.yaml`** and `scripts/cloud/mithril_launch.sh` — **not** the whole
   repo; the code is baked into the image.
5. **Outbound HTTPS** to `api.mithril.ai` and `ghcr.io`.
6. A little local disk — `mithril sky launch` runs a local SkyPilot API server and keeps
   state in `~/.sky/`.

### It does NOT need

- ❌ **Docker** — the pull/run happen on the remote GPU instance.
- ❌ a **registry token** — the GHCR image is public.
- ❌ a **GPU**, the **weights**, or the **BioM3-dev repo**.

### Operational notes

- **Use a unique cluster name every launch.** Mithril retains bid names indefinitely, so
  reuse fails with a *misleading* `ResourcesUnavailableError`. The helper appends a
  timestamp; a web app must do the same.
- **`--down` only fires after a job finishes.** A provision-stage failure leaves the
  instance billing — reconcile with `mithril sky status` / `mithril sky down <c>`.
- **Instances that never get an SSH IP** are a Mithril-side failure: the bid clears, an
  instance is allocated, then SkyPilot waits out a hardcoded 3600 s timeout and cancels
  the bid. Ctrl-C rather than waiting the hour.
- **The local API server accumulates state.** Run
  [`scripts/cloud/mithril_reset.sh`](../scripts/cloud/mithril_reset.sh) when a launch wedges.
