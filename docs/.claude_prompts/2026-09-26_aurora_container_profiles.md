# Handoff: settle the container profile design on Aurora

You are picking up work on BioM3-dev on **Aurora**. The design was drafted on a DGX Spark,
which has no Aurora hardware and no Apptainer, so several questions could not be answered
there. Your job is to answer them from hardware, correct the design where it is wrong, and
record what you find.

## Start here

```bash
cd /flare/NLDesignProtein/$USER/BioM3-dev-space/BioM3-dev   # adjust if your checkout differs
git checkout dev && git pull        # you should land on 5430ee4 or later
```

Read, in this order:

1. `docs/misc/container_interface_design.md` — the design you are validating. Its
   "Open questions" and "Suggested sequence" sections are your task list.
2. `docs/setup/container_validation.md` — the grid of what works where, and the open items
   behind it. Rows `au1` and `au2` are yours.

## Goal being served

A user with only the image (Docker or a converted `.sif`) should be able to bind their own
weights/data/configs/outputs directories and type the **same `biom3_*` command** under
`docker run`, `apptainer exec`, or `mpiexec … apptainer exec`, without knowing any
Aurora-specific environment variables. The proposal is that the image carries named
profiles (`BIOM3_PROFILE=aurora-tile-mpi`, etc.) and the caller selects one.

## Tasks

### 1. Does `apptainer exec` bypass the Docker `ENTRYPOINT`?

The whole design rests on this. The image declares
`ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]`, and both Aurora wrappers use
`apptainer exec`. If `exec` bypasses it, environment setup cannot live in the entrypoint
and must live in Python.

`docker/entrypoint.sh` only prints when `BIOM3_WEIGHTS_BUNDLE` is set, so use that as the
probe. On a login node, no GPU needed:

```bash
SIF=/flare/NLDesignProtein/$USER/biom3_xpu-oneapi_25e440d.sif
BIOM3_WEIGHTS_BUNDLE=does-not-exist apptainer exec "$SIF" true 2>&1 | head
BIOM3_WEIGHTS_BUNDLE=does-not-exist apptainer run  "$SIF" true 2>&1 | head
```

`[entrypoint]` lines mean it ran. Report which of `exec`/`run` produce them. Also check
whether `/.singularity.d/env/*.sh` in the SIF carries the Docker `ENV` values, since that
is the other hook Apptainer honors for `exec`.

### 2. Capture reference environments

The profile values in the design were transcribed from the wrapper scripts. They need to
be checked against what a real run actually has. For each shape below, dump the
environment **from inside the container, as the workload sees it**:

```bash
cd /flare/NLDesignProtein/$USER/BioM3-dev-space/BioM3-dev
mkdir -p outputs/profile_refs

# single node, torchrun inside one container
BIOM3_IMAGE=<your xpu sif> \
  scripts/aurora/apptainer_run.sh bash -lc 'env | sort' > outputs/profile_refs/tile-torchrun.env

# one container per rank under mpiexec (rank 0 only is enough)
NGPU_PER_NODE=12 NGPU_TOTAL=12 BIOM3_RANK_SOURCE=mpi \
BIOM3_IMAGE=/flare/NLDesignProtein/$USER/biom3_xpu-oneapi_25e440d.sif \
  scripts/aurora/apptainer_mpi_run.sh bash -lc 'if [ "${PALS_RANKID:-0}" = 0 ]; then env | sort; fi' \
  > outputs/profile_refs/tile-mpi.env

# RL/node layout, one rank per node
NGPU_PER_NODE=1 NGPU_TOTAL=1 BIOM3_RANK_LAYOUT=node BIOM3_RANK_SOURCE=mpi \
BIOM3_IMAGE=/flare/NLDesignProtein/$USER/biom3_xpu-oneapi_25e440d.sif \
  scripts/aurora/apptainer_mpi_run.sh bash -lc 'env | sort' > outputs/profile_refs/node-mpi.env
```

These are `env` dumps, not training runs — they are cheap and do not need a long
allocation. Then diff them against each other and against the design's profile table, and
report every variable where the table is wrong or incomplete.

### 3. Settle FLAT vs COMPOSITE for the RL path

This is open question 2 and the design explicitly does not guess it.

- `scripts/aurora/apptainer_run.sh` and `apptainer_mpi_run.sh` both force
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, which exposes 12 tiles.
- `scripts/run_gdpo_smoke_multixpu.sh` sets it nowhere and its header states it expects
  `torch.xpu.device_count() == 6`, i.e. the COMPOSITE default.
- `scripts/launchers/aurora_multinode_rl.sh` unsets `ZE_AFFINITY_MASK` "so each rank sees
  all tiles", which under FLAT is 12 and under COMPOSITE is 6.

On a compute node, confirm the counts both ways:

```bash
for h in FLAT COMPOSITE; do
  echo -n "$h: "
  ZE_FLAT_DEVICE_HIERARCHY=$h apptainer exec "$SIF" \
    python -c "import torch; print(torch.xpu.device_count())"
done
```

Then determine what the RL code actually requires — read `src/biom3/rl/` and
`backend/xpu.py`, do not rely on the comments in the job scripts. The answer decides
whether `aurora-node-mpi` is one profile or two.

### 4. Confirm or correct the profile set

The design proposes `aurora-tile-torchrun`, `aurora-tile-mpi`, `aurora-node-mpi`,
`aurora-single-tile`, `container`. That set was inferred from scattered wrappers, launchers
and PBS files. Check it against `jobs/aurora/**` and `scripts/launchers/aurora_*.sh` and
say whether any real run shape is missing or any listed one is fictional.

## Rules

- **Verify in code, not in comments.** Comments and `USAGE` headers in this repo have been
  wrong. `scripts/aurora/apptainer_mpi_run.sh`'s header omitted a required variable
  (`BIOM3_RANK_SOURCE=mpi`), which cost an allocation to rediscover, and a four-line
  comment elsewhere described behavior the code did not have.
- **Do not guess a value to make something pass.** If a variable's correct value is not
  determinable from code or a measured run, say so and stop.
- **Allocation is expensive.** Tasks 1, 2 and 4 need no GPU. Only task 3 needs a compute
  node. Batch the GPU work.
- **Do not change source or scripts** as part of this. The output is findings plus doc
  edits. If you find a bug, write it down as an open item rather than fixing it.
- Source `environment.sh` before running anything, or `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`
  is unset and entrypoint tests fail in a way that looks like a code regression.

## Output

Edit `docs/misc/container_interface_design.md`:

- Replace the "Open questions" section with the answers, or with a narrower question if a
  point is still genuinely open.
- Correct the profile table to the measured values, marking each profile with whether it
  was verified on hardware or is still proposed.
- Add the `apptainer exec` / `ENTRYPOINT` finding to the "Applying it without a wrapper"
  section, since it either confirms or invalidates the Python-hook decision.

Commit with a Conventional Commits message (`docs:`), summary under 72 characters. Leave
the `outputs/profile_refs/*.env` dumps out of git; quote the relevant lines in the doc
instead.

## Context you may need

- Images are published at `ghcr.io/natural-machine/biom3`, tag `<variant>-25e440d` for the
  current source. Build a `.sif` with
  `apptainer build <path>.sif docker://ghcr.io/natural-machine/biom3:xpu-oneapi-25e440d`.
- Two fixes are in source but **not yet in any image**: the Stage 1 XPU multi-rank crash
  (open item 11) and the Stage 1 inference empty-rank crash (item 13). If you run `s1` or
  `emb`, expect them to fail against `25e440d` images; that is known, not new.
- `BIOM3_RANK_SOURCE=mpi` is required for the `xpu-oneapi` multi-node path. Without it you
  get the `pals` fallback and a SIGSEGV during oneCCL setup.
