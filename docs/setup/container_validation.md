# Container validation matrix

Tracks whether each BioM3 container image runs each main entry point on each machine,
launched through the current wrappers (`docker/run.sh`,
`scripts/aurora/apptainer_{run,mpi_run}.sh`). A cell is one row — machine, image variant and
node × device shape — crossed with one entry point.
It is checked off only after that cell's command below has run and met the pass criteria.

Cell IDs are `<row>-<column>`, e.g. `aurora_v2_n2d12-s3pt`. Use the cell ID as the `run_id` and as the
output directory name, so every result is traceable.

## Grid

| Row | Machine | Nodes × devices | Image | emb | s1pt | s3pt | s3ft | s3gft | gen |
| --- | ------- | --------------- | ----- | --- | -- | -- | -- | --- | --- |
| `spark_n1` | DGX Spark | 1 × GB10 | cuda (arm64) | todo | todo | todo | todo | todo | todo |
| `aurora_v1_n1d12` | Aurora | 1 × 12 devices | xpu | todo | todo | todo | todo | todo | todo |
| `aurora_v2_n1d12` | Aurora | 1 × 12 devices | xpu-oneapi | todo | todo | todo | todo | todo | todo |
| `aurora_v2_n2d12` | Aurora | 2 × 12 devices | xpu-oneapi | todo | todo | todo | todo | todo | todo |
| `aurora_v2_n1d1` | Aurora | 1 × 1 devices | xpu-oneapi | todo | todo | todo | todo | todo | todo |
| `aurora_v2_n2d1` | Aurora | 2 × 1 devices | xpu-oneapi | todo | todo | todo | todo | todo | todo |

Status key:

- `pass`: the cell's command ran and met the pass criteria.
- `fail`: it ran and did not. Note the error in the row's log under `outputs/validation/logs/`.
- `todo`: not run yet.
- `n/a`: not applicable by design (see [row notes](#row-notes)).
- `blocked`: a known gap, found by reading the code, stops this cell before it can run (see [open items](#open-items)).

Columns:

| Column | Entry point | Covers |
| ------ | ----------- | ------ |
| `emb` | `biom3_embedding_pipeline` | CSV → Stage 1 (PenCL) → Stage 2 (Facilitator) → compiled HDF5 |
| `s1pt` | `biom3_train_stage1`, via `scripts/stage1_train_*node.sh` | PenCL training from scratch |
| `s3pt` | `biom3_train_stage3`, via `scripts/stage3_train_*node.sh` | ProteoScribe pretraining from scratch on precomputed z_c (HDF5) |
| `s3ft` | `biom3_train_stage3 --finetune True` | ProteoScribe finetuning from `run1_base` on precomputed z_c (HDF5) |
| `s3gft` | `biom3_finetune_stage3` | Generalized finetuning on JSONL records, z_c computed on the device |
| `gen` | `biom3_ProteoScribe_sample` | Sequence generation from z_c |

## Images under test

| Variant | Tag | Rows |
| ------- | --- | ---- |
| cuda | `ghcr.io/natural-machine/biom3:cuda-779859b` (amd64 + arm64; also `cuda-dev`) | `spark_*` |
| xpu | `ghcr.io/natural-machine/biom3:xpu-779859b` (amd64; also `xpu-dev`) | `aurora_v1_*` |
| xpu-oneapi | `ghcr.io/natural-machine/biom3:xpu-oneapi-779859b` (amd64; also `xpu-oneapi-dev`) | `aurora_v2_*` |

If an image is rebuilt, update this table and re-run the affected cells.

The `779859b` images are the first to carry `biom3_fetch_weights`, both test
fixtures, and the Stage 1 XPU and empty-rank fixes. The Aurora `.sif` files must
be rebuilt from these tags before the `aurora_*` rows mean anything:

```bash
apptainer build /flare/NLDesignProtein/biom3_images/biom3-xpu-779859b.sif \
    docker://ghcr.io/natural-machine/biom3:xpu-779859b
apptainer build /flare/NLDesignProtein/biom3_images/biom3-xpu-oneapi-779859b.sif \
    docker://ghcr.io/natural-machine/biom3:xpu-oneapi-779859b
```

## Standard inputs

Every cell uses the `run1_base` weight set (ESM-2, BioBERT, and the PenCL, Facilitator
and ProteoScribe `run1_base` weights), either fetched and mounted as `weights/` or pulled
by the entrypoint with `BIOM3_WEIGHTS_BUNDLE=run1_base`
([docs/setup/weights_bundle.md](weights_bundle.md)). The data inputs are baked into the
image under `/app/tests/_data`, so they need no mount, except where noted.

| Input | Path inside the container | Used by |
| ----- | ------------------------- | ------- |
| Prompts CSV, 1,024 rows | `tests/_data/stage1_inputs/sample_text_seqs_1024.csv` | `emb`, `s1pt` |
| Prompts CSV, 5 rows | `tests/_data/stage1_inputs/sample_text_seqs1.csv` | quick single-rank checks only |
| z_c HDF5, 1,000 rows | `tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5` | `s3pt`, `s3ft` |
| JSONL records, 24 | `tests/_data/stage3_inputs/sample_finetune_records_24.jsonl` | `s3gft` |
| z_c for 5 prompts | `tests/_data/embeddings/test_Facilitator_embeddings.pt` | `gen` |

Getting `run1_base` into the checkout's `weights/` before a row that mounts it:
`biom3_fetch_weights run1_base -o weights` pulls the published bundle (6.4 GB, skipping
files already present by digest), or `scripts/link_weights.sh` symlinks them from a shared
canonical directory (on Spark, `/data/data-share/BioM3-data-share/data/weights`). Rows that
set `BIOM3_WEIGHTS_BUNDLE` instead need no local `weights/` at all.

## Running a row

Every command below is literal. Run one **row preamble** first: it defines `$R` (the
container runner), `$ROW`, `$NGPU` (devices per node) and `$LAUNCH`, and exports
the image and mount settings. After that the **cell commands** are identical on every
machine — copy them as-is. Run everything from the BioM3-dev checkout, since the wrappers
live in the repo. The Aurora multi-node rows do not fit that
shape and carry their own complete commands below.

`$LAUNCH` is empty when `NGPU=1` and is `scripts/launchers/container_singlenode.sh` when
`NGPU` is greater than 1, which is what spawns one rank per device for the entry points
that are not called through a training wrapper. `NGPU` reaches the
container either way: `docker/run.sh` forwards it explicitly, and no Apptainer wrapper
uses `--cleanenv`, so the host environment passes through.

Each command tees to `outputs/validation/logs/$ROW-<column>.log`.

### Row preamble: `spark_n1`

```bash
export ROW=spark_n1 NGPU=1
export BIOM3_IMAGE=ghcr.io/natural-machine/biom3:cuda-779859b
R="docker/run.sh"; LAUNCH=""
mkdir -p outputs/validation/logs
```

In a fresh clone, fetch `run1_base` into this checkout's `weights/` first. `docker/run.sh`
mounts `weights/` read-only, so bind it read-write directly for the fetch:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/weights:/weights" \
    $BIOM3_IMAGE biom3_fetch_weights run1_base -o /weights
```

6.4 GB; a re-run skips each file whose bytes already match the registry's digest.

If instead your `weights/` holds absolute symlinks into a shared directory
(`scripts/link_weights.sh`), every link target has to be mounted at the same path, since a
link inside a mount resolves inside the container — add
`export BIOM3_BIND_EXTRA=/data/data-share,/data/biom3_data` to the preamble. To see where
yours point: `find weights -type l -exec readlink {} + | cut -d/ -f1-4 | sort | uniq -c`.

### Row preamble: `aurora_v1_n1d12`

The `xpu` image, single node, ranks spawned by `torchrun` inside one container. From a
`qsub -I` shell on a compute node.

```bash
export ROW=aurora_v1_n1d12 NGPU=12
export BIOM3_IMAGE=/flare/NLDesignProtein/biom3_images/biom3-xpu-779859b.sif
export BIOM3_BIND_EXTRA=/lus
R="scripts/aurora/apptainer_run.sh"
LAUNCH="scripts/launchers/container_singlenode.sh"
mkdir -p outputs/validation/logs
```

### Row preamble: `aurora_v2_*`

The `xpu-oneapi` image under the host `mpiexec`, one container per rank. All four `v2`
rows share one preamble; only `NODES` and `DEV` change, and `ROW` is derived so it matches
the grid name. From a `qsub -I` shell with at least `NODES` nodes.

```bash
export NODES=2 DEV=12           # 1/12, 2/12, 1/1 or 2/1
export ROW=aurora_v2_n${NODES}d${DEV}
export NGPU_PER_NODE=$DEV NGPU_TOTAL=$((NODES * DEV))
export BIOM3_RANK_SOURCE=mpi
export BIOM3_IMAGE=/flare/NLDesignProtein/biom3_images/biom3-xpu-oneapi-779859b.sif
export BIOM3_BIND_EXTRA=/lus
# multi-node only: drive Slingshot rather than tcp
[ "$NODES" -gt 1 ] && export BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 \
                             BIOM3_FI_PROVIDER=cxi
R="scripts/aurora/apptainer_mpi_run.sh"
MN="--device auto --num_nodes $NODES --devices_per_node $DEV"
mkdir -p outputs/validation/logs
```

`BIOM3_RANK_SOURCE=mpi` is **required**. The default is `pals`, which translates the PALS
rank variables into torch's and lands on `TorchElasticEnvironment`; `mpi` uses
`MPIEnvironment` via mpi4py, the native path for `xpu-oneapi`, whose Intel MPI matches the
host launcher's.

`apptainer_mpi_run.sh` starts one rank per process, so the entry points are called
directly, never through the `scripts/stage*_{single,multi}node.sh` wrappers, which would
spawn ranks a second time. That is why the `v2` cell commands differ from the `v1` ones.

### Cell commands — wrapper rows (`spark_n1`, `aurora_v1_n1d12`)

Identical on both rows once the preamble has run. The training commands pass
`--wandb False`: without it a `WANDB_API_KEY` in your environment is forwarded into the
container and every smoke run is uploaded to the team W&B project.

```bash
# emb
$R $LAUNCH biom3_embedding_pipeline \
    -i tests/_data/stage1_inputs/sample_text_seqs_1024.csv \
    -o outputs/validation/$ROW-emb --prefix emb \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json \
    2>&1 | tee outputs/validation/logs/$ROW-emb.log

# s1pt
$R scripts/stage1_train_singlenode.sh \
    configs/stage1_training/pretrain_scratch_v1.json $NGPU auto $ROW-s1pt \
    --data_path tests/_data/stage1_inputs/sample_text_seqs_1024.csv \
    --epochs 1 --batch_size 2 --valid_size 0.2 --limit_train_batches 4 --wandb False \
    --output_root outputs/validation/$ROW-s1pt \
    2>&1 | tee outputs/validation/logs/$ROW-s1pt.log

# s3pt
$R scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json $NGPU auto $ROW-s3pt \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-s3pt \
    2>&1 | tee outputs/validation/logs/$ROW-s3pt.log

# s3ft
$R scripts/stage3_train_singlenode.sh \
    configs/stage3_training/finetune_v1.json $NGPU auto $ROW-s3ft \
    --finetune True \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-s3ft \
    2>&1 | tee outputs/validation/logs/$ROW-s3ft.log

# s3gft
$R $LAUNCH biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_v1.json \
    --finetune_data_path tests/_data/stage3_inputs/sample_finetune_records_24.jsonl \
    --device auto --num_nodes 1 --devices_per_node $NGPU --run_id $ROW-s3gft \
    --epochs 1 --limit_train_batches 4 --limit_val_batches 2 --wandb False \
    --output_root outputs/validation/$ROW-s3gft \
    2>&1 | tee outputs/validation/logs/$ROW-s3gft.log

# gen
$R $LAUNCH biom3_ProteoScribe_sample \
    -i tests/_data/embeddings/test_Facilitator_embeddings.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/validation/$ROW-gen/generated.pt --fasta --seed 42 \
    2>&1 | tee outputs/validation/logs/$ROW-gen.log
```

### Cell commands — `aurora_v2_*`

Same six entry points, called directly because `mpiexec` already spawned the ranks.

```bash
# emb
$R biom3_embedding_pipeline \
    -i tests/_data/stage1_inputs/sample_text_seqs_1024.csv \
    -o outputs/validation/$ROW-emb --prefix emb \
    --weight_set configs/weights/run1_base.json \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json \
    2>&1 | tee outputs/validation/logs/$ROW-emb.log

# s1pt
$R biom3_train_stage1 \
    --config_path configs/stage1_training/pretrain_scratch_v1.json $MN \
    --run_id $ROW-s1pt --data_path tests/_data/stage1_inputs/sample_text_seqs_1024.csv \
    --epochs 1 --batch_size 2 --valid_size 0.2 --limit_train_batches 4 --wandb False \
    --output_root outputs/validation/$ROW-s1pt \
    2>&1 | tee outputs/validation/logs/$ROW-s1pt.log

# s3pt
$R biom3_train_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v1.json $MN \
    --run_id $ROW-s3pt \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-s3pt \
    2>&1 | tee outputs/validation/logs/$ROW-s3pt.log

# s3ft
$R biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json $MN \
    --run_id $ROW-s3ft --finetune True \
    --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.bin \
    --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
    --epochs 1 --limit_val_batches 1.0 --wandb False \
    --output_root outputs/validation/$ROW-s3ft \
    2>&1 | tee outputs/validation/logs/$ROW-s3ft.log

# s3gft
$R biom3_finetune_stage3 \
    --config_path configs/stage3_training/finetune_generalized_v1.json $MN \
    --run_id $ROW-s3gft \
    --finetune_data_path tests/_data/stage3_inputs/sample_finetune_records_24.jsonl \
    --epochs 1 --limit_train_batches 4 --limit_val_batches 2 --wandb False \
    --output_root outputs/validation/$ROW-s3gft \
    2>&1 | tee outputs/validation/logs/$ROW-s3gft.log

# gen
$R biom3_ProteoScribe_sample \
    -i tests/_data/embeddings/test_Facilitator_embeddings.pt \
    -c configs/inference/stage3_ProteoScribe_sample.json \
    -m weights/ProteoScribe/run1_base_proteoscribe.bin \
    -o outputs/validation/$ROW-gen/generated.pt --fasta --seed 42 \
    2>&1 | tee outputs/validation/logs/$ROW-gen.log
```

## Pass criteria

Every cell:

1. Exit status 0, with no traceback in the log.
2. The log shows the expected backend (`cuda`, `xpu` or `cpu`) and world size (nodes × devices per node).
3. Outputs land under `outputs/validation/<row>-<column>/` on the host. For the `docker/run.sh` row, the caller owns them, not root.

Per column:

| Column | Criteria |
| ------ | -------- |
| `emb` | `emb.PenCL_emb.pt`, `emb.Facilitator_emb.pt` and `emb.compiled_emb.hdf5` exist. The HDF5 has one row per input row. With more than 1 rank: same rows, in input order, with no duplicates. |
| `s1pt`, `s3pt`, `s3ft`, `s3gft` | At least one optimizer step and one validation pass, with finite training and validation loss. A checkpoint is written under `checkpoints/<row>-<column>/` and run artifacts (`args.json`, `run.log`) under `runs/<row>-<column>/`. With more than 1 node, rank 0 writes the checkpoint once. |
| `s3ft` | The log shows the `run1_base` weights loaded with no missing or unexpected keys, and the trainable parameter count matches the finetune flags. |
| `s3gft` | The log shows captions composed from the records and z_c computed on the device, not read from a file. |
| `gen` | 25 sequences (5 prompts × 5 replicas) in `generated.pt` and the FASTA, containing only amino-acid letters, written by rank 0 only. **No cross-run sequence equality is required** — see open item 14. |

## Row notes

- `aurora_v1_n1d12` is the only `xpu` row. Everything else on Aurora is `xpu-oneapi`, so
  the grid answers two questions at once: whether each shape works, and whether
  `xpu-oneapi` alone can replace `xpu`.
- The `v1` row spawns ranks with `torchrun` inside one container; every `v2` row gets one
  container per rank from the host `mpiexec`. That is the reason for two sets of cell
  commands, not a difference between the images.
- The `d1` rows exist to separate device count from node count. `aurora_v2_n2d1` is two
  ranks, so a sharding or collective fault is legible there in a way it is not at 24.

## Open items

1. Retired — no row uses the cpu image. (Was: the cpu image is stale; it was rebuilt as
   `cpu-25e440d` on 2026-09-26, and again as `cpu-779859b` on 2026-09-27.)
2. Retired — Polaris is not in the grid. The finding stands if it returns:
   `scripts/polaris/apptainer_run.sh` does not set `BIOM3_LAUNCHER=container`, so the
   training wrappers dispatch to `polaris_singlenode.sh`, which passes Cray `mpiexec`
   flags the image's OpenMPI rejects.

3. ~~There is no JSONL input for `s3gft` in the image.~~ **Closed 2026-09-27**: added
   `tests/_data/stage3_inputs/sample_finetune_records_24.jsonl` (24 records, 21 KB). It
   covers all three sources (`swissprot`, `pfam`, `supplemental`) and 16 field keys, so
   the schema's `exclude_sources`, `lineage` policy and dropout table are all exercised.
   Needs an image rebuild to be present in the baked `tests/`.
4. ~~The 5-row prompts CSV is too small for `emb` and `s1pt` with more than a few ranks.~~
   **Closed 2026-09-27**: added `tests/_data/stage1_inputs/sample_text_seqs_1024.csv`
   (1,024 rows, 974 KB, SwissProt-derived). At the default `--batch_size 32` that is 32
   batches, so every rank has work up to 32 ranks; the grid's widest row is 24. Needs an
   image rebuild to be present in the baked `tests/`.
5. Retired — the Spark and Polaris multi-node rows are not in the grid.

6. Retired — Mithril is not in the grid. The finding stands if it returns: Stage 1
   inference merges per-rank shards through the filesystem
   (`_merge_rank_shards` in `src/biom3/Stage1/run_PenCL_inference.py`), which cannot work
   where nodes share none.

7. `s3gft` has no single-node wrapper (only `scripts/stage3_finetune_multinode.sh`), so
   the wrapper rows call the entry point directly, through `$LAUNCH`.
8. ~~`gft` fails with `--device auto`.~~ **Closed 2026-09-26.**
   `src/biom3/Stage3/run_ProteoScribe_finetuning.py` never called `resolve_device`
   (unlike its two siblings), so the raw string `auto` reached
   `torch.load(..., map_location="auto")` and every `gft` run died before the first step.
   Fixed in `38725a5` by applying the same resolve/`check_devices_per_node` block the
   Stage 3 trainer uses, first shipped in the `25e440d` images and still present in
   `779859b`. It passed on CUDA before the grid was reset; not re-verified under the
   current grid. The fix also gives `gft` the
   `check_devices_per_node` guard it never had, so an over-large `--devices_per_node`
   now fails early rather than mid-run — worth knowing on Aurora, where a tile count
   that previously slipped through will now be rejected up front.
9. The baked z_c inputs predate `run1_base`. `Stage2_MMD_swissprot_embedding_subset_1000.hdf5`
   dates from March 2026 and `test_Facilitator_embeddings.pt` from February 2026, so their
   z_c come from older PenCL/Facilitator weights. As a result, `ft` starts above the
   from-scratch validation loss (5.51 vs 3.45 measured on CUDA), and `gen` produces full-length,
   low-complexity sequences. Both still show that the machinery runs, but not that the
   outputs make sense. Regenerating both files with `run1_base` would make these checks
   meaningful.

10. `--max_steps` is silently ignored by the `gft` and `pt`/`ft` smoke commands.
   `train_model` only sets `trainer_params['max_steps']` on the `combine` branch
   ([`src/biom3/Stage3/run_PL_training.py:1744`](../../src/biom3/Stage3/run_PL_training.py#L1744));
   the `primary_only` branch sets `max_epochs` instead and drops `max_steps` on the
   floor. `training_strategy` defaults to `auto`, which resolves to `primary_only`
   whenever there are no `--secondary_data_paths`
   ([`src/biom3/Stage3/run_PL_training.py:1140`](../../src/biom3/Stage3/run_PL_training.py#L1140)), so that is the usual
   case. The flag is still accepted and still advertised in `--help`, so it looks like it
   works. `--limit_train_batches` is applied unconditionally
   ([`src/biom3/Stage3/run_PL_training.py:1748`](../../src/biom3/Stage3/run_PL_training.py#L1748)) and is what the commands
   above now use. Found when an `s3gft` diagnostic ran a full 870-batch epoch instead of
   the 20 steps requested. Worth deciding whether `max_steps` should
   warn or apply in epoch mode. Not fixed here — reported only.

11. ~~Stage 1 `s1` fails on XPU with more than one rank.~~ **Fixed in `0d4a374`, in the `779859b` images.** `PL_PEN_CL` computes the RankME effective ranks from singular
   values taken on CPU, then logged them with `sync_dist=True`
   ([`src/biom3/Stage1/PL_wrapper.py:447`](../../src/biom3/Stage1/PL_wrapper.py#L447)).
   Lightning all-reduces whatever it is handed, and an XPU process group is `xccl`-only
   with no backend for a CPU tensor, so the run died at the first validation epoch end
   with `RuntimeError: No backend type associated with device type cpu`. CUDA is
   unaffected (PyTorch registers gloo for CPU alongside NCCL) and one rank is unaffected
   (Lightning skips the reduction at `world_size == 1`), which is why the CUDA single-rank
   run passed.
   Fixed by logging `erank.to(self.device)`.

   **Scope correction:** an earlier version of this item claimed `mask_PL_PEN_CL` had the
   same defect via `performance_metrics(logits.detach().cpu())`. That was wrong.
   `compute_class_metrics` returns sklearn floats, not tensors, and Lightning creates
   those on the module device. Only the erank lines were ever CPU tensors, and they are
   active solely in `PL_PEN_CL` — `mask_PL_PEN_CL` has them commented out. So only
   `dataset_type: default` was affected, not two wrappers.

12. ~~`apptainer_mpi_run.sh`'s `USAGE` header omits `BIOM3_RANK_SOURCE=mpi`.~~
   **Fixed in `0d4a374`.** The body documented the flag but no example used it, so
   following `--help` gave the `pals` default — the fallback for the `xpu` image rather
   than the native path for `xpu-oneapi`. All three examples now set it, and `ENV` lists
   it. No image rebuild needed: the wrappers run on the host.

13. `emb` crashed on every rank that received no rows. **Fixed in `4ea2435`**; the fix is
   host-side, in `scripts/aurora/apptainer_mpi_run.sh`, so it needs no image rebuild. In the `779859b` images. Not
   re-verified under the current grid.

   Two corrections to earlier versions of this item. The fix shipped as `4ea2435` (one line
   in `scripts/aurora/apptainer_mpi_run.sh`), not `4f5e05c` — that earlier attempt read
   `BIOM3_WORLD_SIZE` in `biom3.core` instead and was reverted. Note the bug is specific to
   the `mpiexec` path: `torchrun` sets `WORLD_SIZE` itself, so the `v1` row never had it.

   Original diagnosis, unchanged: The crash (23 ranks raising
   `IndexError: Dimension out of range` from `torch.norm(z_p_tensor, dim=1)`) was a
   symptom, not the defect. The defect: with `BIOM3_RANK_SOURCE=mpi` the MPI wrapper
   skips the PALS-to-torch variable translation, so `WORLD_SIZE` is never set. Rank
   still resolved from `PALS_RANKID`, but world size fell through to the
   `PALS_LOCAL_SIZE × PBS_NODEFILE` fallback, and `/var/spool/pbs` is not bind-mounted
   into the container, so `get_world_size()` returned 1.

   Rank N with `world_size=1` shards as `[N::1]`: rank 0 silently did the whole job and
   every other rank got nothing. Had `world_size` been correct, the empty ranks would
   have returned at the shard barrier and never reached the reporting block at all.

   **This was a correctness bug, not just a crash.** Any multi-rank run under
   `BIOM3_RANK_SOURCE=mpi` was doing single-rank work; only the empty-tensor norm made
   it visible. Lightning programs get their rank from `MPIEnvironment` rather than this
   path, so a passing training run is not evidence that this path is healthy.

   Fixed three ways: read `BIOM3_WORLD_SIZE` (which the wrapper already exports into
   every container) in `_dist_env`; refuse to start when `rank >= world_size`, since that
   combination has no valid interpretation and is otherwise invisible; and keep the
   empty-rank tensors 2-D so the reporting block survives a rank that legitimately holds
   no rows. `pytest tests/ --quick` clean (1382 passed), plus stage1/core/pipeline
   (244 passed). The `emb` cells stay `blocked` until the images are rebuilt.

14. **Unverified.** Stage 3 generation was reported as not bitwise reproducible on XPU
   (10 of 25 sequences identical, the rest differing by single residues). The source is an
   agent whose results are not being carried forward, so treat this as a lead, not a
   finding. It is why the `gen` criterion no longer requires matching sequences across
   runs — if the report is wrong, that criterion should come back.

15. **Unverified.** Single-node Aurora was reported to need no wrapper at all
   (`apptainer exec --cleanenv` plus `torchrun` on the unmodified image), with only
   `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` affecting the outcome. Same source, same
   caveat. Worth re-testing directly, since it would simplify the `v1` row considerably.
