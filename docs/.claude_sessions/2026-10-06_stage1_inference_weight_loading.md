# Session: Stage 1 inference weight loading, `--skip_facilitator`, caption attention mask

**Date:** 2026-10-06
**Branch:** `stage1-inference-loading` (from `dev` at `b1a8363`)

## Goal

Act on a handoff from an agent working in the `embedding-space-perturbations` workspace
(`BIOM3_DEV_HANDOFF.md` there), which reported four problems with Stage 1 inference and the
embedding pipeline. All four were checked against `dev` and confirmed before any change.

## Summary

1. **Stage 1 inference could embed with weights that never loaded.** Fixed.
2. **The pipeline could not run a PenCL weight set that has no Facilitator.** Added
   `--skip_facilitator`, which replaces Stage 2 with the identity map: `z_c` is written as
   a copy of `z_t` and the rest of the pipeline runs on it.
3. **Inference never passed the caption attention mask to BERT**, although Stage 1 pfam
   training has since `ef43803` (2026-09-09). Added `--text_attention_mask`, off by default.
   A weight set can record how its PenCL was trained, and the pipeline warns loudly when
   the flag disagrees with that record.
4. **Two weight files in the main Aurora checkout were Lightning checkpoints under a `.bin`
   name.** The user replaced both with the published files; see "Local weights" below.

## The loading bug

`biom3_PenCL_inference` chose its loader from the file extension. Anything not named `.ckpt`
went through `load_and_prepare_model(strict=False)`, which unwraps a Lightning checkpoint
but keeps its `model.` prefix. For a Lightning checkpoint named `.bin`, no key matched, the
non-strict load reported nothing, and the run embedded with an untrained projection head.
The sign is every `z_t` row norm equal to 22.63 (√512, from the untrained LayerNorm).

This is the same failure as the finetune-embedder bug fixed in July
(`2026-07-24_pencl_finetune_embedder_load_fix.md`), in a path that fix did not cover.

Both loader paths now go through `Stage1.io.load_pencl_weights`:

- The format is read from the file (`core.io.load_state_dict_unwrap_pl`), not its name.
- The load is non-strict but checked: it raises unless every parameter is populated.
  Parameters are tracked by identity, so a tied parameter (the ESM-2 and BERT output heads)
  counts as loaded when any of its names is in the file.
- Missing buffers and unused keys only warn. This is what the old `strict=False` was for:
  `BioM3_PenCL_epoch20.bin` and `PenCL_V09152023_last.ckpt` carry a
  `text_encoder.model.bert.embeddings.position_ids` buffer that current transformers no
  longer keeps.
- Checkpoints no longer go through the PL wrapper. The wrappers hold the network as
  `self.model` and have no parameters or load hooks of their own.

`--load_from_checkpoint` is kept for compatibility. It, or a `.ckpt` path, still selects the
network class from the config's `model_type`; it no longer changes how weights load.

## Changes

| Commit | What |
| ------ | ---- |
| `da524c6` | `fix:` `load_pencl_weights` in `Stage1/io.py`; both paths in `run_PenCL_inference.py` use it; 14 tests |
| `458897c` | `feat:` `--skip_facilitator` in `pipeline/embedding_pipeline.py`; 4 tests |
| `a1a43ec` | `feat:` `--text_attention_mask` through `preprocess.collate_fn(include_mask=...)`, Stage 1 inference and the pipeline; 10 tests |
| `f0ea58f` | `docs:` `emb` pass criterion and open item 17 in `container_validation.md`; this note |
| `a389295` | `feat:` `--skip_facilitator` writes `z_c` as a copy of `z_t` and keeps the HDF5 compile and `--generate` |
| `46121cd` | `feat:` `pencl_trained_with_text_attention_mask` in weight sets; mismatch warning in the pipeline; 14 tests |

`458897c` first made `--skip_facilitator` stop after Stage 1, as the handoff asked. The user
then asked for the identity map instead, so that `z_c` is always populated; `a389295` is that
change, and it also lifts the rule that rejected the flag together with `--generate`.

## The mask and the weight set

`--text_attention_mask` (default off) is the only thing that decides whether a run passes the
mask. A weight set can record how its PenCL was trained under
`pencl_trained_with_text_attention_mask`; `run0_nm_base` and `run1_base` record `false`. When
the flag disagrees with the record the run still follows the flag, and the pipeline logs a
`CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS` banner at the start and again at the end
of the run log. Nothing is checked when the weight set has no such key, when
`--pencl_weights` names a file other than the weight set's, or when `biom3_PenCL_inference`
is run directly, since it does not read a weight set.

## Verification

Run bare metal on an Aurora compute node (`debug` queue), worktree source over the main
checkout's venv, torch 2.13.0a0, transformers 5.14.1.

Job 8906615, weight loading:

| File | Format in the file | Result |
| ---- | ------------------ | ------ |
| published `run1_base_pencl.bin` | raw, 788 keys | 752/752 parameters |
| main checkout's `run1_base_pencl.bin` | Lightning, 788 `model.` keys | 752/752 parameters |
| `run1_base_pencl.ckpt` | Lightning | 752/752 parameters |
| `BioM3_PenCL_epoch20.bin`, `PenCL_V09152023_last.ckpt` | raw / Lightning, 789 keys | loaded, one unused key warned |
| Facilitator weights (negative control) | raw, 6 keys | refused |

Through the entry point, the mislabelled file and the published one give the same
embeddings: cosine 1.000000, largest difference 2e-6, `|z_t|` 12.93 to 14.26.

Job 8906650, mask and pipeline, 64 rows, fp32 (`--no_amp --float32_matmul_precision highest`):

| Comparison | `z_t` |
| ---------- | ----- |
| mask off, old code vs new code, CPU | bit-identical |
| mask off, old code vs new code, XPU | largest difference 3e-6 |
| mask on, `max_padding` vs `dynamic` | largest difference 1.4e-5 |
| mask off, `max_padding` vs `dynamic` | cosine mean 0.978, min 0.805 |
| `run1_base`, mask on vs off | cosine mean 0.872, min 0.695 |

`z_p` is unchanged in every comparison (differences at 2e-6). The last row is why the option
is off by default: `run1_base` was trained without the mask, and turning it on moves `z_t` a
long way.

The same job confirmed that a weight set holding only `pencl_weights` is refused without
`--skip_facilitator`, ran the full pipeline, and ran the tests: Stage 1 inference and
pipeline tests 52 passed, 4 skipped (no CUDA); `pytest tests --quick` 1465 passed, 180
skipped.

Job 8906769, identity map and mask warning, at `46121cd`:

| Check | Result |
| ----- | ------ |
| `--skip_facilitator`, 64 rows | `z_c` equal to `z_t`, in separate storage; the HDF5 embedding equal to `z_t`; 64 rows |
| full pipeline, 64 rows | `z_c` differs from `z_t`, and is bit-identical to the run before these changes |
| `--skip_facilitator --generate`, 5 rows | Stage 3 samples from the copy and writes `run.generated.pt` |
| flag on, weight set records `false` | banner logged twice |
| flag off, weight set records `true` | banner logged twice |
| flag agrees with the weight set | no banner |

Tests in that job: pipeline and Stage 1 inference tests 62 passed, 4 skipped (no CUDA);
`pytest tests --quick` 1479 passed, 180 skipped.

The loader tests fail on the old code (8 of the 9 `prepare_model` cases). `da524c6`,
`458897c` and `a389295` were each checked on their own tree with the stub-based tests.

## Local weights

The real (not symlinked) weight files in the main Aurora checkout were checked against the
published bundle with `biom3_fetch_weights run1_base -o weights --force --dry_run`, which
hashes each file against the registry digest and changes nothing.

| File | Finding |
| ---- | ------- |
| `PenCL/run1_base_pencl.bin` | 3,918,095,585 bytes, a byte-identical copy of the sharepoint `.ckpt`; published is 3,048,812,529 |
| `Facilitator/run1_base_facilitator.bin` | 12,618,834 bytes, a byte-identical copy of the sharepoint `.ckpt`; published is 4,203,629 |
| `ProteoScribe/run1_base_proteoscribe.bin` | matches the published digest |
| `ProteoScribe/blend_sd142k_ep353_ep59.bin` | not published, so no reference; a bare state dict with ProteoScribe's 223 keys and shapes, all values finite |

Every symlinked file the bundle covers matched. The user then replaced the first two with
`biom3_fetch_weights run1_base -o weights --force` and set them read-only again; a second
dry run reports 11 files present and none to fetch. The `.ckpt` symlinks to the sharepoint
checkpoints are unchanged. `scripts/weights_bundle/bundle_specs/run1_base.json` copies
`.bin` sources verbatim, so a bundle built from this checkout before the replacement would
have published Lightning checkpoints as `.bin`.

The published `run1_base_proteoscribe.bin` is not Lightning-wrapped, yet all 223 of its
keys carry the `model.` prefix. It loads because the shared loader strips the prefix
whether or not the file is wrapped.

## Open items

1. **Rerun the invalid `emb` outputs**: `outputs/validation/au1-emb`, `au2-emb` and
   `au2-emb.pre-4ea2435` on Aurora, and anything downstream of them.
2. **Rebuild the images.** The changes are in `src/`, so `xpu-oneapi-abd9941` and the other
   published images still have the loading bug.
3. **The mask in the other caption paths.** `Stage3/finetune_embedder.py:52` and
   `rl/grpo.py:138` also embed captions without the mask. That is right for `run1_base`
   and wrong for weights trained with the mask. Left alone: they need the same option
   before they are used with such weights.

## Reverting

```bash
git revert 46121cd a389295 a1a43ec 458897c da524c6
```

The loader fix `da524c6` stands on its own. The later commits build on each other in the
order given.
