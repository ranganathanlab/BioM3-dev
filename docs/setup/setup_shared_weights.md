# Shared Model Weights

Pretrained model weights required by BioM3 are stored in a shared directory on each machine. This avoids duplicating large files across users and project copies. The local `weights/` directory in each clone of BioM3-dev is populated with symlinks that point to these shared files.

> **No cluster access?** The Run 1 base weights are also published to GHCR as a pullable
> bundle that includes the matching architecture configs. See
> [weights_bundle.md](weights_bundle.md). It populates `weights/` through the same
> `link_weights.sh` described below, so the two approaches are interchangeable.

## Shared weights locations

| Machine | Shared weights path | Permissions |
|---------|-------------------| -------- |
| DGX Spark | `/data/data-share/BioM3-data-share/data/weights` | read only |
| Polaris (ALCF) | `/grand/NLDesignProtein/sharepoint/BioM3-data-share/data/weights` | read only |
| Aurora (ALCF) | `/flare/NLDesignProtein/sharepoint/BioM3-data-share/data/weights` | read only |

The shared directory mirrors the structure of the local `weights/` directory:

```
weights/
  Facilitator/
  LLMs/
  PenCL/
  ProteoScribe/
```

## Populating the local `weights/` directory

The script `scripts/link_weights.sh` creates symlinks in your local `weights/` directory for any files present in the shared directory that you don't already have locally. Existing local files are left untouched and verified against the shared copy via md5 checksum.

```bash
# Preview what will be linked (recommended before first run)
./scripts/link_weights.sh <shared_weights_path> weights --dry-run

# Create symlinks
./scripts/link_weights.sh <shared_weights_path> weights
```

For example, on DGX Spark:

```bash
./scripts/link_weights.sh /data/data-share/BioM3-data-share/data/weights weights --dry-run
./scripts/link_weights.sh /data/data-share/BioM3-data-share/data/weights weights
```

The script will report `MATCH`, `MISMATCH`, or `LINK` for each entry. A `MISMATCH` does not necessarily indicate a problem -- files saved with different versions of PyTorch may differ at the byte level while containing identical tensor data. The script output includes a command to verify tensor-level equivalence when mismatches are found.

## Adding additional weights

The local `weights/` directory supports a mix of symlinked shared files and additional local files. To add new weights that are not in the shared directory, simply place them in the appropriate subdirectory under `weights/`. Running `link_weights.sh` again will skip any files that already exist locally.

The script creates real directories for each component (e.g. `weights/PenCL/`) and symlinks only the individual files within them. This ensures you can always write new local files alongside the shared symlinks.

## Weights required by the test suite

Some tests depend on pretrained weight files that are too large to commit to git. These tests are skipped automatically when the required files are absent (via `check_downloads()` in `tests/conftest.py`). To run the full test suite, the following files must be present in the `weights/` directory:

| File | Stage | Tests |
|------|-------|-------|
| `LLMs/esm2_t33_650M_UR50D.pt` | 1 | `test_stage1_run_PenCL_inference` |
| `PenCL/BioM3_PenCL_epoch20.bin` | 1 | `test_stage1_run_PenCL_inference` |
| `PenCL/run1_base_pencl.ckpt` | 1 | `test_stage1_run_PenCL_inference` (checkpoint loading variant), `test_finetune_embedder` |
| `Facilitator/BioM3_Facilitator_epoch20.bin` | 2 | `test_stage2_run_Facilitator_sample` |
| `ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin` | 3 | `test_model_load_from_bin`, `test_stage3_run_ProteoScribe_sample`, `test_stage3_run_PL_training` |
| `ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.renamed.bin` | 3 | `test_model_load_from_bin`, `test_stage3_run_ProteoScribe_sample` |
| `ProteoScribe/epoch200_full.ckpt/single_model.pth` | 3 | `test_model_load_from_checkpoint_file` |

All of these files are available in the shared weights directory and can be populated by running `link_weights.sh` as described above. They are also published as a bundle: `biom3_fetch_weights --test_weights -o weights` fetches the same set (about 11 GB). Tests that do not depend on these files — including the key-correction tests using committed dummy weights (see `docs/bug_reports/axial_positional_embedding_keys.md`) — will always run.
