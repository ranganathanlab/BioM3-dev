# CLAUDE.md

## Practices

Store session notes in docs/.claude_sessions/

## Project overview

BioM3 is a multi-stage framework for generating novel protein sequences guided by natural language prompts (NeurIPS 2024). It combines protein language models (ESM-2), biomedical text encoders (BioBERT), and diffusion-based sequence generation.

**Pipeline:**
```
Input (CSV: sequences + text)
  → Stage 1: PenCL inference (ESM-2 + BioBERT → joint embeddings z_t, z_p)
  → Stage 2: Facilitator (z_t → z_c, aligned to protein space)
  → Stage 3: ProteoScribe (diffusion model, z_c → generated sequences)
```

## Ecosystem context

BioM3-dev is the core library in a multi-repo ecosystem. See [docs/misc/biom3_ecosystem.md](docs/misc/biom3_ecosystem.md) for full details.

Related repositories:
- **BioM3-data-share** — shared model weights, datasets, and reference databases
- **BioM3-workflow-demo** — end-to-end demo of finetuning and generation
- **BioM3-workspace-template** — *(planned)* workspace configuration template

Machine-specific repo paths are in `.claude/repo_paths.json` (gitignored, not version controlled). This file maps repo names to absolute paths on the current machine.

## Repository layout

```
src/biom3/
  backend/          # Device abstraction (CPU / CUDA / XPU)
  core/             # Shared utilities: model I/O (io.py), state-dict helpers (helpers.py), dataloaders
  dbio/             # Database I/O: readers for SwissProt/Pfam/NCBI taxonomy, enrichment, dataset building
  Stage1/           # PenCL: model.py (encoders + projection), preprocess.py, PL_wrapper.py
  Stage2/           # Facilitator: run_Facilitator_sample.py
  Stage3/           # ProteoScribe: diffusion model, PL training, finetuning, sampling
  rl/               # RL post-training: GRPO, GDPO, DPO; rewards/ package, rollout, preference data
  pipeline/         # End-to-end embedding pipeline (Stage 1 → Stage 2 in one entrypoint)
  geometry/         # Latent-manifold fitting + distance-from-manifold scoring
  split/            # Cluster-aware and stratified dataset splitting
  data_prep/        # HDF5 compilation from embeddings
  benchmarks/       # Stage 3 training/generation benchmarks + plotting
  viz/              # 3D structure rendering, sequence analysis, unmasking-order plots
  app/              # Streamlit web app (installed via the `app` extra)
  cli/              # `biom3 <command>` dispatcher: registry.py (command table), dispatch.py
configs/            # JSON configs for inference, per-stage training, RL, splits, and jobs
  inference/        #   Inference configs; models/ holds shared bases (uses _base_configs composition)
  stage1_training/  #   Stage 1 training; models/ + machines/ bases
  stage2_training/  #   Stage 2 training; models/ + machines/ bases
  stage3_training/  #   Stage 3 training + finetuning; models/ + machines/ bases
  grpo/ dpo/        #   RL post-training configs
  split/ weights/   #   Split specs and named weight sets
  benchmark/ jobs/  #   Benchmark and job-template configs
scripts/            # Bash wrappers (embedding, training, generation, RL, cloud, sync)
docker/             # Dockerfiles (cuda, cpu, xpu, xpu-oneapi), build/push/run wrappers, entrypoint
cloud/              # Mithril job runner (run.mithril.yaml) and its README
requirements/       # pip requirement sets per machine and per image
demos/              # End-to-end demos (dbio dataset building, SH3 embedding pipeline)
data/databases/     # Symlinked reference databases (gitignored, see docs/setup/setup_databases.md)
tests/              # pytest suite (conftest.py, per-stage tests, test data in tests/_data/)
weights/            # Pre-trained model weights (gitignored, see weights/README.md)
docs/               # Setup guides (setup/), CLI reference, dbio/, reinforcement_learning/, misc/
jobs/               # HPC job submission scripts
```

## Entry points

Defined in `pyproject.toml`. See [docs/CLI_reference.md](docs/CLI_reference.md) for argument tables.

Unified command:
- `biom3 <command>` → `biom3.cli:main`. Dispatches a curated subset of the entry points below (e.g. `biom3 embed`, `biom3 generate`, `biom3 finetune`, `biom3 train stage3`) with unchanged arguments. Commands are declared in `src/biom3/cli/registry.py`; the `biom3_*` names stay available. The dispatcher must not modify `sys.argv`, since Lightning re-executes it to launch worker processes.

Inference:
- `biom3_PenCL_inference` → `biom3.Stage1.__main__:run_PenCL_inference`
- `biom3_Facilitator_sample` → `biom3.Stage2.__main__:run_Facilitator_sample`
- `biom3_ProteoScribe_sample` → `biom3.Stage3.__main__:run_ProteoScribe_sample`
- `biom3_embedding_pipeline` → `biom3.pipeline.__main__:run_embedding_pipeline`

Training:
- `biom3_train_stage1` → `biom3.Stage1.__main__:run_stage1_training`
- `biom3_train_stage2` → `biom3.Stage2.__main__:run_stage2_training`
- `biom3_train_stage3` → `biom3.Stage3.__main__:run_stage3_training` (HDF5 z_c; `--finetune True` for finetuning)
- `biom3_finetune_stage3` → `biom3.Stage3.__main__:run_stage3_finetuning` (JSONL records, on-device z_c)

RL post-training:
- `biom3_grpo_train` / `biom3_gdpo_train` / `biom3_dpo_train` → `biom3.rl.__main__:run_{grpo,gdpo,dpo}_train`

Data preparation:
- `biom3_build_dataset` → `biom3.dbio.__main__:run_build_dataset`
- `biom3_build_taxid_index` → `biom3.dbio.__main__:run_build_taxid_index`
- `biom3_csv_to_parquet` → `biom3.dbio.__main__:run_csv_to_parquet`
- `biom3_build_source_{swissprot,pfam,trembl,expasy,smart,brenda}` → `biom3.dbio.__main__:run_build_source_*`
- `biom3_build_pfam_subsets` / `biom3_build_annotation_cache` → `biom3.dbio.__main__:*`
- `biom3_compile_hdf5` → `biom3.data_prep.__main__:run_compile_hdf5`
- `biom3_cluster_split` / `biom3_stratified_cluster_split` → `biom3.split.__main__:*`

Analysis:
- `biom3_fit_manifold` / `biom3_score_manifold` → `biom3.geometry.__main__:*` (see [docs/misc/manifold_distance.md](docs/misc/manifold_distance.md))

Benchmarks and app:
- `biom3_benchmark_stage3_generation` / `biom3_benchmark_stage3_training` / `biom3_plot_benchmark` → `biom3.benchmarks.__main__:*`
- `biom3_app` → `biom3.app:main`

## Building and running

```bash
# Install (editable, from repo root)
pip install -e .

# Smoke only (imports)
pytest tests/test_imports.py

# Fast dev loop — skips entrypoint/training tests (~3 min, CPU, no weights required)
pytest tests/ --quick

# Full test suite (default — includes entrypoint + training tests; weight-gated tests skip if weights missing)
pytest tests/

# Include GPU-only tests
pytest tests/ --include_requires_gpu
```

## Testing conventions

- Test files live in `tests/` with subdirectories per stage.
- Test data is in `tests/_data/`; test outputs go to `tests/_tmp/`.
- CLI arguments for entrypoint tests are stored in `tests/_data/entrypoint_args/*.txt`.
- Custom pytest markers:
  - `@pytest.mark.benchmark` (needs `--benchmark` to run)
  - `@pytest.mark.requires_gpu` (needs `--include_requires_gpu` to run)
  - `@pytest.mark.network` (needs `--network` to run)
  - `@pytest.mark.database_files` (needs `--database_files` to run)
  - `@pytest.mark.slow` (skipped under `--quick`; applied module-wide to entrypoint + training + pipeline tests)
- Tests that require downloaded weights skip gracefully with a message if files are missing.
- For the fast dev loop, prefer `pytest tests/ --quick` over `pytest tests/test_imports.py` — the former covers dbio, viz, Stage 3 sampling/data-splitting/model-IO, and core utilities without needing weights or GPU.

## Code style

### Python conventions
- **Classes**: PascalCase (`ProteinEncoder`, `PEN_CL`, `PL_ProtARDM`)
- **Functions / variables**: snake_case (`load_json_config`, `prepare_model`)
- **Constants**: UPPER_SNAKE_CASE (`DATDIR`, `TMPDIR`, `BACKEND_NAME`)
- **Private helpers**: leading underscore (`_load_state_dict_from_file`)
- Keep type hints lightweight — use them on public function signatures but don't over-annotate internals.
- Avoid adding docstrings, comments, or type annotations to code you didn't change.

### File organization pattern
Each stage follows a consistent layout:
- `model.py` — nn.Module definitions
- `preprocess.py` — datasets and collate functions
- `PL_wrapper.py` — PyTorch Lightning modules
- `run_*.py` — end-to-end scripts (arg parsing, loading, inference/training loop)
- `io.py` — model building and checkpoint loading
- `__main__.py` — thin wrappers that call into `run_*.py`

### Imports
- Group: stdlib → third-party → project (`biom3.*`)
- Device-conditional imports (e.g., `lightning` vs `pytorch_lightning`) go behind `if BACKEND_NAME == _XPU:` guards in `backend/device.py`.

## Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

<optional body with context>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

Examples:
- `feat: add --load_from_checkpoint flag to Stage 1 inference`
- `fix: correct BERT padding to match training config`
- `refactor: extract raw-weight and checkpoint loaders in run_PenCL_inference`

Keep the summary under 72 characters. Use the body for "why", not "what".

## Key architectural details

### Device abstraction
`biom3.backend.device` detects the available backend (CUDA → XPU → CPU) and exposes `get_device()`, `get_backend_name()`. All device-specific code lives in `backend/{cuda,xpu,cpu}.py`. When writing new code, import from `backend.device` rather than hardcoding `torch.device("cuda")`.

### Checkpoint formats
The codebase handles three weight formats:
1. **Raw weights** (`.bin`, `.pt`) — plain `state_dict`, loaded via `core.io.load_and_prepare_model`
2. **Lightning checkpoints** (`.ckpt`) — loaded via `PL_wrapper.load_from_checkpoint`, then unwrap `.model`
3. **DeepSpeed sharded** (directory) — merged via `Stage3.io._load_state_dict_from_sharded_dir`

When loading models, use `core.io.load_and_prepare_model` for raw weights. For Lightning checkpoints, use the stage-specific `prepare_model_from_checkpoint` functions which handle PL wrapper construction and unwrapping.

### Configuration
- **Inference**: JSON files in `configs/inference/` → loaded via `--config_path` with `load_json_config()` → converted to `argparse.Namespace`. Old flat configs in `configs/` still work for backward compatibility.
- **Training**: JSON files in `configs/stage{1,2,3}_training/` → loaded via `--config_path` into argparse defaults. CLI args override JSON values; JSON overrides argparse defaults.
- **Config composition**: All stages (inference and training) use `core.helpers.load_json_config()`, which supports two special keys:
  - `_base_configs`: list of paths loaded *before* the current file (current file overrides them)
  - `_overwrite_configs`: list of paths loaded *after* the current file (they override it)
  - Priority (low → high): `_base_configs` < current file < `_overwrite_configs` < CLI
  - Paths resolve relative to the JSON file's directory. Both keys are stripped from the result.
- **Base configs**: `configs/inference/models/` has shared encoder/model configs (`_base_PenCL.json`, `_base_Facilitator.json`). Each `configs/stage{1,2,3}_training/` directory carries its own `models/` (shared architecture configs, e.g. `_base_ProteoScribe_1block.json`, `_base_ProteoScribe_16blocks.json`) and `machines/` (per-machine device configs: `_aurora.json`, `_polaris.json`, `_spark.json`). Stage 3 inference reuses the training model base configs via `_base_configs`.

### Training output structure
Stage 3 training (`biom3_train_stage3`) organizes outputs under `--output_root` with three key CLI args: `--checkpoints_folder` (default `checkpoints`), `--runs_folder` (default `runs`), and `--run_id` (unique per run, constructed automatically by HPC job templates).

```
{output_root}/
├── {checkpoints_folder}/{run_id}/    ← Lightning/DeepSpeed .ckpt dirs + derived weights
├── {runs_folder}/{run_id}/
│   ├── logs/                         ← lightning_logs/, wandb/
│   └── artifacts/                    ← state_dict.best.pth copy, args.json,
│                                       build_manifest.json, run.log
```
