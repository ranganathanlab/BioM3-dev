<!-- Maintained identically across all BioM3 repos. -->

# BioM3 Ecosystem

## Overview

BioM3 is a multi-stage framework for text-guided protein sequence generation ([NeurIPS 2024](https://openreview.net/forum?id=L1MyyRCAjX)). The project is organized across multiple repositories, each with a distinct role in the development and deployment workflow.

## Repositories

| Repository | Role | Description |
|------------|------|-------------|
| [BioM3-dev](https://github.com/ranganathanlab/BioM3-dev) | Core library | Python package implementing the 3-stage pipeline (PenCL, Facilitator, ProteoScribe), dataset construction (`biom3.dbio`), visualization (`biom3.viz`), and training infrastructure. |
| [BioM3-data-share](https://github.com/natural-machine/BioM3-data-share) | Shared data | Model weights, training datasets, and reference databases. Synced across compute clusters via rsync. Contains download scripts for bioinformatics databases. |
| [BioM3-workflow-demo](https://github.com/natural-machine/BioM3-workflow-demo) | Demo workflows | End-to-end 8-step pipeline demonstrating finetuning and sequence generation on a protein family, with structure prediction and evaluation. |
| [BioM3-workspace-template](https://github.com/natural-machine/BioM3-workspace-template) | Workspace setup | GitHub template repository for setting up new BioM3 workspaces with standardized directory structure and configuration. |

## How the repos relate

```
                  ┌──────────────────────┐
                  │      BioM3-dev       │
                  │   (core library)     │
                  └──────────┬───────────┘
                             │ pip install
                             ▼
                  ┌──────────────────────┐        ┌──────────────────────┐
                  │ BioM3-workflow-demo  │◄───────│  BioM3-data-share    │
                  │   (demo pipeline)    │ sync   │  (weights / data)    │
                  └──────────────────────┘        └──────────────────────┘
                             │                               ▲
                             │                               │ sync
                             │                    ┌──────────┴───────────┐
                             │                    │      BioM3-dev       │
                             │                    │  (also syncs data)   │
                             │                    └──────────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ BioM3-workspace-     │
                  │ template             │
                  │ Same structure as    │
                  │ workflow-demo        │
                  └──────────────────────┘
```

- **BioM3-dev** is the core dependency. It provides the `biom3` Python package that all other repos install.
- **BioM3-data-share** is standalone (no code dependencies). It holds canonical model weights and databases that are synced to each compute cluster.
- **BioM3-workflow-demo** depends on BioM3-dev (installed via pip) and BioM3-data-share (weights and databases symlinked via sync scripts).
- **BioM3-workspace-template** mirrors the structure of BioM3-workflow-demo as a GitHub template for new research workspaces.

## Shared data architecture

BioM3-data-share holds the canonical copy of model weights and datasets. Each compute cluster has a local copy synced via `sync/biom3sync.sh`. BioM3-dev and BioM3-workflow-demo symlink into these paths for weights and databases.

### Per-machine paths

| Machine | BioM3-data-share root | Weights | Databases |
|---------|-----------------------|---------|-----------|
| DGX Spark | `/data/data-share/BioM3-data-share` | `data/weights/` | `databases/` |
| Polaris (ALCF) | `/grand/NLDesignProtein/sharepoint/BioM3-data-share` | `data/weights/` | `databases/` |
| Aurora (ALCF) | `/flare/NLDesignProtein/sharepoint/BioM3-data-share` | `data/weights/` | `databases/` |

Databases (NR, Pfam, SwissProt, etc.) are downloaded per-machine via scripts in `BioM3-data-share/download/` and are **not** synced across clusters.

## Version compatibility

BioM3-dev is in active alpha development on the `0.1.0a` track. Formal semantic versioning will be introduced once BioM3-dev reaches a stable release. See [the tags page](https://github.com/ranganathanlab/BioM3-dev/tags) for the current version.

### Current approach

BioM3-workflow-demo tracks its compatibility with BioM3-dev via `SYNC_LOG.md`, which records paired commit hashes at each sync point. This is the recommended pattern for any repo that depends on BioM3-dev.

### Known-good commits (as of April 2026)

| Repository | Commit | Date |
|------------|--------|------|
| BioM3-dev | `f30d682` | 2026-04-04 |
| BioM3-workflow-demo | `d2b7948` | 2026-04-04 |
| BioM3-data-share | `1417824` | 2026-04-03 |

### Checking for upstream changes

```bash
# From within BioM3-workflow-demo, check what's new in BioM3-dev since last sync:
cd /path/to/BioM3-dev
git log --oneline <last-synced-commit>..HEAD
```

## Machine-specific paths

Each developer's machine has repos cloned to different absolute paths. To support AI-assisted development across repos, each repo contains a `.claude/repo_paths.json` file (gitignored, not version controlled) that maps repo names to local paths:

```json
{
  "biom3_dev": "/home/user/Projects/BioM3-dev",
  "biom3_data_share": "/home/user/Projects/BioM3-data-share",
  "biom3_workflow_demo": "/home/user/Projects/BioM3-workflow-demo",
  "biom3_workspace_template": null,
  "biom3_data_share_system": "/data/data-share/BioM3-data-share"
}
```

This file must be created manually on each machine. See `CLAUDE.md` in each repo for details.

## Cross-repo workflows

### Building a dataset
1. Ensure reference databases are downloaded in BioM3-data-share (`download/` scripts).
2. Run `biom3_build_dataset` (from BioM3-dev) pointing at the databases in BioM3-data-share.
3. Output CSVs go to BioM3-data-share `data/datasets/`.

### Running the demo pipeline
1. Install BioM3-dev (`pip install -e /path/to/BioM3-dev` or from GitHub).
2. Symlink weights and databases from BioM3-data-share into BioM3-workflow-demo.
3. Run the 8-step pipeline via `run_pipeline.py` or individual step scripts.

### Syncing weights and databases
1. On the source machine, run `sync/biom3sync.sh push` in BioM3-data-share.
2. On the target machine, run `sync/biom3sync.sh pull`.
3. Sync logs are written to `.logs/sync.log` (TSV format).

### Keeping workflow-demo in sync with BioM3-dev
1. Check for upstream changes: `git log <last-synced-commit>..HEAD` in BioM3-dev.
2. Update config files and scripts in BioM3-workflow-demo to match new CLI flags or parameters.
3. Record the sync point in `SYNC_LOG.md`.
