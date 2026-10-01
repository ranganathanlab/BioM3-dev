# Contributing to BioM3-dev

Thanks for your interest in contributing! This guide walks through the workflow we use for both internal collaborators and external contributors: cloning the repo, branching off `dev`, committing your changes, and opening a pull request.

For background on the project's layout and how the package is built, see the [README](../README.md) and [docs/misc/biom3_ecosystem.md](./misc/biom3_ecosystem.md).

## Branch model

BioM3-dev uses three tiers of branches:

| Branch | Purpose |
|--------|---------|
| `main` | Release branch. Advances by fast-forward only at version bumps — see [versioning.md](./misc/versioning.md). Never commit here directly. |
| `dev` | Integration branch. All new work is branched from `dev` and merged back into `dev` via pull request. |
| `<your-name>-<topic>` | Your personal feature branch. Short-lived, branched off `dev`, merged back into `dev`. |

The short version: **branch from `dev`, PR back into `dev`**. Releases out of `dev` into `main` are handled separately by maintainers.

## 1. Clone the repository

External contributors should first **fork** the repo on GitHub (top-right "Fork" button on [github.com/ranganathanlab/BioM3-dev](https://github.com/ranganathanlab/BioM3-dev)), then clone your fork:

```bash
git clone https://github.com/<your-username>/BioM3-dev.git
cd BioM3-dev
git remote add upstream https://github.com/ranganathanlab/BioM3-dev.git
```

Internal collaborators with push access to `ranganathanlab/BioM3-dev` can clone the repo directly without forking:

```bash
git clone https://github.com/ranganathanlab/BioM3-dev.git
cd BioM3-dev
```

Then install the package in editable mode (see the [README](../README.md#pip-install) for details, including the `[app]` extra):

```bash
pip install -e .
```

## 2. Create a personal branch from `dev`

Make sure your local `dev` branch is up to date before branching:

```bash
git fetch origin
git checkout dev
git pull origin dev
```

External contributors should also pull from `upstream`:

```bash
git fetch upstream
git checkout dev
git merge upstream/dev
```

Then create your personal branch off `dev`. Use a name that identifies you and the topic — for example `addison-fix-bert-padding` or `jane-feat-streamlit-export`:

```bash
git checkout -b <your-name>-<short-topic>
```

> **Tip:** Keep branches small and topic-scoped. One branch per logical change makes review faster and reduces merge conflicts. If your work spans several unrelated changes, open a separate branch and PR for each.

## 3. Make changes and commit

Edit, add, or remove files as needed. Before committing, run the test suite to confirm nothing regressed:

```bash
source environment.sh
pytest tests/test_imports.py        # quick smoke test, no weights required
pytest tests/                       # full suite (skips tests that need missing weights)
```

Stage your changes and commit. BioM3-dev follows [Conventional Commits](https://www.conventionalcommits.org/) — the type prefix tells reviewers at a glance what kind of change this is:

```bash
git add <files>
git commit -m "<type>: <short summary under 72 chars>"
```

Common types:

| Type | When to use |
|------|-------------|
| `feat` | New user-visible feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructuring with no behavior change |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `docs` | Documentation-only changes |
| `chore` | Build, tooling, or housekeeping |

Examples from the repo's history:

```text
feat: add --load_from_checkpoint flag to Stage 1 inference
fix: correct BERT padding to match training config
refactor: extract raw-weight and checkpoint loaders in run_PenCL_inference
```

Use the commit body (after a blank line) for the *why* if it isn't obvious from the summary. Push your branch to your remote:

```bash
# Internal contributors (push to origin = ranganathanlab/BioM3-dev)
git push -u origin <your-branch-name>

# External contributors (push to your fork)
git push -u origin <your-branch-name>
```

## 4. Open a pull request

Once your branch is pushed, open a PR against the `dev` branch of `ranganathanlab/BioM3-dev`.

**Via the GitHub web UI:** GitHub will prompt you with a "Compare & pull request" banner after pushing. Click it, then **make sure the base branch is set to `dev` (not `main`)**. Fill in the title and description and submit.

**Via the `gh` CLI** (recommended if you have it installed):

```bash
gh pr create --base dev --title "<type>: <short summary>" --body "..."
```

A good PR description includes:

- **What** the change does (one or two sentences).
- **Why** it's needed — the motivation, bug being fixed, or use case being unlocked.
- **How to test** it — the commands a reviewer can run to verify the change.
- Links to any related issues or discussion.

After opening the PR:

1. Wait for CI checks (if configured) and address any failures.
2. Respond to review comments by pushing additional commits to the same branch — the PR updates automatically. Avoid force-pushing once review has started, since it makes review comments harder to follow.
3. Once approved, a maintainer will merge the PR into `dev`. Your branch can then be deleted.

## Things to keep in mind

- **Don't commit large binaries.** Model weights, datasets, and reference databases live in [BioM3-data-share](https://github.com/natural-machine/BioM3-data-share), not in this repo. The `weights/` and `data/databases/` directories are gitignored for this reason.
- **Don't bump the version yourself.** Version bumps happen on release boundaries and are coordinated by maintainers — see [versioning.md](./misc/versioning.md).
- **Match the existing code style.** PascalCase classes, snake_case functions, light type hints on public APIs, no docstrings or inline comments on code you didn't change. Full conventions are in [CLAUDE.md](../CLAUDE.md#code-style).
- **Keep PRs focused.** A reviewer should be able to hold the whole change in their head. If a PR grows beyond that, split it.

If you run into trouble or are unsure about scope before starting work, open an issue first to discuss the approach.
