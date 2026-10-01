"""Commands exposed through the ``biom3`` command line interface.

Each entry maps a command path (``biom3 train stage3`` is ``("train", "stage3")``)
to a module providing ``parse_arguments(argv)`` and ``main(args)``. The tier decides
where a command is listed: ``biom3 --help`` shows the official tier and
``biom3 --help --all`` shows every tier. Commands in every tier are runnable.

``preset_args`` are placed ahead of whatever the user types, so a command can be a
fixed configuration of a more general one.
"""

from dataclasses import dataclass

OFFICIAL = "official"
ADVANCED = "advanced"
EXPERIMENTAL = "experimental"
TIERS = (OFFICIAL, ADVANCED, EXPERIMENTAL)


@dataclass(frozen=True)
class Command:
    path: tuple[str, ...]
    module: str
    summary: str
    tier: str = OFFICIAL
    preset_args: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return " ".join(self.path)


COMMANDS = (
    Command(
        ("embed",),
        "biom3.pipeline.embedding_pipeline",
        "Embed a CSV of sequences and text with PenCL and the Facilitator",
    ),
    Command(
        ("generate",),
        "biom3.Stage3.run_ProteoScribe_sample",
        "Generate protein sequences with ProteoScribe",
    ),
    Command(
        ("finetune",),
        "biom3.Stage3.run_PL_training",
        "Finetune ProteoScribe on precomputed HDF5 embeddings",
        preset_args=("--finetune", "True"),
    ),
    Command(
        ("finetune-generalized",),
        "biom3.Stage3.run_ProteoScribe_finetuning",
        "Finetune ProteoScribe on JSONL records, composing captions each epoch",
    ),
    Command(
        ("train", "stage1"),
        "biom3.Stage1.run_PL_training",
        "Train PenCL",
    ),
    Command(
        ("train", "stage2"),
        "biom3.Stage2.run_PL_training",
        "Train the Facilitator",
    ),
    Command(
        ("train", "stage3"),
        "biom3.Stage3.run_PL_training",
        "Train ProteoScribe on precomputed HDF5 embeddings",
    ),
    Command(
        ("fetch", "weights"),
        "biom3.weights.fetch",
        "Download a published weights bundle",
    ),
    Command(
        ("fetch", "dataset"),
        "biom3.datasets.fetch",
        "Download a published dataset",
    ),
    Command(
        ("stage1", "infer"),
        "biom3.Stage1.run_PenCL_inference",
        "Run PenCL inference on its own",
        tier=ADVANCED,
    ),
    Command(
        ("stage2", "sample"),
        "biom3.Stage2.run_Facilitator_sample",
        "Run the Facilitator on its own",
        tier=ADVANCED,
    ),
    Command(
        ("manifold", "fit"),
        "biom3.geometry.run_fit_manifold",
        "Fit a latent manifold to reference embeddings",
        tier=ADVANCED,
    ),
    Command(
        ("manifold", "score"),
        "biom3.geometry.run_score_manifold",
        "Score embeddings against a fitted manifold",
        tier=ADVANCED,
    ),
    Command(
        ("rl", "grpo"),
        "biom3.rl.run_grpo_train",
        "Post-train ProteoScribe with Group Relative Policy Optimization",
        tier=EXPERIMENTAL,
    ),
    Command(
        ("rl", "gdpo"),
        "biom3.rl.run_gdpo_train",
        "Post-train ProteoScribe with Group Diffusion Policy Optimization",
        tier=EXPERIMENTAL,
    ),
    Command(
        ("rl", "dpo"),
        "biom3.rl.run_dpo_train",
        "Post-train ProteoScribe with Direct Preference Optimization",
        tier=EXPERIMENTAL,
    ),
    Command(
        ("multidomain", "finetune"),
        "biom3.Stage3.multidomain.run_multidomain_finetuning",
        "Train a composed multidomain ProteoScribe decoder",
        tier=EXPERIMENTAL,
    ),
    Command(
        ("multidomain", "sample"),
        "biom3.Stage3.multidomain.run_multidomain_sample",
        "Generate multidomain proteins from a composed checkpoint",
        tier=EXPERIMENTAL,
    ),
)
