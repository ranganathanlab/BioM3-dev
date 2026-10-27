"""Frozen text -> z_c embedding front-end for Stage 3 finetuning.

ProteoScribe is conditioned on z_c. Only the *text* branch of PenCL is needed
to produce it:

    z_t  = text_projection(text_encoder(input_ids))      # Stage 1 (PenCL)
    z_c  = facilitator(z_t)                               # Stage 2 (Facilitator)

The Facilitator is optional. A ProteoScribe pretrained on raw z_t has never
seen facilitated vectors, so conditioning it on them at finetuning time would
feed it a different distribution from the one it was trained on. For those
weights, build the embedder with ``use_facilitator=False`` and the second line
above becomes ``z_c = z_t``. Which one is correct is fixed by how ProteoScribe
was pretrained; it is not a free hyperparameter.

The protein branch (ESM-2) participates only in Stage 1's contrastive loss and
is never used for conditioning, so it is not instantiated here. This keeps the
front-end light enough (BioBERT + two small heads) to run batched on-device
during finetuning.
"""

import torch
import torch.nn as nn

import biom3.Stage1.model as stage1_mod
from biom3.core.io import load_state_dict_unwrap_pl
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


TEXT_BRANCH_PREFIXES = ("text_encoder.", "text_projection.")
PROTEIN_BRANCH_PREFIXES = ("protein_encoder.", "protein_projection.")


class TextToZcEmbedder(nn.Module):
    """Maps tokenized text (BioBERT input_ids) to ProteoScribe's z_c condition.

    Holds only PenCL's text branch (``text_encoder`` + ``text_projection``) and,
    unless ``stage2_args`` is None, the Stage 2 ``facilitator``. Attribute names
    match PenCL so its checkpoint loads with ``strict=False`` (the protein-branch
    keys are ignored). With ``stage2_args=None`` the Facilitator is not built at
    all and ``forward`` returns the raw z_t.

    ``stage2_args`` has no default: which vector conditions ProteoScribe is fixed
    by how it was pretrained, and nothing here can check that, so a caller has to
    say ``None`` deliberately rather than reach the raw-z_t front-end by omission.
    The default lives on :func:`build_text_to_zc_embedder` instead, where it
    preserves the historical behaviour.
    """

    def __init__(self, stage1_args, stage2_args):
        super().__init__()
        self.text_encoder = stage1_mod.TextEncoder(args=stage1_args)
        self.text_projection = stage1_mod.ProjectionHead(
            embedding_dim=stage1_args.text_encoder_embedding,
            args=stage1_args,
        )
        if stage2_args is None:
            self.facilitator = None
        else:
            self.facilitator = stage1_mod.Facilitator(
                in_dim=stage2_args.emb_dim,
                hid_dim=stage2_args.hid_dim,
                out_dim=stage2_args.emb_dim,
                dropout=stage2_args.dropout,
            )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            hidden = self.text_encoder(input_ids)
        else:
            hidden = self.text_encoder(input_ids, attention_mask=attention_mask)
        z_t = self.text_projection(hidden)
        if self.facilitator is None:
            return z_t
        return self.facilitator(z_t)


def _load_frozen_weights(module, weights_path, required_prefixes, label, device=None):
    """Load weights into ``module``, raising if required parameters stay unloaded.

    ``strict=False`` is unavoidable here: a PenCL checkpoint carries the protein
    branch, which this module does not instantiate. But a bare ``strict=False``
    load also swallows the case where *nothing* matches (e.g. a ``model.``
    -prefixed Lightning checkpoint loaded into a bare nn.Module), leaving a
    randomly initialised encoder that trains without any error. Restricting the
    strictness to the keys under ``required_prefixes`` keeps the intended
    tolerance while making a no-op load fail loudly.
    """
    state_dict = load_state_dict_unwrap_pl(weights_path, device=device)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    param_names = {name for name, _ in module.named_parameters()}
    required_missing = [
        k for k in missing if any(k.startswith(p) for p in required_prefixes)
    ]
    unloaded_params = [k for k in required_missing if k in param_names]
    unloaded_buffers = [k for k in required_missing if k not in param_names]

    if unloaded_params:
        raise RuntimeError(
            f"{label} weights at {weights_path} did not populate "
            f"{len(unloaded_params)}/{len(param_names)} required parameters "
            f"(e.g. {unloaded_params[:5]}). The module would train against "
            f"randomly initialised weights. Checkpoint keys look like "
            f"{sorted(state_dict)[:3]}."
        )
    if unloaded_buffers:
        logger.warning(
            "%s: %d buffer(s) not present in the checkpoint: %s",
            label, len(unloaded_buffers), unloaded_buffers[:5],
        )
    logger.info(
        "%s: loaded %d tensors from %s (%d unrelated checkpoint keys ignored)",
        label, len(state_dict) - len(unexpected), weights_path, len(unexpected),
    )


def build_text_to_zc_embedder(
        stage1_args,
        stage2_args,
        pencl_weights: str,
        facilitator_weights: str,
        device=None,
        use_facilitator: bool = True,
    ) -> TextToZcEmbedder:
    """Build the frozen text->z_c embedder and load PenCL (+ Facilitator) weights.

    Both paths accept a raw state dict (``.bin``/``.pt``), a Lightning
    ``.ckpt``, a checkpoint directory, or a DeepSpeed ZeRO shard directory. The
    PenCL load ignores the protein-branch keys but requires every text-branch
    parameter to be populated; the Facilitator load requires all of its own. The
    returned module is frozen and in eval mode so its embeddings are
    deterministic and match inference.

    ``use_facilitator=False`` builds the z_c = z_t front-end for a ProteoScribe
    pretrained on raw z_t: no Facilitator is constructed and none is loaded, so
    ``stage2_args`` and ``facilitator_weights`` are not needed and are refused
    if given, rather than being silently ignored.
    """
    if not pencl_weights:
        raise ValueError(
            "--pencl_weights is required for finetuning: without it the frozen "
            "text->z_c embedder runs on a randomly initialised projection head."
        )
    if use_facilitator and not facilitator_weights:
        raise ValueError(
            "--facilitator_weights is required for finetuning: without it z_c is "
            "produced by a randomly initialised Facilitator. Pass "
            "--use_facilitator False if ProteoScribe was pretrained on raw z_t."
        )
    if not use_facilitator and facilitator_weights:
        raise ValueError(
            "--facilitator_weights was given with --use_facilitator False; the "
            "Facilitator would not be used. Drop one of the two."
        )

    embedder = TextToZcEmbedder(stage1_args, stage2_args if use_facilitator else None)

    logger.info("Loading PenCL text-branch weights from: %s", pencl_weights)
    _load_frozen_weights(
        embedder, pencl_weights,
        required_prefixes=TEXT_BRANCH_PREFIXES,
        label="PenCL text branch",
        device=device,
    )

    if use_facilitator:
        logger.info("Loading Facilitator weights from: %s", facilitator_weights)
        _load_frozen_weights(
            embedder.facilitator, facilitator_weights,
            required_prefixes=("",),
            label="Facilitator",
            device=device,
        )
    else:
        logger.info("No Facilitator: conditioning ProteoScribe on the raw z_t.")

    for p in embedder.parameters():
        p.requires_grad = False
    embedder.eval()
    if device is not None:
        embedder.to(device)
    return embedder

class ProteinToZpEmbedder(nn.Module):
    """Maps a protein sequence to PenCL's z_p, the joint-space protein embedding.

    The mirror of :class:`TextToZcEmbedder`: holds only PenCL's protein branch
    (``protein_encoder`` = ESM-2, plus ``protein_projection``). Attribute names
    match PenCL so its checkpoint loads directly; the text-branch keys are
    ignored. No Facilitator is involved — z_p is already in the joint space that
    the Facilitator maps text *into*.
    """

    def __init__(self, stage1_args):
        super().__init__()
        self.protein_encoder = stage1_mod.ProteinEncoder(args=stage1_args)
        self.protein_projection = stage1_mod.ProjectionHead(
            embedding_dim=stage1_args.protein_encoder_embedding,
            args=stage1_args,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.protein_projection(self.protein_encoder(tokens))

    @torch.no_grad()
    def embed_sequences(self, sequences, device=None, batch_size=64):
        """Embed raw amino-acid strings to z_p, in input order.

        Gaps are stripped here so the returned rows align with ``sequences`` as
        given — callers can zip the original (possibly gapped) strings against
        the result to build a lookup.
        """
        device = device or next(self.parameters()).device
        convert = self.protein_encoder.alphabet.get_batch_converter()
        chunks = []
        for i in range(0, len(sequences), batch_size):
            window = sequences[i:i + batch_size]
            batch = [(str(j), s.replace("-", "")) for j, s in enumerate(window)]
            _, _, tokens = convert(batch)
            chunks.append(self(tokens.to(device)).float().cpu())
        return torch.cat(chunks, dim=0)


def build_protein_to_zp_embedder(
        stage1_args,
        pencl_weights: str,
        device=None,
    ) -> ProteinToZpEmbedder:
    """Build the frozen protein->z_p embedder mirroring the text->z_c one.

    Ignores PenCL's text-branch keys but requires every protein-branch
    parameter to be populated, so a checkpoint that matches nothing fails
    loudly instead of yielding a random projection of stock ESM-2.
    """
    if not pencl_weights:
        raise ValueError(
            "--pencl_weights is required when train_alpha puts any weight on "
            "z_p: without it the protein branch is a randomly initialised "
            "projection of stock ESM-2, which does not share a space with z_c."
        )

    embedder = ProteinToZpEmbedder(stage1_args)

    logger.info("Loading PenCL protein-branch weights from: %s", pencl_weights)
    _load_frozen_weights(
        embedder, pencl_weights,
        required_prefixes=PROTEIN_BRANCH_PREFIXES,
        label="PenCL protein branch",
        device=device,
    )

    for p in embedder.parameters():
        p.requires_grad = False
    embedder.eval()
    if device is not None:
        embedder.to(device)
    return embedder