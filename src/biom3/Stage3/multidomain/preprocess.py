"""Batching for multidomain finetuning.

Each record yields K canvases and K captions, so a batch carries a domain axis:
``num_seqs [B, K, L]`` and ``input_ids [B, K, T]``. Domain order is the record's
N->C order throughout, and rows flatten row-major (example-major, domain-minor)
so ``input_ids.reshape(B * K, T)`` lines up with the raw sequence list.

Captions are tokenized with ``padding="max_length"``. That is the configuration
ProteoScribe's conditioning embeddings were trained under; dynamic padding
produces different embeddings because no attention mask reaches the text encoder
(see docs/bug_reports/bert_embedding_mismatch.md).
"""

import torch

from biom3.Stage3.preprocess import encode_protein_sequence


def encode_captions(text_tokenizer, captions, text_max_length):
    """Batch-encode captions to ``[N, text_max_length]`` input ids.

    The one ``padding="max_length"`` call site in this subpackage, shared by the
    training collate and the sampler. That is the configuration ProteoScribe's
    conditioning embeddings were trained under; dynamic padding produces
    different embeddings because no attention mask reaches the text encoder (see
    docs/bug_reports/bert_embedding_mismatch.md).
    """
    encoded = text_tokenizer(
        list(captions),
        truncation=True,
        max_length=text_max_length,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return encoded["input_ids"]


def _check_canvas_fit(sequence, max_residues, label, index, domain):
    """Reject a domain too long for its canvas, before it becomes a shape error.

    ``pad_ends`` computes a negative pad for an over-long sequence and returns it
    unchanged rather than raising, so the failure would otherwise surface as an
    opaque ``torch.stack`` size mismatch several frames away.
    """
    n_residues = len(sequence.replace("-", ""))
    if n_residues > max_residues:
        raise ValueError(
            f"record {index}, domain {domain}: {label} has {n_residues} residues "
            f"but the canvas holds {max_residues} (image_size**2 minus <START> "
            "and <END>); the data module's length filter should have dropped it"
        )


def _check_domain_count(values, num_domains, label, index):
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"record {index}: {label} must be a list of {num_domains} entries, "
            f"got {type(values).__name__}; the record_schema should map it "
            "through map_domains"
        )
    if len(values) != num_domains:
        raise ValueError(
            f"record {index}: {label} has {len(values)} entries but the run is "
            f"configured for {num_domains} domains"
        )


def make_multidomain_collate_fn(*, text_tokenizer, text_max_length, image_size,
                                num_domains, sequence_key="sequences",
                                caption_key="captions",
                                include_sequences=False):
    """Build a collate fn mapping composed multidomain records to tensors.

    The returned callable takes a batch of
    ``{sequence_key: [str] * K, caption_key: [str] * K}`` dicts, as produced by a
    ``record_schema`` whose two outputs both go through ``map_domains``.

    Returns ``(num_seqs [B, K, image_size**2] float32, input_ids [B, K, T])``.
    With ``include_sequences=True`` a third element is appended: the raw domain
    sequences flattened row-major to ``B * K`` strings, which z_p blending uses
    to key its precomputed lookup.
    """
    image_size = int(image_size)
    num_domains = int(num_domains)
    max_residues = image_size * image_size - 2

    def _collate(batch):
        flat_sequences = []
        flat_captions = []
        for index, sample in enumerate(batch):
            sequences = sample[sequence_key]
            captions = sample[caption_key]
            _check_domain_count(sequences, num_domains, repr(sequence_key), index)
            _check_domain_count(captions, num_domains, repr(caption_key), index)
            for domain, sequence in enumerate(sequences):
                _check_canvas_fit(
                    sequence, max_residues, repr(sequence_key), index, domain)
            flat_sequences.extend(sequences)
            flat_captions.extend(captions)

        num_seqs = torch.stack([
            torch.tensor(encode_protein_sequence(sequence, image_size)).float()
            for sequence in flat_sequences
        ]).reshape(len(batch), num_domains, -1)

        input_ids = encode_captions(
            text_tokenizer, flat_captions, text_max_length
        ).reshape(len(batch), num_domains, -1)

        if include_sequences:
            return num_seqs, input_ids, flat_sequences
        return num_seqs, input_ids

    return _collate
