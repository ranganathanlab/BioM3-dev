"""--text_padding dynamic: crop each training batch to its longest caption.

Items are tokenized to text_max_length and collate_dynamic_text drops the
trailing columns. It must drop only [PAD] columns, keep Swiss-Prot and Pfam
captions at one common length (the MLM forward concatenates them), round to a
multiple of 64, and leave z_t unchanged to float rounding.
"""
import os

import pytest
import torch
from transformers import AutoTokenizer

from biom3.Stage1.model import ProjectionHead, TextEncoder
from biom3.Stage1.preprocess import collate_dynamic_text

TEXT_MODEL = "weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
pytestmark = pytest.mark.skipif(not os.path.exists(TEXT_MODEL),
                                reason=f"Weight files not found: {TEXT_MODEL}")
SWISS = ["PROTEIN NAME: Serine protease.",
         "PROTEIN NAME: Kinase. FUNCTION: Phosphorylates serine residues in response to stress."]
PFAM = ["FAMILY: SH3 domain.",
        "FAMILY: Trypsin. FUNCTION: Hydrolyses peptide bonds. SIMILARITY: peptidase S1."]


def _tok():
    return AutoTokenizer.from_pretrained(TEXT_MODEL)


def _items(tok, swiss, pfam):
    """Items shaped like Pfam_TextSeqPairing_Dataset.__getitem__ output."""
    items = []
    for s, p in zip(swiss, pfam):
        t = tok([s], truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        q = tok([p], truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        tm, qm = t["input_ids"].clone(), q["input_ids"].clone()
        tm[0, 2] = tok.mask_token_id                 # MLM masks real tokens only
        qm[0, 3] = tok.mask_token_id
        prot = torch.ones(1, 1024, dtype=torch.long)
        items.append((t["input_ids"], prot, tm, prot, q["input_ids"], prot, qm, prot,
                      ["True"], t["attention_mask"], q["attention_mask"]))
    return items


def test_crop_drops_only_padding_and_shares_one_length():
    tok = _tok()
    items = _items(tok, SWISS, PFAM)
    full = torch.utils.data.default_collate(items)
    out = collate_dynamic_text(items, multiple=1)
    longest = max(int(full[9].sum(-1).max()), int(full[10].sum(-1).max()))
    for i in (0, 2, 4, 6, 9, 10):
        assert out[i].shape[-1] == longest, (i, out[i].shape)
        assert torch.equal(out[i], full[i][..., :longest])
    for i in (9, 10):
        assert int(full[i][..., longest:].sum()) == 0          # only padding dropped
    for i in (0, 4):
        assert (full[i][..., longest:] == tok.pad_token_id).all()
    for i in (1, 3, 5, 7):
        assert torch.equal(out[i], full[i])                     # proteins untouched


def test_length_rounds_up_to_multiple_and_caps_at_512():
    tok = _tok()
    out = collate_dynamic_text(_items(tok, SWISS, PFAM))        # default multiple 64
    assert out[0].shape[-1] == 64
    long = ["word " * 700]                                      # truncated to 512
    out = collate_dynamic_text(_items(tok, long, PFAM[:1]))
    assert out[0].shape[-1] == 512


def test_z_t_unchanged_by_cropping():
    tok = _tok()

    class _Args:
        text_model_path = TEXT_MODEL
        pretrained_text = True
        trainable_text = False
        bLM_n_layers_to_finetune = 0
        proj_embedding_dim = 512
        dropout = 0.1
        text_max_length = 512

    enc, proj = TextEncoder(args=_Args()).eval(), ProjectionHead(embedding_dim=768, args=_Args()).eval()
    items = _items(tok, SWISS, PFAM)
    full = torch.utils.data.default_collate(items)
    crop = collate_dynamic_text(items)
    with torch.no_grad():
        z_full = proj(enc(full[0], attention_mask=full[9]))
        z_crop = proj(enc(crop[0], attention_mask=crop[9]))
    assert (z_full - z_crop).abs().max().item() < 1e-5
