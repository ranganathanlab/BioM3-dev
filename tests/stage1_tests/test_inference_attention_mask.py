"""--text_attention_mask: pass the caption attention mask at inference.

Training has passed the mask to BERT since the pfam wrappers started threading
it; inference did not, which suits weights trained without it (run1_base) and
is wrong for weights trained with it. The option is off by default.

Checked here: with the option off the collate output is what it always was;
with it on the mask is returned and, fed to the text encoder, makes z_t
independent of padding mode and of batch composition; and both entry points
accept the flag and the pipeline hands it to Stage 1.
"""
import os
from types import SimpleNamespace

import esm
import pytest
import torch
from transformers import AutoTokenizer

import biom3.Stage1.run_PenCL_inference as stage1_mod
from biom3.pipeline.embedding_pipeline import main as pipeline_main
from biom3.pipeline.embedding_pipeline import parse_arguments as parse_pipeline_args
from biom3.Stage1.model import ProjectionHead, TextEncoder
from biom3.Stage1.preprocess import collate_fn
from biom3.Stage1.run_PenCL_inference import parse_arguments as parse_stage1_args

TEXT_MODEL = "weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
needs_text_model = pytest.mark.skipif(
    not os.path.exists(TEXT_MODEL), reason=f"Weight files not found: {TEXT_MODEL}")

# Three captions of deliberately different tokenized lengths.
C1 = "PROTEIN NAME: Serine protease."
C2 = "PROTEIN NAME: Serine protease. FUNCTION: Hydrolyses peptide bonds in the gut."
C3 = ("PROTEIN NAME: Serine protease. FUNCTION: Hydrolyses peptide bonds in the gut. "
      "SUBCELLULAR LOCATION: Secreted into the extracellular space. SIMILARITY: "
      "Belongs to the peptidase S1 family of trypsin-like enzymes.")
SEQ = "MKTAYIAKQR"


class _Args:
    text_model_path = TEXT_MODEL
    pretrained_text = True
    trainable_text = False
    bLM_n_layers_to_finetune = 0
    proj_embedding_dim = 512
    dropout = 0.1
    text_max_length = 512


def _dataset(text_padding):
    """What collate_fn reads from BatchedTextSeqPairingDataset, minus ESM-2."""
    return SimpleNamespace(
        text_tokenizer=AutoTokenizer.from_pretrained(TEXT_MODEL),
        text_max_length=512,
        text_padding=text_padding,
        batch_converter=esm.data.Alphabet.from_architecture("ESM-1b").get_batch_converter(),
        seq_max_length=1024,
    )


def _batch(captions):
    return [(c, SEQ, f"acc{i}") for i, c in enumerate(captions)]


@needs_text_model
@pytest.mark.parametrize("include_raw, length", [(True, 5), (False, 2)])
def test_collate_without_the_mask_is_unchanged(include_raw, length):
    out = collate_fn(_batch([C1, C2]), _dataset("max_padding"), include_raw=include_raw)

    assert len(out) == length
    assert out[0].shape == (2, 512)


@needs_text_model
@pytest.mark.parametrize("include_raw, length", [(True, 6), (False, 3)])
def test_collate_appends_the_mask(include_raw, length):
    dataset = _dataset("dynamic")
    plain = collate_fn(_batch([C1, C2]), dataset, include_raw=include_raw)
    out = collate_fn(_batch([C1, C2]), dataset, include_raw=include_raw, include_mask=True)

    assert len(out) == length
    assert all(torch.equal(a, b) if torch.is_tensor(a) else a == b
               for a, b in zip(plain, out[:-1]))
    input_ids, mask = out[0], out[-1]
    assert mask.shape == input_ids.shape
    assert torch.equal(mask.bool(), input_ids != dataset.text_tokenizer.pad_token_id)
    assert mask[0].sum() < mask[1].sum() == mask.shape[1]


def _z_t(encoder, projection, captions, text_padding, use_mask):
    """z_t of the first caption, through the inference collate and text path."""
    out = collate_fn(_batch(captions), _dataset(text_padding), include_mask=use_mask)
    with torch.no_grad():
        hidden = encoder(out[0], compute_logits=False,
                         attention_mask=out[-1] if use_mask else None)
        return projection(hidden)[0]


@pytest.fixture(scope="module")
def text_path():
    args = _Args()
    torch.manual_seed(0)
    return (TextEncoder(args=args).eval(),
            ProjectionHead(embedding_dim=768, args=args).eval())


@needs_text_model
def test_zt_with_the_mask_ignores_padding_and_batch(text_path):
    encoder, projection = text_path
    reference = _z_t(encoder, projection, [C1, C2], "max_padding", use_mask=True)

    for text_padding in ("max_padding", "dynamic"):
        for captions in ([C1, C2], [C1, C3], [C1, C2, C3]):
            z = _z_t(encoder, projection, captions, text_padding, use_mask=True)
            diff = (z - reference).abs().max().item()
            assert diff < 1e-5, f"{text_padding}, {len(captions)} captions: {diff}"


@needs_text_model
def test_zt_without_the_mask_depends_on_padding(text_path):
    """The control: this is the dependence the option removes."""
    encoder, projection = text_path
    with_c2 = _z_t(encoder, projection, [C1, C2], "dynamic", use_mask=False)
    with_c3 = _z_t(encoder, projection, [C1, C3], "dynamic", use_mask=False)

    assert (with_c2 - with_c3).abs().max().item() > 1e-2


@pytest.mark.parametrize("extra, expected", [([], False), (["--text_attention_mask"], True)])
def test_stage1_flag(extra, expected):
    args = parse_stage1_args([
        "-i", "None",
        "-c", "configs/inference/stage1_PenCL.json",
        "-m", "weights/PenCL/x.bin",
        "-o", "out.pt",
    ] + extra)

    assert args.text_attention_mask is expected


@pytest.mark.parametrize("extra, expected", [([], False), (["--text_attention_mask"], True)])
def test_pipeline_hands_the_flag_to_stage1(tmp_path, monkeypatch, extra, expected):
    seen = []

    def fake_stage1(args, _setup_logging=True):
        seen.append(args.text_attention_mask)
        torch.save({"z_t": torch.zeros(1, 2), "sequence": ["MK"], "acc_id": ["P1"]},
                   args.output_path)

    monkeypatch.setattr(stage1_mod, "main", fake_stage1)

    pipeline_main(parse_pipeline_args([
        "-i", "tests/_data/stage1_inputs/sample_text_seqs1.csv",
        "-o", str(tmp_path / "out"),
        "--prefix", "run",
        "--pencl_weights", "weights/PenCL/x.bin",
        "--pencl_config", "configs/inference/stage1_PenCL.json",
        "--device", "cpu",
        "--skip_facilitator",
    ] + extra))

    assert seen == [expected]
