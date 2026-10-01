"""Data-path tests: map_domains, the collate, and MultiDomainDataModule.

The padding assertion here is load-bearing. The multidomain collate carries its
own tokenizer call rather than sharing the single-domain one, so
``padding="max_length"`` has to be pinned independently — dynamic padding would
silently produce different conditioning embeddings.
"""

import json
import random

import pytest
import torch

from biom3.core.dataloaders import GeneralizedRecordDataset, read_jsonl_records
from biom3.Stage3.multidomain.compose import map_domains
from biom3.Stage3.multidomain.data import MultiDomainDataModule
from biom3.Stage3.multidomain.preprocess import make_multidomain_collate_fn


FIXTURE = "tests/_data/multidomain_smoke.jsonl"
# 16**2 = 256, comfortably above the fixture's longest domain (131 aa).
IMAGE_SIZE = 16
SEQ_LEN = IMAGE_SIZE * IMAGE_SIZE
TEXT_MAX_LENGTH = 16
NUM_DOMAINS = 2

SCHEMA = {
    "sequences": {"compose": "map_domains",
                  "args": {"expect_k": NUM_DOMAINS, "output": {"from": "sequence"}}},
    "captions": {"compose": "map_domains",
                 "args": {"expect_k": NUM_DOMAINS,
                          "output": {"compose": "fields_to_caption",
                                     "args": {"fields_key": "fields",
                                              "default_dropout": 0.0}}}},
}


class _FakeTokenizer:
    """Records the kwargs it was called with, so padding policy is assertable."""

    def __init__(self):
        self.calls = []

    def __call__(self, captions, **kwargs):
        self.calls.append((list(captions), kwargs))
        n = len(captions)
        length = kwargs["max_length"]
        return {"input_ids": torch.arange(n * length).reshape(n, length)}


@pytest.fixture
def records():
    return read_jsonl_records(FIXTURE)


# ── map_domains ───────────────────────────────────────────────────────────


def test_map_domains_preserves_order(records):
    out = map_domains(records[0], {"output": {"from": "pfam_id"}})
    assert out == ["PF00501", "PF13193"]


def test_map_domains_applies_inner_compose(records):
    out = map_domains(records[0], {
        "output": {"compose": "fields_to_caption",
                   "args": {"fields_key": "fields", "default_dropout": 0.0}},
    })
    assert len(out) == NUM_DOMAINS
    assert "AMP-binding enzyme" in out[0]
    assert "C-terminal" in out[1]
    assert out[0] != out[1]


def test_map_domains_enforces_expect_k(records):
    with pytest.raises(ValueError, match="expect_k=3"):
        map_domains(records[0], {"expect_k": 3, "output": {"from": "sequence"}})


def test_map_domains_requires_an_output_spec(records):
    with pytest.raises(ValueError, match="requires an 'output' spec"):
        map_domains(records[0], {})


def test_map_domains_reports_a_missing_domains_key(records):
    with pytest.raises(KeyError, match="has no 'domains' key"):
        map_domains({"sequence": "ACDE"}, {"output": {"from": "sequence"}})


def test_map_domains_draws_dropout_per_domain():
    """Each domain re-rolls independently off the shared rng."""
    record = {"domains": [
        {"fields": {"a": "alpha", "b": "beta", "c": "gamma"}} for _ in range(2)
    ]}
    args = {"output": {"compose": "fields_to_caption",
                       "args": {"fields_key": "fields", "default_dropout": 0.5}}}
    seen = set()
    for seed in range(30):
        out = map_domains(record, args, rng=random.Random(seed))
        seen.add(tuple(out))
    # Identical inputs, independent draws -> the two domains disagree sometimes.
    assert any(a != b for a, b in seen)


def test_map_domains_is_registered():
    from biom3.core.dataloaders import get_compose_function
    assert get_compose_function("map_domains") is map_domains


def test_map_domains_registers_on_package_import():
    """A config naming "map_domains" must resolve without importing compose.

    Registration is an import side effect, so this has to run in a clean
    interpreter — any other test importing ``multidomain.compose`` first would
    populate the registry and mask the failure.
    """
    import subprocess
    import sys

    script = (
        "from biom3.Stage3.multidomain import MultiDomainDataModule\n"
        "from biom3.core.dataloaders import get_compose_function\n"
        "get_compose_function('map_domains')\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"map_domains is not registered by the package import:\n{result.stderr[-800:]}"
    )


# ── collate ───────────────────────────────────────────────────────────────


@pytest.fixture
def dataset(records):
    return GeneralizedRecordDataset(records, schema=SCHEMA, rng=None)


def test_collate_uses_max_length_padding(dataset):
    """Guards the duplicated tokenizer call against dynamic padding."""
    tokenizer = _FakeTokenizer()
    collate = make_multidomain_collate_fn(
        text_tokenizer=tokenizer, text_max_length=TEXT_MAX_LENGTH,
        image_size=IMAGE_SIZE, num_domains=NUM_DOMAINS)
    collate([dataset[0], dataset[1]])

    _, kwargs = tokenizer.calls[0]
    assert kwargs["padding"] == "max_length"
    assert kwargs["max_length"] == TEXT_MAX_LENGTH
    assert kwargs["truncation"] is True


def test_collate_shapes(dataset):
    collate = make_multidomain_collate_fn(
        text_tokenizer=_FakeTokenizer(), text_max_length=TEXT_MAX_LENGTH,
        image_size=IMAGE_SIZE, num_domains=NUM_DOMAINS)
    num_seqs, input_ids = collate([dataset[i] for i in range(3)])
    assert num_seqs.shape == (3, NUM_DOMAINS, SEQ_LEN)
    assert num_seqs.dtype == torch.float32
    assert input_ids.shape == (3, NUM_DOMAINS, TEXT_MAX_LENGTH)


def test_collate_sequence_list_is_row_major(dataset):
    """batch[2] must align with input_ids.reshape(B*K, T), example-major."""
    tokenizer = _FakeTokenizer()
    collate = make_multidomain_collate_fn(
        text_tokenizer=tokenizer, text_max_length=TEXT_MAX_LENGTH,
        image_size=IMAGE_SIZE, num_domains=NUM_DOMAINS, include_sequences=True)
    samples = [dataset[0], dataset[1]]
    _, _, sequences = collate(samples)

    assert len(sequences) == 2 * NUM_DOMAINS
    expected = [s for sample in samples for s in sample["sequences"]]
    assert sequences == expected
    # And the captions were flattened the same way.
    captions, _ = tokenizer.calls[0]
    assert captions == [c for sample in samples for c in sample["captions"]]


def test_collate_rejects_a_wrong_domain_count(dataset):
    collate = make_multidomain_collate_fn(
        text_tokenizer=_FakeTokenizer(), text_max_length=TEXT_MAX_LENGTH,
        image_size=IMAGE_SIZE, num_domains=3)
    with pytest.raises(ValueError, match="configured for 3 domains"):
        collate([dataset[0]])


def test_collate_rejects_a_non_list_output(dataset):
    collate = make_multidomain_collate_fn(
        text_tokenizer=_FakeTokenizer(), text_max_length=TEXT_MAX_LENGTH,
        image_size=IMAGE_SIZE, num_domains=NUM_DOMAINS)
    bad = dict(dataset[0])
    bad["sequences"] = "ACDEF"
    with pytest.raises(TypeError, match="must be a list"):
        collate([bad])


# ── data module ───────────────────────────────────────────────────────────


def _manifest(tmp_path, records, *, train, val, test, fingerprint=None,
              name="split.json"):
    from biom3.split import manifest as split_manifest

    if fingerprint is None:
        fingerprint = split_manifest.compute_fingerprint(
            [r["sequence"] for r in records])
    path = tmp_path / name
    payload = {
        "schema_version": split_manifest.SCHEMA_VERSION,
        "files": [{"path": FIXTURE, "group": "primary", "n_rows": len(records),
                   "fingerprint": fingerprint,
                   "train": train, "val": val, "test": test}],
    }
    path.write_text(json.dumps(payload))
    return str(path)


def _module(tmp_path, records, **overrides):
    kwargs = dict(
        jsonl_path=FIXTURE,
        record_schema=SCHEMA,
        text_model_path=None,
        text_max_length=TEXT_MAX_LENGTH,
        batch_size=2,
        num_workers=0,
        seed=0,
        diffusion_steps=SEQ_LEN,
        image_size=IMAGE_SIZE,
        num_domains=NUM_DOMAINS,
    )
    kwargs.update(overrides)
    # Built lazily: an eagerly-constructed default would write to the same
    # tmp_path and clobber a manifest the caller passed in.
    if "split_manifest_path" not in overrides:
        kwargs["split_manifest_path"] = _manifest(
            tmp_path, records,
            train=list(range(0, 8)), val=[8, 9], test=[10, 11])
    return MultiDomainDataModule(**kwargs)


def _setup(module, tokenizer=None):
    """Run setup with the HF tokenizer load stubbed out."""
    import biom3.Stage3.multidomain.data as data_mod
    real = data_mod.AutoTokenizer
    data_mod.AutoTokenizer = type(
        "_Stub", (), {"from_pretrained": staticmethod(
            lambda *a, **k: tokenizer or _FakeTokenizer())})
    try:
        module.setup()
    finally:
        data_mod.AutoTokenizer = real


def test_requires_a_split_manifest(tmp_path, records):
    with pytest.raises(ValueError, match="requires split_manifest_path"):
        _module(tmp_path, records, split_manifest_path=None)


def test_manifest_indices_are_used(tmp_path, records):
    module = _module(tmp_path, records)
    _setup(module)
    assert len(module.train_dataset) == 8
    assert len(module.val_dataset) == 2
    assert module.split_info[0]["test_indices"] == [10, 11]


def test_manifest_fingerprint_is_validated(tmp_path, records):
    """A manifest built for a different JSONL must be refused."""
    module = _module(tmp_path, records, split_manifest_path=_manifest(
        tmp_path, records, train=[0], val=[1], test=[2],
        fingerprint="n12-deadbeefdeadbeef"))
    with pytest.raises(ValueError):
        _setup(module)


def test_fingerprint_keys_on_the_full_length_sequence(tmp_path, records):
    """It must hash what the splitter clustered on, not the domain canvases."""
    from biom3.split import manifest as split_manifest

    full = split_manifest.compute_fingerprint([r["sequence"] for r in records])
    domain0 = split_manifest.compute_fingerprint(
        [r["domains"][0]["sequence"] for r in records])
    assert full != domain0

    module = _module(tmp_path, records, split_manifest_path=_manifest(
        tmp_path, records, train=[0], val=[1], test=[], fingerprint=full))
    _setup(module)  # would raise if it hashed anything else


def test_length_filter_uses_the_longest_domain(tmp_path, records):
    """Each domain has its own canvas, so the max domain length is the bound."""
    lengths = [max(d["sequence_length"] for d in r["domains"]) for r in records]
    bound = 123
    kept = [i for i in range(8) if lengths[i] <= bound]
    assert 0 < len(kept) < 8, "fixture must straddle the bound for this to bite"

    module = _module(tmp_path, records, diffusion_steps=bound + 2)
    _setup(module)
    assert len(module.train_dataset) == len(kept)


def test_full_length_protein_does_not_drive_the_filter(tmp_path, records):
    """A protein longer than the bound is fine as long as each domain fits."""
    bound = 198
    assert any(r["sequence_length"] > bound for r in records[:8])
    assert all(max(d["sequence_length"] for d in r["domains"]) <= bound
               for r in records[:8])

    module = _module(tmp_path, records, diffusion_steps=bound + 2)
    _setup(module)
    assert len(module.train_dataset) == 8


def test_unique_sequences_flattens_across_domains(tmp_path, records):
    module = _module(tmp_path, records, needs_unique_sequences=True)
    _setup(module)
    unique = module.unique_sequences()
    expected = {d["sequence"] for i in list(range(8)) + [8, 9]
                for d in records[i]["domains"]}
    assert set(unique) == expected
    assert len(unique) == len(set(unique))


def test_unique_sequences_requires_the_flag(tmp_path, records):
    module = _module(tmp_path, records)
    _setup(module)
    with pytest.raises(RuntimeError, match="needs_unique_sequences=True"):
        module.unique_sequences()


def test_rejects_a_record_with_the_wrong_domain_count(tmp_path, records):
    module = _module(tmp_path, records, num_domains=3)
    with pytest.raises(ValueError, match="configured for 3"):
        _setup(module)


def test_dataloader_yields_the_batch_contract(tmp_path, records):
    module = _module(tmp_path, records)
    _setup(module)
    num_seqs, input_ids = next(iter(module.train_dataloader()))
    assert num_seqs.shape == (2, NUM_DOMAINS, SEQ_LEN)
    assert input_ids.shape == (2, NUM_DOMAINS, TEXT_MAX_LENGTH)
