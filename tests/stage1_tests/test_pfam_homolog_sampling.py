"""The Pfam homolog paired with each Swiss-Prot item must be drawn at random.

Pfam_TextSeqPairing_Dataset.extraction_pfam_samples picks one of the protein's
Pfam families, then one member row of that family. The member draw used to be
temp_df.sample(n=1, random_state=seed) -- a generator re-seeded identically on
every call -- so a given family returned the same row for the whole run. On
rank 0 of a 24-rank mid run that meant 4,294 of the shard's 143,077 Pfam rows
(3%) were ever used, one fixed representative per family.
"""
import random

import pandas as pd

from biom3.Stage1.preprocess import Pfam_TextSeqPairing_Dataset


class _Args:
    # present so that re-introducing random_state=self.script_args.seed fails
    # on the assertions below rather than on a missing attribute
    seed = 42


def _dataset(n_members=50):
    """Bare instance holding only what extraction_pfam_samples reads. Skips
    __init__, which loads tokenizers and the ESM alphabet."""
    ds = object.__new__(Pfam_TextSeqPairing_Dataset)
    df = pd.DataFrame({
        "id": [f"F1_{i}" for i in range(n_members)] + [f"F2_{i}" for i in range(n_members)],
        "sequence": ["MKT"] * (2 * n_members),
        "[final]text_caption": ["caption"] * (2 * n_members),
        "pfam_label": ["PF00001"] * n_members + ["PF00002"] * n_members,
    })
    ds.grouped_pfam_df = df.groupby("pfam_label")
    ds.script_args = _Args()
    return ds


def test_draws_vary_within_a_family():
    ds = _dataset()
    random.seed(0)
    ids = [ds.extraction_pfam_samples(["PF00001"])[0] for _ in range(200)]
    # 200 uniform draws from 50 members hit ~49 distinct on average; the old
    # fixed-seed draw hit exactly 1.
    assert len(set(ids)) > 30, f"only {len(set(ids))} distinct homolog(s) in 200 draws"


def test_draws_stay_in_the_requested_family():
    ds = _dataset()
    random.seed(1)
    for _ in range(100):
        accession, _, _, flag, label = ds.extraction_pfam_samples(["PF00002"])
        assert accession.startswith("F2_"), accession
        assert flag == ["True"]
        assert label == "PF00002"


def test_draws_are_reproducible_from_the_run_seed():
    """Random per draw, but the whole sequence is fixed by the run seed, which
    run_PL_training sets once via random.seed(args.seed)."""
    ds = _dataset()
    random.seed(42)
    a = [ds.extraction_pfam_samples(["PF00001", "PF00002"])[0] for _ in range(50)]
    random.seed(42)
    b = [ds.extraction_pfam_samples(["PF00001", "PF00002"])[0] for _ in range(50)]
    assert a == b
    assert len(set(a)) > 1
