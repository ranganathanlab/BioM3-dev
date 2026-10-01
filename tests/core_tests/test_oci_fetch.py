"""Tests for biom3.core.oci and the weights / dataset fetch CLIs built on it."""

import hashlib
import json
import shutil

import pytest

from biom3.core import oci
from biom3.datasets import fetch as dataset_fetch
from biom3.weights import fetch as weights_fetch

REGISTRY = "registry.example/biom3-things"


def _digest(content):
    return "sha256:" + hashlib.sha256(content).hexdigest()


class FakeOras:
    """Stands in for the oras CLI, serving an in-memory bundle."""

    def __init__(self, files):
        self.files = {path: content for path, content in files.items()}
        self.blobs = {_digest(c): c for c in self.files.values()}
        self.fetched = []
        self.corrupt = False

    def __call__(self, cmd):
        if cmd[1:3] == ["manifest", "fetch"]:
            layers = [
                {
                    "digest": _digest(content),
                    "size": len(content),
                    "annotations": {"org.opencontainers.image.title": path},
                }
                for path, content in self.files.items()
            ]
            return json.dumps({"layers": layers})
        if cmd[1:3] == ["blob", "fetch"]:
            out = cmd[cmd.index("--output") + 1]
            digest = cmd[-1].split("@", 1)[1]
            self.fetched.append(digest)
            content = b"corrupted" if self.corrupt else self.blobs[digest]
            with open(out, "wb") as fh:
                fh.write(content)
            return ""
        raise AssertionError(f"unexpected oras call: {cmd}")


@pytest.fixture
def fake_oras(monkeypatch):
    bundle = FakeOras({
        "data/a.csv": b"id,seq\n1,MKV\n",
        "data/sub/b.txt": b"notice\n",
        "other.json": b"{}\n",
        "MANIFEST.json": b'{"files": []}\n',
    })
    monkeypatch.setattr(oci.shutil, "which", lambda name: "/usr/bin/oras")
    monkeypatch.setattr(oci, "_run", bundle)
    return bundle


def test_fetch_strips_prefix_and_leaves_other_files(fake_oras, tmp_path):
    oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/")
    assert (tmp_path / "a.csv").read_bytes() == b"id,seq\n1,MKV\n"
    assert (tmp_path / "sub" / "b.txt").read_bytes() == b"notice\n"
    assert not (tmp_path / "other.json").exists()
    assert not (tmp_path / "MANIFEST.json").exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_present_files_are_not_refetched(fake_oras, tmp_path):
    (tmp_path / "a.csv").write_bytes(b"id,seq\n1,MKV\n")
    oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/")
    assert fake_oras.fetched == [_digest(b"notice\n")]


def test_conflicting_file_needs_force(fake_oras, tmp_path):
    (tmp_path / "a.csv").write_bytes(b"edited locally\n")
    with pytest.raises(RuntimeError, match="a.csv"):
        oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/")
    assert (tmp_path / "a.csv").read_bytes() == b"edited locally\n"

    oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/", force=True)
    assert (tmp_path / "a.csv").read_bytes() == b"id,seq\n1,MKV\n"


def test_digest_mismatch_raises_and_leaves_nothing(fake_oras, tmp_path):
    fake_oras.corrupt = True
    with pytest.raises(RuntimeError, match="expected sha256:"):
        oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/")
    assert not (tmp_path / "a.csv").exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_broken_symlink_in_the_way_is_reported_before_fetching(fake_oras, tmp_path):
    """A weights/ tree of symlinks into an unmounted share dangles, and
    makedirs raises on a broken link even with exist_ok=True."""
    (tmp_path / "sub").symlink_to("/nonexistent/share/sub")
    with pytest.raises(RuntimeError) as exc:
        oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/")
    assert "broken symlink" in str(exc.value)
    assert "/nonexistent/share/sub" in str(exc.value)
    assert fake_oras.fetched == []
    assert not (tmp_path / "a.csv").exists()


def test_blocked_path_is_not_overridden_by_force(fake_oras, tmp_path):
    (tmp_path / "sub").symlink_to("/nonexistent/share/sub")
    with pytest.raises(RuntimeError, match="in the way"):
        oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/", force=True)
    assert fake_oras.fetched == []


def test_directory_where_a_file_belongs_is_reported(fake_oras, tmp_path):
    (tmp_path / "a.csv").mkdir()
    with pytest.raises(RuntimeError, match="is a directory"):
        oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/")
    assert fake_oras.fetched == []


def test_resolvable_symlinked_dir_is_accepted(fake_oras, tmp_path):
    share = tmp_path / "share"
    share.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "sub").symlink_to(share)
    oci.fetch_bundle(REGISTRY, "tag", out, "data/")
    assert (share / "b.txt").read_bytes() == b"notice\n"


def test_dry_run_fetches_nothing(fake_oras, tmp_path):
    oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/", dry_run=True)
    assert fake_oras.fetched == []
    assert not any(tmp_path.iterdir())


def test_include_other_takes_everything_but_the_manifest(fake_oras, tmp_path):
    oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/", include_other=True)
    assert (tmp_path / "a.csv").exists()
    assert (tmp_path / "other.json").exists()
    assert not (tmp_path / "MANIFEST.json").exists()


def test_missing_oras_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(oci.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="oras not found"):
        oci.fetch_bundle(REGISTRY, "tag", tmp_path, "data/")


def test_weights_fetch_takes_only_weights(monkeypatch, tmp_path):
    bundle = FakeOras({
        "weights/PenCL/x.bin": b"weights\n",
        "_base_PenCL.json": b"{}\n",
        "MANIFEST.json": b"{}\n",
    })
    monkeypatch.setattr(oci.shutil, "which", lambda name: "/usr/bin/oras")
    monkeypatch.setattr(oci, "_run", bundle)
    weights_fetch.fetch_bundle("run1_base", tmp_path, registry=REGISTRY)
    assert (tmp_path / "PenCL" / "x.bin").exists()
    assert not (tmp_path / "_base_PenCL.json").exists()


def test_weights_cli_test_weights_switches_registry():
    args = weights_fetch.parse_arguments(["--test_weights", "-o", "w"])
    assert args.registry == weights_fetch.TEST_WEIGHTS_REGISTRY
    assert args.bundle == weights_fetch.TEST_WEIGHTS_TAG


def test_dataset_cli_defaults():
    args = dataset_fetch.parse_arguments(["gfp_demo", "-o", "data"])
    assert args.dataset == "gfp_demo"
    assert args.registry == dataset_fetch.DEFAULT_REGISTRY
    assert not args.force and not args.dry_run


def test_dataset_main_reports_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(oci.shutil, "which", lambda name: None)
    args = dataset_fetch.parse_arguments(["gfp_demo", "-o", str(tmp_path)])
    assert dataset_fetch.main(args) == 1
    assert "oras not found" in capsys.readouterr().err


@pytest.mark.network
def test_fetch_published_gfp_demo(tmp_path):
    if shutil.which("oras") is None:
        pytest.skip("oras not installed")
    dataset_fetch.fetch_dataset("gfp_demo", tmp_path)
    csv_path = tmp_path / "gfp_sample_dataset.csv"
    with open(csv_path) as fh:
        header = fh.readline().strip()
    assert header == "primary_Accession,protein_sequence,[final]text_caption"
    assert (tmp_path / "gfp_sample_dataset.NOTICE.md").exists()
