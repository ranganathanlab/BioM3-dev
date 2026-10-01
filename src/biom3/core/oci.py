"""Fetch the files of a published OCI artifact with oras.

Shared by the weights and dataset fetchers. Stdlib only, so fetching works
before torch or any model code is importable.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

MANIFEST_NAME = "MANIFEST.json"

_ORAS_MISSING = (
    "oras not found on PATH. It ships in the BioM3 container images; on a bare "
    "host install it from https://oras.land/docs/installation"
)


def sha256_file(path, chunk_size=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def format_size(nbytes):
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if nbytes >= scale:
            return f"{nbytes / scale:.2f} {unit}"
    return f"{nbytes} B"


def _require_oras():
    if shutil.which("oras") is None:
        raise RuntimeError(_ORAS_MISSING)


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd[:3])} failed:\n{result.stderr.strip()}"
        )
    return result.stdout


def list_bundle_files(ref):
    """Return [(path, digest, size)] for every file in the bundle.

    Read from the OCI manifest, which is a few KB, so nothing large is
    transferred to find out what a bundle holds.
    """
    manifest = json.loads(_run(["oras", "manifest", "fetch", ref]))
    files = []
    for layer in manifest.get("layers", []):
        title = (layer.get("annotations") or {}).get(
            "org.opencontainers.image.title"
        )
        if title:
            files.append((title, layer["digest"], layer.get("size", 0)))
    return files


def describe_path(path):
    """Describe a path that is in the way, resolving a symlink for the message."""
    if os.path.islink(path):
        state = "symlink" if os.path.exists(path) else "broken symlink"
        return f"{path} -> {os.readlink(path)} ({state})"
    if os.path.isdir(path):
        return f"{path} (is a directory)"
    return f"{path} (not a regular file)"


def blocking_path(path, root):
    """Return the first component of *path* under *root* that is not a directory.

    A broken symlink counts: ``makedirs(..., exist_ok=True)`` raises on one,
    since the path exists without being a directory.
    """
    root, path = os.path.abspath(root), os.path.abspath(path)
    if os.path.lexists(root) and not os.path.isdir(root):
        return root
    if path != root and not path.startswith(root + os.sep):
        return None
    current = root
    rel = os.path.relpath(path, root)
    for part in [] if rel == os.curdir else rel.split(os.sep):
        current = os.path.join(current, part)
        if not os.path.lexists(current):
            return None
        if not os.path.isdir(current):
            return current
    return None


def _plan(files, output_dir, prefix, include_other):
    """Split bundle files into (to_fetch, present, conflicting, ignored, blocked).

    to_fetch and conflicting hold (rel, dest, digest, size) tuples; blocked
    holds descriptions of paths that cannot be written or created.
    """
    to_fetch, present, conflicting, ignored, blocked = [], [], [], [], []
    for path, digest, size in files:
        if path.startswith(prefix):
            rel = path[len(prefix):]
        elif include_other and path != MANIFEST_NAME:
            rel = path
        else:
            ignored.append(path)
            continue
        dest = os.path.join(output_dir, rel)
        entry = (rel, dest, digest, size)
        parent = blocking_path(os.path.dirname(dest), output_dir)
        if parent:
            blocked.append(describe_path(parent))
        elif os.path.isfile(dest):
            if sha256_file(dest) == digest:
                present.append(rel)
            else:
                conflicting.append(entry)
        elif os.path.lexists(dest):
            blocked.append(describe_path(dest))
        else:
            to_fetch.append(entry)
    return to_fetch, present, conflicting, ignored, list(dict.fromkeys(blocked))


def fetch_bundle(registry, tag, output_dir, prefix, include_other=False,
                 force=False, dry_run=False, skipped_note=None):
    """Fetch ``registry:tag`` into ``output_dir``, skipping files already there.

    Bundle files under *prefix* land in *output_dir* with the prefix
    stripped. Other files are taken at their bundle path only when
    *include_other*; MANIFEST.json never is. Each file is matched against the
    registry's own content digest, so an existing file is re-downloaded only
    when its bytes differ. *skipped_note*, if given, reports how many files
    were left out.
    """
    _require_oras()
    ref = f"{registry}:{tag}"

    files = list_bundle_files(ref)
    if not files:
        raise RuntimeError(f"{ref}: manifest lists no files")

    dest_root = os.path.abspath(output_dir)
    to_fetch, present, conflicting, ignored, blocked = _plan(
        files, dest_root, prefix, include_other
    )

    if blocked:
        raise RuntimeError(
            "these paths are in the way, so nothing was downloaded:\n  "
            + "\n  ".join(blocked)
            + "\n\nIf this tree holds symlinks into a shared location, bind or "
            "mount that location at the same path, or fetch into a different "
            "output directory. --force does not remove these paths."
        )

    if conflicting and not force:
        raise RuntimeError(
            "these files differ from the published bundle; pass --force to "
            "replace them:\n  " + "\n  ".join(rel for rel, *_ in conflicting)
        )
    to_fetch.extend(conflicting)

    total = sum(size for *_, size in to_fetch)
    print(f"{ref}", file=sys.stderr)
    print(f"  {len(present)} already present, {len(to_fetch)} to fetch "
          f"({format_size(total)})", file=sys.stderr)
    if ignored and skipped_note:
        print(f"  {len(ignored)} {skipped_note}", file=sys.stderr)
    if dry_run:
        for rel, _, _, size in to_fetch:
            print(f"  would fetch {rel} ({format_size(size)})", file=sys.stderr)
        return dest_root
    if not to_fetch:
        print("  nothing to do", file=sys.stderr)
        return dest_root

    for i, (rel, dest, digest, size) in enumerate(to_fetch, 1):
        parent = os.path.dirname(dest) or "."
        try:
            os.makedirs(parent, exist_ok=True)
        except (FileExistsError, NotADirectoryError, FileNotFoundError) as exc:
            offender = (blocking_path(parent, dest_root)
                        or exc.filename or parent)
            raise RuntimeError(
                f"cannot create the directory for {rel}:\n  "
                + describe_path(offender)
            ) from exc
        tmp = dest + ".partial"
        print(f"  [{i}/{len(to_fetch)}] {rel} ({format_size(size)})",
              file=sys.stderr)
        _run(["oras", "blob", "fetch", "--output", tmp, f"{registry}@{digest}"])
        got = sha256_file(tmp)
        if got != digest:
            os.remove(tmp)
            raise RuntimeError(f"{rel}: digest {got}, expected {digest}")
        os.replace(tmp, dest)

    print(f"  done -> {dest_root}", file=sys.stderr)
    return dest_root
