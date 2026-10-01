#!/usr/bin/env python
"""Assemble a weights or dataset bundle from a declarative spec.

A bundle is a directory of weight, config or data files plus a checksummed
MANIFEST.json, ready to `oras push`. What goes in is defined entirely by a small
JSON spec — nothing about the file set is hard-coded here, and this tool has no
git awareness.

    python build_bundle.py <spec.json> -o <output_dir> [--force]

Spec format:

    {
      "name": "biom3-weights-run1_base",
      "weights": { "<dest under weights/>": "<source path>", ... },
      "configs": { "<dest under configs/>": "<source path>", ... },
      "data":    { "<dest under data/>": "<source path>", ... }
    }

Every section is optional; a dataset bundle has only "data".

Source paths are used as-is if absolute, else resolved against the current
working directory (run from the repo root). A weight source that is a `.ckpt`
file or a directory is flattened to a plain state_dict (Lightning / DeepSpeed
unwrapping); every other file is copied verbatim.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def flatten_checkpoint(src, dst):
    """Unwrap a Lightning/DeepSpeed checkpoint to a flat state_dict."""
    import torch
    from biom3.core.io import load_state_dict_unwrap_pl

    state_dict = load_state_dict_unwrap_pl(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(state_dict, dst)


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def add_entry(top, src, dst):
    if top == "weights" and (os.path.isdir(src) or src.endswith(".ckpt")):
        flatten_checkpoint(src, dst)
    else:
        copy_file(src, dst)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", help="Path to the bundle spec JSON.")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Directory to create the bundle in.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing bundle directory.")
    args = parser.parse_args(argv)

    with open(args.spec) as f:
        spec = json.load(f)

    name = spec["name"]
    bundle_dir = os.path.join(os.path.abspath(args.output_dir), name)

    if os.path.exists(bundle_dir):
        if not args.force:
            sys.exit(f"{bundle_dir} already exists. Pass --force to overwrite.")
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir)

    entries = (
        [("weights", d, s) for d, s in spec.get("weights", {}).items()]
        + [("configs", d, s) for d, s in spec.get("configs", {}).items()]
        + [("data", d, s) for d, s in spec.get("data", {}).items()]
    )
    for top, dest, src in entries:
        src_abs = src if os.path.isabs(src) else os.path.abspath(src)
        if not os.path.exists(src_abs):
            sys.exit(f"missing source for {top}/{dest}: {src_abs}")
        add_entry(top, src_abs, os.path.join(bundle_dir, top, dest))
        print(f"  {top}/{dest}  <-  {src}")

    files = []
    for root, _, names in os.walk(bundle_dir):
        for n in sorted(names):
            path = os.path.join(root, n)
            rel = os.path.relpath(path, bundle_dir)
            if rel == "MANIFEST.json":
                continue
            files.append({
                "path": rel,
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            })
    files.sort(key=lambda r: r["path"])

    with open(os.path.join(bundle_dir, "MANIFEST.json"), "w") as f:
        json.dump({"name": name, "files": files}, f, indent=2)
        f.write("\n")

    total = sum(r["bytes"] for r in files)
    print(f"\n{name}: {len(files)} files, {total / 1e9:.3f} GB")
    print(f"  {bundle_dir}")
    kind = "dataset" if spec.get("data") else "weights"
    print(f"  push:  scripts/weights_bundle/push_bundle.sh {bundle_dir} <tag> "
          f"--repo <repo> --kind {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
