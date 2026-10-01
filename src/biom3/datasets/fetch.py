"""Fetch a published BioM3 dataset from an OCI registry."""

import argparse
import sys

from biom3.core import oci

DEFAULT_REGISTRY = "ghcr.io/natural-machine/biom3-datasets"
DATA_PREFIX = "data/"


def fetch_dataset(dataset, output_dir, registry=DEFAULT_REGISTRY,
                  force=False, dry_run=False):
    """Fetch ``registry:dataset`` into ``output_dir``, skipping files already there.

    The bundle's ``data/`` files land directly in *output_dir*. Each file is
    matched against the registry's own content digest, so an existing file is
    re-downloaded only when its bytes differ.
    """
    return oci.fetch_bundle(registry, dataset, output_dir, DATA_PREFIX,
                            force=force, dry_run=dry_run)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="biom3_fetch_dataset",
        description="Fetch a published BioM3 dataset from an OCI registry.",
        epilog=(
            "examples:\n"
            "  biom3_fetch_dataset gfp_demo -o ./data\n"
            "  biom3_fetch_dataset gfp_demo -o ./data --dry_run\n"
            "\nlist available datasets:\n"
            f"  oras repo tags {DEFAULT_REGISTRY}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="dataset tag to fetch, e.g. gfp_demo")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="directory to write the dataset files into, e.g. ./data")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY,
                        help=f"OCI repository holding the datasets (default: {DEFAULT_REGISTRY})")
    parser.add_argument("--force", action="store_true",
                        help="replace local files whose bytes differ from the published ones")
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would be fetched and exit")
    return parser.parse_args(argv)


def main(args=None):
    if args is None:
        args = parse_arguments()
    try:
        fetch_dataset(args.dataset, args.output_dir, registry=args.registry,
                      force=args.force, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
