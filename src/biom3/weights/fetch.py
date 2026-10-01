"""Fetch a published BioM3 weights bundle from an OCI registry."""

import argparse
import sys

from biom3.core import oci

DEFAULT_REGISTRY = "ghcr.io/natural-machine/biom3-weights"
TEST_WEIGHTS_REGISTRY = "ghcr.io/natural-machine/biom3-test-weights"
TEST_WEIGHTS_TAG = "latest"
WEIGHTS_PREFIX = "weights/"


def fetch_bundle(bundle, output_dir, registry=DEFAULT_REGISTRY,
                 include_configs=False, force=False, dry_run=False):
    """Fetch ``registry:bundle`` into ``output_dir``, skipping files already there.

    Only ``weights/`` entries are taken unless *include_configs*. Each file is
    matched against the registry's own content digest, so an existing file is
    re-downloaded only when its bytes differ.
    """
    return oci.fetch_bundle(
        registry, bundle, output_dir, WEIGHTS_PREFIX,
        include_other=include_configs, force=force, dry_run=dry_run,
        skipped_note="non-weight file(s) skipped (--include_configs to take them)",
    )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="biom3_fetch_weights",
        description="Fetch a published BioM3 weights bundle from an OCI registry.",
        epilog=(
            "examples:\n"
            "  biom3_fetch_weights run1_base -o ./weights\n"
            "  biom3_fetch_weights --test_weights -o ./weights   # everything the tests need\n"
            "  biom3_fetch_weights run1_base -o ./weights --dry_run\n"
            "\nlist available bundles:\n"
            f"  oras repo tags {DEFAULT_REGISTRY}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bundle", nargs="?",
                        help="bundle tag to fetch, e.g. run1_base")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="weights root to merge into, e.g. ./weights")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY,
                        help=f"OCI repository holding the bundles (default: {DEFAULT_REGISTRY})")
    parser.add_argument("--test_weights", action="store_true",
                        help="shortcut for the published set of weights the test suite "
                             f"needs ({TEST_WEIGHTS_REGISTRY})")
    parser.add_argument("--include_configs", action="store_true",
                        help="also take the bundle's non-weight files")
    parser.add_argument("--force", action="store_true",
                        help="replace local files whose bytes differ from the bundle")
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would be fetched and exit")
    args = parser.parse_args(argv)

    if args.test_weights:
        if args.bundle:
            parser.error("give either a bundle tag or --test_weights, not both")
        args.bundle = TEST_WEIGHTS_TAG
        if args.registry == DEFAULT_REGISTRY:
            args.registry = TEST_WEIGHTS_REGISTRY
    elif not args.bundle:
        parser.error("a bundle tag is required (or pass --test_weights)")
    return args


def main(args=None):
    if args is None:
        args = parse_arguments()
    try:
        fetch_bundle(args.bundle, args.output_dir, registry=args.registry,
                     include_configs=args.include_configs, force=args.force,
                     dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
