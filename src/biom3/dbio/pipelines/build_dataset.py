"""Pipeline orchestrator for building fine-tuning datasets."""

import argparse
import csv
import logging
import os
import re
import sys
from datetime import datetime

import pandas as pd

from biom3.backend.device import setup_logger
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    get_file_metadata,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)
from biom3.dbio.config import (
    get_database_path,
    get_training_data_path,
)
from biom3.dbio.readers.swissprot_csv import SwissProtReader, OUTPUT_COLS
from biom3.dbio.readers.pfam_csv import PfamReader
from biom3.dbio.enrich import compose_caption
from biom3.dbio.stats import (
    compute_coverage_stats,
    write_stats_markdown,
)

logger = setup_logger(__name__)

_PFAM_ID_RE = re.compile(r"PF\d{5}")


def _row_has_pfam(pfam_label_cell, pfam_id):
    """Return True if *pfam_id* appears in *pfam_label_cell* (list-stringified)."""
    if pfam_label_cell is None:
        return False
    try:
        if pd.isna(pfam_label_cell):
            return False
    except (TypeError, ValueError):
        pass
    return pfam_id in _PFAM_ID_RE.findall(str(pfam_label_cell))


def _read_release_version(filepath, pattern):
    """Read a release file and extract a version string matching *pattern*."""
    import re
    try:
        with open(filepath) as f:
            text = f.read(2048)
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def _get_database_versions(args):
    """Collect version and file metadata for databases used in this build."""
    versions = {}

    swissprot_path = _resolve_swissprot_path(args)
    pfam_paths = _resolve_pfam_paths(args)

    versions["swissprot_csv"] = get_file_metadata(swissprot_path)
    versions["pfam_csv"] = [get_file_metadata(p) for p in pfam_paths]

    # UniProt release version from reldate.txt
    try:
        db_root = str(get_database_path("swissprot", args.config))
        reldate_path = os.path.join(db_root, "reldate.txt")
        versions["uniprot_release"] = _read_release_version(
            reldate_path, r"Release\s+(\S+)",
        )
    except Exception:
        pass

    # Pfam release version from relnotes.txt
    try:
        db_root = str(get_database_path("pfam", args.config))
        relnotes_path = os.path.join(db_root, "relnotes.txt")
        versions["pfam_release"] = _read_release_version(
            relnotes_path, r"RELEASE\s+(\S+)",
        )
    except Exception:
        pass

    # NCBI taxonomy dump metadata
    if args.add_taxonomy:
        try:
            db_root = str(get_database_path("ncbi_taxonomy", args.config))
            rankedlineage = os.path.join(db_root, "rankedlineage.dmp")
            versions["ncbi_taxonomy"] = get_file_metadata(rankedlineage)
        except Exception:
            pass

    # Provenance TSV (download log with timestamps and MD5s)
    try:
        db_root = str(get_database_path("swissprot", args.config))
        provenance_path = os.path.join(os.path.dirname(db_root), "provenance.tsv")
        if os.path.exists(provenance_path):
            versions["provenance_tsv"] = os.path.realpath(provenance_path)
    except Exception:
        pass

    return versions


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build a fine-tuning dataset for specified Pfam families."
    )
    parser.add_argument(
        "-p", "--pfam_ids", type=str, nargs="+", required=True,
        help="One or more Pfam IDs to extract (e.g. PF00018 PF00169)",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--swissprot", type=str, default=None,
        help="Path to fully_annotated_swiss_prot.csv (default: from config)",
    )
    parser.add_argument(
        "--pfam", type=str, nargs="+", default=None, metavar="PATH",
        help="One or more paths to Pfam protein-text CSVs/Parquets. "
             "Single path: the foundational Pfam_protein_text_dataset.csv "
             "(legacy default). Multiple paths: typically per-family "
             "pfam_full_subsets_*.csv artifacts produced by "
             "biom3_build_pfam_subsets, all sharing the same 11-column "
             "schema. Paths are concatenated before --pfam_ids "
             "filtering. Default: from config.",
    )
    parser.add_argument(
        "--no_dedupe_pfam", action="store_true", default=False,
        help="When --pfam is given multiple paths, preserve duplicate "
             "(primary_Accession, pfam_label) rows instead of dropping "
             "them. Only meaningful with multi-path --pfam; ignored for "
             "single-path inputs.",
    )
    parser.add_argument(
        "--databases_root", type=str, default=None,
        help="Override database root path",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dbio config JSON (default: configs/dbio_config.json)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500_000,
        help="Chunk size for reading the Pfam CSV (default: 500000)",
    )
    parser.add_argument(
        "--enrich_pfam", action="store_true", default=False,
        help="Enrich Pfam captions with UniProt annotations. Requires "
             "either --annotation_cache or --uniprot_dat to supply the "
             "annotation source.",
    )
    parser.add_argument(
        "--uniprot_dat", type=str, nargs="+", default=None,
        metavar="PATH",
        help="Use local UniProt .dat.gz file(s) for enrichment. Accepts "
             "one or more paths (e.g. uniprot_sprot.dat.gz "
             "uniprot_trembl.dat.gz). For full Pfam coverage, include the "
             "TrEMBL file.",
    )
    parser.add_argument(
        "--annotation_cache", type=str, nargs="+", default=None,
        metavar="PATH",
        help="Pre-built annotation Parquet cache(s) for fast enrichment "
             "(built via biom3_build_annotation_cache). Checked before "
             "--uniprot_dat; .dat files are only parsed for accessions "
             "not found in the cache.",
    )
    parser.add_argument(
        "--add_taxonomy", action="store_true", default=False,
        help="Add NCBI taxonomy lineage (local, no API needed)",
    )
    parser.add_argument(
        "--taxonomy_filter", type=str, nargs="*", default=None,
        help='Filter by taxonomy rank (e.g. "superkingdom=Bacteria"). '
             'Literal "None" elements are dropped; an all-"None" list '
             'is treated as if the flag were not passed.',
    )
    parser.add_argument(
        "--taxid_index", type=str, default=None,
        help="Path to pre-built SQLite accession2taxid index (built via biom3_build_taxid_index)",
    )
    parser.add_argument(
        "--taxonomy_dir", type=str, default=None,
        help="Directory containing rankedlineage.dmp (and optionally "
             "prot.accession2taxid.gz when --taxid_index is not set). "
             "When set, overrides the ncbi_taxonomy path resolved from "
             "BIOM3_DATABASES_ROOT or configs/dbio_config.json. Use this "
             "when running biom3_build_dataset with all paths supplied "
             "explicitly (no databases_root config / env var).",
    )
    parser.add_argument(
        "--output_filename", type=str, default="dataset.csv",
        help="Filename for the output dataset CSV (default: dataset.csv). "
             "The annotations file will be named with an '_annotations' suffix.",
    )
    # Source-CSV join layer (opt-in). Each flag enables an additional
    # join against a per-database source CSV produced by the matching
    # biom3_build_source_* builder.
    parser.add_argument(
        "--use_expasy", action="store_true", default=False,
        help="Join ExPASy enzyme data on EC numbers extracted from "
             "annot_catalytic_activity. Adds annot_ec_names and "
             "annot_ec_description columns.",
    )
    parser.add_argument(
        "--expasy_csv", type=str, default=None,
        help="Path to expasy_enzyme.csv. Defaults to the expasy_csv entry "
             "in the dbio config.",
    )
    parser.add_argument(
        "--use_brenda", action="store_true", default=False,
        help="Join BRENDA per-organism kinetics on (EC, organism). Adds "
             "annot_brenda_substrates / km_values / ph_optimum / "
             "temperature_optimum columns.",
    )
    parser.add_argument(
        "--brenda_csv", type=str, default=None,
        help="Path to brenda_kinetics.csv. Defaults to the brenda_csv entry "
             "in the dbio config.",
    )
    parser.add_argument(
        "--organism_match", choices=["strict", "relaxed", "ec_only"],
        default="strict",
        help="BRENDA organism matching strictness. 'strict' (default) "
             "requires species-level match; 'relaxed' falls back to "
             "genus; 'ec_only' accepts any EC-level BRENDA record.",
    )
    parser.add_argument(
        "--use_smart", action="store_true", default=False,
        help="Join SMART domain descriptions on UniProt DR SMART "
             "cross-references. Adds annot_smart_domains column.",
    )
    parser.add_argument(
        "--smart_csv", type=str, default=None,
        help="Path to smart_domains.csv. Defaults to the smart_csv entry "
             "in the dbio config.",
    )
    parser.add_argument(
        "--per_pfam_output", action="store_true", default=False,
        help="Emit one self-contained subdirectory per Pfam ID under "
             "--outdir (each with its own dataset.csv, "
             "dataset_annotations.csv, build_manifest.json, "
             "dataset.stats.md, pfam_ids.csv). No aggregate top-level "
             "dataset is written in this mode. Default: a single "
             "combined dataset at --outdir.",
    )
    parsed = parser.parse_args(args)

    # Normalize --taxonomy_filter: drop literal "None" / "none" elements
    # (commonly produced by unsubstituted shell template variables like
    # `--taxonomy_filter $TAXONOMY_FILTER` when the var is unset). If the
    # surviving list is empty, treat the flag as unset.
    if parsed.taxonomy_filter:
        cleaned = [f for f in parsed.taxonomy_filter if f.strip().lower() != "none"]
        parsed.taxonomy_filter = cleaned if cleaned else None

    return parsed


def _resolve_swissprot_path(args):
    if args.swissprot:
        return args.swissprot
    return str(get_training_data_path("swissprot_csv", args.config))


def _resolve_pfam_paths(args):
    """Resolve Pfam input paths.

    Returns a list of one or more paths. When --pfam is set on the
    CLI, returns the user-provided list verbatim. When unset, falls
    back to the single config-discovered pfam_csv path wrapped in a
    1-element list so callers can iterate uniformly.
    """
    if args.pfam:
        return list(args.pfam)
    return [str(get_training_data_path("pfam_csv", args.config))]


def _resolve_source_csv(args, attr_name, dataset_key):
    """Return the CLI-provided path or fall back to `training_data_path(<dataset_key>)`.

    Used for opt-in source CSVs (ExPASy/BRENDA/SMART). Returns None if no
    CLI arg is given AND the config path doesn't exist on disk, so the
    caller can surface a clear "--<flag>_csv required" error.
    """
    cli_value = getattr(args, attr_name, None)
    if cli_value:
        return cli_value
    try:
        resolved = str(get_training_data_path(dataset_key, args.config))
    except Exception:
        return None
    return resolved if os.path.exists(resolved) else None




def _parse_taxonomy_filters(filter_strs):
    """Parse "rank=value" strings into (rank, include_set) pairs."""
    filters = []
    for s in filter_strs:
        if "=" not in s:
            raise ValueError(f"Invalid taxonomy filter: {s!r}. Expected 'rank=value'.")
        rank, value = s.split("=", 1)
        filters.append((rank.strip(), value.strip()))
    return filters


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Set up dual logging (console + file)
    log_path, file_handler = setup_file_logging(
        args.outdir, logger_prefix="biom3.dbio", log_filename="build.log",
    )
    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Build fine-tuning dataset")
        logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
        logger.info("Command:     %s", " ".join(sys.argv))
        logger.info("Pfam IDs:    %s", " ".join(args.pfam_ids))
        logger.info("Output dir:  %s", os.path.abspath(args.outdir))
        logger.info("=" * 60)

        # Resolve paths
        swissprot_path = _resolve_swissprot_path(args)
        pfam_paths = _resolve_pfam_paths(args)

        # Extract from SwissProt
        logger.info("Extracting from SwissProt...")
        sp_reader = SwissProtReader(swissprot_path)
        df_sp = sp_reader.query_by_pfam(args.pfam_ids)

        # Extract from Pfam input(s). For multi-path runs, query each in
        # turn and concatenate. Dedupe on (primary_Accession, pfam_label)
        # by default; --no_dedupe_pfam preserves the union as a multiset.
        logger.info("Extracting from Pfam (%d input(s))...", len(pfam_paths))
        per_path_dfs = []
        for i, path in enumerate(pfam_paths, 1):
            reader = PfamReader(path, chunk_size=args.chunk_size)
            df_i = reader.query_by_pfam(
                args.pfam_ids, keep_family_cols=True,
            )
            logger.info(
                "Pfam input %d/%d (%s): %s rows",
                i, len(pfam_paths), os.path.basename(path), f"{len(df_i):,}",
            )
            per_path_dfs.append(df_i)

        if per_path_dfs:
            df_pfam = pd.concat(per_path_dfs, ignore_index=True)
        else:
            from biom3.dbio.readers.pfam_csv import OUTPUT_COLS as _PFAM_COLS
            df_pfam = pd.DataFrame(
                columns=_PFAM_COLS + ["family_name", "family_description"],
            )

        if not args.no_dedupe_pfam and len(pfam_paths) > 1:
            pre = len(df_pfam)
            df_pfam = df_pfam.drop_duplicates(
                subset=["primary_Accession", "pfam_label"], keep="first",
            ).reset_index(drop=True)
            logger.info(
                "Pfam (combined, deduped): %s → %s rows from %d input(s)",
                f"{pre:,}", f"{len(df_pfam):,}", len(pfam_paths),
            )
        else:
            logger.info(
                "Pfam (combined): %s rows from %d input(s)",
                f"{len(df_pfam):,}", len(pfam_paths),
            )

        # Step 1: Populate annotation columns on Pfam rows
        from biom3.dbio.enrich import enrich_dataframe

        local_annotations = None
        taxonomy_tree = None
        accession_taxid_map = None

        if args.enrich_pfam or args.add_taxonomy:
            accessions = df_pfam["primary_Accession"].dropna().unique().tolist()
            accession_set = set(accessions)

            if args.enrich_pfam:
                if not args.annotation_cache and not args.uniprot_dat:
                    raise ValueError(
                        "--enrich_pfam requires --annotation_cache or "
                        "--uniprot_dat. The legacy UniProt REST API path was "
                        "removed in v0.1.0a7; build a TrEMBL annotation cache "
                        "via biom3_build_annotation_cache instead."
                    )

                local_annotations = {}

                # Priority 1: Parquet annotation cache (instant lookup)
                if args.annotation_cache:
                    from biom3.dbio.helpers.annotation_cache import load_annotation_cache

                    local_annotations = load_annotation_cache(
                        args.annotation_cache, accession_set,
                    )
                    logger.info("Annotation cache: %s/%s accessions found",
                                f"{len(local_annotations):,}",
                                f"{len(accessions):,}")

                # Priority 2: Raw .dat file parsing (remaining accessions)
                if args.uniprot_dat:
                    remaining = accession_set - set(local_annotations.keys())
                    if remaining:
                        from biom3.dbio.parsers.swissprot_dat import SwissProtDatParser

                        for dat_path in args.uniprot_dat:
                            logger.info("Parsing local .dat file: %s", dat_path)
                            parser = SwissProtDatParser(dat_path)
                            still_remaining = accession_set - set(local_annotations.keys())
                            if not still_remaining:
                                logger.info("All accessions already found, skipping %s", dat_path)
                                break
                            local_annotations.update(parser.parse(still_remaining))
                        logger.info("Local enrichment total: %s/%s accessions found",
                                    f"{len(local_annotations):,}",
                                    f"{len(accessions):,}")

                local_annotations = local_annotations or None

            if args.add_taxonomy:
                taxonomy_tree, accession_taxid_map = _load_taxonomy(
                    args, accessions,
                )

        # Load source-CSV lookups for the join layer (opt-in). CSV paths fall
        # back to config defaults (configs/dbio_config.json training_datasets).
        expasy_lookup = None
        brenda_lookup = None
        smart_lookup = None

        if args.use_expasy:
            expasy_path = _resolve_source_csv(args, "expasy_csv", "expasy_csv")
            if not expasy_path:
                raise ValueError(
                    "--use_expasy requires --expasy_csv or an expasy_csv entry "
                    "in the dbio config (and the file must exist on disk)."
                )
            args.expasy_csv = expasy_path
            from biom3.dbio.enrich import load_expasy_lookup
            expasy_lookup = load_expasy_lookup(expasy_path)
        if args.use_brenda:
            brenda_path = _resolve_source_csv(args, "brenda_csv", "brenda_csv")
            if not brenda_path:
                raise ValueError(
                    "--use_brenda requires --brenda_csv or a brenda_csv entry "
                    "in the dbio config (and the file must exist on disk)."
                )
            args.brenda_csv = brenda_path
            from biom3.dbio.enrich import load_brenda_lookup
            brenda_lookup = load_brenda_lookup(brenda_path)
        if args.use_smart:
            smart_path = _resolve_source_csv(args, "smart_csv", "smart_csv")
            if not smart_path:
                raise ValueError(
                    "--use_smart requires --smart_csv or a smart_csv entry "
                    "in the dbio config (and the file must exist on disk)."
                )
            args.smart_csv = smart_path
            from biom3.dbio.enrich import load_smart_lookup
            smart_lookup = load_smart_lookup(smart_path)

        # Always run enrich_dataframe to copy family columns into annot_* columns
        df_pfam, join_stats = enrich_dataframe(
            df_pfam,
            local_annotations=local_annotations,
            taxonomy_tree=taxonomy_tree,
            accession_taxid_map=accession_taxid_map,
            expasy_lookup=expasy_lookup,
            brenda_lookup=brenda_lookup,
            smart_lookup=smart_lookup,
            organism_match=args.organism_match,
        )

        # Step 2: Compose [final]text_caption from annotation columns (Pfam only).
        # SwissProt rows already have ALL-CAPS captions from the source CSV.
        df_pfam = compose_caption(df_pfam)

        # Track provenance so stats can report per-source row counts, then combine.
        df_combined = pd.concat([
            df_sp.assign(_source="swissprot"),
            df_pfam.assign(_source="pfam"),
        ], ignore_index=True)
        logger.info("Combined dataset: %s rows", f"{len(df_combined):,}")
        logger.info("  SwissProt: %s", f"{len(df_sp):,}")
        logger.info("  Pfam:      %s", f"{len(df_pfam):,}")

        if args.taxonomy_filter:
            df_combined = _apply_taxonomy_filters(df_combined, args)

        # Precompute common build_manifest fields
        resolved_paths = {
            "swissprot_csv": os.path.abspath(_resolve_swissprot_path(args)),
            "pfam_csv": [os.path.abspath(p) for p in _resolve_pfam_paths(args)],
        }
        if args.use_expasy and args.expasy_csv:
            resolved_paths["expasy_csv"] = os.path.abspath(args.expasy_csv)
        if args.use_brenda and args.brenda_csv:
            resolved_paths["brenda_csv"] = os.path.abspath(args.brenda_csv)
        if args.use_smart and args.smart_csv:
            resolved_paths["smart_csv"] = os.path.abspath(args.smart_csv)

        database_versions = _get_database_versions(args)
        if args.use_expasy and args.expasy_csv:
            database_versions["expasy_csv"] = get_file_metadata(args.expasy_csv)
        if args.use_brenda and args.brenda_csv:
            database_versions["brenda_csv"] = get_file_metadata(args.brenda_csv)
        if args.use_smart and args.smart_csv:
            database_versions["smart_csv"] = get_file_metadata(args.smart_csv)

        if args.per_pfam_output:
            # Per-Pfam subdirectories only — no aggregate output at top level.
            for pid in args.pfam_ids:
                subdir = os.path.join(args.outdir, pid)
                os.makedirs(subdir, exist_ok=True)
                mask = df_combined["pfam_label"].apply(
                    lambda s: _row_has_pfam(s, pid)
                )
                df_pid = df_combined[mask].reset_index(drop=True)
                per_pid_row_counts = {
                    "swissprot": int((df_pid["_source"] == "swissprot").sum()),
                    "pfam": int((df_pid["_source"] == "pfam").sum()),
                    "combined": len(df_pid),
                }
                _write_dataset_outputs(
                    df_pid, subdir, [pid], args,
                    start_time=start_time, elapsed_fn=lambda: datetime.now() - start_time,
                    row_counts=per_pid_row_counts,
                    join_stats=join_stats,
                    resolved_paths=resolved_paths,
                    database_versions=database_versions,
                    stats_title=f"dataset/{pid} — coverage stats",
                )
        else:
            aggregate_row_counts = {
                "swissprot": len(df_sp),
                "pfam": len(df_pfam),
                "combined": len(df_combined),
            }
            _write_dataset_outputs(
                df_combined, args.outdir, args.pfam_ids, args,
                start_time=start_time, elapsed_fn=lambda: datetime.now() - start_time,
                row_counts=aggregate_row_counts,
                join_stats=join_stats,
                resolved_paths=resolved_paths,
                database_versions=database_versions,
                stats_title="dataset — coverage stats",
            )

        logger.info("Done in %s", datetime.now() - start_time)
        logger.info("Log saved to %s", log_path)
    finally:
        teardown_file_logging("biom3.dbio", file_handler)


def _write_dataset_outputs(df, outdir, pfam_ids, args, *,
                           start_time, elapsed_fn,
                           row_counts, join_stats, resolved_paths,
                           database_versions, stats_title):
    """Write dataset.csv, dataset_annotations.csv, pfam_ids.csv,
    build_manifest.json, and dataset.stats.md into *outdir*."""
    stem, ext = os.path.splitext(args.output_filename)
    annotations_filename = f"{stem}_annotations{ext}"

    annotations_path = os.path.join(outdir, annotations_filename)
    df_to_save = df.drop(columns=["_source"], errors="ignore")
    df_to_save.to_csv(annotations_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved annotated dataset to %s", annotations_path)

    out_path = os.path.join(outdir, args.output_filename)
    df_to_save[OUTPUT_COLS].to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved dataset to %s", out_path)

    pfam_ids_path = os.path.join(outdir, "pfam_ids.csv")
    pd.DataFrame({"pfam_id": pfam_ids}).to_csv(pfam_ids_path, index=False)

    stats = compute_coverage_stats(
        df,
        source_col="_source" if "_source" in df.columns else None,
        join_metadata=join_stats or None,
    )
    stats_path = os.path.join(outdir, f"{stem}.stats.md")
    write_stats_markdown(stats, stats_path, title=stats_title)
    logger.info("Saved stats report to %s", stats_path)

    outputs = {"row_counts": row_counts}
    if join_stats:
        outputs["join_stats"] = join_stats

    manifest_path = write_manifest(
        args, outdir, start_time, elapsed_fn(),
        outputs=outputs,
        resolved_paths=resolved_paths,
        database_versions=database_versions,
        stats=stats,
    )
    logger.info("Saved build manifest to %s", manifest_path)


def _resolve_taxonomy_dir(args):
    """Resolve the NCBI taxonomy directory.

    Priority: explicit --taxonomy_dir flag > config-resolved
    get_database_path("ncbi_taxonomy", ...). Lets callers run
    biom3_build_dataset with all paths supplied on the CLI without
    requiring BIOM3_DATABASES_ROOT or configs/dbio_config.json.
    """
    if getattr(args, "taxonomy_dir", None):
        return args.taxonomy_dir
    return str(get_database_path("ncbi_taxonomy", args.config))


def _load_taxonomy(args, accessions):
    """Load taxonomy tree and look up accessions."""
    from biom3.dbio.readers.taxonomy import TaxonomyTree, AccessionTaxidMapper

    taxonomy_dir = _resolve_taxonomy_dir(args)
    taxonomy_tree = TaxonomyTree(taxonomy_dir)
    taxonomy_tree.load()

    accession2taxid_path = os.path.join(taxonomy_dir, "prot.accession2taxid.gz")
    mapper = AccessionTaxidMapper(accession2taxid_path)

    if args.taxid_index:
        accession_taxid_map = mapper.lookup_sqlite(accessions, args.taxid_index)
    else:
        accession_taxid_map = mapper.lookup(accessions)

    return taxonomy_tree, accession_taxid_map


def _parse_annot_lineage(lineage_str):
    """Parse an annot_lineage string into a list of taxon names.

    Example input:
        "The organism lineage is Eukaryota, Metazoa, Chordata, ..."
    Returns:
        ["Eukaryota", "Metazoa", "Chordata", ...]
    """
    prefix = "The organism lineage is "
    if lineage_str.startswith(prefix):
        lineage_str = lineage_str[len(prefix):]
    return [t.strip() for t in lineage_str.split(",") if t.strip()]


def _apply_taxonomy_filters(df, args):
    """Filter the combined DataFrame by taxonomy rank constraints.

    Uses two sources for taxonomy data:
    1. NCBI prot.accession2taxid lookup (structured rank->value via TaxonomyTree)
    2. The annot_lineage column populated by UniProt enrichment (flat lineage list)

    Accessions found in the NCBI index use the structured approach (exact rank
    matching). Accessions not in the NCBI index fall back to checking whether
    the filter value appears anywhere in the annot_lineage string.
    """
    from biom3.dbio.readers.taxonomy import TaxonomyTree, AccessionTaxidMapper

    filters = _parse_taxonomy_filters(args.taxonomy_filter)
    taxonomy_dir = _resolve_taxonomy_dir(args)
    taxonomy_tree = TaxonomyTree(taxonomy_dir)
    taxonomy_tree.load()

    accession2taxid_path = os.path.join(taxonomy_dir, "prot.accession2taxid.gz")
    mapper = AccessionTaxidMapper(accession2taxid_path)

    accessions = df["primary_Accession"].dropna().unique().tolist()
    if args.taxid_index:
        acc_to_taxid = mapper.lookup_sqlite(accessions, args.taxid_index)
    else:
        acc_to_taxid = mapper.lookup(accessions)

    # --- Path 1: NCBI structured lookup ---
    taxid_to_accs = {}
    for acc, tid in acc_to_taxid.items():
        taxid_to_accs.setdefault(tid, set()).add(acc)

    all_taxids = set(acc_to_taxid.values())

    for rank, value in filters:
        logger.info("Applying taxonomy filter: %s=%s", rank, value)
        kept_taxids = taxonomy_tree.filter_by_rank(
            all_taxids, rank, include={value},
        )
        all_taxids = kept_taxids

    kept_accs = set()
    for tid in all_taxids:
        kept_accs.update(taxid_to_accs.get(tid, set()))
    ncbi_kept = len(kept_accs)

    # --- Path 2: fallback to annot_lineage for unmapped accessions ---
    ncbi_mapped = set(acc_to_taxid.keys())
    has_lineage_col = "annot_lineage" in df.columns
    fallback_kept = 0

    if has_lineage_col:
        for _, row in df.iterrows():
            acc = row.get("primary_Accession")
            if pd.isna(acc) or acc in ncbi_mapped:
                continue
            lineage_str = row.get("annot_lineage")
            if pd.isna(lineage_str) or not lineage_str:
                continue
            lineage_terms = _parse_annot_lineage(str(lineage_str))
            passes = all(value in lineage_terms for _, value in filters)
            if passes:
                kept_accs.add(acc)
                fallback_kept += 1

    if has_lineage_col:
        logger.info("Taxonomy filter matched: %s via NCBI index, %s via annot_lineage",
                     f"{ncbi_kept:,}", f"{fallback_kept:,}")

    before = len(df)
    df = df[df["primary_Accession"].isin(kept_accs)].copy()
    logger.info("Taxonomy filter: %s -> %s rows", f"{before:,}", f"{len(df):,}")
    return df
