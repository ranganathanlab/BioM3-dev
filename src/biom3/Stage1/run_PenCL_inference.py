"""BioM3 Stage 1: PenCL inference

Mimics the workflow described at
    https://huggingface.co/niksapraljak1/BioM3#stage-1-pencl-inference

Config file:
    configs/inference/stage1_PenCL.json  (uses _base_configs composition)

Example usage (raw weights, built-in test dataset):

biom3_PenCL_inference \
    --input_data_path None \
    --config_path "configs/inference/stage1_PenCL.json" \
    --model_path "./weights/PenCL/BioM3_PenCL_epoch20.bin" \
    --output_path "outputs/pencl_embeddings.pt"

Example usage (custom CSV input, CPU):

biom3_PenCL_inference \
    --input_data_path "data/my_proteins.csv" \
    --config_path "configs/inference/stage1_PenCL.json" \
    --model_path "./weights/PenCL/BioM3_PenCL_epoch20.bin" \
    --output_path "outputs/pencl_embeddings.pt" \
    --device cpu \
    --batch_size 16 \
    --num_workers 4

Example usage (PyTorch Lightning checkpoint):

biom3_PenCL_inference \
    --input_data_path None \
    --config_path "configs/inference/stage1_PenCL.json" \
    --model_path "./weights/PenCL/BioM3_PenCL_epoch20.ckpt" \
    --output_path "outputs/pencl_embeddings.pt"

Example usage (enable the O(n^2) cross-comparison metrics on a subset):

biom3_PenCL_inference \
    --input_data_path "data/large_proteins.csv" \
    --config_path "configs/inference/stage1_PenCL.json" \
    --model_path "./weights/PenCL/BioM3_PenCL_epoch20.bin" \
    --output_path "outputs/pencl_embeddings.pt" \
    --cross_comparison_sample_limit 1000

"""

import copy
import os
import sys
import argparse
import warnings
import yaml
from argparse import Namespace
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm as tqdm
from functools import partial

import biom3.Stage1.preprocess as prep
import biom3.Stage1.model as mod
import biom3.Stage1.PL_wrapper as PL_wrap
from biom3.core.io import load_and_prepare_model
from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)
from biom3.backend.device import setup_logger, set_float32_matmul_precision
from biom3.backend.device import DEVICE_CHOICES, resolve_device
from biom3.core.distributed import (
    barrier,
    init_distributed_if_launched,
    is_main_process,
)

logger = setup_logger(__name__)


# Step 0: Argument Parser Function
def parse_arguments(args):
    parser = argparse.ArgumentParser(description="BioM3 Inference Script (Stage 1)")
    parser.add_argument('-i', '--input_data_path', type=str, required=True,
                        help="Path to input data in csv format. Use None for test case.")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="Path to the JSON configuration file (stage1_config_PenCL_inference.json)")
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help="Path to the pre-trained model weights or checkpoint")
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="Path to save output embeddings")
    
    parser.add_argument('--device', type=str, default="auto",
                        choices=list(DEVICE_CHOICES),
                        help="available device; auto = the detected backend (CUDA, XPU, else CPU)")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="batch size")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="number of dataloading workers")
    parser.add_argument("--load_from_checkpoint", action="store_true",
                        help="Flag to load model_path as a checkpoint. By default, " \
                        "this action is inferred from a .ckpt extension of model_path")
    parser.add_argument("--cross_comparison_sample_limit", type=int, default=0,
                        help="Number of samples used for the O(n^2) cross-comparison "
                             "metrics (dot-product probabilities, homology matrix). "
                             "0 (default) skips them entirely; -1 uses all samples; "
                             "a positive value uses that many. Each metric allocates an "
                             "n x n fp32 matrix (~25 GB at n=80k), so -1 is only safe on "
                             "small datasets. Cross-comparison results are print-only; "
                             "saved embeddings are unaffected.")
    parser.add_argument("--text_padding", type=str, default="max_padding",
                        choices=["max_padding", "dynamic"],
                        help="caption padding. 'max_padding' pads to "
                             "text_max_length, matching training; 'dynamic' pads "
                             "to the batch's longest caption, which makes z_t "
                             "depend on batch composition.")
    parser.add_argument("--float32_matmul_precision", type=str, default=None,
                        choices=["highest", "high", "medium"],
                        help="fp32 matmul precision. 'high' (config default) enables "
                             "TF32 tensor cores; 'highest' forces full fp32 for bitwise "
                             "reproducibility. Overrides the config value when set.")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable autocast and run the forward pass in fp32. "
                             "Autocast (bf16 on xpu, fp16 on cuda) is on by default and "
                             "is a large part of the inference speedup. Disable it when "
                             "comparing runs: bf16 rounding depends on tensor shape, so "
                             "results vary with batch size. Pair with "
                             "--float32_matmul_precision highest for a fully "
                             "deterministic fp32 forward pass.")

    return parser.parse_args(args)


# Step 3: Load Pre-trained Model
def prepare_model(
        config_args,
        model_path,
        device,
        load_from_checkpoint
) -> nn.Module:
    """Initialize the model, load weights, set to eval, and move to device."""
    if load_from_checkpoint:
        model = prepare_model_from_checkpoint(config_args, model_path, device)
    else:
        model = prepare_model_from_raw_weights(config_args, model_path, device)
    model.eval()
    model = model.to(device)
    return model


def prepare_model_from_raw_weights(
        config_args, 
        model_path, 
        device
    ) -> nn.Module:
    """Prepare a model from raw weight file (e.g. .bin or .pt)"""
    model = mod.pfam_PEN_CL(args=config_args)
    model = load_and_prepare_model(
        model, model_path, 
        device=device, 
        strict=False,  # NOTE: Avoids issue.
        eval_mode=True,
        attempt_correction=False,
    )
    return model


def prepare_model_from_checkpoint(
        config_args, 
        model_path, 
        device
    ) -> nn.Module:
    """Prepare a model from a lightning checkpoint."""
    # decide on model architecture
    model_options = {
            'default': mod.PEN_CL,
            'masked': mod.PEN_CL,
            'pfam': mod.pfam_PEN_CL,
            'pfam_ablated': mod.pfam_PEN_CL
    }
    model_class = model_options.get(config_args.model_type, mod.PEN_CL)
    logger.info('Model class: %s', model_class)

    # Model graph (nn.Module)
    model = model_class(args=config_args).to(device)
    model.eval()

    # Decide on PL wrapper
    PL_wrapper_options = {
            'default': PL_wrap.PL_PEN_CL,
            'masked': PL_wrap.mask_PL_PEN_CL,
            'pfam': PL_wrap.pfam_PL_PEN_CL,
            'pfam_ablated': PL_wrap.pfam_PL_PEN_CL
    }
    PL_wrapper_class = PL_wrapper_options.get(
        config_args.model_type, PL_wrap.PL_PEN_CL
    )
    logger.info('PL wrapper class: %s', PL_wrapper_class)
    
    # Get PL model
    PL_model = PL_wrapper_class(
        args=config_args,
        model=model,
        text_tokenizer=model.text_encoder.tokenizer,
        sequence_tokenizer=model.protein_encoder.alphabet
    )
    
    # Load pretrained weights
    logger.info('Loading weights from checkpoint...')
    PL_model = PL_wrapper_class.load_from_checkpoint(
        checkpoint_path=model_path,
        map_location=device,
        args=config_args,
        model=model,
        text_tokenizer=model.text_encoder.tokenizer,
        sequence_tokenizer=model.protein_encoder.alphabet,
        strict=False
    )
    # x = torch.load(model_path)
    # print(x["state_dict"]["model.text_encoder.model.bert.embeddings.position_ids"])

    model = PL_model.model
    return model


# Step 4: Prepare Test Dataset
def load_test_dataset(config_args):
    
    test_dict = {
        'primary_Accession': [
            "P69222", "B5XIP6", "B5XJL3", "B5Y368", "B5YH59"
        ],
        'protein_sequence': [
            "MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR",
            "MVKMIVGLGNPGSKYEKTKHNIGFMAIDNIVKNLDVTFTDDKNFKAQIGSTFINHEKVYFVKPTTFMNNSGIAVKALLTYYNIDITDLIVIYDDLDMEVSKLRLRSKGSAGGHNGIKSIIAHIGTQEFNRIKVGIGRPLKGMTVINHVMGQFNTEDNIAISLTLDRVVNAVKFYLQENDFEKTMQKFNG",
            "MTDYPIKYRLIKTEKHTGARLGEIITPHGTFPTPMFMPVGTQATVKTQSPEELKAIGSGIILSNTYHLWLRPGDELIARSGGLHKFMNWDQPILTDSGGFQVYSLADSRNITEEGVTFKNHLNGSKMFLSPEKAISIQNNLGSDIMMSFDECPQFYQPYDYVKKSIERTSRWAERGLKAHRRPHDQGLFGIVQGAGFEDLRRQSAADLVAMDFPGYSIGGLAVGESHEEMNAVLDFTTPLLPENKPRYLMGVGAPDSLIDGVIRGVDMFDCVLPTRIARNGTCMTSEGRLVVKNAKFAEDFTPLDHDCDCYTCQNYSRAYIRHLLKADETFGIRLTSYHNLYFLVNLMKKVRQAIMDDNLLEFRQDFLERYGYNKSNRNF",
            "MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMGAQMVKEVASKANDAAGDGTTTATVLAQAIVNEGLKAVAAGMNPMDLKRGIDKAVIAAVEELKALSVPCSDSKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLEDELDVVEGMQFDRGYLSPYFINKPDTGAVELESPFILLADKKISNIREMLPVLEAVAKAGKPLVIIAEDVEGEALATLVVNTMRGIVKVAAVKAPGFGDRRKAMLQDIATLTGGTVISEEIGMELEKATLEDLGQAKRVVINKDTTTIIDGVGEESAIQGRVAQIRKQIEEATSDYDREKLQERVAKLAGGVAVIKVGAATEVEMKEKKARVDDALHATRAAVEEGVVAGGGVALVRVAAKLAGLTGQNEDQNVGIKVALRAMEAPLRQIVSNAGEEPSVVANNVKAGDGNYGYNAATEEYGNMIDFGILDPTKVTRSALQYAASVAGLMITTECMVTDLPKGDAPDLGAAGGMGGMGGMGGMM",
            "MGKAIGIDLGTTNSVVAVVVGGEPVVIPNQEGQRTTPSVVAFTDKGERLVGQVAKRQAITNPENTIFSIKRLMGRKYNSQEVQEAKKRLPYKIVEAPNGDAHVEIMGKRYSPPEISAMILQKLKQAAEDYLGEPVTEAVITVPAYFDDSQRQATKDAGRIAGLNVLRIINEPTAAALAYGLDKKKEEKIAVYDLGGGTFDISILEIGEGVIEVKATNGDTYLGGDDFDIRVMDWLIEEFKKQEGIDLRKDRMALQRLKEAAERAKIELSSAMETEINLPFITADASGPKHLLMKLTRAKLEQLVDDLIQKSLEPCKKALSDAGLSQSQIDEVILVGGQTRTPKVQKVVQDFFGKEPHKGVNPDEVVAVGAAIQAAILKGEVKEVLLLDVTPLSLGIETLGGVFTKIIERNTTIPTKKSQIFTTAADNQTAVTIKVYQGEREMAADNKLLGVFELVGIPPAPRGIPQIEVTFDIDANGILHVSAKDLATGKEQSIRITASSGLSEEEIKKMIREAEAHAEEDRRKKQIAEARNEADNMIYTVEKTLRDMGDRISEDERKRIEEAIEKCRRIKDTSNDVNEIKAAVEELAKASHRVAEELYKKAGASQQGAGSTTQSKKEEDVIEAEVEDKDNK"
        ],
        '[final]text_caption': [
                "PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. Stabilizes the binding of IF-2 and IF-3 on the 30S subunit to which N-formylmethionyl-tRNA(fMet) subsequently binds. Helps modulate mRNA selection, yielding the 30S pre-initiation complex (PIC). Upon addition of the 50S ribosomal subunit, IF-1, IF-2 and IF-3 are released leaving the mature 70S translation initiation complex. SUBUNIT: Component of the 30S ribosomal translation pre-initiation complex which assembles on the 30S ribosome in the order IF-2 and IF-3, IF-1 and N-formylmethionyl-tRNA(fMet); mRNA recruitment can occur at any time during PIC assembly. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the IF-1 family. LINEAGE: The organism lineage is Bacteria, Pseudomonadota, Gammaproteobacteria, Enterobacterales, Enterobacteriaceae, Escherichia. FAMILY NAMES: Family names are Translation initiation factor 1A / IF-1.",
                "PROTEIN NAME: Peptidyl-tRNA hydrolase. FUNCTION: The natural substrate for this enzyme may be peptidyl-tRNAs which drop off the ribosome during protein synthesis. CATALYTIC ACTIVITY: an N-acyl-L-alpha-aminoacyl-tRNA + H2O = a tRNA + an N-acyl-L-amino acid + H(+). SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the PTH family. LINEAGE: The organism lineage is Bacteria, Bacillota, Bacilli, Lactobacillales, Streptococcaceae, Streptococcus. FAMILY NAMES: Family names are Peptidyl-tRNA hydrolase.",
                "PROTEIN NAME: Queuine tRNA-ribosyltransferase. FUNCTION: Catalyzes the base-exchange of a guanine (G) residue with the queuine precursor 7-aminomethyl-7-deazaguanine (PreQ1) at position 34 (anticodon wobble position) in tRNAs with GU(N) anticodons (tRNA-Asp, -Asn, -His and -Tyr). Catalysis occurs through a double-displacement mechanism. The nucleophile active site attacks the C1' of nucleotide 34 to detach the guanine base from the RNA, forming a covalent enzyme-RNA intermediate. The proton acceptor active site deprotonates the incoming PreQ1, allowing a nucleophilic attack on the C1' of the ribose to form the product. After dissociation, two additional enzymatic reactions on the tRNA convert PreQ1 to queuine (Q), resulting in the hypermodified nucleoside queuosine (7-(((4,5-cis-dihydroxy-2-cyclopenten-1-yl)amino)methyl)-7-deazaguanosine). CATALYTIC ACTIVITY: 7-aminomethyl-7-carbaguanine + guanosine(34) in tRNA = 7-aminomethyl-7-carbaguanosine(34) in tRNA + guanine. COFACTOR: Binds 1 zinc ion per subunit. PATHWAY: tRNA modification; tRNA-queuosine biosynthesis. SUBUNIT: Homodimer. Within each dimer, one monomer is responsible for RNA recognition and catalysis, while the other monomer binds to the replacement base PreQ1. SIMILARITY: Belongs to the queuine tRNA-ribosyltransferase family. LINEAGE: The organism lineage is Bacteria, Bacillota, Bacilli, Lactobacillales, Streptococcaceae, Streptococcus. FAMILY NAMES: Family names are Queuine tRNA-ribosyltransferase.",
                "PROTEIN NAME: Chaperonin GroEL. FUNCTION: Together with its co-chaperonin GroES, plays an essential role in assisting protein folding. The GroEL-GroES system forms a nano-cage that allows encapsulation of the non-native substrate proteins and provides a physical environment optimized to promote and accelerate protein folding. CATALYTIC ACTIVITY: ATP + H2O + a folded polypeptide = ADP + phosphate + an unfolded polypeptide. SUBUNIT: Forms a cylinder of 14 subunits composed of two heptameric rings stacked back-to-back. Interacts with the co-chaperonin GroES. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the chaperonin (HSP60) family. LINEAGE: The organism lineage is Bacteria, Pseudomonadota, Gammaproteobacteria, Enterobacterales, Enterobacteriaceae, Klebsiella/Raoultella group, Klebsiella. FAMILY NAMES: Family names are TCP-1/cpn60 chaperonin family.",
                "PROTEIN NAME: Chaperone protein DnaK. FUNCTION: Acts as a chaperone. INDUCTION: By stress conditions e.g. heat shock. SIMILARITY: Belongs to the heat shock protein 70 family. LINEAGE: The organism lineage is Bacteria, Nitrospirae, Thermodesulfovibrionia, Thermodesulfovibrionales, Thermodesulfovibrionaceae, Thermodesulfovibrio. FAMILY NAMES: Family names are Hsp70 protein."
        ],
        "pfam_label": [
            "['PF01176’]", 
            "['PF01195’]", 
            "['PF01702’]", 
            "['PF00118’]", 
            "['PF00012’]"
        ]
    }
    
    test_df = pd.DataFrame(test_dict)
    test_dataset = prep.BatchedTextSeqPairingDataset(args=config_args, df=test_df)
    return test_dataset


def load_dataset(input_data_path, config_args, **kwargs):
    """Processes a csv file and constructs a dataset from this data."""
    df = pd.read_csv(
        input_data_path, dtype={"primary_Accession": str}, **kwargs
    )
    df = df.reset_index(drop=True)
    return prep.BatchedTextSeqPairingDataset(args=config_args, df=df)


# Step 5: Compute Homology Probabilities
def compute_homology_matrix(z_p_tensor):
    """
    Compute the homology matrix as cosine similarities between protein latent vectors.
    """
    # Normalize z_p to unit vectors
    z_p_normalized = F.normalize(z_p_tensor, p=2, dim=1)  # L2 normalization

    # Compute cosine similarity matrix
    homology_matrix = torch.matmul(z_p_normalized, z_p_normalized.T)  # (num_samples x num_samples)

    return homology_matrix


def _shard_path(output_path: str, rank: int) -> str:
    return f"{output_path}.rank{rank}.shardtmp"


def _merge_rank_shards(output_path: str, world_size: int):
    """Concatenate per-rank shards in global row order and delete them.

    Each rank writes the rows it embedded plus their global indices; sorting by
    index reproduces the single-process row order regardless of world size.
    """
    idx, z_t, z_p, texts, seqs, accs = [], [], [], [], [], []
    for r in range(world_size):
        path = _shard_path(output_path, r)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing shard from rank {r}: {path}")
        shard = torch.load(path, weights_only=False, map_location="cpu")
        if not shard["indices"]:
            continue  # rank had no batches
        idx.extend(shard["indices"])
        z_t.append(shard["z_t"])
        z_p.append(shard["z_p"])
        texts.extend(shard["texts"])
        seqs.extend(shard["sequences"])
        accs.extend(shard["accessions"])

    order = sorted(range(len(idx)), key=lambda i: idx[i])
    if [idx[i] for i in order] != list(range(len(idx))):
        raise RuntimeError(
            f"Gathered {len(idx)} rows but indices are not a complete 0..N-1 set; "
            "shards are incomplete or overlapping."
        )

    z_t = torch.cat(z_t)[order]
    z_p = torch.cat(z_p)[order]
    texts = [texts[i] for i in order]
    seqs = [seqs[i] for i in order]
    accs = [accs[i] for i in order]

    for r in range(world_size):
        os.remove(_shard_path(output_path, r))

    logger.info("Merged %d rows from %d rank shard(s)", len(idx), world_size)
    return z_t, z_p, texts, seqs, accs


def main(args, _setup_logging=True):
    args.device = resolve_device(args.device)
    # ----- Suppress noisy library warnings -----
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=".*has generative capabilities.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")

    config_args_parser = args
    model_path = config_args_parser.model_path
    batch_size = config_args_parser.batch_size
    num_workers = config_args_parser.num_workers
    load_from_checkpoint = config_args_parser.load_from_checkpoint

    # Opt-in distributed: a no-op unless launched under mpiexec / torchrun, in
    # which case this returns the per-rank device and the run shards by batch.
    rank, local_rank, world_size, resolved_device = init_distributed_if_launched(
        config_args_parser.device
    )
    is_main = is_main_process()

    # Set up dual logging (console + file). Only the main rank owns the file.
    outdir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(outdir, exist_ok=True)
    file_handler = None
    if _setup_logging and is_main:
        log_path, file_handler = setup_file_logging(outdir)
    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("PenCL inference (Stage 1)")
        logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
        logger.info("Command:     %s", " ".join(sys.argv))
        logger.info("=" * 60)

        # Load configuration
        config_dict = load_json_config(config_args_parser.config_path)
        raw_config = copy.deepcopy(config_dict)
        config_args = convert_to_namespace(config_dict)

        # The dataset is built from config_args, not from the CLI namespace, so the
        # padding mode has to be injected here to reach the collate_fn.
        config_args.text_padding = config_args_parser.text_padding
        if config_args.text_padding != "max_padding":
            logger.warning(
                "text_padding=%s: BioM3 training used max_padding, so z_t is "
                "off-distribution here, and under dynamic padding z_t also depends "
                "on batch composition rather than on the caption alone.",
                config_args.text_padding,
            )

        # fp32 matmul precision (TF32): CLI overrides config, config default is "high".
        set_float32_matmul_precision(
            config_args_parser.float32_matmul_precision
            or getattr(config_args, "float32_matmul_precision", "high")
        )

        # Set the device (per-rank when distributed; unchanged single-process)
        device = torch.device(resolved_device)
        if world_size > 1:
            logger.info(
                "Distributed: rank %d/%d (local_rank %d) on %s",
                rank, world_size, local_rank, resolved_device,
            )

        # Infer loading strategy from file extension or specified flag:
        load_from_checkpoint = load_from_checkpoint or model_path.endswith('.ckpt')

        with torch.serialization.safe_globals([Namespace]):
            # Load model
            model = prepare_model(
                config_args=config_args,
                model_path=model_path,
                device=device,
                load_from_checkpoint=load_from_checkpoint
            )

            # Load dataset
            if args.input_data_path.lower() == "none":
                dataset = load_test_dataset(config_args)
            else:
                dataset = load_dataset(
                    args.input_data_path, config_args, 
                    sep=",",
                    quotechar='"',
                    keep_default_na=False,  # empty string instead of nan
                )

        # Shard by BATCH, not by row: every batch keeps exactly the members it would
        # have had single-process, so per-row results are unchanged by world size.
        # (Row-level sharding would repartition batches, and both the ESM batch
        # converter's padding width and bf16 reduction order depend on batch
        # contents.) With world_size == 1 this is the original sequential batching.
        all_batches = [
            list(range(i, min(i + batch_size, len(dataset))))
            for i in range(0, len(dataset), batch_size)
        ]
        my_batches = all_batches[rank::world_size]
        if world_size > len(all_batches) and is_main:
            logger.warning(
                "world_size=%d exceeds the %d batch(es) available; %d rank(s) will "
                "sit idle. Lower batch_size or the rank count to use them.",
                world_size, len(all_batches), world_size - len(all_batches),
            )
        if world_size > 1:
            logger.info(
                "Rank %d: %d of %d batches (%d rows)",
                rank, len(my_batches), len(all_batches),
                sum(len(b) for b in my_batches),
            )

        loader = DataLoader(
            dataset,
            batch_sampler=my_batches,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=partial(prep.collate_fn, dataset=dataset, include_raw=True),
        )

        # Run inference and store accession, text, protein sequence, z_t, and z_p
        z_t_list = []
        z_p_list = []
        text_list = []
        protein_list = []
        acc_id_list = []

        # Determine autocast dtype: use fp16 on CUDA, bf16 on XPU/CPU if available
        use_amp = device.type in ("cuda", "xpu") and not args.no_amp
        amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        logger.info("autocast: %s", amp_dtype if use_amp else "disabled (fp32)")

        index_list = [i for b in my_batches for i in b]

        with torch.inference_mode():
            for item in tqdm.tqdm(loader, disable=not is_main):
                x_t, x_p, texts, sequences, accessions = item
                x_t = x_t.to(device, non_blocking=True)
                x_p = x_p.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(x_t, x_p, compute_masked_logits=False)
                z_t_list.append(outputs["text_joint_latent"].detach().float().cpu())
                z_p_list.append(outputs["seq_joint_latent"].detach().float().cpu())
                # z_t_list.append(outputs[0])
                # z_p_list.append(outputs[1])
                text_list += texts
                protein_list += sequences
                acc_id_list += accessions

        # z_t = torch.cat(z_t_list).cpu()
        # z_p = torch.cat(z_p_list).cpu()
    
        # ORIGINAL CODE FROM HUGGING FACE SEEMS TO PREVENT PARALLELIZATION
        # with torch.no_grad():
        #     for idx in tqdm.trange(len(dataset)):
        #         # print(f"{idx}/{len(dataset)}")
        #         batch = dataset[idx]
        #         x_t, x_p = batch
        #         outputs = model(x_t, x_p, compute_masked_logits=False) # Infer Joint-Embeddings 
        #         z_t = outputs['text_joint_latent']  # Text latent
        #         z_p = outputs['seq_joint_latent']   # Protein latent
        #         z_t_list.append(z_t)
        #         z_p_list.append(z_p)
            
        #         protein_sequence = dataset.protein_sequence_list[idx]
        #         text_prompt = dataset.text_captions_list[idx]
        #         text_list.append(text_prompt)
        #         protein_list.append(protein_sequence)

        # Stack this rank's latent vectors. A rank gets no batches at all when
        # len(all_batches) < world_size, so guard the empty case.
        # 2-D even when empty: the reporting below norms over dim=1, and a bare
        # torch.empty(0) is 1-D.
        z_t_tensor = torch.vstack(z_t_list) if z_t_list else torch.empty(0, 0)
        z_p_tensor = torch.vstack(z_p_list) if z_p_list else torch.empty(0, 0)

        # Distributed: every rank writes its shard, then rank 0 merges in global row
        # order and the others are done. Shards go via disk rather than
        # gather_object_to_main because that helper is built on all_gather_object,
        # which would materialise the whole dataset on every rank.
        if world_size > 1:
            torch.save(
                {
                    "indices": index_list,
                    "z_t": z_t_tensor,
                    "z_p": z_p_tensor,
                    "texts": text_list,
                    "sequences": protein_list,
                    "accessions": acc_id_list,
                },
                _shard_path(args.output_path, rank),
            )
            barrier()
            if not is_main:
                return
            z_t_tensor, z_p_tensor, text_list, protein_list, acc_id_list = (
                _merge_rank_shards(args.output_path, world_size)
            )

        text_prompt_array = np.array(
            [s.encode("utf-8") for s in text_list], dtype=object
        )
        protein_array = np.array(
            [s.encode("utf-8") for s in protein_list], dtype=object
        )
        acc_id_array = np.array(
            [s.encode("utf-8") for s in acc_id_list], dtype=object
        )
    
        # Prepare embedding dict.
        embedding_dict = {
                'z_t': z_t_tensor,
                'z_p': z_p_tensor,
                'text_prompts': text_prompt_array,
                'sequence': protein_array,
                'acc_id': acc_id_array,
        }
    
        # Compute magnitudes (L2 norms) for z_t and z_p
        z_p_magnitude = torch.norm(z_p_tensor, dim=1)  # L2 norm for each protein latent vector
        z_t_magnitude = torch.norm(z_t_tensor, dim=1)  # L2 norm for each text latent vector

        # Print results
        logger.info("\n=== Inference Results ===")
        logger.info("Shape of z_p (protein latent): %s", z_p_tensor.shape)
        logger.info("Shape of z_t (text latent): %s", z_t_tensor.shape)
        logger.info("Magnitudes of z_p vectors: %s", z_p_magnitude)
        logger.info("Magnitudes of z_t vectors: %s", z_t_magnitude)

        # O(n^2) cross-comparison metrics (print-only; the saved embeddings above are
        # unaffected). Skipped unless explicitly requested: each metric allocates an
        # n x n fp32 matrix, which is ~25 GB at n=80k.
        cc_limit = args.cross_comparison_sample_limit or 0
        if cc_limit == 0:
            logger.info(
                "\n=== Cross-comparison metrics skipped "
                "(--cross_comparison_sample_limit 0) ==="
            )
        else:
            n_cc = (
                len(z_p_tensor) if cc_limit < 0
                else min(cc_limit, len(z_p_tensor))
            )
            z_p_cc = z_p_tensor[:n_cc]
            z_t_cc = z_t_tensor[:n_cc]

            # Compute Dot Product scores
            dot_product_scores = torch.matmul(z_p_cc, z_t_cc.T)  # Dot product

            # Normalize scores into probabilities
            protein_given_text_probs = F.softmax(dot_product_scores, dim=0)  # Normalize across rows (proteins), for each text
            text_given_protein_probs = F.softmax(dot_product_scores, dim=1)  # Normalize across columns (texts), for each protein

            # Compute homology probabilities
            homology_matrix = compute_homology_matrix(z_p_cc)

            logger.info(
                "\n=== Cross-comparison subset: k=%d of %d samples ===",
                n_cc, len(z_p_tensor),
            )

            logger.info("\n=== Dot Product Scores Matrix ===")
            logger.info("%s", dot_product_scores)

            logger.info("\n=== Normalized Probabilities ===")
            logger.info("Protein-Normalized Probabilities (Softmax across Proteins for each Text):")
            logger.info("%s", protein_given_text_probs)

            logger.info("Text-Normalized Probabilities (Softmax across Texts for each Protein):")
            logger.info("%s", text_given_protein_probs)

            logger.info("\n=== Homology Matrix (Dot Product of Normalized z_p) ===")
            logger.info("%s", homology_matrix)

        logger.info("\n=== Example raw data elements ===")
        for k in range(min(len(acc_id_array), 3)):
            logger.info("  acc_id[%s]: %s", k, acc_id_array[k])
            logger.info("sequence[%s]: %s", k, protein_array[k])
    
        # Save output
        torch.save(embedding_dict, config_args_parser.output_path)

        # Write manifest
        elapsed = datetime.now() - start_time
        input_display = (
            os.path.abspath(args.input_data_path)
            if args.input_data_path.lower() != "none"
            else "None (test dataset)"
        )
        if _setup_logging:
            write_manifest(
                args, outdir, start_time, elapsed,
                outputs={
                    "num_samples": len(acc_id_array),
                    "embedding_dim": int(z_t_tensor.shape[1]),
                    "output_file": os.path.abspath(args.output_path),
                },
                resolved_paths={
                    "input_data_path": input_display,
                    "model_path": os.path.abspath(args.model_path),
                    "json_config": os.path.abspath(args.config_path),
                },
                config_contents=raw_config,
            )
            logger.info("Done in %s", elapsed)
    finally:
        teardown_file_logging("biom3", file_handler)


# Main Execution
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
