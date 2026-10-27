import torch
from torch.utils.data import random_split, Dataset, DataLoader, Subset, ConcatDataset
from torch.utils.data import default_collate
import pandas as pd
import random
import ast
import dask.dataframe as dd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
import psutil
import time
import copy
import re
import zlib
import json
import hashlib
from datetime import datetime

import numpy as np

import esm
from esm import pretrained
from transformers import AutoTokenizer, AutoModel

from biom3.backend.device import BACKEND_NAME, _XPU, get_device, setup_logger
from biom3.core._dist_env import get_global_rank

if BACKEND_NAME == _XPU:
    from lightning import LightningDataModule
else:
    from pytorch_lightning import LightningDataModule

logger = setup_logger(__name__)


# --- Per-batch stochastic annotation dropout for PenCL retraining ---

_FIELD_PREFIXES = [
    "PROTEIN NAME", "FUNCTION", "CATALYTIC ACTIVITY",
    "BIOPHYSICOCHEMICAL PROPERTIES", "LINEAGE", "FAMILY NAMES",
    "FAMILY NAME", "PARALOG NAME", "PARALOG FUNCTION", "SUBUNIT",
    "SUBCELLULAR LOCATION", "SIMILARITY", "DOMAIN",
    "ACTIVITY REGULATION", "PTM", "TISSUE SPECIFICITY",
    "MISCELLANEOUS", "COFACTOR", "PATHWAY", "BIOTECHNOLOGY",
    "INDUCTION",
]

_RETENTION_PROBS = {
    "PROTEIN NAME": 1.00,
    "FUNCTION": 0.85, "LINEAGE": 0.85,
    "PARALOG NAME": 0.65, "CATALYTIC ACTIVITY": 0.65,
    "FAMILY NAMES": 0.65, "FAMILY NAME": 0.65,
    "PATHWAY": 0.55, "DOMAIN": 0.55,
}
# All others default to 0.40

_FIELD_RE = re.compile(
    r'(?:^|\.\s+)(' + '|'.join(re.escape(p) for p in sorted(_FIELD_PREFIXES, key=len, reverse=True)) + r'):\s*',
    re.IGNORECASE
)


def apply_field_dropout(text, rng=None):
    """Randomly drop annotation fields from structured text.
    For NL text (no field prefixes), applies sentence-level dropout."""
    if rng is None:
        rng = random

    # If text doesn't contain field prefixes, apply sentence dropout
    if not any(f"{prefix}:" in text for prefix in _FIELD_PREFIXES[:3]):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        kept = [s for s in sentences if rng.random() < 0.7]
        return '. '.join(kept) + '.' if kept else text

    # Field-level dropout for structured text
    matches = list(_FIELD_RE.finditer(text))
    if not matches:
        return text

    fields = []
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip().rstrip('.')
        if content:
            fields.append((name, content))

    kept = []
    for name, content in fields:
        prob = _RETENTION_PROBS.get(name, 0.40)
        if rng.random() < prob:
            kept.append(f"{name}: {content}")

    result = '. '.join(kept) + '.' if kept else text
    return result


def check_available_memory():
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available
    available_memory_gb = available_memory / (1024 ** 3)
    return available_memory_gb


#########################################################
# BATCHED VERSION: Dataset iterator with masking tokens #
# Added by A Howe
#########################################################

# Caption padding modes, mapped to the HF tokenizer's own values.
#
# 'max_padding' pads every caption to text_max_length. That is what ALL BioM3
# training used, so it is the default and the in-distribution choice.
# 'dynamic' pads only to the longest caption in the batch.
#
# Without the caption attention mask the two are not interchangeable at
# inference: BERT then attends over every [PAD], so the padding length changes
# z_t. Under 'dynamic' the batch composition therefore leaks into z_t: the same
# caption embeds differently depending on what it was batched with, so results
# are not reproducible unless batch ordering is fixed. Measured on the
# 179,679-row SH3 corpus, a max-padded run and a dynamic-padded bank agree at
# z_t cosine median 0.982 with no row reaching parity, while z_p is
# bit-identical.
#
# With the mask (collate_fn(include_mask=True), inference's
# --text_attention_mask) the pads are ignored and both modes give the same z_t
# to float rounding, so 'dynamic' is safe and faster. Whether to pass the mask
# is a property of the weights: it must match how they were trained.
TEXT_PADDING_MODES = {"max_padding": "max_length", "dynamic": "longest"}
DEFAULT_TEXT_PADDING = "max_padding"


class BatchedTextSeqPairingDataset(Dataset):
    """
    Returns raw (text_caption, protein_sequence, accession_id).
    Tokenization is handled in collate_fn for batching.
    """

    def __init__(self, args, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        self.text_captions = df["[final]text_caption"].tolist()
        self.protein_sequences = df[args.sequence_keyword].tolist()
        self.accession_ids = df[args.id_keyword].tolist()

        self.text_max_length = args.text_max_length
        self.seq_max_length = 1024
        # getattr, not args.text_padding: callers that build this from a bare
        # config namespace (load_test_dataset) carry no such key.
        self.text_padding = getattr(args, "text_padding", DEFAULT_TEXT_PADDING)
        if self.text_padding not in TEXT_PADDING_MODES:
            raise ValueError(
                f"text_padding must be one of {sorted(TEXT_PADDING_MODES)}, "
                f"got {self.text_padding!r}"
            )

        # tokenizers (shared by collate_fn)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            args.text_model_path
        )
        _, self.sequence_tokenizer = pretrained.load_model_and_alphabet(
            args.seq_model_path
        )
        self.batch_converter = self.sequence_tokenizer.get_batch_converter()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.text_captions[idx],
            self.protein_sequences[idx],
            self.accession_ids[idx],
        )

# Set once per process the first time an over-length sequence is truncated.
_LONG_SEQUENCE_WARNED = False


def _warn_long_sequences_once(sequences, accessions, seq_max_length):
    """Warn once per process that a sequence was too long to embed intact.

    Stage 1 training pads to a fixed ``seq_max_length`` by concatenation
    (``torch.ones((1, 1024 - n))``), which raises on anything longer -- so the
    training data was filtered to ``seq_max_length - 2`` residues. Inference
    truncates instead, which also drops the EOS token, so these rows are
    embedded from a token pattern the model never saw.
    """
    global _LONG_SEQUENCE_WARNED
    if _LONG_SEQUENCE_WARNED:
        return
    limit = seq_max_length - 2  # BOS + EOS
    over = [(a, len(s)) for a, s in zip(accessions, sequences) if len(s) > limit]
    if not over:
        return
    _LONG_SEQUENCE_WARNED = True
    acc, length = over[0]
    logger.warning(
        "Sequence longer than %d residues encountered (%s: %d residues). It is "
        "truncated to %d tokens, dropping the EOS token. Stage 1 training was "
        "filtered to <= %d residues, so embedding may be out of distribution.",
        limit, acc, length, seq_max_length, limit,
    )


def collate_fn(
        batch, 
        dataset: BatchedTextSeqPairingDataset, 
        include_raw=False,
        include_mask=False,
):
    """Tokenize a batch. With include_mask the caption attention mask is
    appended as the last element of the returned tuple."""
    texts, sequences, accessions = zip(*batch)

    # -------- TEXT TOKENIZATION --------
    text_inputs = dataset.text_tokenizer(
        list(texts),
        truncation=True,
        max_length=dataset.text_max_length,
        padding=TEXT_PADDING_MODES[dataset.text_padding],
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    # -------- PROTEIN TOKENIZATION --------
    batch_sequences = list(zip(accessions, sequences))
    batch_labels, batch_strs, batch_tokens = dataset.batch_converter(batch_sequences)

    # truncate to max model length, but keep dynamic padding from batch_converter
    if batch_tokens.shape[1] > dataset.seq_max_length:
        _warn_long_sequences_once(sequences, accessions, dataset.seq_max_length)
        batch_tokens = batch_tokens[:, : dataset.seq_max_length]

    if include_raw:
        out = (
            text_inputs["input_ids"], 
            batch_tokens,
            list(texts),        # raw text captions
            list(sequences),    # raw protein sequences
            list(accessions),   # accession IDs
        )
    else:
        out = (text_inputs["input_ids"], batch_tokens)
    if include_mask:
        out += (text_inputs["attention_mask"],)
    return out

########################################
# Dataset iterator with masking tokens #
########################################

class TextSeqPairing_Dataset(Dataset):

    def __init__(self, args: any, df: pd.Series):

        # dataframe
        self.df = df
        self.length = self.df.shape[0]
        self.df_column_names = self.df.columns.tolist()
        self.protein_sequence_list = self.df[args.sequence_keyword].tolist()
        self.text_captions_list = self.df['[final]text_caption'].tolist()
        self.accession_id_list = self.df[args.id_keyword].tolist()

        # parameters
        self.text_max_length = args.text_max_length # max BERT sequence tokenization length
        self.seq_max_length = 1024 # max ESM model

        # tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path) # for text encoder
        _, self.sequence_tokenizer = pretrained.load_model_and_alphabet(args.seq_model_path) # for protein encoder

    def caption_tokenizer(self, batch_captions: list) -> dict:
        
        # transform input text tokens
        text_inputs = self.text_tokenizer(
                            batch_captions,
                            truncation=True,
                            max_length=self.text_max_length,
                            padding='max_length',
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=False
        )
        
        # track the original natural language captions
        text_inputs['orig_captions'] = batch_captions

        return text_inputs
    
    def protein_tokenizer(self, batch_sequences: list) -> dict:
        
        # perpare data for ESM
        batch_converter = self.sequence_tokenizer.get_batch_converter()
        batch_labels, batch_str, batch_tokens = batch_converter(batch_sequences)
        
        # pad sequences
        batch_tokens = torch.cat((
            batch_tokens,
            torch.ones((1,1024-batch_tokens.shape[1])),
            ), dim=-1
        )

        sequence_inputs = {
            'protein_sequence_labels': batch_labels, # UniProtKB id
            'protein_sequence_str': batch_str, # original protein sequence (in amino acids)
            'protein_sequence_tokens': batch_tokens.long() # training data
        }

        return sequence_inputs
    
    
    def __getitem__(self, idx: torch.Tensor) -> (
            dict,
            dict
        ):
        
        protein_sequence = self.protein_sequence_list[idx]
        text_captions = self.text_captions_list[idx]
        accession_id = self.accession_id_list[idx]

        # prepare protein sequence in ESM format (e.g. tuple: (header, sequence)):
        batch_sequences = [
            (accession_id, protein_sequence)
        ]
        
        text_data = self.caption_tokenizer(batch_captions=[text_captions])
        protein_data = self.protein_tokenizer(batch_sequences=batch_sequences)
 
        return (
                text_data['input_ids'],
                protein_data['protein_sequence_tokens']
        )

    def __len__(self):
        return self.length


class MaskTextSeqPairing_Dataset(Dataset):

    def __init__(self, args: any, df: pd.Series):

        # dataframe
        self.df = df
        self.length = self.df.shape[0]
        self.df_column_names = self.df.columns.tolist()
        self.protein_sequence_list = self.df[args.sequence_keyword].tolist()
        self.text_captions_list = self.df['[final]text_caption'].tolist()
        self.accession_id_list = self.df[args.id_keyword].tolist()

        # parameters
        self.text_max_length = args.text_max_length # max BERT sequence tokenization length
        self.seq_max_length = 1024 # max ESM model
        self.mask_prob = 0.15  # probability of masking

        # tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path) # for text encoder
        _, self.sequence_tokenizer = pretrained.load_model_and_alphabet(args.seq_model_path) # for protein encoder

    def caption_tokenizer(self, batch_captions: list) -> dict:

        # transform input text tokens
        text_inputs = self.text_tokenizer(
                            batch_captions,
                            truncation=True,
                            max_length=self.text_max_length,
                            padding='max_length',
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=False
        )

        # apply masking
        text_inputs_masked = self.apply_masking_to_text(text_inputs['input_ids'])
        text_inputs['input_ids_masked'] = text_inputs_masked

        # track the original natural language captions
        text_inputs['orig_captions'] = batch_captions

        return text_inputs

    def protein_tokenizer(self, batch_sequences: list) -> dict:

        # perpare data for ESM
        batch_converter = self.sequence_tokenizer.get_batch_converter()
        batch_labels, batch_str, batch_tokens = batch_converter(batch_sequences)

        # pad sequences
        batch_tokens = torch.cat((
            batch_tokens,
            torch.ones((1,1024-batch_tokens.shape[1])),
            ), dim=-1
        )

        # apply masking
        protein_sequence_tokens_masked = self.apply_masking_to_protein(batch_tokens)

        sequence_inputs = {
            'protein_sequence_labels': batch_labels, # UniProtKB id
            'protein_sequence_str': batch_str, # original protein sequence (in amino acids)
            'protein_sequence_tokens': batch_tokens.long(), # training data
            'protein_sequence_tokens_masked': protein_sequence_tokens_masked.long() # training data for masks
        }

        return sequence_inputs

    def apply_masking_to_text(self, tokens: torch.Tensor) -> torch.Tensor:

        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    self.text_tokenizer.cls_token_id,
                    self.text_tokenizer.sep_token_id,
                    self.text_tokenizer.pad_token_id,
                    self.text_tokenizer.mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:
                masked_tokens[0][i] = self.text_tokenizer.mask_token_id

            else:
                pass

        return masked_tokens


    def apply_masking_to_protein(self, tokens: torch.Tensor) -> torch.Tensor:
        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        # get the special token IDs
        cls_token_id = self.sequence_tokenizer.cls_idx
        sep_token_id = self.sequence_tokenizer.eos_idx
        pad_token_id = self.sequence_tokenizer.padding_idx
        unk_token_id = self.sequence_tokenizer.unk_idx
        mask_token_id = self.sequence_tokenizer.mask_idx

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    cls_token_id,
                    sep_token_id,
                    pad_token_id,
                    unk_token_id,
                    mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:

                masked_tokens[0][i] = self.sequence_tokenizer.mask_idx

            else:
                pass


        return masked_tokens


    def __getitem__(self, idx: torch.Tensor) -> (
            dict,
            dict
        ):

        protein_sequence = self.protein_sequence_list[idx]
        text_captions = self.text_captions_list[idx]
        accession_id = self.accession_id_list[idx]

        # prepare protein sequence in ESM format (e.g. tuple: (header, sequence)):
        batch_sequences = [
            (accession_id, protein_sequence)
        ]

        text_data = self.caption_tokenizer(batch_captions=[text_captions])
        protein_data = self.protein_tokenizer(batch_sequences=batch_sequences)

        return (
                text_data['input_ids'],
                protein_data['protein_sequence_tokens'],
                text_data['input_ids_masked'],
                protein_data['protein_sequence_tokens_masked']
        )


    def __len__(self):
        return self.length


#######################################################
# Dataset iterator with masking tokens + pfam dataset #
#######################################################


class Pfam_TextSeqPairing_Dataset(Dataset):

    def __init__(self, args: any, df: pd.Series, pfam_df: pd.Series):

        self.script_args = args
        # dataframe
        self.df = df
        self.length = self.df.shape[0]
        self.df_column_names = self.df.columns.tolist()
        self.protein_sequence_list = self.df[args.sequence_keyword].tolist()
        self.text_captions_list = self.df['[final]text_caption'].tolist()
        self.accession_id_list = self.df[args.id_keyword].tolist()
        self.pfam_labels_list = self.df['pfam_label'].tolist() # pfam labels found in the swiss-prot

        # Convert the strings into lists
        all_pf_codes = [ast.literal_eval(item) for item in self.pfam_labels_list]
        # Flatten the list of lists
        flat_pf_codes = [code for sublist in all_pf_codes for code in sublist]
        # Get unique PF codes in swiss-prot
        self.unique_pf_codes = list(set(flat_pf_codes))

        # protein family (pfam) database (over 40M sequences)
        self.grouped_pfam_df = pfam_df.groupby('pfam_label')

        pfam_group_keys = set(self.grouped_pfam_df.groups.keys())
        for label in self.unique_pf_codes:
            if label not in pfam_group_keys:
                assert f"Label {label} from swiss-prot is not in pfam_df!"
            else:
                pass

        # parameters
        self.text_max_length = args.text_max_length # max BERT sequence tokenization length
        self.seq_max_length = 1024 # max ESM model
        self.mask_prob = 0.15  # probability of masking

        # tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path) # for text encoder
        _, self.sequence_tokenizer = pretrained.load_model_and_alphabet(args.seq_model_path) # for protein encoder

    def caption_tokenizer(self, batch_captions: list) -> dict:

        # transform input text tokens
        text_inputs = self.text_tokenizer(
                            batch_captions,
                            truncation=True,
                            max_length=self.text_max_length,
                            padding='max_length',
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=False
        )

        # apply masking
        text_inputs_masked = self.apply_masking_to_text(text_inputs['input_ids'])
        text_inputs['input_ids_masked'] = text_inputs_masked

        # track the original natural language captions
        text_inputs['orig_captions'] = batch_captions

        return text_inputs

    def protein_tokenizer(self, batch_sequences: list) -> dict:

        # perpare data for ESM
        batch_converter = self.sequence_tokenizer.get_batch_converter()
        batch_labels, batch_str, batch_tokens = batch_converter(batch_sequences)

        # pad sequences
        batch_tokens = torch.cat((
            batch_tokens,
            torch.ones((1,1024-batch_tokens.shape[1])),
            ), dim=-1
        )

        # apply masking
        protein_sequence_tokens_masked = self.apply_masking_to_protein(batch_tokens)

        sequence_inputs = {
            'protein_sequence_labels': batch_labels, # UniProtKB id
            'protein_sequence_str': batch_str, # original protein sequence (in amino acids)
            'protein_sequence_tokens': batch_tokens.long(), # training data
            'protein_sequence_tokens_masked': protein_sequence_tokens_masked.long() # training data for masks
        }

        return sequence_inputs

    def apply_masking_to_text(self, tokens: torch.Tensor) -> torch.Tensor:

        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    self.text_tokenizer.cls_token_id,
                    self.text_tokenizer.sep_token_id,
                    self.text_tokenizer.pad_token_id,
                    self.text_tokenizer.mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:
                masked_tokens[0][i] = self.text_tokenizer.mask_token_id

            else:
                pass

        return masked_tokens


    def apply_masking_to_protein(self, tokens: torch.Tensor) -> torch.Tensor:

        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        # get the special token IDs
        cls_token_id = self.sequence_tokenizer.cls_idx
        sep_token_id = self.sequence_tokenizer.eos_idx
        pad_token_id = self.sequence_tokenizer.padding_idx
        unk_token_id = self.sequence_tokenizer.unk_idx
        mask_token_id = self.sequence_tokenizer.mask_idx

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    cls_token_id,
                    sep_token_id,
                    pad_token_id,
                    unk_token_id,
                    mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:

                masked_tokens[0][i] = self.sequence_tokenizer.mask_idx

            else:
                pass

        return masked_tokens


    def extraction_pfam_samples(self, pfam_labels: list):
        """
        Extracts pfam samples from the provided labels.

        :param pfam_labels: A list containing pfam labels.
        :return: A tuple containing accession_id, Xp_pfam, Xt_pfam, and bool_pfam_vector.
        """

        # The function assumes that 'nan' is not present in pfam_labels
        bool_pfam_vector = ['True']

        queried_pfam_label = random.choice(pfam_labels)  # Directly get a random element
        temp_df = self.grouped_pfam_df.get_group(queried_pfam_label)

        # Draw the homolog uniformly at random from the family, using the same
        # RNG as the family choice above. This used to be
        # temp_df.sample(n=1, random_state=self.script_args.seed), which builds
        # a fresh generator from the run seed on EVERY call and so returned the
        # same row for a given family every time: on rank 0 of a 24-rank mid
        # run only 4,294 of the shard's 143,077 Pfam rows (3%) were ever used.
        # Reproducibility comes from seeding once per run (run_PL_training
        # seeds random/numpy/torch), and DataLoader workers reseed `random`
        # per worker, so draws stay independent across workers too.
        sampled_row = temp_df.iloc[random.randrange(len(temp_df))]
        accession_id = str(sampled_row['id'])
        Xp_pfam = str(sampled_row['sequence'])
        Xt_pfam = str(sampled_row['[final]text_caption'])

        return (
            accession_id,
            Xp_pfam,
            Xt_pfam,
            bool_pfam_vector,
            queried_pfam_label
        )

    def __getitem__(self, idx: torch.Tensor) -> (
            dict,
            dict
        ):

        # retrieve data samples
        protein_sequence = self.protein_sequence_list[idx] # protein sequences
        text_captions = self.text_captions_list[idx] # text captions
        accession_id = self.accession_id_list[idx] # sequence id from UniProt
        pfam_labels = ast.literal_eval(self.pfam_labels_list[idx]) # protein family labels


        #############################
        # Sample Swiss-prot dataset #
        #############################

        # prepare protein sequence in ESM format (e.g. tuple: (header, sequence)):
        batch_sequences = [
            (accession_id, protein_sequence)
        ]
        # stochastic annotation dropout (PenCL retraining)
        text_captions = apply_field_dropout(text_captions)

        # get swiss-prot protein-text pairing
        text_data = self.caption_tokenizer(batch_captions=[text_captions])
        protein_data = self.protein_tokenizer(batch_sequences=batch_sequences)

        #######################
        # Sample Pfam dataset #
        #######################

        if 'nan' in pfam_labels:
            pfam_accession_id = ''
            pfam_protein_sequence = ''
            pfam_text_captions = ''
            bool_pfam_vector = ['False']
            pfam_text_data = {'input_ids': []}
            pfam_protein_data = {'protein_sequence_tokens': []}
            queried_pfam_label = ''
        else:
            pfam_accession_id, pfam_protein_sequence, pfam_text_captions, bool_pfam_vector, queried_pfam_label = self.extraction_pfam_samples(pfam_labels=pfam_labels)

        # stochastic annotation dropout for Pfam text too
        pfam_text_captions = apply_field_dropout(pfam_text_captions)

        # get pfam protein-text pairing samples...
        pfam_batch_sequences = [(pfam_accession_id, pfam_protein_sequence)]
        pfam_text_data = self.caption_tokenizer(batch_captions=[pfam_text_captions])
        pfam_protein_data = self.protein_tokenizer(batch_sequences=pfam_batch_sequences)

        # attention_mask marks real tokens vs the [PAD]s added to reach
        # text_max_length. The tokenizer already returns it
        # (return_attention_mask=True); it was previously discarded here, so
        # BERT attended over the padding and z_t carried a caption-length
        # signal. Captions stay padded to a fixed length -- the default collate
        # needs uniform shapes -- and the mask handles the pads.
        # False-negative keys, appended AFTER index 10 so collate_dynamic_text's
        # positional field constants keep pointing at the caption tensors.
        # The family key is the family this item was actually paired on
        # (extraction_pfam_samples' random choice), which is exactly what
        # L_PFC treats as the positive -- so it needs no set logic over
        # Swiss-Prot's multi-family label list.
        return (
                text_data['input_ids'],
                protein_data['protein_sequence_tokens'],
                text_data['input_ids_masked'],
                protein_data['protein_sequence_tokens_masked'],
                pfam_text_data['input_ids'],
                pfam_protein_data['protein_sequence_tokens'],
                pfam_text_data['input_ids_masked'],
                pfam_protein_data['protein_sequence_tokens_masked'],
                bool_pfam_vector,
                text_data['attention_mask'],
                pfam_text_data['attention_mask'],
                seq_hash64(protein_sequence),
                seq_hash64(pfam_protein_sequence),
                seq_hash64(queried_pfam_label)
        )


    def __len__(self):
        return self.length


######################
# Default DataModule #
######################


class Default_DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # construct dataset iterator
        dataset_options = {
                'default': TextSeqPairing_Dataset,
                'masked': MaskTextSeqPairing_Dataset,
                'pfam': Pfam_TextSeqPairing_Dataset,
                'pfam_ablated': Pfam_TextSeqPairing_Dataset
        }

        self.dataset_class = dataset_options.get(args.dataset_type, TextSeqPairing_Dataset)
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        if self.trainer is not None:
            logger.info("Number of GPUs: %s", self.trainer.world_size)
            logger.info("Current GPU index: %s", self.trainer.local_rank)

        # Load Swiss-Prot data
        df = self.load_swiss_prot()
        
        # Split the dataframe into train and valid sets
        train_df, valid_df = train_test_split(
            df,
            test_size=self.args.valid_size,
            random_state=self.args.seed
        )
 
        logger.info("Available memory after pfam_df: %s GB", check_available_memory())

        # Define datasets and dataloaders
        self.train_dataset = self.dataset_class(args=self.args, df=train_df)
        self.valid_dataset = self.dataset_class(args=self.args, df=valid_df)

    def load_swiss_prot(self) -> pd.Series:
        # Load and preprocess data (called on each GPU/TPU in DDP)
        logger.info('Load Swiss-Prot data...')

        # Load Swiss-Prot data
        df = pd.read_csv(os.path.expanduser(self.args.data_path))
        df = df[df['protein_sequence'].apply(lambda seq: len(seq) <= 1022)]

        return df

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
                pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True
        )

    def test_dataloader(self):
        # Define test dataloader if needed
        pass


###################
# Pfam DataModule #
###################


# Caption tensors in a Pfam_TextSeqPairing_Dataset item: Swiss-Prot input_ids,
# its MLM-masked copy, Pfam input_ids, its masked copy, and the two attention
# masks. Every item is tokenized to text_max_length; see collate_dynamic_text.
# ---------------------------------------------------------------------------
# False-negative keys.
#
# A contrastive batch contains pairs that are NOT negatives of each other but
# are scored as such: the same protein under a different caption (Swiss-Prot
# has ~8.9 caption variants per accession), and two items drawn on the same
# Pfam family (~25 per row at M = 49,152). The homolog pair (i, i +- N) is
# already handled by an index rule; these two need a key comparison instead.
#
# The keys are 64-bit digests so no global code table has to be built or kept
# in sync across ranks -- equality is all the mask needs, and blake2b is stable
# across processes and runs (unlike hash()). Collision probability over the
# 19,471 families and ~31 M accessions is ~1e-11.
#
# NO_KEY marks a row that must never match anything: the 'nan'-label rows,
# whose Pfam side is an empty string, would otherwise all look identical.
# seq_hash64 matches jobs/loss_trace/trace_stage1_loss.py so the diagnostic and
# the trainer agree on what counts as a duplicate.
NO_KEY = 0


def seq_hash64(s: str) -> int:
    """Signed 64-bit digest of a string, for equality tests only."""
    if not s:
        return NO_KEY
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "big", signed=True)


_PFAM_ITEM_TEXT_FIELDS = (0, 2, 4, 6, 9, 10)
_PFAM_ITEM_ATTN_FIELDS = (9, 10)
DYNAMIC_PAD_MULTIPLE = 64


def collate_dynamic_text(batch, multiple=DYNAMIC_PAD_MULTIPLE):
    """Collate, then crop every caption tensor to the batch's longest caption.

    Items are still tokenized to text_max_length (512), so default_collate can
    stack them; the columns past the longest real caption are all [PAD] and are
    dropped here. That is exactly padding='longest', and since BERT is given the
    attention mask, the embedding is unchanged to float rounding (the padding
    invariance test). MLM masking never touches [PAD], so no masked token is
    lost either. The length is rounded up to `multiple` so the XPU kernels see
    at most 512/multiple distinct text shapes instead of one per length.

    Swiss-Prot and Pfam captions are cropped to one common length because the
    MLM forward concatenates them along the batch dimension.
    """
    out = list(default_collate(batch))
    attn = torch.cat([out[i] for i in _PFAM_ITEM_ATTN_FIELDS], dim=0)
    longest = int(attn.sum(dim=-1).max())
    length = min(attn.shape[-1], -(-longest // multiple) * multiple)
    for i in _PFAM_ITEM_TEXT_FIELDS:
        out[i] = out[i][..., :length]
    return out


PFAM_SPLITS_MANIFEST = "pfam_splits_manifest.json"


def _pfam_source_fingerprint(pfam_data_path):
    st = os.stat(pfam_data_path)
    return {"source_path": os.path.abspath(pfam_data_path),
            "source_size": st.st_size, "source_mtime": st.st_mtime}


def pfam_splits_manifest_matches(splits_dir, pfam_data_path, num_shards):
    """True if splits_dir has a manifest written from this Pfam file for this
    world size; False if it has no manifest. A manifest that does NOT match
    raises. Reads one small file and stats the source, nothing else, so every
    rank can afford to call it.
    """
    path = os.path.join(splits_dir, PFAM_SPLITS_MANIFEST)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        manifest = json.load(fh)
    want = dict(_pfam_source_fingerprint(pfam_data_path), num_shards=num_shards)
    mismatch = {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}
    if mismatch:
        raise ValueError(
            f"{path} does not match this run (manifest, run): {mismatch}. "
            "Refusing to reuse or overwrite it; point pfam_splits_dir elsewhere.")
    return True


def _configured_world_size(args):
    """num_nodes * devices_per_node from the run's settings, or None if unset."""
    try:
        world = int(args.num_nodes) * int(args.devices_per_node)
    except (AttributeError, TypeError, ValueError):
        return None
    return world if world > 0 else None


def pfam_splits_status(splits_dir, pfam_data_path, num_shards):
    """'reuse' if splits_dir holds a complete set of shards written from this
    Pfam file for this world size; 'write' if it holds no manifest.

    A manifest that does NOT match raises instead of regenerating: a pre-sharded
    directory is never overwritten with a different layout behind your back.
    """
    if not pfam_splits_manifest_matches(splits_dir, pfam_data_path, num_shards):
        return "write"
    missing = [ii for ii in range(num_shards)
               if not os.path.exists(os.path.join(splits_dir, f"split_pfam_rank_{ii}.csv"))]
    if missing:
        raise ValueError(f"{splits_dir}: manifest present but {len(missing)} shard(s) "
                         f"missing, first is rank {missing[0]}")
    return "reuse"


def write_pfam_splits(pfam_df, splits_dir, num_shards, pfam_data_path):
    """One shard per rank (row i -> rank i % num_shards), then the manifest.

    The shard loop is unchanged from prepare_data, so a pre-written directory is
    byte-for-byte what training would have written itself. The manifest is
    written last, so an interrupted write is never mistaken for a complete one.
    """
    os.makedirs(splits_dir, exist_ok=True)
    assignments = np.arange(len(pfam_df)) % num_shards
    for ii in range(num_shards):
        split_df = pfam_df.iloc[assignments == ii]
        split_df.to_csv(f"{splits_dir}/split_pfam_rank_{ii}.csv", index=False)
        if ii == 0 or ii == num_shards - 1:
            logger.info("  Split %s: %s rows", ii, len(split_df))
    manifest = dict(_pfam_source_fingerprint(pfam_data_path), num_shards=num_shards,
                    total_rows=int(len(pfam_df)),
                    created=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(splits_dir, PFAM_SPLITS_MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=1)
    logger.info("Saved %s Pfam splits to %s/", num_shards, splits_dir)


def split_by_accession(df, id_col, valid_size, seed):
    """Train/valid split in which every row of an accession lands on ONE side.

    Swiss-Prot has ~8.9 caption variants per accession (5,051,255 rows over
    569,516 accessions). The previous split, train_test_split over rows, sent
    variants of the same protein to both sides; and because it ran on each
    rank's shard-filtered table, the same row could be train on one rank and
    valid on another. Neither validation set was held out.

    The side is a pure function of (seed, accession): crc32 is stable across
    processes, unlike hash(). So every rank agrees, whatever rows its Pfam
    shard lets it keep, and whatever order they arrive in.
    """
    threshold = int(valid_size * 2**32)
    accessions = df[id_col].astype(str)
    valid_accessions = {
        a for a in accessions.unique()
        if zlib.crc32(f"{seed}:{a}".encode()) < threshold
    }
    is_valid = accessions.isin(valid_accessions)
    return df[~is_valid], df[is_valid]


def equalize_across_ranks(train_df, valid_df, seed, device=None):
    """Trim every rank's train and valid frames to the smallest size on any rank.

    Each rank keeps only the Swiss-Prot rows whose Pfam families all appear in
    its own shard. Rare families reach only some ranks, so on the full dataset
    the per-rank tables differ in length, Lightning's DistributedSampler hands
    each rank ceil(len/W) samples, and ranks run different numbers of steps per
    epoch. The first rank to finish goes on to validation and checkpointing
    while the others are still issuing training collectives: a hang, or
    mismatched collectives. The capped mid set never showed it (every family
    reaches every rank) and every study run was capped at 20 steps.

    Shuffles before trimming so the rows dropped are random, not the tail.
    Returns the trimmed frames and a [W, 2] tensor of the original per-rank
    (train, valid) sizes.
    """
    import torch.distributed as dist
    sizes = [len(train_df), len(valid_df)]
    if not (dist.is_available() and dist.is_initialized()):
        return train_df, valid_df, torch.tensor([sizes])
    # float32 so this is the same all_gather the training step already uses on
    # xccl; row counts are exact in float32 below 2**24.
    assert max(sizes) < 2**24, sizes
    local = torch.tensor(sizes, dtype=torch.float32, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    per_rank = torch.stack(gathered).to(torch.long).cpu()
    n_train, n_valid = per_rank.min(dim=0).values.tolist()
    train_df = train_df.sample(frac=1.0, random_state=seed).iloc[:n_train]
    valid_df = valid_df.sample(frac=1.0, random_state=seed).iloc[:n_valid]
    return train_df, valid_df, per_rank


class Pfam_DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # construct dataset iterator
        dataset_options = {
                'default': TextSeqPairing_Dataset,
                'masked': MaskTextSeqPairing_Dataset,
                'pfam': Pfam_TextSeqPairing_Dataset,
                'pfam_ablated': Pfam_TextSeqPairing_Dataset
        }

        self.dataset_class = dataset_options.get(args.dataset_type, TextSeqPairing_Dataset)

        self.OOD_pfam_labels = [
                'PF18369', # Polyketide synthase dimerisation element domain
                'PF04680', # Opioid growth factor receptor repeat
                'PF17988', # VEGFR-2 Transmembrane domain
                'PF12325', # TATA element modulatory factor 1 TATA binding
                'PF03272', # Putative mucin or carbohydrate-binding module
                'PF03938', # Outer membrane protein (OmpH-like)
                'PF17724', # Family of unknown function (DUF5568)
                'PF10696', # Protein of unknown function
                'PF11968', # 25S rRNA (adenine(2142)-N(1))-methyltransferase, Bmt2
                'PF04153' # NOT2/NOT3/NOT5 C-terminal
        ]

        # With pre-written shards (biom3.Stage1.preshard_pfam) prepare_data has
        # nothing to do, so don't expose it. Lightning wraps any overridden
        # prepare_data in _InfiniteBarrier, which builds a Gloo process group
        # over EVERY rank. Gloo connects a full mesh, W^2 connections: 37 s at
        # 768 ranks, and at 3,072 it failed 30 min in with
        # DistNetworkError: Connection reset by peer (run2a, job 8816887).
        # Binding the base-class method makes Lightning's is_overridden() False,
        # so the group is never created. Every rank decides from the manifest
        # alone (one small file, no collective), so all ranks agree; the run2
        # job script has already checked every shard exists before launch.
        self.pfam_splits_prewritten = False
        world = _configured_world_size(args)
        if world is not None and pfam_splits_manifest_matches(
                self._resolve_splits_dir(), args.pfam_data_path, world):
            self.prepare_data = LightningDataModule.prepare_data.__get__(self)
            self.pfam_splits_prewritten = True
            if get_global_rank() == 0:
                logger.info("Pre-written Pfam splits in %s match W=%s; Lightning's "
                            "prepare_data step (and its all-rank Gloo barrier) is skipped",
                            self._resolve_splits_dir(), world)

    def _resolve_splits_dir(self) -> str:
        # Deviation from Rama's layout: prefer a user-specified splits dir so that
        # shard writes land inside the run directory instead of next to the input CSV.
        override = getattr(self.args, 'pfam_splits_dir', None)
        if override and str(override).lower() != 'none':
            return os.path.expanduser(str(override))
        directory_path = os.path.dirname(self.args.pfam_data_path)
        return f"{directory_path}/pfam_temp_splits"

    # Only global rank 0 writes the shards. The default (True) runs prepare_data
    # on local rank 0 of EVERY node, so on 2+ nodes several ranks would write the
    # same split_pfam_rank_*.csv files concurrently. The splits live on shared
    # Lustre, so one writer is both correct and sufficient.
    prepare_data_per_node = False

    def prepare_data(self):
        """Write one pfam shard per rank. Runs on global rank 0 ONLY.

        Deliberately contains NO collectives. Lightning calls this hook on a
        single rank (see _DataConnector.prepare_data) and then everyone meets at
        `strategy.barrier("pre_setup")` before any setup() runs, so the shards
        are guaranteed on disk before any rank reads them.

        An earlier version called dist.barrier() here. That worked only because
        Stage 1 ran under SingleDeviceStrategy, where every rank is its own world
        of one and therefore every rank ran prepare_data. Under real DDP it
        deadlocks: rank 0 waits on a barrier the other 23 ranks never reach.
        """
        import torch.distributed as dist

        # Real distributed world size: this must match the number of shards the
        # ranks will later look for in setup().
        if dist.is_initialized():
            num_gpus = dist.get_world_size()
        else:
            num_gpus = self.trainer.world_size if self.trainer else 1

        splits_dir = self._resolve_splits_dir()

        # Shards written ahead of time (biom3.Stage1.preshard_pfam) are reused:
        # at 3,072 ranks writing them here costs ~87 min of every job while
        # every other rank waits.
        if pfam_splits_status(splits_dir, self.args.pfam_data_path, num_gpus) == "reuse":
            logger.info("Reusing %s pre-written Pfam splits in %s (manifest matches)",
                        num_gpus, splits_dir)
            return

        logger.info('Upload Pfam Database and split it over %s dataframes', num_gpus)
        # Load Swiss-Prot data
        df = self.load_swiss_prot()

        # Load Pfam data
        pfam_df = self.load_pfam_database()

        # Ensure Pfam labels match with Swiss-Prot
        pfam_unique_labels = set(pfam_df['pfam_label'].tolist())
        df = df[df['pfam_label'].apply(lambda x: all(label in pfam_unique_labels for label in ast.literal_eval(x)))]

        # Fast modular split: assign each row to a rank based on index
        # This is ~100x faster than stratified_split on 44M rows
        write_pfam_splits(pfam_df, splits_dir, num_gpus, self.args.pfam_data_path)

        # After saving the splits to disk
        del pfam_df, df
        gc.collect()
        # No barrier here: Lightning's strategy.barrier("pre_setup") runs between
        # prepare_data and setup, and only this rank executes prepare_data.

    def setup(self, stage=None):

        import torch.distributed as dist
        if dist.is_initialized():
            logger.info("Number of GPUs: %s", dist.get_world_size())
            logger.info("Current GPU index: %s", dist.get_rank())
        elif self.trainer is not None:
            logger.info("Number of GPUs: %s", self.trainer.world_size)
            logger.info("Current GPU index: %s", self.trainer.local_rank)

        # Load Swiss-Prot data
        df = self.load_swiss_prot()

        # Load Pfam data
        # Determine the GPU index — use real distributed rank for mpiexec mode
        if dist.is_initialized():
            gpu_idx = dist.get_rank()
        else:
            gpu_idx = self.trainer.local_rank if self.trainer else 0

        logger.info("Available memory before pfam_df: %s GB", check_available_memory())

        # Load the corresponding split pfam_df for this GPU
        splits_dir = self._resolve_splits_dir()
        columns_to_extract = ['id', 'pfam_label', 'sequence', '[final]text_caption']
        pfam_df = dd.read_csv(
                f"{splits_dir}/split_pfam_rank_{gpu_idx}.csv",
                dtype={6: 'str'},
                usecols=columns_to_extract
        ).compute()

        logger.info('Finished loading Pfam data...')
        pfam_df = pfam_df[pfam_df['sequence'].apply(lambda seq: len(seq) <= 1022)]

        # Ensure Pfam labels match with Swiss-Prot
        pfam_unique_labels = set(pfam_df['pfam_label'].tolist())
        df = df[df['pfam_label'].apply(lambda x: all(label in pfam_unique_labels for label in ast.literal_eval(x)))]

        # Split by accession, identically on every rank (see split_by_accession).
        train_df, valid_df = split_by_accession(
            df, self.args.id_keyword, self.args.valid_size, self.args.seed)

        # Every rank must run the same number of steps (see equalize_across_ranks).
        device = self.trainer.strategy.root_device if self.trainer is not None else None
        train_df, valid_df, per_rank = equalize_across_ranks(
            train_df, valid_df, self.args.seed, device=device)
        if gpu_idx == 0:
            for col, name, kept in ((0, "train", len(train_df)), (1, "valid", len(valid_df))):
                sizes = per_rank[:, col]
                logger.info(
                    "Swiss-Prot %s rows per rank after shard filtering: min %d, max %d, "
                    "mean %.0f over %d ranks; every rank uses %d (trims up to %d rows, %.2f%%)",
                    name, sizes.min().item(), sizes.max().item(), sizes.float().mean().item(),
                    len(sizes), kept, sizes.max().item() - kept,
                    100.0 * (sizes.max().item() - kept) / max(1, sizes.max().item()))

        logger.info("Available memory after pfam_df: %s GB", check_available_memory())

        # Define datasets and dataloaders
        self.train_dataset = self.dataset_class(args=self.args, df=train_df, pfam_df=pfam_df)
        self.valid_dataset = self.dataset_class(args=self.args, df=valid_df, pfam_df=pfam_df)

    def load_swiss_prot(self) -> pd.Series:
        # Load and preprocess data (called on each GPU/TPU in DDP)
        logger.info('Load Swiss-Prot data...')

        # Load Swiss-Prot data
        df = pd.read_csv(os.path.expanduser(self.args.data_path))
        df = df[df['protein_sequence'].apply(lambda seq: len(seq) <= 1022)]

        # remove OOD Test samples
        logger.info('Removing SwissProt OOD samples:')
        logger.info('SwissProt size: %s', df.shape[0])
        for ii, label in enumerate(self.OOD_pfam_labels):
            df = df[~df['pfam_label'].str.contains(label)]
            logger.info('SwissProt Size: %s', df.shape[0])

        logger.info('-' * 20)

        return df


    def load_pfam_database(self) -> pd.Series:

        logger.info('Load Pfam data...')
        # Step 1: Load Pfam dataset
        columns_to_extract = ['id', 'pfam_label', 'sequence', '[final]text_caption']
        pfam_df = dd.read_csv(self.args.pfam_data_path, dtype={6: 'str'}, usecols=columns_to_extract).compute()

        logger.info('Finished loading Pfam data with size %s...', pfam_df.shape[0])
        pfam_df = pfam_df[pfam_df['sequence'].apply(lambda seq: len(seq) <= 1022)]

        # remove OOD Test samples
        logger.info('Removing Pfam OOD samples:')
        logger.info('Pfam Size: %s', pfam_df.shape[0])
        for ii, label in enumerate(self.OOD_pfam_labels):
            pfam_df = pfam_df[~pfam_df['pfam_label'].str.contains(label)]
            logger.info('Pfam size: %s', pfam_df.shape[0])
        logger.info('-' * 20)

        return pfam_df


    def stratified_split(
            self,
            df: pd.Series,
            label_col: str,
            num_splits: int
            ) -> list:


        # count the number of instances for each class
        class_counts = df[label_col].value_counts()

        # Find classes that have only one instance
        small_classes = class_counts[class_counts < self.args.num_gpus].index

        # Duplicate the rows of singleton classes
        duplicated_rows = df[df['pfam_label'].isin(small_classes)]

        # Initialize an empty list to hold the smaller dfs
        smaller_dfs = []

        # Calculate the test size for each split
        test_size = 1.0 / num_splits

        # Duplicate rows for small classes so that each has at least num_gpus instances
        for small_class in tqdm(small_classes):

            small_class_df = df[df['pfam_label'] == small_class]
            num_duplications = self.args.num_gpus - small_class_df.shape[0]
            duplicated_small_class_df = pd.concat([small_class_df] * num_duplications, ignore_index=True)
            duplicated_rows = pd.concat([duplicated_rows, duplicated_small_class_df])


        # Append duplicated rows to the original DataFrame
        df = pd.concat([df, duplicated_rows], ignore_index=True)
        for ii in range(num_splits - 1):
            # Perform the stratified split
            train, test = train_test_split(df, stratify=df[label_col], test_size=test_size, random_state=self.args.seed)

            # Append the test (smaller set) to the list
            smaller_dfs.append(test)

            # Update df to be the remaining larger set for the next iteration
            df = train

            # Update test_size for the next iteration
            test_size = 1.0 / (num_splits - (ii + 1))

        # Append the last remaining set
        smaller_dfs.append(df)

        logger.info('Pfam database splits: %s', [len(temp_df) for temp_df in smaller_dfs])

        return smaller_dfs


    def _collate_fn(self):
        # 'dynamic' crops each batch's captions to its longest (rounded up to a
        # multiple of 64); 'max_padding' keeps every caption at text_max_length.
        mode = getattr(self.args, 'text_padding', DEFAULT_TEXT_PADDING)
        if mode not in TEXT_PADDING_MODES:
            raise ValueError(f"text_padding must be one of {sorted(TEXT_PADDING_MODES)}, got {mode!r}")
        return collate_dynamic_text if mode == 'dynamic' else None

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
                pin_memory=True,
                collate_fn=self._collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=self._collate_fn(),
        )

    def test_dataloader(self):
        # Define test dataloader if needed
        pass



################################
# Facilitator Dataset Iterator #
################################


class Facilitator_Dataset(Dataset):

    def __init__(self, args: any, dataset: dict):

        device = get_device() if args.num_gpus >= 1 else torch.device('cpu')

        # Check if text_embeddings is a list and convert to a tensor
        if isinstance(dataset['text_embedding'], list):
            # Convert list elements to tensors if they are not already
            text_emb_tensors = [torch.tensor(emb).to(device) if not isinstance(emb, torch.Tensor) else emb.to(device) for emb in dataset['text_embedding']]
            # Stack the list of tensors
            self.text_embeddings = torch.stack(text_emb_tensors)
        else:
            self.text_embeddings = dataset['text_embedding'].to(device)

        # Check if protein_embeddings is a list and convert to a tensor
        if isinstance(dataset['protein_embedding'], list):
            # Convert list elements to tensors if they are not already
            protein_emb_tensors = [torch.tensor(emb).to(device) if not isinstance(emb, torch.Tensor) else emb.to(device) for emb in dataset['protein_embedding']]
            # Stack the list of tensors
            self.protein_embeddings = torch.stack(protein_emb_tensors)
        else:
            self.protein_embeddings = dataset['protein_embedding'].to(device)


    def __getitem__(self, idx: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):


        z_t = self.text_embeddings[idx]
        z_p = self.protein_embeddings[idx] 

        return (
                z_t,
                z_p
        )


    def __len__(self):
        return len(self.text_embeddings)

###########################
# Facilitator Data Module #
###########################



class Facilitator_DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
       
        self.OOD_pfam_labels = [
                'PF18369', # Polyketide synthase dimerisation element domain
                'PF04680', # Opioid growth factor receptor repeat
                'PF17988', # VEGFR-2 Transmembrane domain
                'PF12325', # TATA element modulatory factor 1 TATA binding
                'PF03272', # Putative mucin or carbohydrate-binding module
                'PF03938', # Outer membrane protein (OmpH-like)
                'PF17724', # Family of unknown function (DUF5568)
                'PF10696', # Protein of unknown function
                'PF11968', # 25S rRNA (adenine(2142)-N(1))-methyltransferase, Bmt2
                'PF04153' # NOT2/NOT3/NOT5 C-terminal
        ]
        

        # prepare embeddings
        #self.embedding_data = torch.load(args.swissprot_data_path)
        # dataset iterator
        #dataset = Facilitator_Dataset(args=args, dataset=self.embedding_data)
        # create a clone of the dataset
        #cloned_dataset = copy.deepcopy(dataset)

        # Get indices and split them
        #indices = list(range(len(dataset)))
        #train_indices, valid_indices = train_test_split(indices, test_size=args.valid_size, random_state=args.seed)
        
        # create full dataloader
        #self.all_dataloader = DataLoader(cloned_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create PyTorch DataLoader using the indices
        #self.train_sampler = Subset(dataset, train_indices)
        #self.valid_sampler = Subset(dataset, valid_indices)
        #train_dataloader = DataLoader(train_sampler, batch_size=args.batch_size, shuffle=True)
        #valid_dataloader = DataLoader(test_sampler, batch_size=args.batch_size, shuffle=False)
    
        ##########################################
        # Load Stage 1 SwissProt+Pfam Embeddings #
        ##########################################
    
        # initialize the embedding data to None
        self.swissprot_data, self.pfam_data = None, None
    
        # get both the swissprot and pfam dataset iterator in one
        if (args.swissprot_data_path != 'None') and (args.pfam_data_path != 'None'):
            logger.info('Load both SwissProt and Pfam dataset...')
            self.train_dataset, self.valid_dataset, self.all_swiss_dataloader, self.all_pfam_dataloader = self.load_both()

        # get the swissprot dataset iterator
        elif args.pfam_data_path == 'None':
            logger.info('Load SwissProt dataset...')
            self.train_dataset, self.valid_dataset, self.all_swiss_dataloader = self.load_swissprot()
            self.all_pfam_dataloader = None

        # get the pfam dataset iterator 
        elif args.swissprot_data_path == 'None':
            logger.info('Load Pfam dataset...')
            self.train_dataset, self.valid_dataset, self.all_pfam_dataloader = self.load_pfam()
            self.all_swiss_dataloader = None
            


    def load_swissprot(self):

        # prepare embeddings
        self.swissprot_data = torch.load(self.args.swissprot_data_path)
        
        # dataset iterator
        swiss_dataset = Facilitator_Dataset(args=self.args, dataset=self.swissprot_data)      
        # create a clone of the dataset
        cloned_swiss_dataset = copy.deepcopy(swiss_dataset)

        # Get indices and split them
        indices = list(range(len(swiss_dataset)))
        train_indices, valid_indices = train_test_split(indices, test_size=self.args.valid_size, random_state=self.args.seed)
        
        # Create Pytorch iterator using the indices
        swiss_train_subset = Subset(swiss_dataset, train_indices)
        swiss_valid_subset = Subset(swiss_dataset, valid_indices)

        # Create Pytorch dataloader on all samples
        swiss_all_dataloader = DataLoader(cloned_swiss_dataset, batch_size=self.args.batch_size, shuffle=False)

        
        return (
                swiss_train_subset,
                swiss_valid_subset,
                swiss_all_dataloader
        )
    
    
    def load_pfam(self):

        # prepare embeddings
        self.pfam_data = torch.load(self.args.pfam_data_path)
        
        # dataset iterator
        pfam_dataset = Facilitator_Dataset(args=self.args, dataset=self.pfam_data)      
        # create a clone of the dataset
        cloned_pfam_dataset = copy.deepcopy(pfam_dataset)

        # Get indices and split them
        indices = list(range(len(pfam_dataset)))
        train_indices, valid_indices = train_test_split(indices, test_size=self.args.valid_size, random_state=self.args.seed)
        
        # Create Pytorch Dataloader using the indices
        pfam_train_subset = Subset(pfam_dataset, train_indices)
        pfam_valid_subset = Subset(pfam_dataset, valid_indices)

        # Create Pytorch dataloader on all samples
        pfam_all_dataloader = DataLoader(cloned_pfam_dataset, batch_size=self.args.batch_size, shuffle=False)

        return (
                pfam_train_subset,
                pfam_valid_subset,
                pfam_all_dataloader
        )
    

    def load_both(self):

        # get swissprot
        swissprot_train_subset, swissprot_valid_subset, swissprot_all_dataloader = self.load_swissprot()

        # get pfam
        pfam_train_subset, pfam_valid_subset, pfam_all_dataloader = self.load_pfam()
    
        # combined subsets 
        combined_train_subset = ConcatDataset([swissprot_train_subset, pfam_train_subset])
        combined_valid_subset = ConcatDataset([swissprot_valid_subset, pfam_valid_subset])

        return (
                combined_train_subset,
                combined_valid_subset,
                swissprot_all_dataloader,
                pfam_all_dataloader
        )


    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                #self.train_sampler,
                batch_size=self.args.batch_size,
                #num_workers=self.args.num_workers,
                shuffle=True,
                #pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset,
                #self.valid_sampler,
                batch_size=self.args.batch_size,
                #num_workers=self.args.num_workers,
                #pin_memory=True
        )

    def test_dataloader(self):
        # Define test dataloader if needed
        pass


