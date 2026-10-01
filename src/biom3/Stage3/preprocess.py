# Standard libraries
import os
import numpy as np
import random
from pathlib import Path
import pandas as pd

# Data handling
import h5py
from typing import Dict, List, Tuple, Union, Optional, Any

# PyTorch essentials
import torch
from torch.utils.data import Dataset, DataLoader

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

# If needed for sequence processing
from collections import Counter
import string
import re

# Multiprocessing for parallel preprocessing (optional)
import multiprocessing as mp
from functools import partial

# Progress tracking
from tqdm import tqdm

# JSON line parsing for finetuning datasets
import json

def get_mnist_dataset(args: any) -> DataLoader:
    """
    Create a DataLoader for the MNIST dataset with appropriate transformations.
    
    This function prepares the MNIST dataset in one of two formats:
    1. 'normal' mode: Binary 2D images resized to args.image_size
    2. 'sequence' mode: Flattened 1D binary sequences created from images
    
    Both modes apply binarization by thresholding pixel values at 0.5.
    
    Args:
        args: Configuration object containing:
            - dataset: Mode selection ('normal' or 'sequence')
            - data_root: Path to store/find MNIST data
            - image_size: Target size for resizing images
            - workers: Number of dataloader workers
            - batch_size: Batch size for training
            - download: Whether to download dataset if not present
            
    Returns:
        DataLoader configured with the transformed MNIST dataset
        
    Raises:
        SystemExit: If dataset parameter is not 'normal' or 'sequence'
    """
    if args.dataset == 'normal':
        logger.debug("download=%s", args.download)
        transform = Compose([ToTensor(), Resize(args.image_size), lambda x: x > 0.5])
        train_dataset = MNIST(root=args.data_root, download=True, transform=transform, train=True)
        train_dataloader = DataLoader(
                train_dataset,
                num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True
        )
    elif args.dataset == 'sequence':
        transform = Compose([ToTensor(), Resize(args.image_size), lambda x: x > 0.5, T.Lambda(lambda x: torch.flatten(x).unsqueeze(0))])
        train_dataset = MNIST(root=args.data_root, download=True, transform=transform, train=True)
        train_dataloader = DataLoader(
                train_dataset,
                num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True
        )
    else:
        logger.error('Please pick either normal or sequence')
        quit()
    return train_dataloader




"""Protein sequence preprocessing utilities for tokenization and padding."""

#@jit(nopython=True)
def pad_ends(
        seqs: list,
        max_seq_length: int
    ) -> list:
    """
    Pad a protein sequence to reach a specified maximum length.

    This function adds gap characters ('-') to the end of a sequence
    until it reaches the target length. This ensures uniform sequence
    length for batch processing.

    Args:
        seqs: A protein sequence represented as a list of amino acid characters
        max_seq_length: The target length to pad the sequence to

    Returns:
        A padded sequence with length equal to max_seq_length
    """
    padded_seqs = [] # add padded gaps at the end of each sequence
    #for seq in seqs:
    seq_length = len(seqs)
    # number of padded tokens
    pad_need = max_seq_length - seq_length
    # add number of padded tokens to the end
    padded_seqs = seqs + ['-']*pad_need
    return padded_seqs

def create_num_seqs(seq_list: list) -> list:
    """
    Convert a protein sequence from amino acid letters to numerical indices.

    This function maps each amino acid in the input sequence to a unique
    numerical index based on a predefined vocabulary. The vocabulary includes
    standard amino acids, special tokens (<START>, <END>), padding characters,
    and accommodates rare/nonstandard amino acids (X, U, Z, B, O).

    Args:
        seq_list: A protein sequence represented as a list of amino acid characters

    Returns:
        A list of integers representing the encoded sequence
    """
    # tokenizer
    tokens = ['*', '<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>', '-']
    #tokens = [ '<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>', '-']
    # needed to lose these to the token list
    tokens = tokens + ['X', 'U', 'Z', 'B', 'O']
    token2int = {x:ii for ii, x in enumerate(tokens)}
    # empty list to hold num rep. seqs.
    num_seq_list = [token2int[aa] for aa in seq_list]
    return num_seq_list

        
def prepare_protein_data(
        args: any,
        data_dict: dict
    ) -> (
            list,
            list
    ):
    """
    Prepare protein sequences and their corresponding text embeddings for model training.
    
    This function processes raw protein sequences by:
    1. Removing gap characters (-)
    2. Adding <START> and <END> tokens to each sequence
    3. Filtering sequences that exceed the maximum length (determined by diffusion_steps)
    4. Padding sequences to a uniform length for batch processing
    5. Converting sequences to numerical representation
    6. Extracting corresponding text embeddings based on the facilitator type
    
    Args:
        args: Configuration object containing:
            - sequence_keyname: Key to access sequences in data_dict
            - diffusion_steps: Maximum allowed sequence length
            - image_size: Used to calculate target padded length (image_size^2)
            - facilitator: Type of embedding to use ('MSE', 'MMD', or 'Default')
        data_dict: Dictionary containing sequences and various embedding types
        
    Returns:
        tuple: (
            num_seq_list: List of numerically encoded, filtered, and padded sequences
            text_emb: List of text embeddings corresponding to the filtered sequences
        )
        
    Raises:
        ValueError: If an unsupported facilitator type is provided
    """
    logger.debug("Data keys: %s", list(data_dict.keys()))
    logger.info('Prepare dataset')
    # prepare sequences
    seq_list = [seq.replace('-','') for seq in data_dict[args.sequence_keyname]]
    seq_list = [['<START>'] + list(seq) + ['<END>'] for seq in seq_list]
    seq_lens = [len(seq) for seq in seq_list]
    # Determine the maximum sequence length based on context window size
    max_seq_len = int(args.diffusion_steps)
    # Get indices of sequences that meet the criteria
    valid_indices = [i for i, seq in enumerate(seq_list) if len(seq) <= max_seq_len]
    # Filter num_seq_list based on these indices
    filter_seq_list = [seq_list[i] for i in valid_indices]
    max_seq_len = int(args.image_size * args.image_size)
    padded_seq_list = pad_ends(
            seqs=filter_seq_list,
            max_seq_length=max_seq_len
    )
    num_seq_list = create_num_seqs(padded_seq_list) # numerical representations
    # prepare class labels
    #class_label_list = df.label.values.tolist()
    if args.facilitator in ['MSE', 'MMD']:
        text_emb = data_dict['text_to_protein_embedding']
    elif args.facilitator in ['Default']:
        text_emb = data_dict['text_embedding']
    else:
        raise ValueError(f"Unexpected value for 'facilitator': {args.facilitator}")
    text_emb = [text_emb[i] for i in valid_indices]
    # prune sequence and texts out based on length
    logger.info('Finished preparing dataset')
    #
    return (
            num_seq_list,
            text_emb
    )


class protein_dataset(Dataset):
    """
    Dataset for handling protein sequences paired with text embeddings.
    
    This class implements a PyTorch Dataset to efficiently load and batch
    protein sequences that have been converted to numerical representation,
    along with their corresponding text embeddings. It handles the conversion
    of list data to tensors as needed.
    
    Attributes:
        num_seqs (torch.Tensor): Tensor containing numerical representations of protein sequences
        text_emb (torch.Tensor): Tensor containing text embeddings corresponding to each sequence
    """
    def __init__(
            self,
            num_seq_list: list,
            text_emb: torch.Tensor
    ):
        """
        Initialize the protein dataset with sequences and embeddings.
        
        Args:
            num_seq_list: List or tensor of numerically encoded protein sequences
            text_emb: Tensor of text embeddings corresponding to sequences
        """
        if not torch.is_tensor(num_seq_list):
            self.num_seqs = torch.tensor(num_seq_list).float()
        else:
            pass
        self.text_emb = text_emb
        #if not torch.is_tensor(class_label_list):
        #    self.class_label = torch.tensor(class_label_list).float()
    
    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of protein sequences in the dataset
        """
        return len(self.num_seqs)
    
    def __getitem__(self, idx: any) -> (
            torch.FloatTensor,
            torch.FloatTensor
    ):
        """
        Retrieve a specific sample by index.
        
        This method fetches a protein sequence and its corresponding text 
        embedding at the specified index, handling tensor conversion if needed.
        
        Args:
            idx: Index of the sample to retrieve (can be a tensor or standard index)
            
        Returns:
            tuple: (
                num_seqs: FloatTensor of the numerically encoded protein sequence
                text_emb: FloatTensor of the text embedding for the sequence
            )
        """
        # convert and return the data batch samples
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # sequences
        num_seqs = self.num_seqs[idx]
        # class labels
        text_emb = self.text_emb[idx]
        return (
                num_seqs,
                text_emb
        )


def HDF5_prepare_protein_data(args, sequences):
    """
    Process a single protein sequence from an HDF5 file for model training.
    
    This function handles protein sequence data stored in HDF5 format by:
    1. Decoding the byte string to UTF-8 text
    2. Removing gap characters (-)
    3. Adding <START> and <END> tokens
    4. Filtering based on maximum allowed length
    5. Padding to a uniform length (determined by image_size^2)
    6. Converting to numerical representation
    
    Unlike the standard prepare_protein_data function, this version:
    - Processes a single sequence rather than a batch
    - Does not handle text embeddings
    - Returns the result in a list with a single element
    
    Args:
        args: Configuration object containing:
            - diffusion_steps: Maximum allowed sequence length
            - image_size: Used to calculate target padded length (image_size^2)
        sequences: Byte string containing the protein sequence from HDF5
        
    Returns:
        list: A list containing one element - the numerically encoded, 
              filtered, and padded sequence
    """
    diffusion_steps = args["diffusion_steps"]
    image_size = args["image_size"]
    #print('Prepare dataset')
    sequences = sequences.decode('utf-8')
    # Prepare sequences: Remove gaps and add start/end tokens
    seq_list = [seq.replace('-', '') for seq in sequences]
    seq_list = ['<START>'] + [aa for aa in seq_list] + ['<END>']
    seq_lens = [len(seq) for seq in seq_list]
    # Determine the maximum sequence length based on context window size
    max_seq_len = int(diffusion_steps)
    # Get indices of sequences that meet the criteria
    valid_indices = [i for i, seq in enumerate(seq_list) if len(seq) <= max_seq_len]
    # Filter sequences based on these indices
    filter_seq_list = [seq_list[i] for i in valid_indices]
    # Padding sequences to a uniform length
    padded_seq_list = pad_ends(seqs=filter_seq_list, max_seq_length=int(image_size * image_size))
    num_seq_list = create_num_seqs(padded_seq_list)  # Numerical representations
    return [num_seq_list]


#########################################################
# Generalized finetuning dataset support                #
# cleaned record --> composed caption --> (on-GPU) z_c  #
#########################################################

def encode_protein_sequence(sequence: str, image_size: int) -> list:
    """Tokenize one protein sequence to padded numerical form for ProteoScribe.

    Mirrors :func:`HDF5_prepare_protein_data` for a single sequence: strip gaps,
    add ``<START>``/``<END>``, pad to ``image_size**2`` with gap tokens, and map
    to integer indices. Length filtering is handled upstream by the data module.
    """
    seq_list = ['<START>'] + [aa for aa in sequence.replace('-', '')] + ['<END>']
    padded = pad_ends(seqs=seq_list, max_seq_length=int(image_size * image_size))
    return create_num_seqs(padded)


def make_seq_caption_collate_fn(*, text_tokenizer, text_max_length, image_size,
                                sequence_key="sequence", caption_key="caption",
                                include_sequences=False):
    """Build a collate fn mapping composed records to model-ready tensors.

    The returned callable takes a batch of ``{sequence_key: str, caption_key: str}``
    dicts (as emitted by :class:`~biom3.core.dataloaders.GeneralizedRecordDataset`),
    tokenizes each sequence with :func:`encode_protein_sequence` and stacks them,
    and batch-encodes the captions with the BioBERT tokenizer using
    ``padding="max_length"`` — the configuration ProteoScribe's conditioning
    embeddings were trained under (see docs/bug_reports/bert_embedding_mismatch.md).

    Returns ``(num_seqs [B, image_size**2] float32, input_ids [B, text_max_length])``,
    the batch contract expected by :class:`PL_ProtARDM_Finetune`.

    With ``include_sequences=True`` a third element is appended: the raw
    sequence strings, verbatim from the record. z_p-blended finetuning uses
    them to key its precomputed z_p lookup.
    """
    image_size = int(image_size)

    def _collate(batch):
        num_seqs = torch.stack([
            torch.tensor(
                encode_protein_sequence(sample[sequence_key], image_size)
            ).float()
            for sample in batch
        ])
        text_inputs = text_tokenizer(
            [sample[caption_key] for sample in batch],
            truncation=True,
            max_length=text_max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        if include_sequences:
            return (num_seqs, text_inputs["input_ids"],
                    [sample[sequence_key] for sample in batch])
        return num_seqs, text_inputs["input_ids"]

    return _collate





