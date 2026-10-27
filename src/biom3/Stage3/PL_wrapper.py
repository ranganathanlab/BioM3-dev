# ----- PyTorch Core -----
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import OneHotCategorical

# ----- Retrieve available device -----
from biom3.backend.device import BACKEND_NAME, _XPU

# ----- PyTorch Lightning Framework -----
if BACKEND_NAME == _XPU:
    # lightning imports (from local installation)
    import lightning as pl
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping

# ----- Distributed Training -----
#from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP  # Old import
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # Current FSDP implementation
from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap
)
import deepspeed  # DeepSpeed integration

# ----- Data & File Handling -----
import h5py  # HDF5 file format support
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Subset, DistributedSampler
from sklearn.model_selection import train_test_split

# ----- Optimization & Scheduling -----
from transformers.optimization import Adafactor  # Efficient optimizer for transformers
from deepspeed.ops.adam import DeepSpeedCPUAdam  # Memory-efficient Adam implementation
from transformers import get_cosine_schedule_with_warmup  # Learning rate scheduler
from transformers import AutoTokenizer  # BioBERT tokenizer for finetuning

# ----- Standard Libraries & Utilities -----
import functools
import hashlib
import math
import copy

# ----- Custom Modules & Project-Specific Imports -----
from biom3.Stage3.DSEma import moving_average, clone_zero_model  # EMA implementation
import biom3.Stage3.transformer_training_helper as trainer_tools
import biom3.Stage3.eval_metrics as eval_funcs
import biom3.Stage3.preprocess as prep
from biom3.core.dataloaders import (
    GeneralizedRecordDataset,
    JsonlRecordStore,
    read_jsonl_records,
)
from biom3.backend.device import print_gpu_initialization, setup_logger

logger = setup_logger(__name__)


def _make_distributed_sampler(dataset, *, shuffle: bool, seed: int):
    """Return a DistributedSampler with drop_last=True if torch.distributed is
    initialized, else None.

    drop_last=True at the sampler level is required for distributed training:
    without it, sklearn's train_test_split produces an effective train set
    whose size is rarely a multiple of (world_size * batch_size). The leftover
    samples get distributed unevenly across ranks (e.g. 11 ranks get 71 batches,
    1 rank gets 72), causing the next DDP gradient allreduce to deadlock on a
    genuinely mismatched collective. Sampler-level drop_last truncates the shard
    so every rank gets exactly the same number of samples.

    Pair with Trainer(use_distributed_sampler=False) so Lightning doesn't
    auto-wrap the DataLoader with its own (non-drop_last) sampler.
    """
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=True,
            seed=seed,
        )
    return None


class PL_ProtARDM(pl.LightningModule):
    """
    PyTorch Lightning module for Protein Autoregressive Diffusion Models.
    
    This class implements a Lightning wrapper around the protein diffusion model,
    handling training, validation, optimization, and performance tracking. It follows
    the autoregressive diffusion process with conditional sampling for protein sequence
    generation guided by text embeddings.
    
    The model uses a masked autoregressive approach where sequence tokens are
    progressively generated in a randomized order, with each prediction conditioned
    on previously generated tokens and a text embedding that guides the generation.
    
    Key features:
    - Supports multiple optimizers (AdamW, Adafactor, Adam, DeepSpeedCPUAdam)
    - Configurable learning rate schedulers (cosine warmup, exponential decay)
    - Comprehensive metrics tracking during training and validation
    - Compatible with distributed training via DeepSpeed
    """

    def __init__(
            self,
            args: any,
            model: nn.Module,
            #ema_model: nn.Module,
        ):
        """
        Initialize the PyTorch Lightning module for protein diffusion.
        
        Args:
            args: Configuration object containing training and model parameters
            model: Neural network module implementing the diffusion model architecture
            #ema_model: Optional Exponential Moving Average model for improved inference stability
        """
        super().__init__()
        #self.save_hyperparameters()

        # arguments
        self.script_args = args

        # the whole model
        self.model = model
        #self.ema_model = ema_model

        #clone_zero_model(self.model, self.ema_model, zero_stage=3)
        ##self.ema_model = copy.deepcopy(self.model)

    @property
    def loss_positions(self) -> str:
        """Which unsampled positions drive the gradients: 'all' or 'non_pad'."""
        return getattr(self.script_args, 'loss_positions', 'all')

    @property
    def normalize_zc(self) -> bool:
        """Whether the conditioning vector is scaled to unit length for the model."""
        return bool(getattr(self.script_args, 'normalize_zc', False))

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y_c: torch.Tensor,
            ema=False,
        ) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        
        This method routes the input through either the primary model
        or the EMA model (if enabled) to generate logits for the masked 
        protein sequence positions.
        
        Args:
            x: Tensor of shape [batch_size, 1, seq_length] containing the partially
               masked protein sequence
            t: Tensor of shape [batch_size] containing timestep indices
            y_c: Tensor containing conditional text embeddings to guide generation
            ema: Whether to use the EMA model for inference (if available)
            
        Returns:
            Tensor of logits for each possible amino acid at each position
        """
        if ema:
            logits = self.ema_model(x=x, t=t.view(-1,), y_c=y_c)
        else:
            logits = self.model(x=x, t=t.view(-1,), y_c=y_c)
        return logits
        #return F.softmax(logits, dim=1)


    #def on_train_batch_end(self, *args, **kwargs):
    #    clone_zero_model(self.model, self.ema_model, zero_stage=3)
    #    #moving_average(self.model, self.ema_model, beta=0.0, zero_stage=3)



    def configure_optimizers(self, ):
        """
        Configure optimizers and learning rate schedulers.
        
        This method sets up the training optimization based on the arguments provided:
        - Chooses between different optimizers (AdamW, Adafactor, Adam, DeepSpeedCPUAdam)
        - Configures appropriate learning rate schedulers based on script_args
          including cosine warmup or exponential decay options
        
        Returns:
            dict or optimizer: Configuration dictionary containing optimizer and
                              scheduler settings, or just the optimizer if no
                              scheduler is specified
        """
        choose_optim = self.script_args.choose_optim
        lr = self.script_args.lr
        weight_decay = self.script_args.weight_decay
        scheduler_gamma = self.script_args.scheduler_gamma
        traindata_len = self.script_args.traindata_len
        epochs = self.script_args.epochs

        if choose_optim == 'AdamW':

            if isinstance(self, FSDP):
                logger.info("Enter FSDP")
                optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

            else:
                optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        elif choose_optim == 'AdaFactor':
            optimizer = Adafactor(self.parameters(), lr=lr, weight_decay=weight_decay, relative_step=False)

        elif choose_optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        elif choose_optim == 'DeepSpeedCPUAdam':
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler_gamma is not None:
            if isinstance(scheduler_gamma, str):
                if 'coswarmup' == scheduler_gamma.lower():
                    logger.info('Using cossine warmup scheduler with decay')
                    num_warmup_steps=traindata_len
                    num_training_steps=traindata_len * epochs
                    logger.info('Num_warmup_steps=%s', num_warmup_steps)
                    logger.info('Num_training_steps=%s', num_training_steps)

                    def _get_cosine_schedule_with_warmup_lr_lambda(
                        current_step: int, num_warmup_steps: int, num_training_steps: int, num_cycles: float
                    ):
                        if current_step < num_warmup_steps:
                            return float(current_step) / float(max(1, num_warmup_steps))
                        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

                    lr_lambda = functools.partial(
                        _get_cosine_schedule_with_warmup_lr_lambda,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps,
                        num_cycles=0.5,
                    )
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1),
                            "interval": "step",
                        },
                    }

                    #return {
                    #    "optimizer": optimizer,
                    #    "lr_scheduler": {
                    #        "scheduler": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps),
                    #        "interval": "step",
                    #    },
                    #}
            else:
                logger.info('Using Exponential learning rate decay / epoch with factor: %s', scheduler_gamma)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma, verbose=True),
                        "interval": "epoch",
                    },
                }

        else:
            return optimizer

        #else:
        #    print("Please make choose_option variable from these options: 'AdamW', 'AdaFactor', 'Adam', 'DeepSpeedCPUAdam'")

    def common_step(
            self,
            realization: torch.Tensor,
            realization_idx: any,
            stage: str) -> dict:
        """
        Common processing step used in both training and validation.
        
        This method handles:
        1. Reshaping input data
        2. Computing loss via the ELBO objective
        3. Calculating and logging metrics
        4. Managing memory utilization
        
        Args:
            realization: Input protein sequences, either as a tensor or a list
                        containing [sequence, text_embedding]
            realization_idx: Index of the current batch
            stage: String indicating the current stage ('train', 'val', or 'EMA_val')
            
        Returns:
            dict: Dictionary containing the computed loss
        """
        if isinstance(realization, list):

            # class labels
            y_c = realization[1]#.long()
            if self.normalize_zc:
                y_c = F.normalize(y_c.float(), dim=-1)

            # input samples
            realization = realization[0]
            batch_size, seq_length = realization.size()

        realization = realization.reshape(batch_size, 1, seq_length).long()

        train_tuple = self.cond_elbo_objective(
                realization=realization,
                y_c=y_c,
                realization_idx=realization_idx,
                stage=stage,
                ema=True if 'ema' in stage.lower() else False,
        )

        if len(train_tuple) == 1:
            loss = train_tuple[0]
        else:
            loss = train_tuple[0]
            metrics = train_tuple[1]
        loss_terms = train_tuple[2] if len(train_tuple) > 2 else {}

        # if realization_idx == 0:
        #     if self.global_rank == 0:
        #         gpu_memory_usage = print_gpu_initialization()
        #     else:
        #         gpu_memory_usage = 0.0
        #     self.log(f"{stage}_gpu_memory_usage", gpu_memory_usage, sync_dist=True)

        # For train: log per-step, no cross-rank sync (grad allreduce already syncs).
        # For val: skip per-step logging; accumulate locally and sync once at epoch end.
        # Per-step sync_dist=True during val races with DeepSpeed ZeRO's grad
        # allreduce on XPU/oneCCL and deadlocks (see docs/bug_reports/).
        is_val = 'val' in stage
        log_kwargs = dict(on_step=not is_val, on_epoch=True, sync_dist=is_val)

        self.log(f"{stage}_loss", loss, prog_bar=True, **log_kwargs)
        for name, value in loss_terms.items():
            self.log(f"{stage}_loss_{name}", value, **log_kwargs)
        if len(train_tuple) > 1:
            self.log(f"{stage}_prev_hard_acc",    metrics[0], prog_bar=True, **log_kwargs)
            self.log(f"{stage}_prev_soft_acc",    metrics[1], **log_kwargs)
            self.log(f"{stage}_fut_hard_acc",     metrics[2], prog_bar=True, **log_kwargs)
            self.log(f"{stage}_fut_soft_acc",     metrics[3], **log_kwargs)
            self.log(f"{stage}_current_hard_acc", metrics[4], prog_bar=True, **log_kwargs)
            self.log(f"{stage}_current_soft_acc", metrics[5], **log_kwargs)
            self.log(f"{stage}_current_ppl",      metrics[6], **log_kwargs)
            self.log(f"{stage}_prev_ppl",         metrics[7], **log_kwargs)
            self.log(f"{stage}_fut_ppl",          metrics[8], **log_kwargs)
            self.log(f"{stage}_pos_entropy",      metrics[9], **log_kwargs)

        if BACKEND_NAME == _XPU:
            torch.xpu.memory.empty_cache()
        elif BACKEND_NAME == "cuda":
            torch.cuda.empty_cache()
        return {'loss': loss}

    def training_step(
            self,
            realization: torch.Tensor,
            realization_idx: any):
        """
        Perform a single training step.
        
        This method is called by PyTorch Lightning during training.
        It delegates to common_step with the 'train' stage flag.
        
        Args:
            realization: Input protein sequences
            realization_idx: Index of the current batch
            
        Returns:
            dict: Dictionary containing the computed loss
        """
        return self.common_step(realization, realization_idx, stage='train')

    def validation_step(
            self,
            realization: torch.Tensor,
            realization_idx: any):
        """
        Perform a single validation step.
        
        This method is called by PyTorch Lightning during validation.
        It delegates to common_step with the 'val' stage flag.
        
        Args:
            realization: Input protein sequences
            realization_idx: Index of the current batch
        """
        self.common_step(realization, realization_idx, stage='val')
        #self.common_step(realization, realization_idx, stage='EMA_val')

    def on_fit_start(self):
        if self.global_rank == 0:
            logger.info("Conditioning vector scaled to unit length: %s", self.normalize_zc)
        # Diagnostic: confirm which torch.distributed backend is actually in use.
        # On Aurora frameworks/2025.3.1 this should print 'xccl'. If it prints
        # 'ccl' or 'gloo' the DeepSpeedStrategy process_group_backend kwarg
        # isn't taking effect and oneCCL collectives aren't going through the
        # native xccl path.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = torch.distributed.get_backend()
            import logging
            logging.getLogger().info(
                f"[rank {self.global_rank}] torch.distributed backend = {backend}"
            )

    def on_before_optimizer_step(self, optimizer):
        norms = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        if norms:
            total_norm = torch.stack(norms).norm(2)
            self.log("train_grad_norm", total_norm, on_step=True, on_epoch=False)

    def apply_OneHotCat(self, probs: torch.Tensor) -> any:
        """
        Create a OneHotCategorical distribution from input probabilities.
        
        This helper method converts model output probabilities into a
        proper distribution object for sampling and probability calculations.
        
        Args:
            probs: Tensor of shape [batch_size, num_classes, seq_length] 
                  containing probabilities
                  
        Returns:
            OneHotCategorical distribution with appropriate shape
        """
        return OneHotCategorical(probs=probs.permute(0,2,1))
        #return OneHotCategorical(probs=F.softmax(probs.permute(0,2,1), dim=-1))

    def cond_elbo_objective(
            self,
            realization: torch.Tensor,
            y_c: torch.Tensor,
            realization_idx: any,
            stage: str,
            ema=False,
        ):
        """
        Calculate the conditional Evidence Lower Bound (ELBO) objective.
        
        This method implements the core diffusion training objective:
        1. Generates a random sampling path and timestep
        2. Creates masks for different parts of the sequence
        3. Computes model predictions on masked sequences
        4. Calculates log probabilities and loss
        5. Computes performance metrics
        
        Args:
            realization: Tensor of shape [batch_size, 1, seq_length] containing 
                        protein sequences
            y_c: Tensor containing conditional text embeddings
            realization_idx: Index of the current batch
            stage: String indicating the current stage ('train', 'val')
            ema: Whether to use the EMA model for inference
            
        Returns:
            tuple: (loss, metrics) where metrics is a tuple of performance measurements
        """
        # Use Lightning's rank-local device (xpu:N for rank N) rather than the
        # CLI --device string ("xpu"), which resolves to xpu:0 on every rank and
        # crashes under plain DDP with "Expected all tensors to be on the same
        # device". DeepSpeedStrategy previously hid this by moving tensors
        # transparently before use.
        device = self.device

        bs, channel, seq_length = realization.size()

        # get a batch of random sampling paths
        sampled_random_path = trainer_tools.sample_random_path(bs, seq_length, device=device)
        # sample a set of random smapling steps for each individual training sequences in the current batch
        idx = trainer_tools.sample_random_index_for_sampling(bs, seq_length, device=device, option='random')
        # we create a mask that masks the location were we've already sampled
        random_path_mask = trainer_tools.create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
        # create a mask that masks the location where we are currently sampling
        current_path_mask = trainer_tools.create_sampling_location_mask(sampled_random_path, idx, bs, seq_length)
        # future sampling locations (i.e. >t)
        future_path_mask = trainer_tools.create_mask_at_future_path_index(sampled_random_path, idx, bs, seq_length)
        # tokenize realization
        real_tokens, bs, seq_length = trainer_tools.create_token_labels(self.script_args, realization)
        #real_tokens = realization.clone().squeeze(1)
        # mask realizations
        real_token_masked = trainer_tools.mask_realizations(real_tokens, random_path_mask)
        # conditional probs
        #probs = self(x=real_token_masked, t=idx, y_c=y_c, ema=ema)
        logits = self(x=real_token_masked, t=idx, y_c=y_c, ema=ema)
        logits = logits.float() # TN: Cast to float32 before softmax, added to fix "Expected parameter probs (Tensor of shape (32, 1024, 29)) of distribution OneHotCategorical() to satisfy the constraint Simplex(), but found invalid values" error

        conditional_prob = OneHotCategorical(logits=logits.permute(0,2,1))
        #conditional_prob = self.apply_OneHotCat(probs=probs)
        # evaluate the value of the log prob for the given realization
        log_prob = trainer_tools.log_prob_of_realization(self.script_args, conditional_prob, real_tokens)

        # compute an average over all the unsampled
        #log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob.to(device), real_token_masked.to(device))
        log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob, real_token_masked)
        #log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob, real_token_masked, real_tokens)


        # compute an average loss i.e. negative average log-likelihood over the batch elements
        log_prob_weighted = trainer_tools.weight_log_prob(log_prob_unsampled, idx, seq_length)
        # compute an average loss i.e. negative average log-likelihood over the batch elements
        loss = trainer_tools.compute_average_loss_for_batch(log_prob_weighted)

        # the same loss split by whether the target is padding; one of them drives the gradients
        loss_terms = {'all': loss}
        loss_terms.update(trainer_tools.unsampled_loss_terms(
            log_prob, real_token_masked, real_tokens, prep.PAD_TOKEN_ID))
        loss = loss_terms[self.loss_positions]

        #if 'val' in stage:
        probs = F.softmax(logits, dim=1)
        metrics = self.performance_step(
                    real_tokens=real_tokens.cpu(),
                    idx=idx.cpu(),
                    sampled_random_path=sampled_random_path.cpu().float(),
                    probs=probs.cpu().float(),
                    conditional_prob=conditional_prob)

        return loss, metrics, loss_terms


    @torch.no_grad()
    def performance_step(
                    self,
                    real_tokens: torch.Tensor,
                    idx: torch.Tensor,
                    sampled_random_path: torch.Tensor,
                    probs: torch.Tensor,
                    conditional_prob: torch.Tensor
                    ) -> tuple:
        """
        Calculate performance metrics for the current batch.
        
        This method computes various accuracy and perplexity metrics for
        different parts of the sequence (previous, current, and future tokens)
        based on the diffusion timestep.
        
        Note: This method is decorated with @torch.no_grad() to prevent
        gradient computation during metric calculation.
        
        Args:
            real_tokens: Ground truth protein sequences
            idx: Tensor containing timestep indices for each sequence
            sampled_random_path: Random sampling paths for the batch
            probs: Softmax probabilities from the model
            conditional_prob: OneHotCategorical distribution for sampling
            
        Returns:
            tuple: A tuple containing various performance metrics:
                - prev_B_hard_acc: Hard accuracy for previous tokens
                - prev_B_soft_acc: Soft accuracy for previous tokens
                - fut_B_hard_acc: Hard accuracy for future tokens
                - fut_B_soft_acc: Soft accuracy for future tokens
                - current_B_hard_acc: Hard accuracy for current tokens
                - current_B_soft_acc: Soft accuracy for current tokens
                - current_ppl: Perplexity for current tokens
                - prev_ppl: Perplexity for previous tokens
                - fut_ppl: Perplexity for future tokens
                - pos_entropy: Average positional entropy
        """
        # create numerical token sequence
        sample_seq = torch.argmax(trainer_tools.sample_from_conditional(conditional_prob).cpu(), dim=1)

        # eval prev positions in terms of time
        prev_B_hard_acc, prev_B_soft_acc, fut_B_hard_acc, fut_B_soft_acc, current_B_hard_acc, current_B_soft_acc = eval_funcs.compute_acc_given_time_pos(
                real_tokens=real_tokens,
                sample_seq=sample_seq,
                sample_path=sampled_random_path,
                idx=idx
        )

        # compute ppl given time position
        current_ppl, prev_ppl, fut_ppl = eval_funcs.compute_ppl_given_time_pos(
                probs=probs,
                sample_path=sampled_random_path,
                idx=idx
        )

        # average positional entropy
        pos_entropy = trainer_tools.compute_pos_entropy(probs=probs).mean().item()

        metric_evals = (
                prev_B_hard_acc,
                prev_B_soft_acc,
                fut_B_hard_acc,
                fut_B_soft_acc,
                current_B_hard_acc,
                current_B_soft_acc,
                current_ppl,
                prev_ppl,
                fut_ppl,
                pos_entropy
        )

        return metric_evals



class PFamDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for protein sequence data from PFam and/or SwissProt databases.

    This module handles data loading, preprocessing, train/validation splitting,
    and DataLoader creation for protein sequence modeling. It supports loading
    from multiple data sources (SwissProt and PFam) and can merge them automatically
    when both are provided.

    The module uses protein_dataset from the prep module to create PyTorch Dataset
    objects and handles the entire data pipeline from raw data to batched DataLoaders.
    """
    def __init__(self, args):
        """
        Initialize the PFam DataModule and prepare the datasets.

        This method:
        1. Loads protein data from specified sources
        2. Preprocesses sequences and embeddings
        3. Splits data into training and validation sets
        4. Creates dataset objects for both splits

        Args:
            args: Configuration object containing:
                - data paths (swissprot_data_root, pfam_data_root)
                - valid_size: Fraction of data to use for validation
                - seed: Random seed for reproducible splitting
                - batch_size: Number of samples per batch
                - num_workers: Number of worker processes for data loading
        """
        super().__init__()
        self.args = args
        #df = pd.read_csv(args.data_root)
        #data = torch.load(args.data_root)
        data = self.load_data()
        num_seq_list, text_emb_list = prep.prepare_protein_data(
                args=args,
                data_dict=data
        )
        logger.info('Performing 80/20 random train/val split')
        num_seq_list_train, num_seq_list_val, text_emb_train, text_emb_val = train_test_split(num_seq_list,
                                                                                            text_emb_list,
                                                                                            test_size=args.valid_size,
                                                                                            #stratify=class_label_list,
                                                                                            random_state=args.seed)
        logger.info('Number of training samples: %s', len(num_seq_list_train))
        logger.info('Number of validation samples: %s', len(num_seq_list_val))
        self.train_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list_train,
            text_emb=text_emb_train
        )
        self.val_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list_val,
            text_emb=text_emb_val
        )

    def load_data(self):
        """
        Load protein sequence data from specified sources.

        This method checks the provided paths for SwissProt and PFam data
        and loads them accordingly. If both sources are available, it merges
        the data from both sources.

        Returns:
            dict: Dictionary containing protein sequences and their associated
                  text embeddings

        Raises:
            FileNotFoundError: If specified data files cannot be found
            ValueError: If neither SwissProt nor PFam data is available
        """
        try:
            logger.debug("swissprot_data_root=%s, pfam_data_root=%s", self.args.swissprot_data_root, self.args.pfam_data_root)
            if self.args.swissprot_data_root != "None":
                swissprot_data = torch.load(self.args.swissprot_data_root)
            else:
                swissprot_data=None
            if self.args.pfam_data_root != "None":
                pfam_data = torch.load(self.args.pfam_data_root)
            else:
                pfam_data=None
            if (self.args.swissprot_data_root != "None") and (self.args.pfam_data_root != "None"):
                return self.merge_and_append_values(dict1=swissprot_data, dict2=pfam_data)
            elif self.args.swissprot_data_root == "None":
                return pfam_data
            elif self.args.pfam_data_root == "None":
                return swissprot_data
            else:
                raise ValueError('Both SwissProt and Pfam datasets are unavailable.')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e}")

    def merge_and_append_values(self, dict1, dict2):
        """
        Merge two dictionaries by concatenating their values.

        This utility method combines data from two sources (e.g., SwissProt and PFam)
        by merging corresponding values for each key. If the values are lists,
        they are concatenated. If not, they are converted to lists and then merged.

        Args:
            dict1: First dictionary (e.g., SwissProt data)
            dict2: Second dictionary (e.g., PFam data)

        Returns:
            dict: A merged dictionary with combined values from both sources
        """
        merged_dict = {}
        # Combine all keys from both dictionaries
        all_keys = set(dict1) | set(dict2)
        for key in all_keys:
            values = []
            if key in dict1:
                values.append(dict1[key])
            if key in dict2:
                values.append(dict2[key])
            # Merge values for each key
            # This merges lists or appends non-list values
            merged_dict[key] = [item for sublist in values for item in (sublist if isinstance(sublist, list) else [sublist])]
        return merged_dict

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        Uses an explicit DistributedSampler with drop_last=True under
        distributed training so every rank gets the same number of samples.
        See _make_distributed_sampler for the rationale. Pair with
        Trainer(use_distributed_sampler=False).
        """
        sampler = _make_distributed_sampler(
            self.train_dataset, shuffle=True, seed=self.args.seed
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            sampler=sampler,
            shuffle=(sampler is None),
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.
        """
        sampler = _make_distributed_sampler(
            self.val_dataset, shuffle=False, seed=self.args.seed
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            sampler=sampler,
            shuffle=False,
        )

ALPHA_BLEND = "blend"
EVAL_SPREAD = "spread"
_ALPHA_ALIASES = {"zc": 0.0, "zp": 1.0}


def normalize_alpha_spec(value):
    """Normalize an alpha spec to a float in [0, 1] or the string ``"blend"``.

    alpha is the weight on z_p in ``y = alpha * z_p + (1 - alpha) * z_c``.
    Accepts the names ``zc`` / ``zp`` / ``blend``, or any numeric literal in
    [0, 1] for a constant blend.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text == ALPHA_BLEND:
            return ALPHA_BLEND
        if text in _ALPHA_ALIASES:
            return _ALPHA_ALIASES[text]
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"train_alpha must be 'zc', 'zp', 'blend', or a number in "
                f"[0, 1] (the weight on z_p); got {value!r}"
            ) from None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"a constant train_alpha must be in [0, 1]; got {value}"
        )
    return value


def resolve_eval_alpha(value):
    """Normalize a validation-alpha spec to a float in [0, 1] or ``"spread"``.

    ``"spread"`` assigns each validation example its own alpha, deterministically
    (see :func:`deterministic_alpha`), so the metric covers the whole [0, 1]
    operating range instead of a single point. A constant is still allowed for
    targeted evaluation at one alpha. The ``"blend"`` *training* schedule is
    rejected: it resamples per batch, which would make val loss — and therefore
    best-checkpoint selection — jitter epoch to epoch.
    """
    if isinstance(value, str) and value.strip().lower() == EVAL_SPREAD:
        return EVAL_SPREAD
    spec = normalize_alpha_spec(value)
    if spec == ALPHA_BLEND:
        raise ValueError(
            "eval_alpha must be 'spread', 'zc', 'zp', or a constant in [0, 1]; "
            "not the 'blend' training schedule (a resampled validation alpha "
            "makes val loss incomparable across epochs)."
        )
    return float(spec)


def deterministic_alpha(key: str) -> float:
    """A stable pseudo-random alpha in [0, 1) for a string key.

    Keyed on the sequence so the same example always draws the same alpha —
    across epochs, processes, and DDP ranks — which is what keeps the spread
    validation metric comparable epoch to epoch. ``hash()`` is unsuitable here:
    it is salted per interpreter run.
    """
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def alpha_spec_uses_zp(spec) -> bool:
    """Whether a normalized alpha spec ever puts weight on z_p."""
    return spec == ALPHA_BLEND or float(spec) > 0.0


def sample_blend_alpha(n, device=None):
    """Rama's train_blend schedule {alpha=1: .25, alpha=0: .25, U(0,1): .5}.

    Returns weights on z_p shaped ``[n, 1]``. Shared by both training paths so
    the HDF5 and generalized finetuning runs draw from the same distribution.
    """
    r = torch.rand(n, device=device)
    a = torch.rand(n, device=device)
    a = torch.where(r < 0.25, torch.ones_like(a), a)
    a = torch.where(r >= 0.75, torch.zeros_like(a), a)
    return a.view(n, 1)


class HDF5Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for protein sequences stored in HDF5 format.
    
    This dataset loads protein sequences and their corresponding text embeddings 
    from a specified group within an HDF5 file. The protein sequences undergo 
    preprocessing via the HDF5_prepare_protein_data function to convert them into 
    numerical representation suitable for model input.
    
    The dataset is designed to work with HDF5 files containing at least:
    - 'sequence': Protein sequences stored as byte strings
    - 'text_to_protein_embedding': Numerical embeddings for each sequence
    - 'sequence_length': Length information for each sequence (optional)
    
    Note: This dataset keeps the HDF5 file open during its lifetime, so it's
    important to call the close() method when done to release resources.
    """
    def __init__(self, args, file_path, group_name, zp_path=None,
                 train_alpha=0.0, eval_alpha=EVAL_SPREAD):
        """
        Initialize the HDF5 protein dataset.

        Args:
            args: Configuration object containing preprocessing parameters
            file_path: Path to the HDF5 file containing protein data
            group_name: Name of the group within the HDF5 file to access
                       (e.g., 'MMD_data' or 'MSE_data')
            zp_path: Stage 2 Facilitator output (.pt) holding the z_p rows this
                       HDF5 was compiled from. Required when train_alpha puts
                       any weight on z_p.
            train_alpha: Weight on z_p for training rows — 'zc', 'zp', 'blend',
                       or a constant in [0, 1]
            eval_alpha: Weight on z_p for rows registered by set_eval_indices —
                       'spread' or a constant in [0, 1]
        """
        self.file_path = file_path
        self.f = h5py.File(file_path, 'r')
        self.group = self.f[group_name]
        self.transform = prep.HDF5_prepare_protein_data
        self.args = args
        # Store data as attributes for easier access
        self.tensor_data = self.group["text_to_protein_embedding"]
        self.sequences = self.group["sequence"]
        self.train_alpha = normalize_alpha_spec(train_alpha)
        self.eval_alpha = resolve_eval_alpha(eval_alpha)
        self._eval_indices = frozenset()
        self.z_p = None
        if alpha_spec_uses_zp(self.train_alpha):
            if zp_path is None:
                raise ValueError(
                    "train_alpha puts weight on z_p, so zp_path is required: "
                    "pass the Stage 2 Facilitator .pt this HDF5 was compiled "
                    "from"
                )
            self.z_p = self._load_zp(zp_path)

    def _load_zp(self, zp_path):
        """Load z_p and check it is row-aligned with this HDF5.

        biom3_compile_hdf5 copies the Facilitator .pt through row for row, so
        row i of z_p pairs with row i here — but only for the .pt this file was
        actually compiled from. A different run has the same shape and would
        pair z_p and z_c from unrelated proteins with nothing to signal it.
        """
        from biom3.split import manifest as split_manifest

        payload = torch.load(zp_path, weights_only=False)
        if 'z_p' not in payload:
            raise KeyError(
                f"{zp_path} has no 'z_p' key (keys: {sorted(payload)}); it "
                f"should be the .pt written by biom3_Facilitator_sample"
            )
        z_p = payload['z_p'].cpu().float()
        if z_p.shape[0] != len(self):
            raise ValueError(
                f"z_p has {z_p.shape[0]} rows but {self.file_path} has "
                f"{len(self)}; they must be row-aligned to blend"
            )
        if 'sequence' in payload:
            theirs = split_manifest.compute_fingerprint(payload['sequence'])
            ours = split_manifest.compute_fingerprint(self.sequences)
            if theirs != ours:
                raise ValueError(
                    f"{zp_path} does not match {self.file_path} (sequence "
                    f"fingerprint {theirs} vs {ours}); z_p must come from the "
                    f"Facilitator run this HDF5 was compiled from"
                )
        else:
            logger.warning(
                "%s carries no 'sequence' key, so z_p alignment was checked on "
                "row count alone", zp_path)
        logger.info("Loaded z_p from %s: %d rows, dim %d",
                    zp_path, z_p.shape[0], z_p.shape[1])
        return z_p

    def set_eval_indices(self, indices):
        """Mark the rows served to validation.

        Train and val are Subsets over one shared dataset instance, so the row
        itself has to know which split it belongs to: without this every val
        access would redraw a random alpha and val loss would stop being
        comparable across epochs.
        """
        self._eval_indices = frozenset(int(i) for i in indices)

    def _sequence_str(self, idx):
        seq = self.sequences[idx]
        return seq.decode() if isinstance(seq, bytes) else str(seq)

    def _alpha_for(self, idx):
        """Weight on z_p for one row."""
        if idx in self._eval_indices:
            if self.eval_alpha == EVAL_SPREAD:
                return deterministic_alpha(self._sequence_str(idx))
            return float(self.eval_alpha)
        if self.train_alpha == ALPHA_BLEND:
            return float(sample_blend_alpha(1).item())
        return float(self.train_alpha)

    def get_sequence_lengths(self):
        """
        Retrieve the lengths of all sequences in the dataset.
        
        Returns:
            numpy.ndarray: Array containing the length of each sequence
        """
        # Assumes there is a dataset in your HDF5 file that stores sequence lengths
        return self.group['sequence_length'][:]
        
    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of protein sequences in the dataset
        """
        return len(self.group['text_to_protein_embedding'])
        
    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        
        This method:
        1. Loads the text embedding directly from the HDF5 file
        2. Processes the protein sequence through the transform function
        3. Returns both as a tuple
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (
                x_p: Tensor containing the preprocessed protein sequence 
                z_c: Tensor containing the text-to-protein embedding
            )
        """
        z_c = torch.from_numpy(self.tensor_data[idx])
        x_p = torch.tensor(
            self.transform(
                args=self.args,
                sequences=self.sequences[idx]
            )[0]
        )
        if self.z_p is not None:
            a = self._alpha_for(idx)
            z_c = a * self.z_p[idx] + (1.0 - a) * z_c
        # Return a dictionary containing your data
        return (
            x_p, # sequence
            z_c # text_to_protein_embedding
        )
        
    def close(self):
        """
        Close the HDF5 file.
        
        This method should be called when the dataset is no longer needed
        to properly release file handles and system resources.
        """
        self.f.close()



class HDF5DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for HDF5-formatted protein sequence datasets.

    Supports loading from a primary data source and optionally one or more
    secondary sources.  Each source is split into train/validation, filtered
    by sequence length, and the splits from all sources are concatenated.

    Key features:
    - Memory-efficient access to large sequence datasets via HDF5
    - Supports combining data from multiple sources
    - Automatic filtering of sequences that exceed maximum length
    - Proper resource management with HDF5 file closing
    """
    def __init__(
        self, *,
        batch_size,
        num_workers,
        valid_size,
        seed,
        diffusion_steps,
        image_size,
        primary_path=None,
        secondary_paths=None,
        group_name='data',
        split_manifest_path=None,
        zp_path=None,
        train_alpha=0.0,
        eval_alpha=EVAL_SPREAD,
        # Deprecated aliases
        swissprot_path=None,
        pfam_path=None,
    ):
        super().__init__()
        # Handle deprecated aliases
        if swissprot_path is not None and primary_path is None:
            primary_path = swissprot_path
        if pfam_path is not None and secondary_paths is None:
            secondary_paths = [pfam_path]

        self.args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "valid_size": valid_size,
            "seed": seed,
            "diffusion_steps": diffusion_steps,
            "image_size": image_size,
        }
        self.primary_path = primary_path
        self.secondary_paths = secondary_paths or []
        self.group_name = group_name
        self.split_manifest_path = split_manifest_path
        self.zp_path = zp_path
        self.train_alpha = normalize_alpha_spec(train_alpha)
        self.eval_alpha = resolve_eval_alpha(eval_alpha)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.seed = seed
        self.min_seq_length = diffusion_steps - 2

    def setup(self, stage=None):
        train_datasets = []
        val_datasets = []

        all_paths = []
        if self.primary_path is not None:
            all_paths.append(self.primary_path)
        all_paths.extend(self.secondary_paths)

        if alpha_spec_uses_zp(self.train_alpha) and len(all_paths) > 1:
            raise ValueError(
                f"z_p blending supports a single HDF5 (primary_path); got "
                f"{len(all_paths)} paths. The secondary sources have no z_p of "
                f"their own, so they would train on pure z_c while the primary "
                f"blends."
            )

        manifest = None
        if self.split_manifest_path is not None:
            from biom3.split import manifest as split_manifest
            manifest = split_manifest.read_manifest(self.split_manifest_path)
            if len(manifest["files"]) != len(all_paths):
                raise ValueError(
                    f"split manifest describes {len(manifest['files'])} file(s) "
                    f"but {len(all_paths)} data path(s) were provided; the "
                    f"manifest must be regenerated for this set of inputs"
                )
            logger.info("Using curated split manifest %s", self.split_manifest_path)

        self.split_info = []
        for file_index, path in enumerate(all_paths):
            dataset = HDF5Dataset(
                args=self.args,
                file_path=path,
                group_name=self.group_name,
                zp_path=self.zp_path if file_index == 0 else None,
                train_alpha=self.train_alpha if file_index == 0 else 0.0,
                eval_alpha=self.eval_alpha,
            )
            test_idx = []
            if manifest is not None:
                from biom3.split import manifest as split_manifest
                entry = manifest["files"][file_index]
                split_manifest.validate_file_entry(
                    entry,
                    n_rows=len(dataset),
                    fingerprint=split_manifest.compute_fingerprint(dataset.sequences),
                )
                train_idx, val_idx, test_idx = entry["train"], entry["val"], entry["test"]
            else:
                train_idx, val_idx = self.split_indices(dataset)
            train_idx = self.filter_by_sequence_length(dataset, train_idx)
            val_idx = self.filter_by_sequence_length(dataset, val_idx)
            dataset.set_eval_indices(val_idx)
            train_datasets.append(Subset(dataset, train_idx))
            val_datasets.append(Subset(dataset, val_idx))
            self.split_info.append({
                "path": path,
                "train_indices": train_idx,
                "val_indices": val_idx,
                "test_indices": test_idx,
            })
            logger.info("Loaded dataset from %s (%d train, %d val, %d held-out test)",
                        path, len(train_idx), len(val_idx), len(test_idx))

        if not train_datasets:
            raise ValueError("No dataset paths provided")

        # Combine training and validation subsets
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    def split_indices(self, dataset):
        """
        Create train and validation indices for a dataset.

        This utility method creates a random split of the dataset indices
        based on the validation size specified in the arguments.

        Args:
            dataset: Dataset to split

        Returns:
            tuple: (train_indices, validation_indices)
        """
        # Create indices for splitting
        train_size = int((1 - self.valid_size) * len(dataset))
        indices = np.arange(len(dataset))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        return indices[:train_size], indices[train_size:]

    def filter_by_sequence_length(self, dataset, indices):
        """
        Filter indices to include only sequences within the length threshold.

        This method removes sequences that are too long for the model's
        context window (determined by diffusion_steps).

        Args:
            dataset: Dataset containing the sequences
            indices: Indices to filter

        Returns:
            list: Filtered indices containing only valid-length sequences
        """
        # Assumes 'get_sequence_lengths' method exists in HDF5Dataset
        sequence_lengths = dataset.get_sequence_lengths()
        filtered_indices = [idx for idx in indices if sequence_lengths[idx] <= self.min_seq_length]
        logger.info("Original indices count: %s, Filtered indices count: %s", len(indices), len(filtered_indices))
        return filtered_indices

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.  Uses an explicit
        DistributedSampler with drop_last=True under distributed training.
        See _make_distributed_sampler.
        """
        sampler = _make_distributed_sampler(self.train_dataset, shuffle=True, seed=self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=(sampler is None),
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.
        """
        sampler = _make_distributed_sampler(self.val_dataset, shuffle=False, seed=self.seed)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )

    def teardown(self, stage=None):
        """
        Clean up resources when the DataModule is no longer needed.

        This method properly closes all open HDF5 files to prevent resource
        leaks, especially important for large datasets.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        # Close HDF5 files properly
        if stage == 'fit' or stage is None:
            for dataset in self.train_dataset.datasets:
                dataset.dataset.close()
            for dataset in self.val_dataset.datasets:
                dataset.dataset.close()


# Backward compatibility alias
HDF5_PFamDataModule = HDF5DataModule


class PL_ProtARDM_Finetune(PL_ProtARDM):
    """ProteoScribe finetuning module that embeds text prompts to z_c on-device.

    Identical to :class:`PL_ProtARDM` except that each batch arrives as
    ``(num_seqs, input_ids)`` rather than ``(num_seqs, z_c)``. A frozen
    text->z_c front-end (PenCL text branch + Facilitator) computes z_c on the
    device under ``no_grad`` before the inherited diffusion training step runs.

    The embedder is intentionally kept *outside* the LightningModule's module
    tree (stored in a one-element list): its frozen parameters then stay out of
    ``self.parameters()`` (so the optimizer and DeepSpeed never see them) and out
    of saved checkpoints (rebuilt from PenCL/Facilitator weights on resume). It
    runs in fp32 eval mode, so its embeddings match Stage 1/2 inference run
    without autocast. With ``text_attention_mask`` the batch carries the caption
    attention mask after ``input_ids`` and the embedder passes it to BERT.
    """

    def __init__(self, args, model, embedder, zp_lookup=None, train_alpha=0.0,
                 eval_alpha=EVAL_SPREAD, text_attention_mask=False):
        super().__init__(args=args, model=model)
        embedder.eval()
        for p in embedder.parameters():
            p.requires_grad = False
        self._embedder_ref = [embedder]
        self.zp_lookup = zp_lookup
        self.text_attention_mask = text_attention_mask
        self.train_alpha = normalize_alpha_spec(train_alpha)
        self.eval_alpha = resolve_eval_alpha(eval_alpha)

    def _train_alpha(self, n, device):
        """Per-example weight on z_p for a *training* batch, shaped ``[n, 1]``."""
        if self.train_alpha != ALPHA_BLEND:
            return torch.full((n, 1), float(self.train_alpha), device=device)
        return sample_blend_alpha(n, device=device)

    def _eval_alpha(self, sequences, device):
        """Per-example weight on z_p for a *validation* batch, shaped ``[n, 1]``.

        A constant applies one alpha to every example; ``"spread"`` gives each
        its own deterministic alpha keyed on the sequence, so the metric spans
        the operating range yet stays identical across epochs. Best-checkpoint
        selection then reflects the whole range, not a single point.
        """
        if self.eval_alpha == EVAL_SPREAD:
            vals = [deterministic_alpha(s) for s in sequences]
            return torch.tensor(vals, dtype=torch.float32, device=device).view(-1, 1)
        return torch.full((len(sequences), 1), self.eval_alpha, device=device)

    @property
    def embedder(self):
        return self._embedder_ref[0]

    def _is_training_batch(self):
        # self.trainer raises when unattached, so read the backing attribute.
        trainer = getattr(self, "_trainer", None)
        return self.training if trainer is None else trainer.training

    def on_after_batch_transfer(self, batch, dataloader_idx):
        num_seqs, input_ids = batch[0], batch[1]
        extras = list(batch[2:])
        embedder = self.embedder
        if next(embedder.parameters()).device != input_ids.device:
            embedder.to(input_ids.device)
        with torch.no_grad():
            if self.text_attention_mask:
                if not extras:
                    raise RuntimeError(
                        "text_attention_mask needs the caption attention mask in "
                        "the batch; build the data module with "
                        "text_attention_mask=True so the collate emits it."
                    )
                z_c = embedder(input_ids, attention_mask=extras.pop(0))
            else:
                z_c = embedder(input_ids)
        if self.zp_lookup is None:
            return [num_seqs, z_c]
        if not extras:
            raise RuntimeError(
                "z_p blending needs the raw sequences in the batch; build the "
                "data module with needs_unique_sequences=True so the collate "
                "emits them."
            )
        sequences = extras[0]
        z_p = torch.stack([self.zp_lookup[s] for s in sequences]).to(z_c)
        if self._is_training_batch():
            alpha = self._train_alpha(z_c.size(0), z_c.device)
        else:
            alpha = self._eval_alpha(sequences, z_c.device).to(z_c)
        y = alpha * z_p + (1.0 - alpha) * z_c
        return [num_seqs, y]


class _SequenceView:
    """Indexable view over a record source exposing each row's raw sequence.

    Lets :func:`biom3.split.manifest.compute_fingerprint` sample sequences by row
    index without materializing them all (it only reads a handful of positions),
    so it stays cheap even over a lazy :class:`JsonlRecordStore`.
    """

    def __init__(self, source, key):
        self._source = source
        self._key = key

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx):
        return self._source[idx][self._key]


class GeneralizedDataModule(pl.LightningDataModule):
    """Schema-driven finetuning DataModule over cleaned records in JSONL.

    Each record is ``{sequence, fields: {key: raw_description, ...}, <length_field>}``.
    A ``record_schema`` composes a ``caption`` (via the ``fields_to_caption``
    compose function) and passes the sequence through; on each access the
    composition is re-rolled (per-key dropout + optional shuffle). The collate
    tokenizes both into ``(num_seqs, input_ids)`` for :class:`PL_ProtARDM_Finetune`.

    Records are read eagerly (:func:`~biom3.core.dataloaders.read_jsonl_records`)
    by default, or lazily (:class:`~biom3.core.dataloaders.JsonlRecordStore`) when
    ``lazy=True``. Train/val splitting and sequence-length filtering mirror
    :class:`HDF5DataModule`; lengths come from ``length_field`` when present and
    are otherwise computed from the (ungapped) sequence.
    """

    def __init__(
        self, *,
        jsonl_path,
        record_schema,
        text_model_path,
        text_max_length,
        batch_size,
        num_workers,
        valid_size,
        seed,
        diffusion_steps,
        image_size,
        sequence_key="sequence",
        caption_key="caption",
        length_field="sequence_length",
        lazy=False,
        split_manifest_path=None,
        needs_unique_sequences=False,
        text_attention_mask=False,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.record_schema = record_schema
        self.text_model_path = text_model_path
        self.text_max_length = text_max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.seed = seed
        self.diffusion_steps = diffusion_steps
        self.image_size = image_size
        self.sequence_key = sequence_key
        self.caption_key = caption_key
        self.length_field = length_field
        self.lazy = lazy
        self.split_manifest_path = split_manifest_path
        self.needs_unique_sequences = needs_unique_sequences
        self.text_attention_mask = text_attention_mask
        self.min_seq_length = diffusion_steps - 2

    def setup(self, stage=None):
        if self.lazy:
            source = JsonlRecordStore(
                self.jsonl_path, scalar_fields=[self.length_field],
            )
            lengths = list(source.get_scalar(self.length_field))
        else:
            source = read_jsonl_records(self.jsonl_path)
            lengths = [rec.get(self.length_field) for rec in source]

        dataset = GeneralizedRecordDataset(
            source, schema=self.record_schema, rng=None,
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_path)
        self._collate_fn = prep.make_seq_caption_collate_fn(
            text_tokenizer=self.text_tokenizer,
            text_max_length=self.text_max_length,
            image_size=self.image_size,
            sequence_key=self.sequence_key,
            caption_key=self.caption_key,
            include_sequences=self.needs_unique_sequences,
            include_mask=self.text_attention_mask,
        )

        lengths = self._resolve_lengths(source, dataset, lengths)
        if self.split_manifest_path is not None:
            train_idx, val_idx, test_idx = self._manifest_split(source)
        else:
            train_idx, val_idx = self._split_indices(len(dataset))
            test_idx = []
        train_idx = self._filter_by_length(train_idx, lengths)
        val_idx = self._filter_by_length(val_idx, lengths)
        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)
        self.split_info = [{
            "path": self.jsonl_path,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }]
        if self.needs_unique_sequences:
            # Keys must match what the collate emits, so read the sequence the
            # same way it does: the raw record value for a passthrough spec,
            # otherwise through the composed dataset.
            raw_key = self._raw_sequence_key()
            seqs = (
                (source[i][raw_key] if raw_key is not None
                 else dataset[i][self.sequence_key])
                for i in list(train_idx) + list(val_idx)
            )
            self._unique_sequences = list(dict.fromkeys(seqs))
        else:
            self._unique_sequences = None
        logger.info(
            "Loaded generalized dataset from %s (%d train, %d val, %d held-out test)",
            self.jsonl_path, len(train_idx), len(val_idx), len(test_idx))

    def unique_sequences(self):
        """Deduplicated train+val sequences, for the one-off z_p precompute."""
        if self._unique_sequences is None:
            raise RuntimeError(
                "unique_sequences() requires the data module to be built with "
                "needs_unique_sequences=True, which run_ProteoScribe_finetuning "
                "sets whenever train_alpha puts weight on z_p."
            )
        return self._unique_sequences

    def _manifest_split(self, source):
        """Read train/val/test row indices from a curated split manifest.

        Validates that the manifest was built for this exact JSONL (row count +
        content fingerprint over the raw sequences) before trusting its indices.
        The test split is returned for record-keeping but held out of training.
        """
        from biom3.split import manifest as split_manifest

        manifest = split_manifest.read_manifest(self.split_manifest_path)
        if len(manifest["files"]) != 1:
            raise ValueError(
                f"split manifest describes {len(manifest['files'])} file(s) but the "
                f"generalized dataloader reads a single JSONL; regenerate the manifest"
            )
        raw_key = self._raw_sequence_key()
        if raw_key is None:
            raise ValueError(
                "split_manifest_path requires the sequence output to be a plain "
                "passthrough ({'from': <key>}); fingerprint validation needs the "
                "raw record sequences"
            )
        entry = manifest["files"][0]
        split_manifest.validate_file_entry(
            entry,
            n_rows=len(source),
            fingerprint=split_manifest.compute_fingerprint(
                _SequenceView(source, raw_key)
            ),
        )
        logger.info("Using curated split manifest %s", self.split_manifest_path)
        return entry["train"], entry["val"], entry["test"]

    def _raw_sequence_key(self):
        """Raw record key for the sequence when it is a plain passthrough.

        Returns the source key for ``sequence_key`` when its schema spec is a
        simple ``{"from": key}`` / bare-string passthrough, else ``None`` (the
        sequence is composed and must be read through the dataset).
        """
        spec = self.record_schema.get(self.sequence_key)
        if isinstance(spec, str):
            return spec
        if isinstance(spec, dict) and "from" in spec:
            return spec["from"]
        return None

    def _resolve_lengths(self, source, dataset, lengths):
        """Fill in any missing ``length_field`` values from the ungapped sequence.

        Reads the raw sequence directly when it is a passthrough output (avoiding
        a throwaway caption composition per record); otherwise falls back to the
        composed dataset output.
        """
        lengths = list(lengths)
        raw_key = self._raw_sequence_key()
        fallbacks = 0
        for i, length in enumerate(lengths):
            if length is None:
                if raw_key is not None:
                    seq = source[i][raw_key]
                else:
                    seq = dataset[i][self.sequence_key]
                lengths[i] = len(seq.replace('-', ''))
                fallbacks += 1
        if fallbacks:
            logger.info("Computed %d/%d sequence lengths from sequence (missing %r)",
                        fallbacks, len(lengths), self.length_field)
        return lengths

    def _split_indices(self, n):
        train_size = int((1 - self.valid_size) * n)
        indices = np.arange(n)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        return indices[:train_size], indices[train_size:]

    def _filter_by_length(self, indices, lengths):
        filtered = [idx for idx in indices if lengths[idx] <= self.min_seq_length]
        logger.info("Original indices count: %s, Filtered indices count: %s",
                    len(indices), len(filtered))
        return filtered

    def train_dataloader(self):
        sampler = _make_distributed_sampler(self.train_dataset, shuffle=True, seed=self.seed)
        # persistent_workers is left False so workers re-spawn each epoch and torch
        # reseeds them, which is what makes GeneralizedRecordDataset's per-worker RNG
        # vary across epochs (see its docstring caveat).
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        sampler = _make_distributed_sampler(self.val_dataset, shuffle=False, seed=self.seed)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

