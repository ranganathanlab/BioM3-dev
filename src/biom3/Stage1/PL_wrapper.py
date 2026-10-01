# pytorch fucntions
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributed as dist

# PL functions
from biom3.backend.device import BACKEND_NAME, _XPU

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning import Trainer, seed_everything
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything

# misc functions
from contextlib import nullcontext
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import time

# other learning packages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# our packages
import biom3.Stage1.preprocess as prep
import biom3.Stage1.model as mod
from biom3.backend.device import print_gpu_initialization, print_memory_usage


def _safe_barrier():
    # No-op when the default process group is not initialized (single-GPU,
    # CPU smoke tests, inference). Safe replacement for `dist.barrier()`.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()



class _GatherGrad(torch.autograd.Function):
    """all_gather whose backward is ONE collective, not an all-to-all.

    torch.distributed.nn.functional.all_gather -- what
    LightningModule.all_gather(sync_grads=True) ultimately calls -- reduces its
    gradient with reduce_scatter only when the backend is NCCL. For every other
    backend it emulates reduce_scatter with an all-to-all of world_size tensors
    (torch/distributed/nn/functional.py::_AllGather.backward). Aurora runs xccl,
    so every gathered tensor took the emulation path: an all-to-all is O(W)
    messages per rank, hence O(W^2) across the fabric, once per gather per step.

    That is the quadratic term in the measured s_step = 7.21 + 1.93e-4*W^2
    (8.85s at 96 ranks, 35.71s at 384 -- 4x the ranks bought no throughput).

    The gradient of an all_gather is each rank's slice of the incoming gradient
    summed over ranks, so one all_reduce and a slice is exact. all_reduce moves
    twice what reduce_scatter would, which is irrelevant next to removing W^2,
    and unlike reduce_scatter it is known to work on every backend we run.
    """

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
        tensor = tensor.contiguous()
        out = tensor.new_empty((world,) + tuple(tensor.shape))
        dist.all_gather_into_tensor(out, tensor, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_out[ctx.rank], None


def _gather_with_grad(tensor, group=None):
    """[B, ...] on each rank -> [W, B, ...] everywhere, gradients intact.

    Matches LightningModule.all_gather(sync_grads=True) in shape and value, and
    like it returns a leading axis of 1 when not running distributed.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor.unsqueeze(0)
    return _GatherGrad.apply(tensor, group)


def _log_reduced(module, scalars, prog_bar_keys=(), on_step=True, on_epoch=True):
    """self.log() every scalar, but with ONE cross-rank reduction for all of them.

    Lightning's sync_dist=True issues a separate collective per logged value.
    The Pfam training step logs 18 (5 losses + 12 sklearn metrics + memory), and
    at scale it is the NUMBER of collectives, not their payload, that costs:
    measured s_step went 8.85s at 96 ranks to 35.71s at 384. Stacking the
    scalars into a single all_reduce yields identical values -- Lightning's
    default sync_dist reduction is also a mean -- for one collective, not 18.
    """
    keys = list(scalars)
    vals = torch.stack([
        torch.as_tensor(scalars[k], dtype=torch.float32,
                        device=module.device).detach().reshape(())
        for k in keys
    ])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        vals = vals / dist.get_world_size()
    for k, v in zip(keys, vals):
        module.log(k, v, prog_bar=(k in prog_bar_keys or 'f1' in k),
                   on_step=on_step, on_epoch=on_epoch, sync_dist=False)


def _gather_four(module, a, b, c, d):
    """all_gather four [B, D] tensors in one collective, preserving layout.

    A [W, B, D] gather viewed as (-1, D) is rank-major; so is slicing a
    [W, 4B, D] gather on the middle axis. The results are bit-identical to four
    separate gathers, for a quarter of the collectives.
    """
    B, D = a.shape[0], a.shape[-1]
    fused = torch.cat((a, b, c, d), dim=0)
    out = _gather_with_grad(fused).reshape(-1, 4 * B, D)
    return tuple(out[:, i * B:(i + 1) * B, :].reshape(-1, D) for i in range(4))


def _gather_keys(module, swiss_cols, pfam_cols):
    """[M, K] int64 false-negative keys, in the same layout as z_*_all.

    Each rank holds [B] keys for its Swiss-Prot rows and [B] for its Pfam rows;
    they are stacked to [2B, K], gathered in ONE collective, then split back so
    the result is [swiss block ; pfam block], rank-major -- the layout
    _contrastive_row_index assumes.

    Cheap: 2 int64 columns over M = 49,152 rows is 786 KB, against ~200 MB for
    the four embedding tensors already gathered every step. Keys carry no
    gradient, so this does not go through _gather_with_grad.
    """
    if not swiss_cols:
        return None
    swiss = torch.stack(swiss_cols, dim=-1)                      # [B, K]
    pfam = torch.stack(pfam_cols, dim=-1)                        # [B, K]
    B, K = swiss.shape
    out = module.all_gather(torch.cat((swiss, pfam), dim=0), sync_grads=False)
    out = out.reshape(-1, 2 * B, K)                              # [W, 2B, K]
    return torch.cat((out[:, :B].reshape(-1, K), out[:, B:].reshape(-1, K)), dim=0)


def _stage1_keys(module, swiss_seq, pfam_seq, family):
    """Assemble and gather whatever false-negative keys the config enables."""
    args = module.script_args
    sc, pc = [], []
    if getattr(args, 'mask_same_sequence', False):
        sc.append(swiss_seq.long()); pc.append(pfam_seq.long())
    if getattr(args, 'mask_same_family', False):
        # Both halves of a pair carry the family the pair was drawn on.
        sc.append(family.long()); pc.append(family.long())
    return _gather_keys(module, sc, pc)


def _contrastive_row_index(module, micro_batch, world_size, rank, device):
    """Global row indices this rank owns in the gathered [2*W*B, D] batch.

    all_gather returns [W, B, D] and the caller does .view(-1, D), so the layout
    is rank-major: rank r owns Swiss-Prot rows [r*B, (r+1)*B) and, after the
    Swiss/Pfam concat, Pfam rows [N + r*B, N + (r+1)*B) with N = W*B.
    """
    N = world_size * micro_batch
    swiss = torch.arange(rank * micro_batch, (rank + 1) * micro_batch, device=device)
    return torch.cat([swiss, swiss + N])


def _sharded_inter_intra(model, z_p_all, z_t_all, micro_batch, gather_fn, keys=None):
    """Row-sharded L_GC and L_PFC, equal to the dense pair (see
    tests/stage1_tests/test_sharded_contrastive.py -- values AND gradients).

    The dense path builds the full M x M similarity matrix on every rank, which
    is O(W^2): ~2 GB at 512 ranks, 32 GB at 2048. This computes only this rank's
    [2B, M] rows.
    """
    import torch.distributed as dist

    M = z_p_all.shape[0]
    N = M // 2
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    row_index = _contrastive_row_index(model, micro_batch, world_size, rank,
                                       z_p_all.device)

    # targets is a ROW-wise softmax and the protein-side term consumes targets.T,
    # so every column entry needs a different row's normaliser. Each rank can
    # only compute its own rows' normalisers; gather them. sync_grads=True is
    # required -- targets is differentiable in the dense path, so detaching here
    # would match the forward and silently change the backward.
    lz = model.inter_row_logsumexp(z_p_all, z_t_all, row_index, keys)  # [2B]
    # One gather, not two: gathering [2B] gives [W, 2B], and column-slicing that
    # is exactly what gathering the two halves separately produced.
    lz_all = gather_fn(lz).reshape(-1, 2 * micro_batch)              # [W, 2B]
    lz_swiss = lz_all[:, :micro_batch].reshape(-1)                   # [N]
    lz_pfam = lz_all[:, micro_batch:].reshape(-1)                    # [N]
    row_logZ = torch.cat([lz_swiss, lz_pfam])                        # [M]

    loss_align, logits = model.compute_inter_loss_sharded(
        z_p_all, z_t_all, N, row_index, row_logZ, keys)
    loss_intra, cosine = model.compute_intra_loss_sharded(
        z_p_all, N, row_index, keys)
    return loss_align, logits, loss_intra, cosine, row_index


UNIFORMITY_MODALITIES = {'protein': ('protein',), 'text': ('text',),
                         'both': ('protein', 'text')}


def _uniformity_losses(model, args, z_p_all, z_t_all, micro_batch, impl, gather_fn):
    """L_unif per selected modality, on the gathered [M, D] batch.

    Every rank returns the same global scalars. Dense builds all M rows;
    sharded builds this rank's [2B, M] rows and gathers their [2B] logsumexps
    (one collective for both modalities -- logsumexp is order-invariant, so the
    rank-major layout of the gather does not matter).
    """
    t = args.uniformity_t
    z = {'protein': z_p_all, 'text': z_t_all}
    names = UNIFORMITY_MODALITIES[args.uniformity_on]
    if impl != 'sharded':
        return {n: model.compute_uniformity_loss(z[n], t) for n in names}
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    rows = _contrastive_row_index(model, micro_batch, world_size, rank, z_p_all.device)
    lse = torch.stack([model.uniformity_row_logsumexp(z[n], rows, t) for n in names])
    lse_all = gather_fn(lse)                                         # [W, K, 2B]
    return {n: model.uniformity_from_row_logsumexp(lse_all[:, k].reshape(-1), t)
            for k, n in enumerate(names)}


def _performance_metrics_sharded(module, logits_rows, logits_cols, row_index):
    """performance_metrics() for row-sharded logits.

    logits_rows [R, M]: this rank's text anchors against every protein.
    logits_cols [M, R]: every text against this rank's protein anchors.

    The dense version assumes a square matrix whose diagonal is the correct
    pairing (y_true = arange(M)). Here local row k is global row row_index[k],
    so the correct column is row_index[k], not k.
    """
    lr = logits_rows.cpu().float()
    lc = logits_cols.cpu().float()
    p_text = F.softmax(lr, dim=-1)        # [R, M] text anchor -> proteins
    p_seq = F.softmax(lc.T, dim=-1)       # [R, M] protein anchor -> texts
    p_tot = (p_seq + p_text) / 2
    y_true = row_index.detach().cpu()
    out = {}
    for pred, source in ((torch.argmax(p_text, dim=-1), 'text'),
                         (torch.argmax(p_seq, dim=-1), 'seq'),
                         (torch.argmax(p_tot, dim=-1), 'total')):
        out.update(module.compute_class_metrics(outputs=pred, targets=y_true,
                                                source=source))
    return out


######################
# Default PL wrapper #
######################

class PL_PEN_CL(pl.LightningModule):


    def __init__(
            self,
            args: any,
            model: nn.Module,
            text_tokenizer: any,
            sequence_tokenizer: any
        ):

        super().__init__()
        # arguments
        self.script_args = args
        
        # model components
        self.model = model

        # tokenizers
        self.text_tokenizer = text_tokenizer
        self.sequence_tokenizer = sequence_tokenizer
        
        # validation tracker for outputs
        self.val_text_joint_latents = []
        self.val_seq_joint_latents = []

        # prediction tracker for outputs
        self.predict_text_joint_latents = []
        self.predict_seq_joint_latents = []
            
    def forward(
            self,
            x_t: torch.Tensor,
            x_s: torch.Tensor
        ) -> (
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
        ):

        outputs = self.model(
                        x_t=x_t,
                        x_s=x_s
        )

        return (
                outputs['text_joint_latent'],
                outputs['seq_joint_latent'],
        )

    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: any,
        ) -> dict:
         
        if isinstance(batch, list):
            # split the 
            text_batch, protein_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch
        )
        _safe_barrier()

        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True)
        _safe_barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True)
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])
      
        # compute loss values
        loss, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        # track loss ...
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # track metrics
        metric_dict = self.performance_metrics(logits=logits)
        for key in metric_dict:
            values = metric_dict[key]

            final_key = 'train_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, on_step=True, on_epoch=True, sync_dist=True)
        
        if batch_idx == 0:
            gpu_memory_usage = print_gpu_initialization()
            self.log(f'gpu_memory_usage', gpu_memory_usage, sync_dist=True)

        return {'loss': loss}


    def validation_step(
            self,
            batch: list,
            batch_idx: any
        ) -> dict:

        # split the batch
        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch
        )
        
        _safe_barrier()
        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True).view(-1, z_t.shape[-1])
        _safe_barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True).view(-1, z_s.shape[-1])
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])

        # compute loss values
        loss, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        
        # track validation loss ...
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)

        # copmute validation metrics
        metric_dict = self.performance_metrics(logits=logits.detach().cpu())

        for key in metric_dict:
            values = metric_dict[key]
            final_key = 'valid_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, sync_dist=True)
        
        # collect joint embedding
        self.val_text_joint_latents.append(z_t_all.detach().cpu())
        self.val_seq_joint_latents.append(z_s_all.detach().cpu())

        return {'valid_loss': loss}

    def on_validation_epoch_end(self):
        
        # collect and aggregate outputs from all validation steps
        val_z_t_joint = torch.cat(self.val_text_joint_latents, dim=0)
        val_z_s_joint = torch.cat(self.val_seq_joint_latents, dim=0)
        
        # compute singular values 
        text_log_sigma_k, S_text = self.compute_singular(val_z_t_joint.detach().cpu())
        protein_log_sigma_k, S_protein = self.compute_singular(val_z_s_joint.detach().cpu())
        
        # save image pngs for tracking dimensionality collapse
        self.save_png_to_tensorboard(
                data=text_log_sigma_k.numpy(),
                title='text',
        )
        self.save_png_to_tensorboard(
                data=protein_log_sigma_k.numpy(),
                title='protein'
        )

        # free memory
        self.val_text_joint_latents.clear()
        self.val_seq_joint_latents.clear()
        

        # compute effective rank (RankME):
        erank_text = self.compute_effective_rank(sigma_ks=S_text)
        erank_protein = self.compute_effective_rank(sigma_ks=S_protein)
        
        # log erank metrics. The singular values are computed on CPU above, so
        # the effective ranks land there too; sync_dist all-reduces whatever it
        # is given, and an XPU process group has no backend for a CPU tensor.
        self.log('valid_erank_text', erank_text.to(self.device), sync_dist=True)
        self.log('valid_erank_protein', erank_protein.to(self.device), sync_dist=True)


    def configure_optimizers(self,):

        params = [
                {"params": self.model.protein_encoder.parameters(), "lr": self.script_args.protein_encoder_lr},
                {"params": self.model.text_encoder.parameters(), "lr": self.script_args.text_encoder_lr},
                {"params": itertools.chain(
                    self.model.protein_projection.parameters(),
                    self.model.text_projection.parameters()
                    ),
                "lr": self.script_args.head_lr,
                "weight_decay": self.script_args.weight_decay}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=self.script_args.weight_decay)

        return {
                "optimizer": optimizer,
        }
    
    @torch.no_grad()
    def compute_class_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            source: str
        ) -> dict:

        # convert torch tensors to numpy array
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()

        # compute the metrics
        accuracy = accuracy_score(targets_np, outputs_np.round())
        precision = precision_score(targets_np, outputs_np.round(), average='micro')
        recall = recall_score(targets_np, outputs_np.round(), average='micro')
        f1 = f1_score(targets_np, outputs_np.round(), average='micro')

        return {
                f'{source}_accuracy': accuracy,
                f'{source}_precision': precision,
                f'{source}_recall': recall,
                f'{source}_f1': f1
        }

    @torch.no_grad()
    def performance_metrics(self, logits: torch.Tensor) -> tuple:

        logits = logits.cpu().float()

        # get probs
        p_text = F.softmax(logits, dim=-1) # prob of a given text captions aligning well with seq. pairs
        p_seq = F.softmax(logits.T, dim=-1) # prob of a given seq aligning well with text pairs
        p_tot = (p_seq + p_text) / 2 # total prob

        # get class labels
        y_pred_text = torch.argmax(p_text, dim=-1)
        y_pred_seq = torch.argmax(p_seq, dim=-1)
        y_pred = torch.argmax(p_tot, dim=-1)
        y_true = torch.arange(y_pred_text.shape[0])
        
        # compute class metrics
        text_metrics = self.compute_class_metrics(
                                    outputs=y_pred_text,
                                    targets=y_true,
                                    source='text'
        )
        seq_metrics = self.compute_class_metrics(
                                    outputs=y_pred_seq,
                                    targets=y_true,
                                    source='seq'
        )
        total_metrics = self.compute_class_metrics(
                                    outputs=y_pred,
                                    targets=y_true,
                                    source='total'
        )

        # combine dicts into one
        combined_dict = {}
        combined_dict.update(text_metrics)
        combined_dict.update(seq_metrics)
        combined_dict.update(total_metrics)

        return combined_dict
    
    @torch.no_grad()
    def compute_singular(self, inputs: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):

        # goal of this function: track for dimensionality collapse
        # inputs dim: (batch_size, emb_dim)

        mean_inputs = torch.mean(inputs, dim=0) # average over batch dimension
        norm_inputs = inputs - mean_inputs # normalize vectors
       
       # compute correlation matrix  #TODO: double check work...
        C = torch.zeros((norm_inputs.shape[-1], norm_inputs.shape[-1]))
        for sample_idx in range(norm_inputs.shape[0]):
            norm_vector = norm_inputs[sample_idx, :].unsqueeze(0)
            C += norm_vector.T @ norm_vector
        C *= 1/norm_vector.shape[0]

        _, S, _ = torch.linalg.svd(C, full_matrices=False)
        
        # return singular value indexes 
        log_sigma_k, _ = torch.sort(torch.log(S), descending=True)
        return (
                log_sigma_k,
                S
        )
        
    def compute_effective_rank(self, sigma_ks: torch.Tensor) -> torch.Tensor:
        """
        references:
            - Roy et al. The effective rank: a measure of effective dimensionality
            - Garrido et al. RankMe: Assessing the Downstream Performnace of Pretrained SS Reps by their Rank.
        """
        # sort the singular values
        sigma_ks, _ = torch.sort(sigma_ks, descending=True)
        
        # copute L1 norm for sing values.
        l1_norm_sigma = torch.norm(sigma_ks, p=1)
        
        # compute singular value distribution
        p_k = sigma_ks / l1_norm_sigma  + torch.finfo(torch.float).eps
        
        # compute Shannon entropy
        entropy = - torch.sum(p_k * torch.log(p_k))
    
        # get effective rank (RankME):
        erank = torch.exp(entropy)

        return erank

    def save_png_to_tensorboard(
            self,
            data: np.single,
            title: str,
            x_axis_label: str='Singular Value Rank Index',
            y_axis_label: str='Log of singular values',
            ):
    
        current_epoch = self.trainer.current_epoch
        
        # Plot the line
        fig, ax = plt.subplots(dpi=300)
        ax.plot(data)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_ylim([-25,3])

        # Log the plot in TensorBoard 
        self.logger.experiment.add_figure(f'{title}_SingularValues_{current_epoch}', fig, current_epoch)

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: torch.Tensor,
            dataloder_idx: bool=False
        ) -> (  
                torch.Tensor,
                torch.Tensor
        ):


        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
            outputs = self(
                    x_t=text_batch,
                    x_s=protein_batch,
            )
        
        z_t_joint, z_p_joint = outputs

        self.predict_text_joint_latents.append(z_t_joint.detach().cpu())
        self.predict_seq_joint_latents.append(z_p_joint.detach().cpu())

        return outputs

    def on_predict_epoch_end(self, outputs=None):

        self.predict_text_joint_latents = torch.cat(self.predict_text_joint_latents).cpu()
        self.predict_seq_joint_latents = torch.cat(self.predict_seq_joint_latents).cpu()



##########################
# Masked-task PL wrapper #
##########################

class mask_PL_PEN_CL(pl.LightningModule):


    def __init__(
            self,
            args: any,
            model: nn.Module,
            text_tokenizer: any,
            sequence_tokenizer: any
    ):

        super().__init__()
        # arguments
        self.script_args = args
        
        # model components
        self.model = model
    
        # tokenizers
        self.text_tokenizer = text_tokenizer
        self.sequence_tokenizer = sequence_tokenizer
        
        # validation tracker for outputs
        self.val_text_joint_latents = []
        self.val_seq_joint_latents = []
    
        # prediction tracker for outputs
        self.predict_text_joint_latents = []
        self.predict_seq_joint_latents = []

    def forward(
            self,
            x_t: torch.Tensor,
            x_s: torch.Tensor,
            compute_masked_logits: bool=False
        ) -> (
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
        ):

        outputs = self.model(
                        x_t=x_t,
                        x_s=x_s,
                        compute_masked_logits=compute_masked_logits
        )
        
        if compute_masked_logits:
            # forward pass for computing logits for masked language objective
            return (
                    outputs['text_masked_logits'],
                    outputs['protein_masked_logits']
            )
        else:
            # forward pass for computing latent embeddings in the joint space
            return (
                outputs['text_joint_latent'],
                outputs['seq_joint_latent'],
            )

    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: any,
        ) -> dict:
         
        if isinstance(batch, list):
            # split the data
            text_batch, protein_batch, text_mask_batch, protein_mask_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch,
                    compute_masked_logits=False
        )
        _safe_barrier()

        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True)
        _safe_barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True)
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])
      
        # compute loss values
        loss_align, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        # compute mask language model logits
        logits_t_mask, logits_s_mask = self(
                x_t=text_mask_batch,
                x_s=protein_mask_batch,
                compute_masked_logits=True
        )
        
        # compute mask language loss for biomedical expert model
        loss_text_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_t_mask,
                targets=text_batch,
                targets_masked=text_mask_batch,
                mask_token_id=self.text_tokenizer.mask_token_id
        )
        
        # compute mask language loss for protein expert model
        loss_sequence_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_s_mask,
                targets=protein_batch,
                targets_masked=protein_mask_batch,
                mask_token_id=self.sequence_tokenizer.mask_idx
        )
        
        
        # total loss
        loss = loss_align + loss_text_mask + loss_sequence_mask

        # track loss ...
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_align', loss_align, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_text_mask', loss_text_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_seq_mask', loss_sequence_mask, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        # track metrics
        metric_dict = self.performance_metrics(logits=logits)
        for key in metric_dict:
            values = metric_dict[key]

            final_key = 'train_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, on_step=True, on_epoch=True, sync_dist=True)
        
        if batch_idx == 0:
            gpu_memory_usage = print_gpu_initialization()
            self.log(f'gpu_memory_usage', gpu_memory_usage, sync_dist=True)

        return {'loss': loss}


    def validation_step(
            self,
            batch: list,
            batch_idx: any
        ) -> dict:

        # split the batch
        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch, text_mask_batch, protein_mask_batch = batch
        
        # forward pass 
        z_t, z_s = self(
                    x_t=text_batch,
                    x_s=protein_batch
        )
        
        _safe_barrier()
        # gather all tensors
        z_t_all = self.all_gather(z_t, sync_grads=True).view(-1, z_t.shape[-1])
        _safe_barrier()
        z_s_all = self.all_gather(z_s, sync_grads=True).view(-1, z_s.shape[-1])
        
        # stack the embeddings
        z_t_all = z_t_all.view(-1, z_t.shape[-1])
        z_s_all = z_s_all.view(-1, z_s.shape[-1])

        # compute loss values
        loss_align, logits = self.model.compute_loss(
                protein_embeddings=z_s_all,
                text_embeddings=z_t_all
        )
        
        # compute mask language model logits
        logits_t_mask, logits_s_mask = self(
                x_t=text_mask_batch,
                x_s=protein_mask_batch,
                compute_masked_logits=True
        )
        
        # compute mask language loss for biomedical expert model
        loss_text_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_t_mask,
                targets=text_batch,
                targets_masked=text_mask_batch,
                mask_token_id=self.text_tokenizer.mask_token_id
        )
        
        # compute mask language loss for protein expert model
        loss_sequence_mask = self.model.compute_masked_lang_loss(
                logits_masked=logits_s_mask,
                targets=protein_batch,
                targets_masked=protein_mask_batch,
                mask_token_id=self.sequence_tokenizer.mask_idx
        )
        
        # total loss
        loss = loss_align + loss_text_mask + loss_sequence_mask

        # track validation loss ...
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.log('valid_loss_align', loss_align, prog_bar=True, sync_dist=True)
        self.log('valid_loss_text_mask', loss_text_mask, prog_bar=False, sync_dist=True)
        self.log('valid_loss_seq_mask', loss_sequence_mask, prog_bar=False, sync_dist=True)

        # copmute validation metrics
        metric_dict = self.performance_metrics(logits=logits.detach().cpu())

        for key in metric_dict:
            values = metric_dict[key]
            final_key = 'valid_' + key
            self.log(final_key, metric_dict[key], prog_bar=True if 'f1' in key else False, sync_dist=True)
        
        # collect joint embedding
        self.val_text_joint_latents.append(z_t_all.detach().cpu())
        self.val_seq_joint_latents.append(z_s_all.detach().cpu())

        return {'valid_loss': loss}

    def on_validation_epoch_end(self):
        
 #       # collect and aggregate outputs from all validation steps
 #      val_z_t_joint = torch.cat(self.val_text_joint_latents, dim=0)
 #       val_z_s_joint = torch.cat(self.val_seq_joint_latents, dim=0)
        
        # compute singular values 
 #       text_log_sigma_k, S_text = self.compute_singular(val_z_t_joint.detach().cpu())
 #       protein_log_sigma_k, S_protein = self.compute_singular(val_z_s_joint.detach().cpu())
        
        # save image pngs for tracking dimensionality collapse
 #       self.save_png_to_tensorboard(
 #               data=text_log_sigma_k.numpy(),
 #               title='text',
 #       )
 #       self.save_png_to_tensorboard(
 #               data=protein_log_sigma_k.numpy(),
 #               title='protein'
 #       )

        # free memory
        self.val_text_joint_latents.clear()
        self.val_seq_joint_latents.clear()
        

        # compute effective rank (RankME):
 #       erank_text = self.compute_effective_rank(sigma_ks=S_text)
 #       erank_protein = self.compute_effective_rank(sigma_ks=S_protein)
        
        # log erank metrics
 #       self.log('valid_erank_text', erank_text, sync_dist=True)
 #       self.log('valid_erank_protein', erank_protein, sync_dist=True)


    def configure_optimizers(self,):

        params = [
                {"params": self.model.protein_encoder.parameters(), "lr": self.script_args.protein_encoder_lr},
                {"params": self.model.text_encoder.parameters(), "lr": self.script_args.text_encoder_lr},
                {"params": itertools.chain(
                    self.model.protein_projection.parameters(),
                    self.model.text_projection.parameters()
                    ),
                "lr": self.script_args.head_lr,
                "weight_decay": self.script_args.weight_decay}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=self.script_args.weight_decay)

        return {
                "optimizer": optimizer,
        }
    
    @torch.no_grad()
    def compute_class_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            source: str
        ) -> dict:

        # convert torch tensors to numpy array
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()

        # compute the metrics
        accuracy = accuracy_score(targets_np, outputs_np.round())
        precision = precision_score(targets_np, outputs_np.round(), average='micro')
        recall = recall_score(targets_np, outputs_np.round(), average='micro')
        f1 = f1_score(targets_np, outputs_np.round(), average='micro')

        return {
                f'{source}_accuracy': accuracy,
                f'{source}_precision': precision,
                f'{source}_recall': recall,
                f'{source}_f1': f1
        }

    @torch.no_grad()
    def performance_metrics(self, logits: torch.Tensor) -> tuple:

        logits = logits.cpu().float()

        # get probs
        p_text = F.softmax(logits, dim=-1) # prob of a given text captions aligning well with seq. pairs
        p_seq = F.softmax(logits.T, dim=-1) # prob of a given seq aligning well with text pairs
        p_tot = (p_seq + p_text) / 2 # total prob

        # get class labels
        y_pred_text = torch.argmax(p_text, dim=-1)
        y_pred_seq = torch.argmax(p_seq, dim=-1)
        y_pred = torch.argmax(p_tot, dim=-1)
        y_true = torch.arange(y_pred_text.shape[0])
        
        # compute class metrics
        text_metrics = self.compute_class_metrics(
                                    outputs=y_pred_text,
                                    targets=y_true,
                                    source='text'
        )
        seq_metrics = self.compute_class_metrics(
                                    outputs=y_pred_seq,
                                    targets=y_true,
                                    source='seq'
        )
        total_metrics = self.compute_class_metrics(
                                    outputs=y_pred,
                                    targets=y_true,
                                    source='total'
        )

        # combine dicts into one
        combined_dict = {}
        combined_dict.update(text_metrics)
        combined_dict.update(seq_metrics)
        combined_dict.update(total_metrics)

        return combined_dict
    
    @torch.no_grad()
    def compute_singular(self, inputs: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):

        # goal of this function: track for dimensionality collapse
        # inputs dim: (batch_size, emb_dim)

        mean_inputs = torch.mean(inputs, dim=0) # average over batch dimension
        norm_inputs = inputs - mean_inputs # normalize vectors
       
       # compute correlation matrix  #TODO: double check work...
        C = torch.zeros((norm_inputs.shape[-1], norm_inputs.shape[-1]))
        for sample_idx in tqdm(range(norm_inputs.shape[0])):
            norm_vector = norm_inputs[sample_idx, :].unsqueeze(0)
            C += norm_vector.T @ norm_vector
        C *= 1/norm_vector.shape[0]

        _, S, _ = torch.linalg.svd(C, full_matrices=False)
        
        # return singular value indexes 
        log_sigma_k, _ = torch.sort(torch.log(S), descending=True)
        return (
                log_sigma_k,
                S
        )
        
    def compute_effective_rank(self, sigma_ks: torch.Tensor) -> torch.Tensor:
        """
        references:
            - Roy et al. The effective rank: a measure of effective dimensionality
            - Garrido et al. RankMe: Assessing the Downstream Performnace of Pretrained SS Reps by their Rank.
        """
        # sort the singular values
        sigma_ks, _ = torch.sort(sigma_ks, descending=True)
        
        # copute L1 norm for sing values.
        l1_norm_sigma = torch.norm(sigma_ks, p=1)
        
        # compute singular value distribution
        p_k = sigma_ks / l1_norm_sigma  + torch.finfo(torch.float).eps
        
        # compute Shannon entropy
        entropy = - torch.sum(p_k * torch.log(p_k))
    
        # get effective rank (RankME):
        erank = torch.exp(entropy)

        return erank

    def save_png_to_tensorboard(
            self,
            data: np.single,
            title: str,
            x_axis_label: str='Singular Value Rank Index',
            y_axis_label: str='Log of singular values',
            ):
    
        current_epoch = self.trainer.current_epoch
        
        # Plot the line
        fig, ax = plt.subplots(dpi=300)
        ax.plot(data)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_ylim([-25,3])

        # Log the plot in TensorBoard 
        self.logger.experiment.add_figure(f'{title}_SingularValues_{current_epoch}', fig, current_epoch)

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: torch.Tensor,
            dataloder_idx: bool=False
        ) -> (  
                torch.Tensor,
                torch.Tensor
        ):


        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
            outputs = self(
                    x_t=text_batch,
                    x_s=protein_batch,
                    compute_masked_logits=False
            )
        
        z_t_joint, z_p_joint = outputs

        self.predict_text_joint_latents.append(z_t_joint.detach().cpu())
        self.predict_seq_joint_latents.append(z_p_joint.detach().cpu())

        return outputs

    def on_predict_epoch_end(self, outputs=None):

        self.predict_text_joint_latents = torch.cat(self.predict_text_joint_latents).cpu()
        self.predict_seq_joint_latents = torch.cat(self.predict_seq_joint_latents).cpu()



########################
# Pfam-task PL wrapper #
########################


class pfam_PL_PEN_CL(pl.LightningModule):


    def __init__(
            self,
            args: any,
            model: nn.Module,
            text_tokenizer: any,
            sequence_tokenizer: any
    ):

        super().__init__()
        # arguments
        self.script_args = args
        
        # model components
        self.model = model
    
        # tokenizers
        self.text_tokenizer = text_tokenizer
        self.sequence_tokenizer = sequence_tokenizer
        
        # validation tracker for outputs
        self.val_text_joint_latents = []
        self.val_seq_joint_latents = []
    
        # predictions...
        self.predict_text_joint_latents = [] 
        self.predict_seq_joint_latents = []

    def forward(
            self,
            x_t: torch.Tensor,
            x_p: torch.Tensor,
            compute_masked_logits: bool=False,
            x_t_mask: torch.Tensor=None
        ) -> (
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
        ):

        outputs = self.model(
                        x_t=x_t,
                        x_s=x_p,
                        compute_masked_logits=compute_masked_logits,
                        x_t_mask=x_t_mask
        )
        
        if compute_masked_logits:
            # forward pass for computing logits for masked language objective
            return (
                    outputs['text_masked_logits'],
                    outputs['protein_masked_logits']
            )
        else:
            # forward pass for computing latent embeddings in the joint space
            return (
                outputs['text_joint_latent'],
                outputs['seq_joint_latent'],
            )


    def on_train_batch_start(self, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_end_time = time.time()
        batch_time = batch_end_time - self.batch_start_time
        #print(f'Rank={dist.get_rank()}: time to process batch is {batch_time}')
        #self.log(f'batch_time_rank_{dist.get_rank()}', batch_time, on_step=True, on_epoch=False)

    def _embedding_geometry(self, z_p_all, z_t_all, split):
        """Mean embedding norms and the temperature they actually imply.

        The contrastive softmax is sharpened by ||z_t|| ||z_p|| / tau, not by tau
        alone, and nothing in the objective constrains the norms: the model can
        buy back confidence by growing them instead of improving directions.
        """
        p = z_p_all.detach().norm(dim=-1).mean()
        t = z_t_all.detach().norm(dim=-1).mean()
        return {f'{split}_z_p_norm': p, f'{split}_z_t_norm': t,
                f'{split}_tau_eff': self.model.temperature / (p * t)}

    def _add_uniformity(self, loss, z_p_all, z_t_all, micro_batch, impl, split):
        """loss + weight * mean_modality(L_unif); a no-op when the weight is 0,
        and when the gathered batch is a single pair (M = 2): no pair survives
        the self/homolog exclusion, so L_unif would be NaN.

        With log_uniformity and weight 0 the term is measured but not trained on,
        so a control run reports the same metric as a weighted one.
        """
        weight = getattr(self.script_args, 'uniformity_weight', 0.0)
        log_only = weight <= 0 and getattr(self.script_args, 'log_uniformity', False)
        if weight <= 0 and not log_only:
            return loss, {}
        if z_p_all.shape[0] <= 2:
            return loss, {}
        with torch.no_grad() if log_only else nullcontext():
            losses = _uniformity_losses(self.model, self.script_args, z_p_all, z_t_all,
                                        micro_batch, impl, _gather_with_grad)
        if not log_only:
            loss = loss + weight * sum(losses.values()) / len(losses)
        return loss, {f'{split}_loss_unif_{n}': v for n, v in losses.items()}

    def training_step(self, batch: torch.Tensor, batch_idx: any) -> dict:
        """
        Execute a single training step.

        Given a batch of data, this function processes both Swiss-Prot and Pfam data through the model, computes 
        various loss values including inter-modal, intra-modal, and masked language model losses for both text 
        and protein sequences. This function also computes and logs various metrics and GPU memory usage.

        Parameters:
        - batch: The input data batch. This can include multiple types of data.
        - batch_idx: Index of the current batch.

        Steps:
        1. Split the data into Swiss-Prot and Pfam batches, if the batch is a list.
        2. Forward pass the Swiss-Prot data through the model.
        3. Synchronize and gather embeddings from all GPUs.
        4. Forward pass the Pfam data through the model.
        5. Synchronize and gather Pfam embeddings from all GPUs.
        6. Concatenate Swiss-Prot and Pfam embeddings.
        7. Compute inter-modal and intra-modal loss values.
        8. Compute masked language model logits for the concatenated batch.
        9. Compute masked language loss for both text and protein sequences.
        10. Compute and log the total loss and individual loss components.
        11. Compute and log performance metrics.
        12. Log GPU memory usage at the start of training.

        Returns:
        - Dictionary containing the total loss value.

        Note: 
        This function is intended to be used within a distributed (multi-GPU) training context, as evident 
        from the use of barriers and gathering operations. It's designed to handle batches that contain both 
        Swiss-Prot and Pfam data, both being biological datasets used in multi-modal protein embeddings. 
        The function utilizes both inter-modal (between modalities) and intra-modal (within the same modality) 
        contrastive losses, as well as masked language modeling objectives similar to BERT's MLM objective.
        """

        # Check if the batch is a list and split data if so.
        if isinstance(batch, list):
            # NB: *_mask_batch are the masked-LM corrupted tokens; *_attn_mask
            # are the BERT attention masks marking real tokens vs [PAD].
            text_batch, protein_batch, text_mask_batch, protein_mask_batch, \
            pfam_text_batch, pfam_protein_batch, pfam_text_mask_batch, pfam_protein_mask_batch, \
            bool_pfam_vector, text_attn_mask, pfam_text_attn_mask, \
            swiss_seq_key, pfam_seq_key, family_key = batch
    

        #print(f'rank={dist.get_rank()}: text size {text_batch.shape}')
        
        #start_time_forward_pass = time.time()
        # Forward pass with Swiss-Prot data.
        z_t_swiss, z_p_swiss = self(
            x_t=text_batch,
            x_p=protein_batch,
            compute_masked_logits=False,
            x_t_mask=text_attn_mask
        )
        # Timer end and log
        #end_time_forward_pass = time.time()
        #print(f"Rank={dist.get_rank()}: Time taken for Swiss-Prot forward pass: {end_time_forward_pass - start_time_forward_pass} seconds.")

        # Forward pass with Pfam data. The barriers Run 1 placed around these
        # forwards are removed: all_gather is itself a synchronisation point, so
        # they bought nothing and cost a collective each.
        z_t_pfam, z_p_pfam = self(
            x_t=pfam_text_batch,
            x_p=pfam_protein_batch,
            compute_masked_logits=False,
            x_t_mask=pfam_text_attn_mask
        )

        # Gather all four embedding sets from all GPUs in one collective.
        z_t_swiss_all, z_p_swiss_all, z_t_pfam_all, z_p_pfam_all = _gather_four(
            self, z_t_swiss, z_p_swiss, z_t_pfam, z_p_pfam)

        # Concatenate Swiss-Prot and Pfam embeddings.
        z_t_all = torch.cat((z_t_swiss_all, z_t_pfam_all), dim=0)
        z_p_all = torch.cat((z_p_swiss_all, z_p_pfam_all), dim=0)
        keys = _stage1_keys(self, swiss_seq_key, pfam_seq_key, family_key)
        
        # Timer start
        #start_time_loss_computation = time.time()

        # Compute inter-modal loss.
        _impl = getattr(self.script_args, 'contrastive_impl', 'dense')
        if _impl == 'sharded':
            loss_align, logits, loss_intra, cosine_similarity, _rows = _sharded_inter_intra(
                self.model, z_p_all, z_t_all, z_t_swiss.shape[0],
                _gather_with_grad, keys,
            )
        else:
            loss_align, logits = self.model.compute_inter_loss(
                protein_embeddings=z_p_all,
                text_embeddings=z_t_all,
                batch_size=z_p_all.shape[0] // 2,
                keys=keys
            )
        # Timer end and log
        #end_time_loss_computation = time.time()
        #print(f"Rank={dist.get_rank()}: Time taken for loss computation: {end_time_loss_computation - start_time_loss_computation} seconds.")


        # Compute intra-modal loss (the sharded branch already produced it).
        if _impl != 'sharded':
            loss_intra, cosine_similarity = self.model.compute_intra_loss(
                protein_embeddings=z_p_all,
                batch_size=z_p_all.shape[0] // 2,
                keys=keys
            )

        # Concatenate batches for masked language modeling.
        all_text_batch = torch.cat((text_batch, pfam_text_batch), dim=0)
        all_protein_batch = torch.cat((protein_batch, pfam_protein_batch), dim=0)
        all_text_mask_batch = torch.cat((text_mask_batch, pfam_text_mask_batch), dim=0)
        all_protein_mask_batch = torch.cat((protein_mask_batch, pfam_protein_mask_batch), dim=0)
        
        #TODO: timer start
        #start_time_mask_comp = time.time()

        # Compute masked language model logits.
        logits_t_mask, logits_s_mask = self(
            x_t=all_text_mask_batch,
            x_p=all_protein_mask_batch,
            compute_masked_logits=True,
            x_t_mask=torch.cat((text_attn_mask, pfam_text_attn_mask), dim=0)
        )
        #end_time_mask_comp = time.time()
        #print(f"Rank={dist.get_rank()}: Time taken for mask predictions: {end_time_mask_comp - start_time_mask_comp} seconds.")


        # Compute masked language model loss for text data.
        loss_text_mask = self.model.compute_masked_lang_loss(
            logits_masked=logits_t_mask,
            targets=all_text_batch,
            targets_masked=all_text_mask_batch,
            mask_token_id=self.text_tokenizer.mask_token_id
        )

        # Compute masked language model loss for protein data.
        loss_sequence_mask = self.model.compute_masked_lang_loss(
            logits_masked=logits_s_mask,
            targets=all_protein_batch,
            targets_masked=all_protein_mask_batch,
            mask_token_id=self.sequence_tokenizer.mask_idx
        )


        if self.script_args.dataset_type == 'pfam':
            # Aggregate all computed losses.
            loss = loss_align + loss_intra + loss_text_mask + loss_sequence_mask
        
        elif self.script_args.dataset_type == 'pfam_ablated':
            # Aggregate all losses besides PFC.
            loss = loss_align + loss_text_mask + loss_sequence_mask
        else:
            # Add an assertion here
            assert self.script_args.dataset_type in ['pfam', 'pfam_ablated'], "Unexpected dataset_type value"
            sys.stderr.write("Unexpected dataset_type value\n")
            sys.exit(1)

        loss, unif_scalars = self._add_uniformity(
            loss, z_p_all, z_t_all, z_t_swiss.shape[0], _impl, 'train')

        # Compute additional performance metrics.
        if _impl == 'sharded':
            # sharded logits are (rows [R,M], cols [M,R]), not a square matrix
            metric_dict = _performance_metrics_sharded(self, logits[0], logits[1], _rows)
        else:
            metric_dict = self.performance_metrics(logits=logits)

        # Every scalar below used to carry sync_dist=True -- 18 collectives per
        # step. One reduction covers all of them; the logged values are the same.
        scalars = {
            'train_loss': loss,
            'train_loss_align': loss_align,
            'train_loss_intra': loss_intra,
            'train_loss_text_mask': loss_text_mask,
            'train_loss_seq_mask': loss_sequence_mask,
        }
        scalars.update(unif_scalars)
        scalars.update(self._embedding_geometry(z_p_all, z_t_all, 'train'))
        for key, value in metric_dict.items():
            scalars['train_' + key] = value
        scalars['memory_usage'] = print_memory_usage()
        if batch_idx == 0:
            scalars['gpu_memory_usage'] = print_gpu_initialization()
        _log_reduced(self, scalars, prog_bar_keys=(
            'train_loss', 'train_loss_align', 'train_loss_intra'))
 
        return {'loss': loss}


    def validation_step(
            self,
            batch: torch.Tensor,
            batch_idx: any,
        ) -> dict:

        """
        `validation_step()`: Validates a single batch of data and computes loss and performance metrics.

        Parameters:
        - `self`: Reference to the current instance of the model or module.
        - `batch`: Input data, which might contain text and protein sequences, their corresponding masks, and additional data from both Swiss-Prot and Pfam datasets.
        - `batch_idx`: Identifier for the current batch.

        Functionality:
        1. Extracts and processes data from the given batch.
        2. Computes embeddings for Swiss-Prot and Pfam datasets.
        3. Concatenates these embeddings to form a unified representation.
        4. Computes various loss values: inter-modal, intra-modal, and masked language losses for both biomedical texts and protein sequences.
        5. Logs the computed loss values and other performance metrics, highlighting metrics such as F1-score.
        6. Collects and appends the joint embeddings of the batch for potential future use.

        Returns:
        - A dictionary with the total validation loss for the current batch.
        """

        if isinstance(batch, list):
            # split the data
            # NB: *_mask_batch are the masked-LM corrupted tokens; *_attn_mask
            # are the BERT attention masks marking real tokens vs [PAD].
            text_batch, protein_batch, text_mask_batch, protein_mask_batch, \
            pfam_text_batch, pfam_protein_batch, pfam_text_mask_batch, pfam_protein_mask_batch, \
            bool_pfam_vector, text_attn_mask, pfam_text_attn_mask, \
            swiss_seq_key, pfam_seq_key, family_key = batch

        
        # forward pass over the swiss-prot data
        z_t_swiss, z_p_swiss = self(
                                  x_t=text_batch,
                                  x_p=protein_batch,
                                  compute_masked_logits=False,
                                  x_t_mask=text_attn_mask
        )

        # foward pass over the pfam data
        z_t_pfam, z_p_pfam = self(
                                x_t=pfam_text_batch,
                                x_p=pfam_protein_batch,
                                compute_masked_logits=False,
                                x_t_mask=pfam_text_attn_mask
        )

        # gather all four embedding sets in one collective (see _gather_four)
        z_t_swiss_all, z_p_swiss_all, z_t_pfam_all, z_p_pfam_all = _gather_four(
            self, z_t_swiss, z_p_swiss, z_t_pfam, z_p_pfam)
           
        # concatenate swiss-prot <> pfam embeddings
        z_t_all = torch.cat((z_t_swiss_all, z_t_pfam_all), dim=0)
        z_p_all = torch.cat((z_p_swiss_all, z_p_pfam_all), dim=0)
        keys = _stage1_keys(self, swiss_seq_key, pfam_seq_key, family_key)

        # Validation must take the same sharded branch as training. The dense
        # call below it builds the full M x M matrix on EVERY rank, M = 2*W*B:
        # 151 MB at 384 ranks, 2.4 GB at 1536 -- an O(W^2) ceiling that has
        # nothing to do with the model size.
        _impl = getattr(self.script_args, 'contrastive_impl', 'dense')
        if _impl == 'sharded':
            loss_align, logits, loss_intra, cosine_similarity, _rows = _sharded_inter_intra(
                self.model, z_p_all, z_t_all, z_t_swiss.shape[0],
                _gather_with_grad, keys,
            )
        else:
            # compute inter-modal loss values
            loss_align, logits = self.model.compute_inter_loss(
                                                protein_embeddings=z_p_all,
                                                text_embeddings=z_t_all,
                                                batch_size=z_p_all.shape[0] // 2,
                                                keys=keys
            )

            # compute intra-modal loss values
            loss_intra, cosine_similarity = self.model.compute_intra_loss(
                                                protein_embeddings=z_p_all,
                                                batch_size=z_p_all.shape[0] // 2,
                                                keys=keys
            )

        # concatenate batch samples
        all_text_batch = torch.cat((text_batch, pfam_text_batch), dim=0)
        all_protein_batch = torch.cat((protein_batch, pfam_protein_batch), dim=0)
        all_text_mask_batch = torch.cat((text_mask_batch, pfam_text_mask_batch), dim=0)
        all_protein_mask_batch = torch.cat((protein_mask_batch, pfam_protein_mask_batch), dim=0)

        # compute mask language model logits
        logits_t_mask, logits_s_mask = self(
                    x_t=all_text_mask_batch,
                    x_p=all_protein_mask_batch,
                    compute_masked_logits=True,
                    x_t_mask=torch.cat((text_attn_mask, pfam_text_attn_mask), dim=0)
        )

        # compute mask language loss for biomedical expert model
        loss_text_mask = self.model.compute_masked_lang_loss(
                    logits_masked=logits_t_mask,
                    targets=all_text_batch,
                    targets_masked=all_text_mask_batch,
                    mask_token_id=self.text_tokenizer.mask_token_id
        )

        # compute mask language loss for protein expert model
        loss_sequence_mask = self.model.compute_masked_lang_loss(
                    logits_masked=logits_s_mask,
                    targets=all_protein_batch,
                    targets_masked=all_protein_mask_batch,
                    mask_token_id=self.sequence_tokenizer.mask_idx
        )


        # total loss
        #loss = loss_align + loss_intra + loss_text_mask + loss_sequence_mask
        
        if self.script_args.dataset_type == 'pfam':
            # Aggregate all computed losses.
            loss = loss_align + loss_intra + loss_text_mask + loss_sequence_mask
        
        elif self.script_args.dataset_type == 'pfam_ablated':
            # Aggregate all losses besides PFC.
            loss = loss_align + loss_text_mask + loss_sequence_mask
        else:
            # Add an assertion here
            assert self.script_args.dataset_type in ['pfam', 'pfam_ablated'], "Unexpected dataset_type value"
            sys.stderr.write("Unexpected dataset_type value\n")
            sys.exit(1)

        loss, unif_scalars = self._add_uniformity(
            loss, z_p_all, z_t_all, z_t_swiss.shape[0], _impl, 'valid')

        # track metrics
        if _impl == 'sharded':
            metric_dict = _performance_metrics_sharded(self, logits[0], logits[1], _rows)
        else:
            metric_dict = self.performance_metrics(logits=logits.detach().cpu())

        # one reduction for all of them, as in training_step
        scalars = {
            'valid_loss': loss,
            'valid_loss_align': loss_align,
            'valid_loss_intra': loss_intra,
            'valid_loss_text_mask': loss_text_mask,
            'valid_loss_seq_mask': loss_sequence_mask,
        }
        scalars.update(unif_scalars)
        scalars.update(self._embedding_geometry(z_p_all, z_t_all, 'valid'))
        for key, value in metric_dict.items():
            scalars['valid_' + key] = value
        scalars['memory_usage'] = print_memory_usage()
        _log_reduced(self, scalars, prog_bar_keys=(
            'valid_loss', 'valid_loss_align', 'valid_loss_intra'))


        # collect joint embedding
        #self.val_text_joint_latents.append(z_t_all.detach().cpu())
        #self.val_seq_joint_latents.append(z_p_all.detach().cpu())

        return {'valid_loss': loss}


 #   def on_validation_epoch_end(self):
 #       print('Enter validation end of epoch analysis...')
#
 #       # collect and aggregate outputs from all validation steps
 #       val_z_t_joint = torch.cat(self.val_text_joint_latents, dim=0)
 #       val_z_s_joint = torch.cat(self.val_seq_joint_latents, dim=0)
 #       
 #       # compute singular values 
 #       print('Compute singular values...')
 #       text_log_sigma_k, S_text = self.compute_singular(val_z_t_joint.detach().cpu())
 #       protein_log_sigma_k, S_protein = self.compute_singular(val_z_s_joint.detach().cpu())
 #      
 #       # save image pngs for tracking dimensionality collapse
 #       self.save_png_to_tensorboard(
 #               data=text_log_sigma_k.numpy(),
 #               title='text',
 #       )
 #       self.save_png_to_tensorboard(
 #               data=protein_log_sigma_k.numpy(),
 #               title='protein'
 #       )
 #
 #       # free memory
 #       self.val_text_joint_latents.clear()
 #       self.val_seq_joint_latents.clear()
 #       
 #       
 #       # compute effective rank (RankME):
 #       print('Compute eranks')
 #       erank_text = self.compute_effective_rank(sigma_ks=S_text)
 #       erank_protein = self.compute_effective_rank(sigma_ks=S_protein)
 #       
 #       # log erank metrics
 #       self.log('valid_erank_text', erank_text, sync_dist=True)
 #       self.log('valid_erank_protein', erank_protein, sync_dist=True)

    def configure_optimizers(self,):

        params = [
                {"params": self.model.protein_encoder.parameters(), "lr": self.script_args.protein_encoder_lr},
                {"params": self.model.text_encoder.parameters(), "lr": self.script_args.text_encoder_lr},
                {"params": itertools.chain(
                    self.model.protein_projection.parameters(),
                    self.model.text_projection.parameters()
                    ),
                "lr": self.script_args.head_lr,
                "weight_decay": self.script_args.weight_decay}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=self.script_args.weight_decay)

        return {
                "optimizer": optimizer,
        }
    
    @torch.no_grad()
    def compute_class_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            source: str
        ) -> dict:

        # convert torch tensors to numpy array
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()

        # compute the metrics
        accuracy = accuracy_score(targets_np, outputs_np.round())
        precision = precision_score(targets_np, outputs_np.round(), average='micro')
        recall = recall_score(targets_np, outputs_np.round(), average='micro')
        f1 = f1_score(targets_np, outputs_np.round(), average='micro')

        return {
                f'{source}_accuracy': accuracy,
                f'{source}_precision': precision,
                f'{source}_recall': recall,
                f'{source}_f1': f1
        }

    @torch.no_grad()
    def performance_metrics(self, logits: torch.Tensor) -> tuple:

        logits = logits.cpu().float()

        # get probs
        p_text = F.softmax(logits, dim=-1) # prob of a given text captions aligning well with seq. pairs
        p_seq = F.softmax(logits.T, dim=-1) # prob of a given seq aligning well with text pairs
        p_tot = (p_seq + p_text) / 2 # total prob

        # get class labels
        y_pred_text = torch.argmax(p_text, dim=-1)
        y_pred_seq = torch.argmax(p_seq, dim=-1)
        y_pred = torch.argmax(p_tot, dim=-1)
        y_true = torch.arange(y_pred_text.shape[0])
        
        # compute class metrics
        text_metrics = self.compute_class_metrics(
                                    outputs=y_pred_text,
                                    targets=y_true,
                                    source='text'
        )
        seq_metrics = self.compute_class_metrics(
                                    outputs=y_pred_seq,
                                    targets=y_true,
                                    source='seq'
        )
        total_metrics = self.compute_class_metrics(
                                    outputs=y_pred,
                                    targets=y_true,
                                    source='total'
        )

        # combine dicts into one
        combined_dict = {}
        combined_dict.update(text_metrics)
        combined_dict.update(seq_metrics)
        combined_dict.update(total_metrics)

        return combined_dict
    
    @torch.no_grad()
    def compute_singular(self, inputs: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):

        # goal of this function: track for dimensionality collapse
        # inputs dim: (batch_size, emb_dim)

        mean_inputs = torch.mean(inputs, dim=0) # average over batch dimension
        norm_inputs = inputs - mean_inputs # normalize vectors
       
       # compute correlation matrix  #TODO: double check work...
        C = torch.zeros((norm_inputs.shape[-1], norm_inputs.shape[-1]))
        for sample_idx in range(norm_inputs.shape[0]):
            norm_vector = norm_inputs[sample_idx, :].unsqueeze(0)
            C += norm_vector.T @ norm_vector
        C *= 1/norm_vector.shape[0]

        _, S, _ = torch.linalg.svd(C, full_matrices=False)
        
        # return singular value indexes 
        log_sigma_k, _ = torch.sort(torch.log(S), descending=True)
        return (
                log_sigma_k,
                S
        )
        
    def compute_effective_rank(self, sigma_ks: torch.Tensor) -> torch.Tensor:
        """
        references:
            - Roy et al. The effective rank: a measure of effective dimensionality
            - Garrido et al. RankMe: Assessing the Downstream Performnace of Pretrained SS Reps by their Rank.
        """
        # sort the singular values
        sigma_ks, _ = torch.sort(sigma_ks, descending=True)
        
        # copute L1 norm for sing values.
        l1_norm_sigma = torch.norm(sigma_ks, p=1)
        
        # compute singular value distribution
        p_k = sigma_ks / l1_norm_sigma  + torch.finfo(torch.float).eps
        
        # compute Shannon entropy
        entropy = - torch.sum(p_k * torch.log(p_k))
    
        # get effective rank (RankME):
        erank = torch.exp(entropy)

        return erank

    def save_png_to_tensorboard(
            self,
            data: np.single,
            title: str,
            x_axis_label: str='Singular Value Rank Index',
            y_axis_label: str='Log of singular values',
            ):
    
        current_epoch = self.trainer.current_epoch
        
        # Plot the line
        fig, ax = plt.subplots(dpi=300)
        ax.plot(data)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_ylim([-25,3])

        # Log the plot in TensorBoard 
        self.logger.experiment.add_figure(f'{title}_SingularValues_{current_epoch}', fig, current_epoch)
        
        # Close the figure to free up memory
        plt.close(fig)

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: torch.Tensor,
            dataloder_idx: bool=False
        ) -> (  
                torch.Tensor,
                torch.Tensor
        ):


        if isinstance(batch, list):
            # mean loss
            text_batch, protein_batch = batch
            outputs = self(
                    x_t=text_batch,
                    x_p=protein_batch,
                    compute_masked_logits=False
            )
        
        z_t_joint, z_p_joint = outputs

        self.predict_text_joint_latents.append(z_t_joint.detach().cpu())
        self.predict_seq_joint_latents.append(z_p_joint.detach().cpu())

        return outputs

    def on_predict_epoch_end(self, outputs=None):

        self.predict_text_joint_latents = torch.cat(self.predict_text_joint_latents).cpu()
        self.predict_seq_joint_latents = torch.cat(self.predict_seq_joint_latents).cpu()


##########################
# Facilitator PL wrapper #
##########################

class PL_Facilitator(pl.LightningModule):

    def __init__(
            self, 
            args: any
    ):

        super().__init__()

        # arguments
        self.args = args

        # model 
        self.model = mod.Facilitator(
                    in_dim=self.args.emb_dim,
                    hid_dim=self.args.hid_dim,
                    out_dim=self.args.emb_dim,
                    dropout=self.args.dropout
        )
        
        self.text_to_protein_joint_embeddings = []

    def forward(
            self,
            z_t: torch.Tensor,
    ) -> torch.Tensor:

        # reconfigure z_t to z_p (additional alignment)
        z_t_to_p = self.model(z_t)

        return z_t_to_p

    

    def training_step(self, batch: torch.Tensor, batch_id: any) -> dict:

        # check if the batch is a list and split data if so 
        if isinstance(batch, list):
            text_embeddings, protein_embeddings = batch

        # forward pass with the model
        z_t_to_p = self(z_t=text_embeddings)

        # compute loss
        loss = self.model.compute_loss(
                output=z_t_to_p,
                target=protein_embeddings,
                loss_option=self.args.loss_type
        )
    
        # log the total loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {'loss': loss}

 
    def validation_step(self, batch: torch.Tensor, batch_id: any) -> dict:

        # check if the batch is a list and split data if so 
        if isinstance(batch, list):
            text_embeddings, protein_embeddings = batch

        # forward pass with the model
        z_t_to_p = self(z_t=text_embeddings)
    
        # compute loss
        loss = self.model.compute_loss(
                output=z_t_to_p,
                target=protein_embeddings,
                loss_option=self.args.loss_type
        )
    
        # log the total loss
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {'loss': loss}

    
    def configure_optimizers(self,):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
        return {
                "optimizer": optimizer
        }
   

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = None) -> torch.Tensor:
        """
        Defines a single prediction (inference) step.
        """

        # Unpack the batch if it comes in a list format.
        # Here, we only take text embeddings for prediction as an example.
        if isinstance(batch, list):
            text_embeddings, _ = batch  # We ignore the second element (protein_embeddings)
        else:
            text_embeddings = batch

        # Perform forward pass to get transformed text embeddings (z_t_to_p)
        z_t_to_p = self(z_t=text_embeddings)
        self.text_to_protein_joint_embeddings.append(z_t_to_p.detach().cpu())

        return z_t_to_p

    def on_predict_epoch_end(self, outputs=None):
        
        self.text_to_protein_joint_embeddings = torch.cat(self.text_to_protein_joint_embeddings).cpu()
