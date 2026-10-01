import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForMaskedLM
import torch.distributed as dist
import esm
from torch.nn.utils.weight_norm import weight_norm

# Sentinel for "this row has no usable key" (a 'nan'-label row whose Pfam side
# is empty). It must never compare equal to anything, including another
# NO_KEY. Kept in step with preprocess.NO_KEY by
# tests/stage1_tests/test_false_negative_mask.py rather than by an import, so
# model.py does not pull in the data stack.
NO_KEY = 0


"""
functions and classes adapted from the following:
    1. https://keras.io/examples/vision/nl_image_search/
    2. https://colab.research.google.com/drive/1hYHb0FTdKQCXZs3qCwVZnSuVGrZU2Z1w?usp=sharing
"""

class ProteinEncoder(nn.Module):
    """
    Encoder for protein sequence to a fixed size vector --> z_s
    """
    
    def __init__(self, args: any):
        super().__init__()

        #self.script_args = args
        self.seq_model_path = args.seq_model_path
        self.pretrained = args.pretrained_seq
        self.trainable = args.trainable_seq
        self.n_layers_to_finetune = args.pLM_n_layers_to_finetune
        self.rep_layer = args.rep_layer
        self.model, self.alphabet = self.get_ESM_model() # get model and alphabet (ESM)
        
        for p in self.model.parameters():
            if self.trainable and self.n_layers_to_finetune == 0:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        # Make the last n_layers_to_finetune layers trainable
        if self.trainable and self.n_layers_to_finetune != 0:
            for layer in self.model.layers[-self.n_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # Use the [CLS] token hidden representation as the sentence's embedding
        # for the downstream latent alignment.
        self.target_token_idx = 0

    def get_ESM_model(self):

        return esm.pretrained.load_model_and_alphabet(
                os.path.expanduser(
                    self.seq_model_path
                )
        )
    
    def forward(self, x_s: torch.Tensor, compute_logits: bool=False):
        # drop channel depth
        x_s = x_s.squeeze(1)

        outputs = self.model(
                x_s,
                repr_layers=[self.rep_layer],
                return_contacts=False
        )
        
        # mask langauge model objective 
        if compute_logits:
            logits = outputs['logits']
            return logits
       
        # fine-tuning cls token for protein sequence alignment with biomedical text 
        cls_hidden = outputs['representations'][self.rep_layer][:,self.target_token_idx,:]
        return cls_hidden
    
class TextEncoder(nn.Module):

    """
    Encoder for protein's natural text to a fixed size vector --> z_t
    """

    def __init__(self, args: any):
        super().__init__()
 
        self.model_name = args.text_model_path
        self.pretrained = args.pretrained_text
        self.trainable = args.trainable_text
        self.n_layers_to_finetune = args.bLM_n_layers_to_finetune 
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)

        if self.pretrained:
            #self.model = AutoModel.from_pretrained(self.model_name)
            self.model = BertForMaskedLM.from_pretrained(self.model_name)

        else:
            #self.model = AutoModel.from_config(self.model_name)
            self.model = BertForMaskedLM.from_config(self.model_name)

        for p in self.model.parameters():
            if self.trainable and self.n_layers_to_finetune == 0:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Make the last n_layers_to_finetune layers trainable
        if self.trainable and self.n_layers_to_finetune != 0:
            for layer in self.model.bert.encoder.layer[-self.n_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        
        # Use the [CLS] token hidden representation as the sentence's embedding
        # for the downstream latent alignment.
        self.target_token_idx = 0

    def forward(self, inputs: torch.Tensor, compute_logits: bool=False,
                attention_mask: torch.Tensor=None) -> torch.Tensor:
        """Encode a batch of tokenised captions.

        Captions are padded to a fixed text_max_length so every tensor in the
        batch has the same shape (the DataLoader's default collate requires
        that). `attention_mask` marks the real tokens; without it BERT attends
        over every [PAD], so the hidden state -- and therefore z_t -- varies
        with how much padding a caption happened to receive, injecting caption
        length as a signal. Passing the mask is the standard fix and is what
        the tokenizer already returns.
        """
        # drop channel depth
        inputs = inputs.squeeze(1)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(1)

        if compute_logits:
            # compute the masked language model logits
            outputs = self.model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            return logits

        else:
            # Use the underlying BERT model directly to skip the MLM head
            outputs = self.model.bert(inputs, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    """
    g(.) which maps z_t --> h_t or z_s --> h_s
    
    Note: h is the joint embedding representation, h_t
    is the joint embedding for the text caption, and
    h_s is the joint embedding for the protein sequence.
    """

    def __init__(self, embedding_dim: int, args: any):

        super().__init__()
        self.projection_dim = args.proj_embedding_dim
        self.dropout = args.dropout
        self.embedding_dim = embedding_dim

        # model graph
        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        projection = self.projection(z)
        h = self.gelu(projection)
        h = self.fc(h)
        h = self.dropout(h)
        h = h + projection
        h = self.layer_norm(h)
        return h


#############################
# Default/mask architecture #
#############################

class PEN_CL(nn.Module):

    """
    Protein Embeddings with Natural language using Constrastive Learing (PEN-CL).
    """

    def __init__(self, args: any):

        super().__init__()
        self.protein_embedding = args.protein_encoder_embedding
        self.text_embedding = args.text_encoder_embedding
        self.temperature = args.temperature

        # protein sequence expert
        self.protein_encoder = ProteinEncoder(args=args)
        # natural text expert
        self.text_encoder = TextEncoder(args=args)
        
        # projection heads g_seq( . ) --> joint embedding space
        self.protein_projection = ProjectionHead(
                embedding_dim=self.protein_embedding,
                args=args
        )

        # projection heads g_text( . ) --> joint embedding space
        self.text_projection = ProjectionHead(
                embedding_dim=self.text_embedding,
                args=args
        )

    def forward(
            self,
            x_t: torch.Tensor,
            x_s: torch.Tensor,
            compute_masked_logits: bool=False,
            x_t_mask: torch.Tensor=None
        ) -> dict:

        if compute_masked_logits:
            # forward pass for computing logits for masked langauge objective
            protein_logits = self.protein_encoder(x_s, compute_logits=True)
            text_logits = self.text_encoder(x_t, compute_logits=True,
                                            attention_mask=x_t_mask)

            return {
                    'text_masked_logits': text_logits,
                    'protein_masked_logits': protein_logits
            }

        else:
            # split the tuple into 2 dicts... 
            # getting protein sequence and text inputs ...
            z_t = self.text_encoder(x_t, compute_logits=False,
                                    attention_mask=x_t_mask)
            z_s = self.protein_encoder(x_s, compute_logits=False)

            # "joint" sequence and text embedding (with same dimension)
            z_t_joint = self.text_projection(z_t)
            z_s_joint = self.protein_projection(z_s)

            return {
                    'text_joint_latent': z_t_joint,
                    'seq_joint_latent': z_s_joint,
            }


    def compute_loss(
            self,
            protein_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor
        ) -> (
                torch.Tensor,
                torch.Tensor
        ):

        # calc. the loss for constrsative multi-modal learning
        # note: only account for "inter" representations
        
        # cosine similarity
        logits = (text_embeddings @ protein_embeddings.T) / self.temperature
        protein_similarity = protein_embeddings @ protein_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T
        
        # ground truth
        targets = F.softmax(
                (protein_similarity + text_similarity) / (2 * self.temperature), dim=-1
        )

        # prediction
        text_loss = self.cross_entropy(logits, targets, reduction='none')
        protein_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (protein_loss + text_loss) / 2.0
        
        return (
                loss.mean(),
                logits.cpu()
        )

    def cross_entropy(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            reduction: str='none'
        ) -> torch.Tensor:

        # compute categorical cross entropy
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            assert False, print('Choose either "none" or "mean" for reduction argument')

    def compute_masked_lang_loss(
            self,
            logits_masked: torch.Tensor,
            targets: torch.Tensor,
            targets_masked: torch.Tensor,
            mask_token_id: torch.Tensor
        ) -> torch.Tensor:
        # compute the masked langauge objective loss for masked logits
        
        loss_func = nn.CrossEntropyLoss(reduction='none')

        loss_mask = loss_func(
                            logits_masked.permute(0, 2, 1), # (batch_size, vocab_size, seq_len)
                            targets.squeeze(1) # (batch_size, seq_len)
        )

        # list to append loss values
        batch_loss = []

        for ii, target_mask_sample in enumerate(targets_masked):
            
            # locate mask positions. Keep this a 1-D bool TENSOR: the old
            # `.tolist()` produced a nested list (target_mask_sample is
            # [1, seq_len]), and indexing a 1-D tensor with a nested list is the
            # deprecated "non-tuple sequence for multidimensional indexing" path.
            # PyTorch warns that it will become x[torch.tensor(seq)], which on a
            # 1-D tensor raises "too many indices" -- so this was a future hard
            # failure, not just noise. .tolist() also forced a device->host sync
            # on every loop iteration.
            masked_positions = (target_mask_sample == mask_token_id).reshape(-1)
            # extract the loss values at those masked positions
            loss_mask_sample = loss_mask[ii][masked_positions]
            
            # append mean loss value for a given batch sample
            if loss_mask_sample.numel() > 0:
                batch_loss.append(torch.mean(loss_mask_sample).unsqueeze(0))
        
        # Guard on batch_loss, not on loss_mask_sample. The latter is the last
        # loop variable: if the final sample happened to have no masked
        # positions, every other sample's loss was silently discarded and this
        # returned 0.0. It also raised NameError when the batch was empty.
        if batch_loss:
            loss_mask_mean = torch.mean(torch.cat(batch_loss))
        else:
            # no masked positions anywhere in the batch
            loss_mask_mean = torch.tensor(0.0, device=logits_masked.device)


        return loss_mask_mean



#####################
# Pfam architecture #
#####################



class pfam_PEN_CL(nn.Module):

    """
    Protein Embeddings with Natural lanauge using Constrastive Learing (PEN-CL) while including pfam constrastive learning.
    """

    def __init__(self, args: any):

        super().__init__()

        self.protein_embedding = args.protein_encoder_embedding
        self.text_embedding = args.text_encoder_embedding
        self.temperature = args.temperature

        # protein sequence expert
        self.protein_encoder = ProteinEncoder(args=args)
        # natural text expert
        self.text_encoder = TextEncoder(args=args)
        
        # projection heads g_seq( . ) --> joint embedding space
        self.protein_projection = ProjectionHead(
                embedding_dim=self.protein_embedding,
                args=args
        )

        # projection heads g_text( . ) --> joint embedding space
        self.text_projection = ProjectionHead(
                embedding_dim=self.text_embedding,
                args=args
        )

    def forward(
            self,
            x_t: torch.Tensor,
            x_s: torch.Tensor,
            compute_masked_logits: bool=False,
            x_t_mask: torch.Tensor=None
        ) -> dict:

        if compute_masked_logits:
            # forward pass for computing logits for masked langauge objective
            protein_logits = self.protein_encoder(x_s, compute_logits=True)
            text_logits = self.text_encoder(x_t, compute_logits=True,
                                            attention_mask=x_t_mask)

            return {
                    'text_masked_logits': text_logits,
                    'protein_masked_logits': protein_logits
            }

        else:
            # split the tuple into 2 dicts... 
            # getting protein sequence and text inputs ...
            z_t = self.text_encoder(x_t, compute_logits=False,
                                    attention_mask=x_t_mask)
            z_s = self.protein_encoder(x_s, compute_logits=False)

            # "joint" sequence and text embedding (with same dimension)
            z_t_joint = self.text_projection(z_t)
            z_s_joint = self.protein_projection(z_s)

            return {
                    'text_joint_latent': z_t_joint,
                    'seq_joint_latent': z_s_joint,
            }

    def compute_inter_loss(
            self,
            protein_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
            batch_size: int,
            keys: torch.Tensor = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Compute the inter-modal contrastive InfoNCE loss between protein and text embeddings.

        Parameters:
        - protein_embeddings: A tensor representing the embeddings of the protein sequences.
        - text_embeddings: A tensor representing the embeddings of the text descriptions.
        - batch_size: The number of samples in the batch.

        Steps:
        1. Generate a masking matrix to identify off-diagonal elements.
        2. Compute cosine similarities (i.e., logits) between text and protein embeddings.
        3. Compute self-similarities for both protein and text embeddings.
        4. Mask off-diagonal elements between swiss-prot and pfam in the similarity matrices.
        5. Define ground truth by averaging the masked protein and text similarity matrices.
        6. Compute the contrastive loss for the protein and text embeddings using the ground truth.

        Returns:
        - Mean contrastive loss for the given batch of protein and text embeddings.
        - The logits (cosine similarity matrix between text and protein embeddings).

        Note: This function assumes a specific structure in the input batches, where corresponding positive samples 
        in the protein and text embeddings are arranged in a particular way, allowing for masking and contrastive loss calculation.
        """
        
        # get off-diagonal masking matrix
        mask = torch.zeros((2*batch_size, 2*batch_size))
        # mask the bottom left quadrant diagonal
        mask[batch_size:, :batch_size] = torch.eye(batch_size)
        # mask the top right quadrant
        mask[:batch_size, batch_size:] = torch.eye(batch_size)
        # convert to correct device and convert to boolean
        mask = mask.to(protein_embeddings.device).bool()
        # same-sequence / same-family false negatives; the diagonal is L_GC's
        # positive, so it is cleared after the OR (a key always equals itself).
        _M = 2 * batch_size
        _idx = torch.arange(_M, device=protein_embeddings.device)
        mask |= self._key_equality_mask(keys, _idx, _M)
        mask[_idx, _idx] = False

        # matrix multiplication between model embeddings
        logits = (text_embeddings @ protein_embeddings.T) / self.temperature
        protein_similarity = protein_embeddings @ protein_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T

        # mask the off-diagonal between swiss-prot and pfam
        mask_protein_similarity = self.set_inf(protein_similarity, mask)
        mask_text_similarity = self.set_inf(text_similarity, mask)
        mask_logits = self.set_inf(logits, mask)

        # ground truth
        targets = F.softmax(
            (mask_protein_similarity + mask_text_similarity) / (2 * self.temperature), dim=-1
        )

        # compute loss
        text_loss = self.cross_entropy(mask_logits, targets, reduction='none')
        protein_loss = self.cross_entropy(mask_logits.T, targets.T, reduction='none')
        loss = (protein_loss + text_loss) / 2.0

        return (
            loss.mean(),
            mask_logits.detach().cpu()
        )


    # ------------------------------------------------------------------
    # Row-sharded contrastive losses.
    #
    # The dense implementations above materialise the full M x M similarity
    # matrix on EVERY rank, where M = 2 * world_size * micro_batch. That is
    # O(W^2) per rank: 2 GB of activations at 512 ranks, 32 GB at 2048, 512 GB
    # at 8192 -- i.e. unusable beyond ~1k ranks.
    #
    # InfoNCE decomposes by anchor row, so a rank only needs the rows it owns,
    # each against ALL candidates: [2B, M] instead of [M, M]. That is O(W) per
    # rank -- 2 MB at 2048 ranks. The result is mathematically identical; see
    # tests/stage1_tests/test_sharded_contrastive.py for the equivalence check.
    #
    # `row_index` holds this rank's global row indices into the gathered batch.
    # ------------------------------------------------------------------

    def _key_equality_mask(self, keys, index, M, transpose=False):
        """[R, M] (or [M, R]) True where a row and a candidate share a key.

        `keys` is [M, K] int64, one column per enabled rule -- sequence
        identity (the same protein under a different caption) and Pfam family
        (two items drawn on the same family). Both are false negatives the
        index-rule homolog mask does not reach: at M = 49,152 a row has ~25
        same-family candidates and a 5.6% chance of an identical-sequence one,
        against the single (i, i +- N) pair the index rule covers.

        A row whose key is NO_KEY matches nothing; guarding the row side is
        enough, since that also kills NO_KEY == NO_KEY.

        Callers MUST clear the row's positive column afterwards -- a key always
        equals itself, so the diagonal (L_GC) or the homolog column (L_PFC)
        would otherwise be masked and the loss would lose its numerator.
        """
        R = index.numel()
        out = torch.zeros((M, R) if transpose else (R, M), dtype=torch.bool,
                          device=index.device)
        if keys is None:
            return out
        sel = keys[index]                                        # [R, K]
        for c in range(keys.shape[1]):
            row_k = sel[:, c].unsqueeze(1)                        # [R, 1]
            hit = (row_k == keys[:, c].unsqueeze(0)) & (row_k != NO_KEY)
            out |= hit.T if transpose else hit
        return out

    def _homolog_mask_rows(self, row_index, M, keys=None):
        """mask[row_index, :] for the inter-loss homolog mask, without building M x M.

        The dense mask marks (i, i+N) and (i+N, i): a Swiss-Prot entry and the
        Pfam homolog curated to match it. Those are false negatives and are
        excluded from the contrastive denominator.

        With `keys`, same-sequence and same-family candidates join them. L_GC's
        positive is the diagonal, so (i, i) is cleared last and unconditionally.
        """
        N = M // 2
        ar = torch.arange(row_index.numel(), device=row_index.device)
        out = self._key_equality_mask(keys, row_index, M)
        partner = torch.where(row_index < N, row_index + N, row_index - N)
        out[ar, partner] = True
        out[ar, row_index] = False
        return out

    def _homolog_mask_cols(self, col_index, M, keys=None):
        """mask[:, col_index] -- the transpose slice, same rule."""
        N = M // 2
        ar = torch.arange(col_index.numel(), device=col_index.device)
        out = self._key_equality_mask(keys, col_index, M, transpose=True)
        partner = torch.where(col_index < N, col_index + N, col_index - N)
        out[partner, ar] = True
        out[col_index, ar] = False
        return out

    def compute_inter_loss_sharded(
            self,
            protein_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
            batch_size: int,
            row_index: torch.Tensor,
            row_logZ: torch.Tensor,
            keys: torch.Tensor = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Row-sharded equivalent of compute_inter_loss.

        `row_logZ` is logsumexp((mp + mt) / 2tau) for EVERY row of the gathered
        batch -- shape [M]. Each rank can only compute this for its own rows, so
        the caller all_gathers it. It is needed because `targets` is a row-wise
        softmax and the protein-side term consumes targets.T, i.e. COLUMN slices
        of a row-normalised matrix: every column entry belongs to a different
        row's normaliser.
        """
        z_p, z_t = protein_embeddings, text_embeddings
        M = z_p.shape[0]
        tau = self.temperature
        two_tau = 2.0 * tau

        mask_rows = self._homolog_mask_rows(row_index, M, keys)    # [R, M]
        mask_cols = self._homolog_mask_cols(row_index, M, keys)    # [M, R]

        # --- text side: this rank's rows against all candidates ---
        ml_rows = self.set_inf((z_t[row_index] @ z_p.T) / tau, mask_rows)
        s_rows = self.set_inf(z_p[row_index] @ z_p.T, mask_rows) \
               + self.set_inf(z_t[row_index] @ z_t.T, mask_rows)
        targets_rows = F.softmax(s_rows / two_tau, dim=-1)          # [R, M]
        text_loss = (-targets_rows * F.log_softmax(ml_rows, dim=-1)).sum(1)

        # --- protein side: same rows of the TRANSPOSE ---
        ml_cols = self.set_inf((z_t @ z_p[row_index].T) / tau, mask_cols)   # [M, R]
        s_cols = self.set_inf(z_p @ z_p[row_index].T, mask_cols) \
               + self.set_inf(z_t @ z_t[row_index].T, mask_cols)
        # targets.T[r, j] = targets[j, r] = exp(s[j, r]/2tau - logZ[j])
        targets_cols_T = torch.exp(s_cols / two_tau - row_logZ.unsqueeze(1)).T  # [R, M]
        protein_loss = (-targets_cols_T * F.log_softmax(ml_cols.T, dim=-1)).sum(1)

        loss = (protein_loss + text_loss) / 2.0
        # Return both slices: metrics need the transpose direction, and the full
        # M x M matrix the dense path returns does not exist here by design.
        return loss.mean(), (ml_rows.detach(), ml_cols.detach())

    def inter_row_logsumexp(
            self,
            protein_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
            row_index: torch.Tensor,
            keys: torch.Tensor = None,
        ) -> torch.Tensor:
        """logsumexp((mp + mt)/2tau) for this rank's rows -- all_gather this."""
        z_p, z_t = protein_embeddings, text_embeddings
        M = z_p.shape[0]
        mask_rows = self._homolog_mask_rows(row_index, M, keys)
        s_rows = self.set_inf(z_p[row_index] @ z_p.T, mask_rows) \
               + self.set_inf(z_t[row_index] @ z_t.T, mask_rows)
        return torch.logsumexp(s_rows / (2.0 * self.temperature), dim=-1)

    def compute_intra_loss_sharded(
            self,
            protein_embeddings: torch.Tensor,
            batch_size: int,
            row_index: torch.Tensor,
            keys: torch.Tensor = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Row-sharded equivalent of compute_intra_loss. Rows only -- no transpose."""
        z_p = protein_embeddings
        M = z_p.shape[0]
        R = row_index.numel()
        ar = torch.arange(R, device=z_p.device)

        cs = (z_p[row_index] @ z_p.T) / self.temperature            # [R, M]
        # dense pos_mask = eye(M).roll(M//2, dims=0): row i pairs with (i - M//2) % M
        pos_col = (row_index - M // 2) % M
        # L_PFC's positive is the homolog column, so the key mask is cleared
        # THERE, not on the diagonal -- the opposite of the inter loss.
        drop = self._key_equality_mask(keys, row_index, M)
        drop[ar, row_index] = True                                  # exclude i == j
        drop[ar, pos_col] = False                                   # keep the positive
        cs = self.set_inf(cs, drop)
        pos = cs[ar, pos_col]
        nll = -pos + torch.logsumexp(cs, dim=-1)
        return nll.mean(), cs.detach()

    def compute_intra_loss(  
            self,
            protein_embeddings,
            batch_size,
            keys: torch.Tensor = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the intra-modal contrastive InfoNCE loss for protein embeddings.

        Parameters:
        - protein_embeddings: A tensor representing the embeddings of the protein sequences.
        - batch_size: Batch size used for training.

        Steps:
        1. Normalize the protein embeddings using L2 normalization.
        2. Compute the cosine similarity between the normalized embeddings.
        3. Mask the diagonal of the cosine similarity matrix to avoid using a protein's similarity with itself.
        4. Define positive examples by rolling the mask. The positive example for a given protein embedding is determined by an embedding half the batch size away.
        5. Compute the InfoNCE loss using the masked cosine similarity matrix.

        Returns:
        - Mean InfoNCE loss for the given batch of protein embeddings.
        - The cosine similarity matrix.

        Note: The underlying assumption is that in each batch, corresponding positive samples for a given protein embedding 
        lie half the batch size away. The function computes the negative log likelihood loss between these positive samples 
        and the entire batch.
        """
        
        # l2 normalization
        #norm_protein_embeddings = F.normalize(protein_embeddings, p=2, dim=1)
        norm_protein_embeddings = protein_embeddings

        # cosine similarity
        cosine_similarity = (norm_protein_embeddings @ norm_protein_embeddings.T) / self.temperature
        
        # mask cosine similarity matrix
        sample_size = protein_embeddings.shape[0]
        mask = torch.eye(sample_size, device=cosine_similarity.device, dtype=torch.bool)

        # Find positive example -> batch_size //2 away from the original example (swiss-prot<>pfam)
        # Built from a CLEAN eye: `mask` picks up the key hits just below.
        pos_mask = mask.roll(shifts=mask.shape[0]//2, dims=0)

        _idx = torch.arange(sample_size, device=cosine_similarity.device)
        mask = (mask | self._key_equality_mask(keys, _idx, sample_size)) & ~pos_mask
        #cosine_similarity.masked_fill_(mask, float(-9e15))
        cosine_similarity = self.set_inf(cosine_similarity, mask)

        # InfoNCE loss
        nll = -cosine_similarity[pos_mask] + torch.logsumexp(cosine_similarity, dim=-1)
        
        return (
            nll.mean(),
            cosine_similarity.cpu(),
        )

    # ------------------------------------------------------------------
    # Uniformity (Wang & Isola, 2020):
    #   L_unif = log mean_{i != j} exp(-t ||x_i - x_j||^2),  x = z / ||z||
    # On the unit sphere ||x_i - x_j||^2 = 2 - 2 x_i.x_j, so this is a
    # log-mean-exp of 2t * cosine, minus 2t. Pairs (i, i) and the Swiss-Prot /
    # Pfam homolog pair (i, i +- N) are excluded, so the term never pushes
    # apart the pairs L_intra pulls together.
    #
    # The loss is one global log-mean-exp, so it is assembled from per-row
    # logsumexps: dense computes all M rows, sharded computes this rank's rows
    # and all_gathers them (see PL_wrapper._uniformity_losses).
    # ------------------------------------------------------------------

    def uniformity_row_logsumexp(
            self,
            embeddings: torch.Tensor,
            row_index: torch.Tensor,
            t: float,
        ) -> torch.Tensor:
        """logsumexp_j 2t * cos(z_i, z_j) over valid j, for each i in row_index."""
        z = F.normalize(embeddings.float(), dim=-1)
        M = z.shape[0]
        mask = self._homolog_mask_rows(row_index, M)
        mask[torch.arange(row_index.numel(), device=row_index.device), row_index] = True
        s = (2.0 * t) * (z[row_index] @ z.T)
        return torch.logsumexp(s.masked_fill(mask, float('-inf')), dim=-1)

    @staticmethod
    def uniformity_from_row_logsumexp(row_lse: torch.Tensor, t: float) -> torch.Tensor:
        """Combine per-row logsumexps (all M rows, any order) into L_unif."""
        M = row_lse.numel()
        return torch.logsumexp(row_lse, dim=0) - np.log(M * (M - 2)) - 2.0 * t

    def compute_uniformity_loss(
            self,
            embeddings: torch.Tensor,
            t: float = 2.0,
        ) -> torch.Tensor:
        """Dense L_unif over the full gathered batch [M, D], M = 2N."""
        rows = torch.arange(embeddings.shape[0], device=embeddings.device)
        return self.uniformity_from_row_logsumexp(
            self.uniformity_row_logsumexp(embeddings, rows, t), t)

    def set_inf(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor
        ) -> torch.Tensor:
        # Determine replacement value based on tensor dtype
        if tensor.dtype == torch.float32:
            replace_value = -9e15
        elif tensor.dtype == torch.float16:
            replace_value = -1e4
        else:
            raise ValueError("Unsupported tensor dtype for this operation.")

        # Use masked_fill_ to replace positions in tensor where mask is True with the specified value
        tensor.masked_fill_(mask, replace_value)

        return tensor

    def cross_entropy(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            reduction: str='none'
        ) -> torch.Tensor:

        # compute categorical cross entropy
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            assert False, print('Choose either "none" or "mean" for reduction argument')

    def compute_masked_lang_loss(
            self,
            logits_masked: torch.Tensor,
            targets: torch.Tensor,
            targets_masked: torch.Tensor,
            mask_token_id: torch.Tensor
        ) -> torch.Tensor:
        
        """
        Compute the masked language model loss for BERT-like architectures.

        Given a batch of logits predicted for masked positions and their corresponding target tokens, this function 
        computes the cross-entropy loss between the predicted logits and the true labels, but only for positions 
        that have been masked in the input.

        Parameters:
        - logits_masked: Predicted token logits for masked positions from the model.
                         Shape: (batch_size, seq_len, vocab_size).
        - targets: True token IDs for each position in the input sequence.
                   Shape: (batch_size, seq_len).
        - targets_masked: Token IDs for the input sequence, including masked positions.
                          Shape: (batch_size, seq_len).
        - mask_token_id: The ID corresponding to the [MASK] token in the vocabulary.

        Steps:
        1. Compute the cross-entropy loss between predicted logits and true labels across all positions.
        2. For each sample in the batch, locate the positions that were masked.
        3. Extract the loss values corresponding to these masked positions.
        4. Compute and return the mean of these extracted loss values across the batch.

        Returns:
        - Mean cross-entropy loss for masked positions across the batch.

        Note: This function focuses exclusively on masked positions in the input, as is typical for the MLM objective 
        in BERT-like models. It disregards unmasked positions.
        """

        # compute the masked langauge objective loss for masked logits
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_mask = loss_func(
                            logits_masked.permute(0, 2, 1), # (batch_size, vocab_size, seq_len)
                            targets.squeeze(1) # (batch_size, seq_len)
        )

        # list to append loss values
        batch_loss = []

        for ii, target_mask_sample in enumerate(targets_masked):
            
            # locate mask positions. Keep this a 1-D bool TENSOR: the old
            # `.tolist()` produced a nested list (target_mask_sample is
            # [1, seq_len]), and indexing a 1-D tensor with a nested list is the
            # deprecated "non-tuple sequence for multidimensional indexing" path.
            # PyTorch warns that it will become x[torch.tensor(seq)], which on a
            # 1-D tensor raises "too many indices" -- so this was a future hard
            # failure, not just noise. .tolist() also forced a device->host sync
            # on every loop iteration.
            masked_positions = (target_mask_sample == mask_token_id).reshape(-1)
            # extract the loss values at those masked positions
            loss_mask_sample = loss_mask[ii][masked_positions]
            
            # append mean loss value for a given batch sample
            if loss_mask_sample.numel() > 0:
                batch_loss.append(torch.mean(loss_mask_sample).unsqueeze(0))
        
        # Guard on batch_loss, not on loss_mask_sample. The latter is the last
        # loop variable: if the final sample happened to have no masked
        # positions, every other sample's loss was silently discarded and this
        # returned 0.0. It also raised NameError when the batch was empty.
        if batch_loss:
            loss_mask_mean = torch.mean(torch.cat(batch_loss))
        else:
            # no masked positions anywhere in the batch
            loss_mask_mean = torch.tensor(0.0, device=logits_masked.device)

        return loss_mask_mean


###############
# Facilitator #
###############


class Facilitator(nn.Module):

    def __init__(self,
                 in_dim: int,  # Input dimension
                 hid_dim: int,  # Hidden layer dimension
                 out_dim: int,  # Output dimension
                 dropout: float = 0.  # Dropout rate
                 ):
        super().__init__()

        # Main neural network structure
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),  # Weight-normalized linear layer
            nn.GELU(),  # GELU activation function
            nn.Dropout(dropout, inplace=True),  # Dropout layer
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)  # Weight-normalized output layer
        )

    def forward(self, x):
        # Forward pass through the network
        return self.main(x)

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, loss_option='MSE') -> torch.Tensor:
        # Compute loss based on the chosen loss_option ('MSE' or 'MMD')
        if loss_option == 'MSE':
            return Facilitator.compute_MSE(output, target)
        elif loss_option == 'MMD':
            return Facilitator.compute_mmd(output, target)
        else:
            return ValueError("Invalid loss option")
    
    @staticmethod
    def compute_MSE(output, target):
        # Compute Mean Squared Error between output and target
        mse_loss = nn.MSELoss()
        loss = mse_loss(output, target)
        return loss

    @staticmethod
    def compute_kernel(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        Compute the Gaussian RBF kernel between tensors x and y
        """

        # Get the sizes of each mini-batch
        x_size, y_size = x.shape[0], y.shape[0]

        # Dimension based on z size
        dim = x.shape[1]

        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)

        x_core = x.expand(x_size, y_size, dim)
        y_core = y.expand(x_size, y_size, dim)

        # Gaussian RBF kernel computation
        return torch.exp(-(x_core - y_core).pow(2).mean(2) / dim)

    @staticmethod
    def compute_mmd(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between two distributions.
        Args:
            x: Samples from first distribution (z_t_to_p ~ q(z_p))
            y: Samples from second distribution (z_p ~ p(z_p))
        Returns:
            MMD_loss: The MMD loss between the sampled distributions
        """

        x_kernel = Facilitator.compute_kernel(x, x)
        y_kernel = Facilitator.compute_kernel(y, y)
        xy_kernel = Facilitator.compute_kernel(x, y)

        # Calculate MMD loss
        return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
