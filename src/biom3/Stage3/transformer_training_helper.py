# ----- Standard Libraries -----
import itertools
from pathlib import Path

# ----- Numerical Processing -----
import numpy as np

# ----- Progress Visualization -----
from tqdm.auto import tqdm

# ----- PyTorch Core -----
import torch
import torch.nn.functional as F
import torch.nn as nn

# ----- PyTorch Probability Distributions -----
from torch.distributions import OneHotCategorical
from torch.distributions import Categorical

# ----- Custom Modules -----
import biom3.Stage3.eval_metrics as eval_funcs


def sample_random_path(
        batch_size: int,
        seq_length: int,
        device: str='device'
    ) -> torch.Tensor:
    """
    Generate random sampling paths for autoregressive diffusion process.
    
    This function creates a batch of random permutations representing the order
    in which tokens will be generated in the diffusion process. Each sequence
    in the batch gets its own unique random sampling path.
    
    For example, if seq_length is 5, a random path might be [3, 0, 4, 1, 2],
    indicating that the 4th position (0-indexed) should be sampled first,
    then the 1st position, etc.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_length: Length of each sequence
        device: Device on which to create the tensors ('cpu', 'cuda', etc.)
        
    Returns:
        torch.Tensor: Tensor of shape [batch_size, seq_length] containing
                      random permutations for each sequence in the batch
    """
    # Create a batch of random sampling paths
    random_paths = torch.stack(
            [torch.randperm(seq_length, device=device) for _ in range(batch_size)],
            dim=0  # Use dim instead of axis for PyTorch convention
    )
    # Alternative: sequential paths (commented out)
    # random_paths = torch.stack(
    #        [torch.arange(seq_length, device=device) for _ in range(batch_size)],
    #        dim=0
    # )
    return random_paths

def create_mask_at_random_path_index(
        sample_random_path: torch.Tensor,
        idx: any,
        batch_size: int,
        seq_length: int
    ) -> torch.Tensor:
    """
    Create masks indicating positions already sampled in the diffusion process.
    
    This function generates binary masks where 1s represent positions that have 
    already been sampled (positions with indices less than the current timestep),
    and 0s represent positions that have not yet been sampled or are currently
    being sampled.
    
    Args:
        sample_random_path: Tensor of shape [batch_size, seq_length] containing 
                           random permutations for sampling order
        idx: Current timestep indices, can be a scalar or tensor of shape [batch_size]
        batch_size: Number of sequences in the batch 
        seq_length: Length of each sequence
        
    Returns:
        torch.Tensor: Binary mask of shape [batch_size, seq_length] where:
                     1 = position already sampled
                     0 = position not yet sampled or currently being sampled
    """
    # create a mask that has 1s everywhere we've sampled and 0's everywhere else
    mask = (sample_random_path < idx)
    return mask

def create_sampling_location_mask(
        sampled_random_path: torch.Tensor,
        idx: any,
        batch_size: int,
        seq_length: int
    ) -> torch.Tensor:
    """
    Create masks indicating current sampling positions in the diffusion process.
    
    This function generates binary masks where 1s represent the positions currently
    being sampled (positions with indices equal to the current timestep).
    
    Args:
        sampled_random_path: Tensor of shape [batch_size, seq_length] containing 
                            random permutations for sampling order
        idx: Current timestep indices, can be a scalar or tensor of shape [batch_size]
        batch_size: Number of sequences in the batch
        seq_length: Length of each sequence
        
    Returns:
        torch.Tensor: Binary mask of shape [batch_size, seq_length] where:
                     1 = position currently being sampled
                     0 = all other positions
    """
    # create a binary mask that has 1 at the current location for us to sample
    sampling_location_mask = (sampled_random_path == idx).long()
    return sampling_location_mask

def create_mask_at_future_path_index(
        sampled_random_path: torch.Tensor,
        idx: any,
        batch_size: int,
        seq_length: int
    ) -> torch.Tensor:
    """
    Create masks indicating future sampling positions in the diffusion process.
    
    This function generates binary masks where 1s represent positions that will be 
    sampled in the future (positions with indices greater than the current timestep),
    and 0s represent positions already sampled or currently being sampled.
    
    Args:
        sampled_random_path: Tensor of shape [batch_size, seq_length] containing 
                            random permutations for sampling order
        idx: Current timestep indices, can be a scalar or tensor of shape [batch_size]
        batch_size: Number of sequences in the batch
        seq_length: Length of each sequence
        
    Returns:
        torch.Tensor: Binary mask of shape [batch_size, seq_length] where:
                     1 = position to be sampled in the future
                     0 = position already sampled or currently being sampled
    """
    # create a mask that has 1s everywhere were are not going to be sampling and
    # 0's everywhere we previously and currently sampled
    sampling_future_mask = (sampled_random_path > idx).long()
    return sampling_future_mask

def sample_from_conditional(conditional_prob: any) -> torch.Tensor:
    """
    Sample tokens from the conditional probability distribution.
    
    This function draws samples from the provided categorical distribution,
    which represents probabilities over possible tokens at each position.
    
    Args:
        conditional_prob: OneHotCategorical distribution representing token
                         probabilities at each position
        
    Returns:
        torch.Tensor: Sampled one-hot encoded tokens with shape [batch_size, seq_length, vocab_size]
    """
    # sample from the categorical dist.
    return conditional_prob.sample().permute(0,2,1)

def compute_entropy(conditional_prob: any) -> torch.Tensor:
    """
    Compute the entropy of the model's predicted probability distribution.
    
    Entropy measures the uncertainty or randomness in the distribution - higher
    entropy indicates more uncertainty in token predictions.
    
    Args:
        conditional_prob: OneHotCategorical distribution representing token
                         probabilities at each position
        
    Returns:
        torch.Tensor: Entropy values for each position in the sequence
    """
    # we can directly compute the entropy of the categorical distribution
    return conditional_prob.entropy()


class exp_weight_time_sample:
    """
    Exponentially-weighted time step sampler for diffusion models.
    
    This class implements a sampler that preferentially selects earlier time steps
    based on an exponential decay function. This biased sampling strategy can be 
    useful for diffusion training, where earlier time steps might require more
    attention than later ones.
    
    The probability of sampling time step t is proportional to exp(-decay_rate * t),
    which means earlier time steps (smaller t) have higher probability of being chosen.
    
    Attributes:
        timesteps (int): Total number of diffusion steps
        decay_rate (float): Rate of exponential decay for sampling weights
        weights (torch.Tensor): Normalized sampling weights for each time step
    """
    def __init__(self, timesteps: int, decay_rate: float):
        """
        Initialize the exponentially-weighted time step sampler.
        
        Args:
            timesteps: Total number of diffusion steps
            decay_rate: Controls how quickly the sampling probability decreases
                        with increasing time step. Higher values mean stronger
                        preference for early time steps.
        """
        self.timesteps = timesteps
        self.decay_rate = decay_rate
        # Compute the weight based on the exp function
        self.weights = torch.tensor(
                [torch.exp(-torch.tensor([i])*decay_rate) for i in range(self.timesteps)]
        )
        # Normalize weights to form a valid probability distribution
        self.weights /= self.weights.sum()
        
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample time steps according to the exponential weighting.
        
        Args:
            batch_size: Number of time step samples to generate
            
        Returns:
            torch.Tensor: Tensor of shape [batch_size] containing sampled time steps
        """
        # Generate random samples according to the computed weights
        samples = torch.multinomial(self.weights, batch_size, replacement=True)
        return samples

def sample_random_index_for_sampling(
        batch_size: int,
        seq_length: int,
        device: str='xpu',
        option: str='random'
    ) -> any:
    """
    Sample indices representing the current sampling positions in the diffusion process.
    
    This function generates indices that determine which positions in the sequence
    should be sampled next. It supports two sampling strategies:
    - 'random': Uniform random sampling across all positions
    - 'weighted': Biased sampling favoring earlier positions using exponential weighting
    
    Args:
        batch_size: Number of sequences in the batch
        seq_length: Length of each sequence
        device: Device on which to create the tensors ('cpu', 'cuda', etc.)
        option: Sampling strategy to use ('random' or 'weighted')
        
    Returns:
        torch.Tensor: Tensor of shape [batch_size, 1] containing the sampled indices
    """
    if option == 'random':
        # Sample a random index where we want to sample next
        idx = torch.randint(
                low=0,
                high=seq_length+1,
                size=(batch_size,1),
                device=device,
                requires_grad=False
        )
    elif option == 'weighted':
        time_sampler = exp_weight_time_sampler(timesteps=seq_length+1, decay_rate=0.005)
        # Sample a weighted random index where we want to sample next
        idx = time_sampler.sample(batch_size=batch_size).unsqueeze(1).to(device)
    return idx

def log_prob_of_realization(
        args: any,
        conditional_prob: any,
        real_tokens: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate log probabilities of actual tokens under the model's predicted distribution.
    
    This function computes the log probability of the ground truth tokens (real_tokens)
    according to the conditional probability distribution produced by the model.
    These log probabilities are used to calculate the loss function for training.
    
    Args:
        args: Configuration object (device information is used)
        conditional_prob: OneHotCategorical distribution from the model's predictions
        real_tokens: Ground truth tokens, shape [batch_size, seq_length]
        
    Returns:
        torch.Tensor: Log probabilities of the real tokens, shape [batch_size, seq_length]
    """
    # Compute the log-prob of a given realization
    #log_prob = conditional_prob._categorical.log_prob(real_tokens.to(args.device))
    log_prob = conditional_prob._categorical.log_prob(real_tokens)
    #log_prob = conditional_prob.log_prob(real_tokens.to(args.device))
    return log_prob

def log_prob_of_unsampled_locations(
        log_prob: torch.Tensor,
        token_mask: torch.Tensor
        ) -> torch.Tensor:
    """
    Compute log probabilities for positions that haven't been sampled yet.
    
    This function filters the log probabilities to only include positions that
    have not yet been generated (where token_mask is 0), zeroing out positions
    that have already been sampled. It then sums these log probabilities across
    the sequence dimension for each sample in the batch.
    
    Args:
        log_prob: Tensor of shape [batch_size, seq_length] containing log probabilities 
                 of the real tokens under the model's distribution
        token_mask: Binary mask of shape [batch_size, seq_length] where:
                    1 = position already sampled
                    0 = position not yet sampled
        
    Returns:
        torch.Tensor: Summed log probabilities of unsampled positions for each 
                     sample in the batch, shape [batch_size]
    """
    # Compute the total log prob of the unsampled locations, taking sum over log-probs
    log_prob_unsampled = ((token_mask == 0)*1 * log_prob)
    return log_prob_unsampled.sum(1)

def unsampled_loss_terms(
        log_prob: torch.Tensor,
        token_mask: torch.Tensor,
        real_tokens: torch.Tensor,
        pad_token_id: int
    ) -> dict:
    """
    Split the loss over unsampled positions into sequence and padding terms.

    Every unsampled position is one of two kinds, by its true token: padding,
    the pad tokens that fill the tail of a sequence up to the model's fixed
    length, or non-pad, the <START>, residue and <END> tokens of the sequence
    itself. Padding is a property of the target. It is unrelated to the mask
    token (0) that marks unsampled positions in the model's input, and a pad
    that has already been sampled is context, not a target.

    Each sequence's term is the mean negative log-probability over the
    positions it actually contributes: its unsampled non-pad positions, or its
    unsampled pad positions. That count is not a function of the diffusion time.
    The unsampled positions are drawn from the whole fixed-length window, so how
    many of them fall on the sequence depends on its length and on the draw, and
    two sequences at the same diffusion time contribute different numbers. A
    sequence that contributes none is left out of the batch average.

    Args:
        log_prob: Tensor of shape [batch_size, seq_length] containing log probabilities
                 of the real tokens under the model's distribution
        token_mask: The model's input tokens, shape [batch_size, seq_length],
                    where 0 = position not yet sampled
        real_tokens: Ground truth tokens, shape [batch_size, seq_length]
        pad_token_id: Index of the pad token in the vocabulary

    Returns:
        dict: scalar losses under the keys 'non_pad' and 'pad'
    """
    unsampled = token_mask == 0
    is_pad = real_tokens == pad_token_id
    zeros = torch.zeros_like(log_prob)
    terms = {}
    for name, positions in (('non_pad', unsampled & ~is_pad), ('pad', unsampled & is_pad)):
        count = positions.sum(1)
        total = torch.where(positions, log_prob, zeros).sum(1)
        per_sequence = -total / count.clamp(min=1)
        contributes = count > 0
        terms[name] = (per_sequence * contributes).sum() / contributes.sum().clamp(min=1)
    return terms

def weight_log_prob(
        log_prob_unsampled: torch.Tensor,
        idx: any,
        seq_length
    ) -> torch.Tensor:
    """
    Weight the unsampled log probabilities to normalize for varying timesteps.
    
    As the diffusion process progresses, the number of unsampled positions decreases.
    This function normalizes the summed log probabilities by dividing by the number
    of remaining positions, ensuring consistent scaling regardless of timestep.
    
    Args:
        log_prob_unsampled: Summed log probabilities of unsampled positions,
                           shape [batch_size]
        idx: Current timestep indices, shape [batch_size, 1]
        seq_length: Length of each sequence
        
    Returns:
        torch.Tensor: Weighted log probabilities, shape [batch_size]
    """
    # Compute the average log-prob over the unsampled locations
    log_prob_weighted = 1/(seq_length - idx.squeeze(1) + 1) * log_prob_unsampled
    return log_prob_weighted

def compute_average_loss_for_batch(log_prob_weighted: torch.Tensor) -> torch.Tensor:
    """
    Compute the final loss by taking the negative mean of weighted log probabilities.
    
    This function converts the weighted log probabilities into a scalar loss value
    by taking their negative mean across the batch dimension. The negative sign 
    converts the maximization of log probability into the minimization of loss.
    
    Args:
        log_prob_weighted: Weighted log probabilities, shape [batch_size]
        
    Returns:
        torch.Tensor: Scalar loss value (negative mean of weighted log probabilities)
    """
    # Compute a (negative) average over the batch elements to compute an unbiased estimator of the loss
    loss = -log_prob_weighted.mean()
    return loss

def create_token_labels(args, realization) -> (
        torch.Tensor,
        int,
        int
    ):
    """
    Convert raw input data to numerical token representations for the model.
    
    This function takes the input realization (e.g., binary MNIST images or protein sequences)
    and maps them to appropriate numerical token indices based on the task type.
    
    For MNIST:
    - 0 = mask (absorbing state)
    - 1 = background pixel
    - 2 = foreground pixel
    
    For proteins:
    - Uses the raw numerical representation of amino acids
    
    Args:
        args: Configuration object containing task type ('MNIST' or 'proteins')
        realization: Input data tensor of shape [batch_size, channel, seq_length]
        
    Returns:
        tuple: (
            real_tokens: Tensor of shape [batch_size, seq_length] containing numerical token indices
            bs: Batch size
            seq_length: Sequence length
        )
    """
    bs, channel, seq_length = realization.size()
    temp_real = realization.reshape(bs, channel, seq_length)*1
    if args.task == 'MNIST':
        real_tokens = (temp_real == 1)*2 + (temp_real == 0)*1  # numerical token labels for mnist
    elif args.task == 'proteins':
        real_tokens = temp_real
        # real_tokens = temp_real + 1
    # background --> label 1
    # foreground --> label 2
    # mask (absorbing state) --> label 0
    return (
            real_tokens.squeeze(1),
            bs,
            seq_length
    )


def mask_realizations(
        real_tokens: torch.Tensor,
        random_path_mask: torch.Tensor
    ) -> torch.Tensor:
    """
    Mask positions in the sequence that haven't been sampled yet.
    
    This function processes the real tokens by replacing positions that have not
    yet been sampled (according to the random_path_mask) with mask tokens (0).
    This creates the partially-masked input needed for the diffusion model to
    make conditional predictions.
    
    The masking process works by:
    1. Identifying positions where random_path_mask is False (not yet sampled)
    2. Setting those positions to the mask token value (0)
    3. Leaving already sampled positions unchanged
    
    Args:
        real_tokens: Tensor of shape [batch_size, seq_length] containing token indices
        random_path_mask: Binary mask of shape [batch_size, seq_length] where:
                         True/1 = position already sampled
                         False/0 = position not yet sampled
        
    Returns:
        torch.Tensor: Masked sequence tensor of same shape as real_tokens,
                     with unsampled positions set to 0
    """
    out_real_tokens = real_tokens.clone()
    # batch size
    bs = random_path_mask.shape[0]
    # convert random path to boolean
    bool_rand_path_mask = random_path_mask.to(dtype=torch.bool)
    # positional masks
    # mask the future sample positions
    future_mask_positions = ((~bool_rand_path_mask)*1).squeeze(1)
    for ii in range(bs):
        mask_positions = future_mask_positions[ii].nonzero().tolist()
        # insert mask tokens
        out_real_tokens[ii, mask_positions] = 0
    return out_real_tokens

def predict_conditional_prob(
        model: nn.Module,
        real_token_masked: torch.Tensor,
        idx: any,
        args: any
    ) -> (
            any,
            torch.Tensor
    ):
    """
    Generate conditional probabilities for the next token prediction.
    
    This function:
    1. Passes the masked sequence through the model to get logits
    2. Converts logits to probabilities using softmax
    3. Creates a OneHotCategorical distribution for sampling
    
    The resulting distribution represents the model's prediction of which
    token should be placed at each masked position, conditioned on the
    already-revealed tokens and the current timestep.
    
    Args:
        model: Neural network model for token prediction
        real_token_masked: Partially masked sequence tensor with shape [batch_size, seq_length]
        idx: Current timestep indices with shape [batch_size, 1]
        args: Configuration object (contains device information)
        
    Returns:
        tuple: (
            conditional_prob: OneHotCategorical distribution for sampling
            probs: Softmax probabilities tensor with shape [batch_size, num_tokens, seq_length]
        )
    """
    #logits = model(x=real_token_masked.to(args.device), t=idx.view(-1,))
    logits = model(x=real_token_masked, t=idx.view(-1,))
    probs = F.softmax(
            logits,
            dim=1
    )
    conditional_prob = OneHotCategorical(probs=probs.permute(0,2,1))
    return (
            conditional_prob,
            probs
    )


"""
Here, we compute the previous position tokens, current token position, and future token positions, where
past, current, and future are defined by the time trajectory.
"""

@torch.no_grad()
def sample_from_conditional(conditional_prob: any) -> torch.Tensor:
    """
    Sample tokens from the model's predicted probability distribution.

    This function draws categorical samples from the model's output distribution
    and arranges them in the format expected by the rest of the pipeline.
    The @torch.no_grad decorator ensures no gradients are computed during sampling.

    Args:
        conditional_prob: OneHotCategorical distribution representing token
                         probabilities at each position

    Returns:
        torch.Tensor: One-hot encoded samples with shape [batch_size, seq_length, vocab_size]
    """
    # Draw a sample from the categorical distribution
    cond_prob_sample = conditional_prob.sample().permute(0,2,1)
    return cond_prob_sample

@torch.no_grad()
def sample_recover(
        real_tokens: torch.Tensor,
        cond_prob_sample: torch.Tensor,
        current_path_mask: torch.Tensor
    ) -> float:
    """
    Compute accuracy at the current sampling position for each sequence.

    This function measures how often the model's prediction matches the ground
    truth at the specific position being sampled in the current timestep.
    Accuracy is calculated over the batch by comparing predicted tokens to
    real tokens at the designated "current" positions.

    Args:
        real_tokens: Ground truth token indices with shape [batch_size, seq_length]
        cond_prob_sample: Model's sampled token probabilities with shape
                          [batch_size, seq_length, vocab_size]
        current_path_mask: Binary mask indicating current sampling positions with shape
                          [batch_size, seq_length]

    Returns:
        float: Accuracy as a fraction representing the portion of sequences where
               the model correctly predicted the token at the current position
    """
    # Remove from GPU (note: these operations don't modify the original tensors)
    real_tokens.cpu()
    cond_prob_sample.cpu()
    current_path_mask.cpu()

    # Find current sampling index for each sequence in the batch
    current_tensor_pos = torch.argmax((current_path_mask == 1)*1, dim=-1)

    # Check if model predictions match the ground truth label at current sampling index
    match_preds = [(
        real_tokens[seq_idx, ii] == torch.argmax(cond_prob_sample, dim=1)[seq_idx, ii]
        ).item()*1 for seq_idx, ii in enumerate(current_tensor_pos.cpu().numpy())
    ]

    # Return accuracy as fraction of correct predictions
    return sum(match_preds)/len(match_preds)

@torch.no_grad()
def compute_prev_token_acc(
        cond_real_tokens: torch.Tensor,
        cond_prob_sample: torch.Tensor,
        path_mask: torch.Tensor
    ) -> np.ndarray:
    """
    Compute accuracy of model predictions at previously sampled positions.
    
    This function evaluates how well the model's predictions match the ground truth
    at positions that have already been sampled (according to path_mask). This
    measures the model's ability to correctly "reconstruct" tokens it has already seen.
    
    The function:
    1. Identifies previously sampled positions for each sequence
    2. Extracts real and predicted tokens at those positions
    3. Computes the match rate for each sequence
    4. Returns the average accuracy across all sequences
    
    Args:
        cond_real_tokens: Ground truth token indices with shape [batch_size, seq_length]
        cond_prob_sample: Model's sampled token probabilities with shape 
                          [batch_size, seq_length, vocab_size]
        path_mask: Binary mask indicating previously sampled positions with shape
                  [batch_size, seq_length]
        
    Returns:
        float: Mean accuracy across the batch for previously sampled positions
    """
    # Move to CPU for processing
    cond_real_tokens.cpu()
    cond_prob_sample.cpu()
    path_mask.cpu()
    
    # Get predicted token indices by taking argmax
    cond_sample_tokens = torch.argmax(cond_prob_sample, dim=1)
    
    matches = []
    for ii, sample_pos in enumerate(path_mask):
        # Extract real tokens at previously sampled positions
        temp_real_tokens = cond_real_tokens[ii, sample_pos.nonzero()].squeeze(1)
        # Extract predicted tokens at the same positions
        temp_sample_tokens = cond_sample_tokens[ii, sample_pos.nonzero()].squeeze(1)
        # Record whether predictions match ground truth
        matches.append(
                (temp_real_tokens == temp_sample_tokens).tolist()
        )
    
    # Calculate accuracy for each sequence
    acc = []
    for match in matches:
        try:
            acc.append(sum(match*1)/len(match))
        except ZeroDivisionError:
            # Handle case where no positions have been sampled yet
            acc.append(0)
            
    # Return mean accuracy across all sequences
    return np.mean(acc)



@torch.no_grad()
def compute_future_token_acc(
        cond_real_tokens: torch.Tensor,
        cond_prob_sample: torch.Tensor,
        path_mask: torch.Tensor
    ) -> np.ndarray:
    """
    Compute accuracy of model predictions at future sampling positions.
    
    This function evaluates how well the model's predictions match the ground truth
    at positions that will be sampled in future timesteps (according to path_mask).
    This measures the model's ability to predict tokens it hasn't explicitly seen yet.
    
    The function:
    1. Identifies future sampling positions for each sequence
    2. Extracts real and predicted tokens at those positions
    3. Computes the match rate for each sequence
    4. Returns the average accuracy across all sequences
    
    Args:
        cond_real_tokens: Ground truth token indices with shape [batch_size, seq_length]
        cond_prob_sample: Model's sampled token probabilities with shape 
                          [batch_size, seq_length, vocab_size]
        path_mask: Binary mask indicating future sampling positions with shape
                  [batch_size, seq_length]
        
    Returns:
        float: Mean accuracy across the batch for future sampling positions
    """
    # Move to CPU for processing
    cond_real_tokens.cpu()
    cond_prob_sample.cpu()
    path_mask.cpu()
    
    # Get predicted token indices by taking argmax
    cond_sample_tokens = torch.argmax(cond_prob_sample, dim=1)
    
    matches = []
    for ii, sample_pos in enumerate(path_mask):
        # Extract real tokens at future sampling positions
        temp_real_tokens = cond_real_tokens[ii, sample_pos.nonzero()].squeeze(1)
        # Extract predicted tokens at the same positions
        temp_sample_tokens = cond_sample_tokens[ii, sample_pos.nonzero()].squeeze(1)
        # Record whether predictions match ground truth
        matches.append(
                (temp_real_tokens == temp_sample_tokens).tolist()
        )
    
    # Calculate accuracy for each sequence
    acc = []
    for match in matches:
        try:
            acc.append(sum(match*1)/len(match))
        except ZeroDivisionError:
            # Handle case where no future positions exist
            acc.append(0)
    
    # Return mean accuracy across all sequences
    # Note: Fixed indentation error - this was inside the loop in the original
    return np.mean(acc)


@torch.no_grad()
def compute_pos_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate the average positional entropy of the model's predicted probabilities.

    Entropy measures the uncertainty in the model's predictions - higher values
    indicate more uniform probability distributions (higher uncertainty), while
    lower values indicate more peaked distributions (higher confidence).

    This function computes the information-theoretic entropy for each position
    and token class using the formula: -p*log(p), then averages across all
    positions and examples in the batch to get a single scalar metric.

    The @torch.no_grad decorator ensures no gradients are computed during this
    calculation, making it more memory-efficient for evaluation.

    Args:
        probs: Tensor of shape [batch_size, vocab_size, seq_length] containing
              softmax probabilities from the model

    Returns:
        torch.Tensor: Scalar tensor containing the average positional entropy
    """
    # Calculate average positional entropy using the formula: -p*log(p)
    # First average across vocab dimension, then across batch dimension
    pos_entropy = torch.mean(torch.mean(-probs * torch.log(probs), dim=1), dim=0)
    return pos_entropy

def elbo_objective(
        model: nn.Module,
        realization: torch.Tensor,
        args: any
        ) -> (
                torch.Tensor,
                float,
                float,
                float,
                torch.Tensor
        ):
    """
    Implement the Evidence Lower Bound (ELBO) objective for diffusion model training.
    
    This function executes the complete training procedure for a single batch:
    1. Generates random sampling paths and timesteps
    2. Creates appropriate masks for different sequence positions
    3. Tokenizes and masks the input data
    4. Obtains model predictions for the masked positions
    5. Computes the loss (negative log-likelihood)
    6. Calculates various performance metrics
    
    The ELBO objective represents a lower bound on the log-likelihood of the data
    under the diffusion model, and maximizing this objective (minimizing the loss)
    trains the model to predict the original data distribution.
    
    Args:
        model: Neural network module implementing the diffusion model
        realization: Input data tensor of shape [batch_size, channel, seq_length]
        args: Configuration object containing model parameters and device info
        
    Returns:
        tuple: (
            loss: Tensor containing the negative ELBO (training loss)
            acc: Accuracy at current sampling positions
            prev_acc: Accuracy at previously sampled positions
            future_acc: Accuracy at future sampling positions
            pos_entropy: Average entropy of the model's predictions
        )
    """
    bs, channel, seq_length = realization.size()
    # get a batch of random sampling paths
    sampled_random_path = sample_random_path(bs, seq_length, device=args.device)
    # sample a set of random sampling steps for each individual training image in the current batch
    idx = sample_random_index_for_sampling(bs, seq_length, device=args.device, option='random')
    # we create a mask that masks the locations wher we've already sampled
    random_path_mask = create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
    # create a mask that masks the locations where are currently sampling
    current_path_mask = create_sampling_location_mask(sampled_random_path, idx, bs, seq_length)
    # future samplign locations (i.e. >t)
    future_path_mask = create_mask_at_future_path_index(sampled_random_path, idx, bs, seq_length)
    # tokenize realizations
    real_tokens, bs, seq_length = create_token_labels(args, realization)
    # mask realizations
    real_token_masked = mask_realizations(real_tokens, random_path_mask)
    # conditional probs
    conditional_prob, probs = predict_conditional_prob(model, real_token_masked, idx, args)
    # evaluate the value of the log prob for the given realization
    log_prob = log_prob_of_realization(args, conditional_prob, real_tokens)
    # compute an average over all the unsampled locations for each image in the batch
    #log_prob_unsampled = log_prob_of_unsampled_locations(log_prob.to(args.device), real_token_masked.to(args.device))
    log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, real_token_masked)
    # compute an average over all the unsampled locations for each image in the batch
    log_prob_weighted = weight_log_prob(log_prob_unsampled, idx, seq_length)
    # compute an average loss i.e. negative average log likelihood over teh batch elements
    loss = compute_average_loss_for_batch(log_prob_weighted)
    # compute metrics
    cond_prob_sample = sample_from_conditional(conditional_prob)
    acc = sample_recover(real_tokens, cond_prob_sample, current_path_mask)
    prev_acc = compute_prev_token_acc(real_tokens, cond_prob_sample, random_path_mask)
    future_acc = compute_future_token_acc(real_tokens, cond_prob_sample, future_path_mask)
    # average positional entropy
    pos_entropy = compute_pos_entropy(probs=probs)
    return (
            loss,
            acc,
            prev_acc,
            future_acc,
            pos_entropy
    )

def cond_predict_conditional_prob(
        model: nn.Module,
        real_token_masked: torch.Tensor,
        y_c: torch.Tensor,
        idx: any,
        args: any
    ) -> (
            any,
            torch.Tensor
    ):
    """
    Generate conditional probabilities guided by both masked sequence and external condition.
    
    This function extends the basic predict_conditional_prob by incorporating
    an additional conditional input (y_c), which could be a class embedding,
    text embedding, or other guiding signal. The model uses this condition
    to better guide its predictions toward sequences that satisfy both the
    partially revealed tokens and the external condition.
    
    The conditioning process allows for controlled generation, where the
    diffusion model's output is steered toward specific characteristics
    defined by the conditional input.
    
    Args:
        model: Neural network model for conditional token prediction
        real_token_masked: Partially masked sequence tensor with shape [batch_size, seq_length]
        y_c: Conditional embedding tensor (e.g., text embedding) to guide generation
        idx: Current timestep indices with shape [batch_size, 1]
        args: Configuration object (contains device information)
        
    Returns:
        tuple: (
            conditional_prob: OneHotCategorical distribution for sampling
            probs: Softmax probabilities tensor with shape [batch_size, num_tokens, seq_length]
        )
    """
    #logits = model(x=real_token_masked.to(args.device), t=idx.view(-1,), y_c=y_c)
    logits = model(x=real_token_masked, t=idx.view(-1,), y_c=y_c)
    probs = F.softmax(
            logits,
            dim=1
    )
    conditional_prob = OneHotCategorical(probs=probs.permute(0,2,1))
    #conditional_prob = Categorical(probs=probs.permute(0,2,1))
    return (
            conditional_prob,
            probs
    )


def cond_elbo_objective(
        model: nn.Module,
        realization: torch.Tensor,
        y_c: torch.Tensor,
        args: any,
        iteration: int
        ) -> (
                torch.Tensor,
                tuple
        ):
    """
    Implement the conditional Evidence Lower Bound (ELBO) objective for text-guided diffusion.
    
    This function extends the standard ELBO objective to incorporate conditional 
    information (y_c), typically text embeddings, that guide the diffusion process.
    The conditioning allows the model to generate sequences that satisfy both the
    autoregressive constraints and the external guidance signal.
    
    The function:
    1. Samples random paths and timesteps for the diffusion process
    2. Creates masks for different sequence positions (past, current, future)
    3. Applies the model to predict token probabilities conditioned on both
       partially revealed sequences and the text embedding
    4. Computes the loss (negative conditional log-likelihood)
    5. Periodically evaluates performance metrics (based on iteration count)
    
    Args:
        model: Neural network module implementing the diffusion model
        realization: Input data tensor of shape [batch_size, channel, seq_length]
        y_c: Conditional embedding tensor (e.g., text embedding) to guide generation
        args: Configuration object containing model parameters and device info
        iteration: Current training iteration, used to decide when to compute metrics
        
    Returns:
        tuple: (
            loss: Tensor containing the negative conditional ELBO (training loss)
            metric_evals: Tuple of performance metrics (accuracies, perplexities)
                         or (None) if not an evaluation iteration
        )
    """
    bs, channel, seq_length = realization.size()

    # get a batch of random sampling paths
    sampled_random_path = sample_random_path(bs, seq_length, device=args.device)
    # sample a set of random sampling steps for each individual training samples in the current batch
    idx = sample_random_index_for_sampling(bs, seq_length, device=args.device, option='random')
    # we create a mask that masks the locations wher we've already sampled
    random_path_mask = create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
    # create a mask that masks the locations where are currently sampling
    current_path_mask = create_sampling_location_mask(sampled_random_path, idx, bs, seq_length)
    # future samplign locations (i.e. >t)
    future_path_mask = create_mask_at_future_path_index(sampled_random_path, idx, bs, seq_length)
    # tokenize realizations
    real_tokens, bs, seq_length = create_token_labels(args,realization)
    #real_tokens = realizations.clone().squeeze(1)
    # mask realizations
    real_token_masked = mask_realizations(real_tokens, random_path_mask)
    # conditional probs
    conditional_prob, probs = cond_predict_conditional_prob(model, real_token_masked, y_c, idx, args)
    # evaluate the value of the log prob for the given realization
    log_prob = log_prob_of_realization(args, conditional_prob, real_tokens)
    # compute an average over all the unsampled locations for each image in the batch
    #log_prob_unsampled = log_prob_of_unsampled_locations(log_prob.to(args.device), real_token_masked.to(args.device))
    log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, real_token_masked)
    #log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, real_token_masked, real_tokens)

    # compute an average over all the unsampled locations for each image in the batch
    log_prob_weighted = weight_log_prob(log_prob_unsampled, idx, seq_length)
    # compute an average loss i.e. negative average log likelihood over teh batch elements
    loss = compute_average_loss_for_batch(log_prob_weighted)

    # compute metrics
    if iteration % args.enter_eval == 0:
        with torch.no_grad():
            # compute accuracy given time position
            sample_seq = torch.argmax(sample_from_conditional(conditional_prob), dim=1) # create numerical token sequences

            # convert to cpu
            real_tokens = real_tokens.cpu()
            sample_seq = sample_seq.cpu()
            idx = idx.cpu()
            sampled_random_path = sampled_random_path.cpu()
            probs = probs.cpu()

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
            pos_entropy = compute_pos_entropy(probs=probs).mean().item()

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
    else:
        metric_evals = (None)

    return (
            loss,
            metric_evals
    )

