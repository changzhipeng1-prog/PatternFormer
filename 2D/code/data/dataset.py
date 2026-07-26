"""
Dataset and DataLoader utilities for V3 (2D PDE, 145-node FEM mesh).

PDEContextDataset:
    Wraps the p->solutions lookup table.  Each __getitem__ samples:
        - one target p index
        - k random context p indices (k ~ Uniform[1, max_context_p])
    It does NOT build embeddings; that is deferred to the model's forward
    (or to a pre-build step in the collate function when SequenceBuilder is available).

    Returns raw index data so the collate_fn can access the lookup and
    build the actual embedding sequences via SequenceBuilder.

length_bucket_collate (factory):
    Returns a collate_fn that:
        1. Calls SequenceBuilder for each sample in the batch.
        2. Pads all sequences to the batch's longest actual_len.
        3. Builds the 2D causal + padding attention mask [B, L_max, L_max].
        4. Stacks all batch tensors.

    The 2D attention mask:
        mask[b, i, j] = 0     if position j is visible to position i
                       -inf   otherwise
        Visibility rule: j <= i  (causal) AND j < actual_len[b]  (not padding).
        This is constructed as a float tensor and passed directly to
        Qwen's base transformer via attention_mask argument
        (HuggingFace accepts 2D/4D float masks).

EpochShuffleSampler:
    Shuffles the full set of target p indices once per epoch (deterministic per seed).
"""
import sys
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class PDEContextDataset(Dataset):
    """
    Dynamic context-sampling dataset from a p->solutions lookup table.

    Args:
        lookup:           dict from generate_p_solutions_dataset.py
                          keys: p_values [N], solutions_by_p list[[K_i,145]]
        candidate_p_idx:  LongTensor [M]  — p indices to sample from (train/val/test split)
        max_context_p:    int  — max number of context blocks per sample
        seed:             int
    """

    def __init__(self, lookup: dict, candidate_p_idx: torch.Tensor,
                 max_context_p: int = 10, seed: int = 42,
                 no_context_prob: float = 0.0):
        self.lookup           = lookup
        self.p_values         = lookup["p_values"]              # [N] float
        self.solutions_by_p   = lookup["solutions_by_p"]        # list[[K_i,145]]
        self.candidate_p_idx  = (
            candidate_p_idx.tolist()
            if isinstance(candidate_p_idx, torch.Tensor)
            else list(candidate_p_idx)
        )
        self.max_context_p = int(max_context_p)
        self.base_seed     = int(seed)
        self.epoch         = 0
        self.no_context_prob = float(no_context_prob)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.candidate_p_idx)

    def __getitem__(self, target_idx) -> dict:
        """
        Args:
            target_idx: an integer from candidate_p_idx (the target p bucket index)

        Returns dict:
            target_p_idx:    int
            target_p_val:    float
            target_sols:     Tensor [K_t, 145]
            context_p_idx:   list[int]   length k
            context_p_vals:  list[float]
            context_sols:    list[Tensor [K_i, 145]]
        """
        # target_idx may be:
        #   - int target p index
        #   - tuple(target_p_idx, repeat_id) when sampler repeats per epoch
        if isinstance(target_idx, (tuple, list)):
            target_p_idx = int(target_idx[0])
            repeat_id = int(target_idx[1]) if len(target_idx) > 1 else 0
        else:
            target_p_idx = int(target_idx)
            repeat_id = 0

        rng = np.random.RandomState(
            self.base_seed
            + self.epoch * 1_000_003
            + target_p_idx * 97
            + repeat_id * 1_009
        )

        # Sample context p indices (exclude target).
        # Optional zero-context mode is useful for "p -> all solutions" fine-tuning.
        pool = [x for x in self.candidate_p_idx if x != target_p_idx]
        use_no_context = (
            self.no_context_prob > 0.0
            and rng.rand() < self.no_context_prob
        )
        if use_no_context or len(pool) == 0:
            ctx_idxs = []
        else:
            k = rng.randint(1, min(self.max_context_p, len(pool)) + 1)
            ctx_idxs = rng.choice(pool, size=k, replace=False).tolist()

        target_sols = self.solutions_by_p[target_p_idx]           # [K_t, 1024]
        target_p_val = float(self.p_values[target_p_idx].item())

        ctx_sols  = [self.solutions_by_p[i] for i in ctx_idxs]
        ctx_p_vals = [float(self.p_values[i].item()) for i in ctx_idxs]

        return {
            "target_p_idx":   target_p_idx,
            "target_p_val":   target_p_val,
            "target_sols":    target_sols,
            "context_p_idx":  ctx_idxs,
            "context_p_vals": ctx_p_vals,
            "context_sols":   ctx_sols,
        }


# -----------------------------------------------------------------------
# Sampler
# -----------------------------------------------------------------------

class EpochShuffleSampler(Sampler):
    """
    Yields every p index in candidate_p_idx exactly once per epoch, shuffled.

    DDP-aware: when torch.distributed is initialised, each rank receives a
    non-overlapping shard of the shuffled indices so the 8 ranks process
    disjoint data within each step (true data-parallel).

    The global shuffle is identical across ranks (same seed + epoch), so
    sharding is deterministic and reproducible.
    """

    def __init__(self, candidate_p_idx: torch.Tensor, seed: int = 42, repeat_factor: int = 1):
        self.indices = (
            candidate_p_idx.clone().cpu().long()
            if isinstance(candidate_p_idx, torch.Tensor)
            else torch.tensor(list(candidate_p_idx), dtype=torch.long)
        )
        self.seed  = int(seed)
        self.epoch = 0
        self.repeat_factor = max(1, int(repeat_factor))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        import torch.distributed as dist

        # Repeat the full target-p set with independent shuffles so one target p
        # can appear multiple times per epoch under different sampled contexts.
        chunks = []
        for rep in range(self.repeat_factor):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 10_000 + rep)
            order = torch.randperm(len(self.indices), generator=g)
            # Keep repeat_id so dataset can sample different context per repeat.
            rep_pairs = [(int(self.indices[i].item()), rep) for i in order.tolist()]
            chunks.extend(rep_pairs)
        shuffled = chunks     # list[(target_p_idx, repeat_id)]

        if dist.is_available() and dist.is_initialized():
            rank       = dist.get_rank()
            world_size = dist.get_world_size()
            # Pad to a multiple of world_size so every rank sees the same number of
            # batches; otherwise per-rank sync_policy_grads / all_reduce counts differ
            # and NCCL can deadlock or desync.
            n = len(shuffled)
            pad = (-n) % world_size
            if pad:
                shuffled.extend(shuffled[:pad])
            per_rank = len(shuffled) // world_size
            start = rank * per_rank
            shuffled = shuffled[start : start + per_rank]

        return iter(shuffled)

    def __len__(self):
        import torch.distributed as dist
        n = len(self.indices) * self.repeat_factor
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            return math.ceil(n / world_size)
        return n


class WeightedEpochSampler(Sampler):
    """
    Weighted sampler that over-represents p values with many solutions.

    For each of `repeat_factor` rounds, samples len(indices) items WITH
    replacement according to per-p `weights`.  Each sample receives a unique
    virtual_rep_id = round * N + draw_position so that PDEContextDataset picks
    a different random context combination for every occurrence.

    DDP-aware: the full pair list is split across ranks after shuffling.
    """

    def __init__(
        self,
        candidate_p_idx: torch.Tensor,
        weights: torch.Tensor,
        seed: int = 42,
        repeat_factor: int = 1,
    ):
        self.indices = (
            candidate_p_idx.clone().cpu().long()
            if isinstance(candidate_p_idx, torch.Tensor)
            else torch.tensor(list(candidate_p_idx), dtype=torch.long)
        )
        weights = torch.as_tensor(weights, dtype=torch.float64).cpu()
        assert len(weights) == len(self.indices), "weights must match indices length"
        self.weights = weights / weights.sum()   # normalised
        self.seed = int(seed)
        self.epoch = 0
        self.repeat_factor = max(1, int(repeat_factor))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        import torch.distributed as dist

        N = len(self.indices)
        pairs = []

        for rep in range(self.repeat_factor):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 10_000 + rep)
            # Sample WITH replacement so high-weight p values appear more often.
            drawn = torch.multinomial(self.weights, num_samples=N,
                                      replacement=True, generator=g)
            for i, pos in enumerate(drawn.tolist()):
                virtual_rep_id = rep * N + i   # unique → unique context RNG seed
                pairs.append((int(self.indices[pos].item()), virtual_rep_id))

        # Final global shuffle so different round types are interleaved across batches.
        g_final = torch.Generator()
        g_final.manual_seed(self.seed + self.epoch * 10_000 + 99999)
        perm = torch.randperm(len(pairs), generator=g_final).tolist()
        pairs = [pairs[i] for i in perm]

        if dist.is_available() and dist.is_initialized():
            rank       = dist.get_rank()
            world_size = dist.get_world_size()
            per_rank   = math.ceil(len(pairs) / world_size)
            start      = rank * per_rank
            end        = min(start + per_rank, len(pairs))
            pairs      = pairs[start:end]

        return iter(pairs)

    def __len__(self):
        import torch.distributed as dist
        n = len(self.indices) * self.repeat_factor
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            return math.ceil(n / world_size)
        return n


# -----------------------------------------------------------------------
# Collate (factory)
# -----------------------------------------------------------------------

def make_collate_fn(seq_builder, max_seq_len: int = 256, solution_dim: int = 145):
    """
    Factory that returns a collate_fn closing over seq_builder.

    The returned function:
        1. Calls seq_builder.build() for each sample.
        2. Pads sequences to the longest actual_len in the batch.
        3. Additionally collects raw target block data (canonical_target_sols,
           target_p_vals, target_k_counts, target_block_offsets) so that
           V2PDEModel.forward() can rebuild target block embeddings with full
           gradient tracking for InputProjector and SpecialTokenEmbeddings.

    Returns batch dict:
        inputs_embeds:          [B, L_max, hidden_dim]   bfloat16
                                (pre-built, used for context region only;
                                 target block will be recomputed in forward())
        attention_mask:         [B, L_max]               long (1=real, 0=pad)
        loss_mask:              [B, L_max]               bool
        cls_targets:            [B, L_max]               long
        reg_targets:            [B, L_max, latent_dim]   float32
        p_targets:              [B, L_max]               float32
        actual_lens:            [B]                      long
        target_block_offsets:   [B]                      long  (start of target block)
        target_sols_padded:     [B, K_max, solution_dim]  float32 (padded raw solutions)
        target_k_counts:        [B]                      long  (actual K per sample)
        target_p_vals:          [B]                      float32
    """

    def collate_fn(batch: list) -> dict:
        samples = []
        for item in batch:
            built = seq_builder.build(
                context_p_vals=item["context_p_vals"],
                context_sols=item["context_sols"],
                target_p_val=item["target_p_val"],
                target_sols=item["target_sols"],
            )
            samples.append(built)

        # Determine padding length for this batch
        actual_lens = [s["actual_len"] for s in samples]
        L_max = max(actual_lens)
        B = len(samples)
        hidden_dim = samples[0]["inputs_embeds"].shape[-1]
        latent_dim = samples[0]["reg_targets"].shape[-1]

        # Allocate padded tensors
        inputs_embeds  = torch.zeros(B, L_max, hidden_dim, dtype=torch.bfloat16)
        attention_mask = torch.zeros(B, L_max, dtype=torch.long)
        loss_mask      = torch.zeros(B, L_max, dtype=torch.bool)
        cls_targets    = torch.full((B, L_max), fill_value=-100, dtype=torch.long)
        reg_targets    = torch.zeros(B, L_max, latent_dim, dtype=torch.float32)
        p_targets      = torch.zeros(B, L_max, dtype=torch.float32)

        for i, (s, L) in enumerate(zip(samples, actual_lens)):
            inputs_embeds[i, :L]  = s["inputs_embeds"]
            attention_mask[i, :L] = 1
            loss_mask[i, :L]      = s["loss_mask"]
            cls_targets[i, :L]    = s["cls_targets"]
            reg_targets[i, :L]    = s["reg_targets"]
            p_targets[i, :L]      = s["p_targets"]

        # ---- Raw target block data (for in-forward gradient-tracked embedding) ----
        k_counts = [s["canonical_target_sols"].shape[0] for s in samples]
        K_max    = max(k_counts) if k_counts else 1

        target_sols_padded   = torch.zeros(B, K_max, solution_dim, dtype=torch.float32)
        target_k_counts      = torch.tensor(k_counts, dtype=torch.long)
        target_p_vals_tensor = torch.tensor(
            [s["target_p_val"] for s in samples], dtype=torch.float32
        )
        target_block_offsets = torch.tensor(
            [s["target_block_offset"] for s in samples], dtype=torch.long
        )

        for i, s in enumerate(samples):
            K_t = k_counts[i]
            target_sols_padded[i, :K_t] = s["canonical_target_sols"]

        return {
            "inputs_embeds":        inputs_embeds,                            # [B, L_max, D]
            "attention_mask":       attention_mask,                           # [B, L_max]
            "loss_mask":            loss_mask,                                # [B, L_max]
            "cls_targets":          cls_targets,                              # [B, L_max]
            "reg_targets":          reg_targets,                              # [B, L_max, lat]
            "p_targets":            p_targets,                                # [B, L_max]
            "actual_lens":          torch.tensor(actual_lens, dtype=torch.long),
            "target_block_offsets": target_block_offsets,                    # [B]
            "target_sols_padded":   target_sols_padded,                      # [B, K_max, 1024]
            "target_k_counts":      target_k_counts,                         # [B]
            "target_p_vals":        target_p_vals_tensor,                    # [B]
        }

    return collate_fn


# -----------------------------------------------------------------------
# 2D attention mask builder (used inside V2PDEModel.forward)
# -----------------------------------------------------------------------

def build_2d_causal_mask(actual_lens: torch.Tensor, L_max: int,
                          device: torch.device) -> torch.Tensor:
    """
    Build a 4D boolean attention mask for HuggingFace Transformers.

    Convention: True  → attend  (additive value = 0)
                False → block   (additive value = -inf)

    The mask encodes:
        causal constraint:  j <= i
        padding constraint: j < actual_len[b]

    Args:
        actual_lens: [B]  int64 — real sequence lengths
        L_max:       int
        device:      torch.device

    Returns:
        [B, 1, L_max, L_max] bool tensor
    """
    B = actual_lens.shape[0]
    i_idx = torch.arange(L_max, device=device).unsqueeze(1)   # [L, 1]
    j_idx = torch.arange(L_max, device=device).unsqueeze(0)   # [1, L]

    causal = (j_idx <= i_idx)                                  # [L, L]

    # padding: j < actual_len[b]  → [B, L]
    lens = actual_lens.to(device).unsqueeze(1)                 # [B, 1]
    not_pad = (j_idx < lens)                                   # [B, L]

    # combine explicitly to [B, 1, L, L]:
    #   causal:  [1, 1, L, L]
    #   not_pad: [B, 1, 1, L]  (only key dimension j is padded)
    # -> broadcast result [B, 1, L, L]
    mask = causal.unsqueeze(0).unsqueeze(1) & not_pad.unsqueeze(1).unsqueeze(2)
    return mask                                                 # bool [B, 1, L, L]
