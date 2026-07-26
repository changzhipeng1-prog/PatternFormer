"""
Dataset and DataLoader utilities for 1D_a2a4 V2 training.

Data format: list of dicts  [{params: [a4, a2], solutions: [arr1024, ...]}, ...]

PDEContextDataset:
    Each __getitem__ returns one target sample + k random context samples.

WeightedEpochSampler:
    Inverse-frequency weighting by k class to handle class imbalance
    (k=4 dominates 77%, k=5 only 0.6%).

make_collate_fn:
    Builds batches with 2D causal+padding attention masks.
    params_targets shape: [B, L, 2]  (a4, a2 per position)
"""
import sys
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class PDEContextDataset(Dataset):
    """
    Dynamic context-sampling dataset for (a4, a2) → solutions mapping.

    Args:
        data:           list of dicts with 'params'=[a4,a2] and 'solutions'=[arr,...]
        indices:        LongTensor — indices into data to use (train/val/test split)
        max_context_p:  max number of context blocks per sample
        seed:           int
        no_context_prob: probability of empty context (for ablation)
    """

    def __init__(self, data: list, indices: torch.Tensor,
                 max_context_p: int = 8, seed: int = 42,
                 no_context_prob: float = 0.0):
        self.data       = data
        self.indices    = (indices.tolist() if isinstance(indices, torch.Tensor)
                           else list(indices))
        self.max_context_p   = int(max_context_p)
        self.base_seed       = int(seed)
        self.epoch           = 0
        self.no_context_prob = float(no_context_prob)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item) -> dict:
        if isinstance(item, (tuple, list)):
            target_idx = int(item[0])
            repeat_id  = int(item[1]) if len(item) > 1 else 0
        else:
            target_idx = int(item)
            repeat_id  = 0

        rng = np.random.RandomState(
            self.base_seed
            + self.epoch * 1_000_003
            + target_idx * 97
            + repeat_id  * 1_009
        )

        pool = [x for x in self.indices if x != target_idx]
        use_no_context = (
            self.no_context_prob > 0.0 and rng.rand() < self.no_context_prob
        )
        if use_no_context or len(pool) == 0:
            ctx_idxs = []
        else:
            k = rng.randint(1, min(self.max_context_p, len(pool)) + 1)
            ctx_idxs = rng.choice(pool, size=k, replace=False).tolist()

        def _load(idx):
            d = self.data[idx]
            a4, a2 = float(d['params'][0]), float(d['params'][1])
            sols = torch.stack([
                torch.as_tensor(s, dtype=torch.float32)
                for s in d['solutions']
            ])   # [K, 1024]
            return (a4, a2), sols

        target_params, target_sols = _load(target_idx)
        ctx_params = []
        ctx_sols   = []
        for ci in ctx_idxs:
            cp, cs = _load(ci)
            ctx_params.append(cp)
            ctx_sols.append(cs)

        return {
            "target_idx":    target_idx,
            "target_params": target_params,   # (a4, a2)
            "target_sols":   target_sols,     # [K_t, 1024]
            "ctx_params":    ctx_params,      # list[(a4,a2)]
            "ctx_sols":      ctx_sols,        # list[[K_i, 1024]]
        }


class EpochShuffleSampler(Sampler):
    """DDP-aware epoch shuffle sampler."""

    def __init__(self, indices: torch.Tensor, seed: int = 42, repeat_factor: int = 1):
        self.indices = (indices.clone().cpu().long()
                        if isinstance(indices, torch.Tensor)
                        else torch.tensor(list(indices), dtype=torch.long))
        self.seed   = int(seed)
        self.epoch  = 0
        self.repeat_factor = max(1, int(repeat_factor))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        import torch.distributed as dist
        chunks = []
        for rep in range(self.repeat_factor):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 10_000 + rep)
            order = torch.randperm(len(self.indices), generator=g)
            chunks.extend([(int(self.indices[i].item()), rep) for i in order.tolist()])

        if dist.is_available() and dist.is_initialized():
            rank, ws = dist.get_rank(), dist.get_world_size()
            pad = (-len(chunks)) % ws
            if pad:
                chunks.extend(chunks[:pad])
            per_rank = len(chunks) // ws
            chunks = chunks[rank * per_rank: (rank + 1) * per_rank]

        return iter(chunks)

    def __len__(self):
        import torch.distributed as dist
        n = len(self.indices) * self.repeat_factor
        if dist.is_available() and dist.is_initialized():
            return math.ceil(n / dist.get_world_size())
        return n


class WeightedEpochSampler(Sampler):
    """
    Weighted sampler for k-class imbalance.
    num_samples controls total draws per epoch (default: len(indices)).
    """

    def __init__(self, indices: torch.Tensor, weights: torch.Tensor,
                 seed: int = 42, repeat_factor: int = 1,
                 num_samples: int = None):
        self.indices = (indices.clone().cpu().long()
                        if isinstance(indices, torch.Tensor)
                        else torch.tensor(list(indices), dtype=torch.long))
        weights = torch.as_tensor(weights, dtype=torch.float64).cpu()
        assert len(weights) == len(self.indices)
        self.weights       = weights / weights.sum()
        self.seed          = int(seed)
        self.epoch         = 0
        self.repeat_factor = max(1, int(repeat_factor))
        self.num_samples   = int(num_samples) if num_samples is not None else len(self.indices)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        import torch.distributed as dist
        N = self.num_samples
        pairs = []
        for rep in range(self.repeat_factor):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 10_000 + rep)
            drawn = torch.multinomial(self.weights, num_samples=N,
                                      replacement=True, generator=g)
            for i, pos in enumerate(drawn.tolist()):
                pairs.append((int(self.indices[pos].item()), rep * N + i))

        g2 = torch.Generator()
        g2.manual_seed(self.seed + self.epoch * 10_000 + 99999)
        perm = torch.randperm(len(pairs), generator=g2).tolist()
        pairs = [pairs[i] for i in perm]

        if dist.is_available() and dist.is_initialized():
            rank, ws = dist.get_rank(), dist.get_world_size()
            # Pad to divisible by ws so all ranks get identical counts.
            # (ceil-slicing gives rank ws-1 fewer samples → different batch
            #  counts → mismatched manual all_reduce calls → NCCL hang.)
            pad = (-len(pairs)) % ws
            if pad:
                pairs.extend(pairs[:pad])
            per_rank = len(pairs) // ws
            pairs    = pairs[rank * per_rank: (rank + 1) * per_rank]

        return iter(pairs)

    def __len__(self):
        import torch.distributed as dist
        n = self.num_samples * self.repeat_factor
        if dist.is_available() and dist.is_initialized():
            return math.ceil(n / dist.get_world_size())
        return n


def make_collate_fn(seq_builder, max_seq_len: int = 256):
    """
    Factory returning a collate_fn for 1D_a2a4 batches.

    Batch dict keys:
        inputs_embeds:          [B, L_max, D]        bfloat16
        attention_mask:         [B, L_max]            long
        loss_mask:              [B, L_max]            bool
        cls_targets:            [B, L_max]            long
        reg_targets:            [B, L_max, latent]    float32
        params_targets:         [B, L_max, 2]         float32  (a4, a2 per position)
        actual_lens:            [B]                   long
        target_block_offsets:   [B]                   long
        target_sols_padded:     [B, K_max, 1024]      float32
        target_k_counts:        [B]                   long
        target_params:          [B, 2]                float32  (a4, a2 per sample)
    """

    def collate_fn(batch: list) -> dict:
        samples = []
        for item in batch:
            built = seq_builder.build(
                context_params=item["ctx_params"],
                context_sols=item["ctx_sols"],
                target_params=item["target_params"],
                target_sols=item["target_sols"],
            )
            samples.append(built)

        actual_lens = [s["actual_len"] for s in samples]
        L_max   = max(actual_lens)
        B       = len(samples)
        hid     = samples[0]["inputs_embeds"].shape[-1]
        lat     = samples[0]["reg_targets"].shape[-1]

        inputs_embeds  = torch.zeros(B, L_max, hid,  dtype=torch.bfloat16)
        attention_mask = torch.zeros(B, L_max,       dtype=torch.long)
        loss_mask      = torch.zeros(B, L_max,       dtype=torch.bool)
        cls_targets    = torch.full((B, L_max), -100, dtype=torch.long)
        reg_targets    = torch.zeros(B, L_max, lat,  dtype=torch.float32)
        params_targets = torch.zeros(B, L_max, 2,    dtype=torch.float32)

        for i, (s, L) in enumerate(zip(samples, actual_lens)):
            inputs_embeds[i, :L]   = s["inputs_embeds"]
            attention_mask[i, :L]  = 1
            loss_mask[i, :L]       = s["loss_mask"]
            cls_targets[i, :L]     = s["cls_targets"]
            reg_targets[i, :L]     = s["reg_targets"]
            params_targets[i, :L]  = s["params_targets"]   # [L, 2]

        k_counts = [s["canonical_target_sols"].shape[0] for s in samples]
        K_max    = max(k_counts) if k_counts else 1

        target_sols_padded   = torch.zeros(B, K_max, 1024, dtype=torch.float32)
        target_k_counts      = torch.tensor(k_counts, dtype=torch.long)
        target_params_tensor = torch.tensor(
            [list(s["target_params"]) for s in samples], dtype=torch.float32
        )   # [B, 2]
        target_block_offsets = torch.tensor(
            [s["target_block_offset"] for s in samples], dtype=torch.long
        )

        for i, s in enumerate(samples):
            K_t = k_counts[i]
            target_sols_padded[i, :K_t] = s["canonical_target_sols"]

        return {
            "inputs_embeds":        inputs_embeds,
            "attention_mask":       attention_mask,
            "loss_mask":            loss_mask,
            "cls_targets":          cls_targets,
            "reg_targets":          reg_targets,
            "params_targets":       params_targets,
            "actual_lens":          torch.tensor(actual_lens, dtype=torch.long),
            "target_block_offsets": target_block_offsets,
            "target_sols_padded":   target_sols_padded,
            "target_k_counts":      target_k_counts,
            "target_params":        target_params_tensor,
        }

    return collate_fn


def build_2d_causal_mask(actual_lens: torch.Tensor, L_max: int,
                          device: torch.device) -> torch.Tensor:
    """[B, 1, L_max, L_max] bool, True=attend."""
    B    = actual_lens.shape[0]
    i_idx = torch.arange(L_max, device=device).unsqueeze(1)
    j_idx = torch.arange(L_max, device=device).unsqueeze(0)
    causal  = (j_idx <= i_idx)
    not_pad = (j_idx < actual_lens.to(device).unsqueeze(1))
    return causal.unsqueeze(0).unsqueeze(1) & not_pad.unsqueeze(1).unsqueeze(2)
