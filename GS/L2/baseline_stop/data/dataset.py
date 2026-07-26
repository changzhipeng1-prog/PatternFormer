"""Dataset / DataLoader for Gray-Scott 2D (image solutions, 2D parameter).

Adapted from 2D_multisolution/qwen/data/dataset.py:
  - p is a [2] vector (rho,mu); solutions are [K_i,2,128,128]
  - collate pads target_sols to [B,K_max,2,128,128] and p_targets to [B,L,2]
Samplers and the 2D causal-mask builder are carried over unchanged.
"""
import sys, os, math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class PDEContextDataset(Dataset):
    def __init__(self, lookup: dict, candidate_p_idx, max_context_p: int = 4,
                 seed: int = 42, no_context_prob: float = 0.0,
                 context_mode: str = "random", context_pool_idx=None):
        self.p_values = lookup["p_values"]              # [P,2]
        self.solutions_by_p = lookup["solutions_by_p"]  # list[[K_i,2,128,128]]
        self.candidate_p_idx = (candidate_p_idx.tolist()
                                if isinstance(candidate_p_idx, torch.Tensor)
                                else list(candidate_p_idx))
        self.max_context_p = int(max_context_p)
        self.base_seed = int(seed)
        self.epoch = 0
        self.no_context_prob = float(no_context_prob)
        # context_mode: "random" (any candidate) or "local" (nearest in (rho,mu)).
        # Local context exploits that nearby params share solution structure
        # (empirically ~4x more similar than random) -> a far stronger prior.
        self.context_mode = context_mode
        # pool to draw context FROM (the "known library"); defaults to candidates.
        # For local-mode eval pass the TRAIN params here so a test target draws
        # context from solved (train) params, matching real deployment.
        if context_pool_idx is None:
            pool_idx = list(self.candidate_p_idx)
        elif isinstance(context_pool_idx, torch.Tensor):
            pool_idx = context_pool_idx.tolist()
        else:
            pool_idx = list(context_pool_idx)
        self.context_pool_idx = pool_idx
        pv = self.p_values[torch.as_tensor(pool_idx, dtype=torch.long)].float()  # [M,2]
        self._pool_p = pv
        self._p_scale = pv.std(0).clamp_min(1e-8)       # per-dim scale for distance

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.candidate_p_idx)

    def _nearest_pool(self, target_p_idx):
        """context_pool indices sorted by normalized (rho,mu) distance, target excluded."""
        tp = self.p_values[target_p_idx].float()
        d = ((self._pool_p - tp) / self._p_scale).pow(2).sum(1).sqrt()
        order = torch.argsort(d).tolist()
        return [self.context_pool_idx[i] for i in order
                if self.context_pool_idx[i] != target_p_idx]

    def __getitem__(self, target_idx) -> dict:
        if isinstance(target_idx, (tuple, list)):
            target_p_idx = int(target_idx[0])
            repeat_id = int(target_idx[1]) if len(target_idx) > 1 else 0
        else:
            target_p_idx = int(target_idx); repeat_id = 0

        rng = np.random.RandomState(self.base_seed + self.epoch * 1_000_003
                                    + target_p_idx * 97 + repeat_id * 1_009)
        use_no_context = self.no_context_prob > 0.0 and rng.rand() < self.no_context_prob
        if self.context_mode == "local":
            # nearest 2*max_context_p params, then sample k of them (locality + variety)
            near = self._nearest_pool(target_p_idx)[: max(1, 2 * self.max_context_p)]
            pool = near
        else:
            pool = [x for x in self.context_pool_idx if x != target_p_idx]
        if use_no_context or len(pool) == 0:
            ctx_idxs = []
        else:
            k = rng.randint(1, min(self.max_context_p, len(pool)) + 1)
            ctx_idxs = rng.choice(pool, size=k, replace=False).tolist()

        return {
            "target_p_idx": target_p_idx,
            "target_p_val": self.p_values[target_p_idx],                 # [2]
            "target_sols": self.solutions_by_p[target_p_idx],           # [K_t,2,128,128]
            "context_p_idx": ctx_idxs,
            "context_p_vals": [self.p_values[i] for i in ctx_idxs],     # list[[2]]
            "context_sols": [self.solutions_by_p[i] for i in ctx_idxs],
        }


class EpochShuffleSampler(Sampler):
    def __init__(self, candidate_p_idx, seed: int = 42, repeat_factor: int = 1):
        self.indices = (candidate_p_idx.clone().cpu().long()
                        if isinstance(candidate_p_idx, torch.Tensor)
                        else torch.tensor(list(candidate_p_idx), dtype=torch.long))
        self.seed = int(seed); self.epoch = 0
        self.repeat_factor = max(1, int(repeat_factor))

    def set_epoch(self, epoch): self.epoch = int(epoch)

    def __iter__(self):
        import torch.distributed as dist
        chunks = []
        for rep in range(self.repeat_factor):
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch * 10_000 + rep)
            order = torch.randperm(len(self.indices), generator=g)
            chunks.extend([(int(self.indices[i].item()), rep) for i in order.tolist()])
        shuffled = chunks
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            pad = (-len(shuffled)) % world
            if pad: shuffled.extend(shuffled[:pad])
            per = len(shuffled) // world
            shuffled = shuffled[rank * per: rank * per + per]
        return iter(shuffled)

    def __len__(self):
        import torch.distributed as dist
        n = len(self.indices) * self.repeat_factor
        if dist.is_available() and dist.is_initialized():
            return math.ceil(n / dist.get_world_size())
        return n


def make_collate_fn(seq_builder, in_ch=2, img=128):
    def collate_fn(batch: list) -> dict:
        samples = [seq_builder.build(
            context_p_vals=item["context_p_vals"], context_sols=item["context_sols"],
            target_p_val=item["target_p_val"], target_sols=item["target_sols"]
        ) for item in batch]

        actual_lens = [s["actual_len"] for s in samples]
        L = max(actual_lens); B = len(samples)
        D = samples[0]["inputs_embeds"].shape[-1]
        lat = samples[0]["reg_targets"].shape[-1]
        pdim = samples[0]["p_targets"].shape[-1]

        inputs_embeds = torch.zeros(B, L, D, dtype=torch.bfloat16)
        attention_mask = torch.zeros(B, L, dtype=torch.long)
        loss_mask = torch.zeros(B, L, dtype=torch.bool)
        cls_targets = torch.full((B, L), -100, dtype=torch.long)
        reg_targets = torch.zeros(B, L, lat, dtype=torch.float32)
        p_targets = torch.zeros(B, L, pdim, dtype=torch.float32)
        for i, (s, l) in enumerate(zip(samples, actual_lens)):
            inputs_embeds[i, :l] = s["inputs_embeds"]
            attention_mask[i, :l] = 1
            loss_mask[i, :l] = s["loss_mask"]
            cls_targets[i, :l] = s["cls_targets"]
            reg_targets[i, :l] = s["reg_targets"]
            p_targets[i, :l] = s["p_targets"]

        k_counts = [s["canonical_target_sols"].shape[0] for s in samples]
        K_max = max(k_counts) if k_counts else 1
        target_sols_padded = torch.zeros(B, K_max, in_ch, img, img, dtype=torch.float32)
        for i, s in enumerate(samples):
            target_sols_padded[i, :k_counts[i]] = s["canonical_target_sols"]
        target_p_vals = torch.stack([s["target_p_val"] for s in samples], 0)   # [B,2]

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
            "p_targets": p_targets,
            "actual_lens": torch.tensor(actual_lens, dtype=torch.long),
            "target_block_offsets": torch.tensor([s["target_block_offset"] for s in samples], dtype=torch.long),
            "target_sols_padded": target_sols_padded,                          # [B,K_max,2,128,128]
            "target_k_counts": torch.tensor(k_counts, dtype=torch.long),
            "target_p_vals": target_p_vals,                                    # [B,2]
        }
    return collate_fn


def build_2d_causal_mask(actual_lens: torch.Tensor, L_max: int, device) -> torch.Tensor:
    i_idx = torch.arange(L_max, device=device).unsqueeze(1)
    j_idx = torch.arange(L_max, device=device).unsqueeze(0)
    causal = (j_idx <= i_idx)
    lens = actual_lens.to(device).unsqueeze(1)
    not_pad = (j_idx < lens)
    return causal.unsqueeze(0).unsqueeze(1) & not_pad.unsqueeze(1).unsqueeze(2)
