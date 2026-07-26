"""Train GSPDEModel (conv-AE + Qwen+LoRA) — Gray-Scott 2D multi-solution generator.

Phase 2 (the AE is pretrained separately via train_ae.py).
Single GPU:  python train.py --smoke 5
DDP:         torchrun --nproc_per_node=6 train.py
Two optimizer groups (projector lr / LoRA lr), grad-accum, plateau on val MSE.
"""
import os, sys, argparse, time, json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from model.v3_model import GSPDEModel
from data.dataset import PDEContextDataset, EpochShuffleSampler, make_collate_fn
from data.sequence_builder import SequenceBuilder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr_projector", type=float, default=None)
    p.add_argument("--lr_lora", type=float, default=None)
    p.add_argument("--unet_checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    # with-context (Phase 2) and no-context (Phase 3) are SEPARATE runs, never mixed:
    p.add_argument("--no_context_prob", type=float, default=0.0,
                   help="0.0 = pure with-context (Phase 2); 1.0 = pure no-context (Phase 3)")
    p.add_argument("--val_no_context_prob", type=float, default=None,
                   help="defaults to --no_context_prob (eval in the same mode as train)")
    p.add_argument("--resume_from", type=str, default=None,
                   help="checkpoint dir to resume (Phase 3 inits from Phase 2 best_model)")
    p.add_argument("--ss_prob", type=float, default=None,
                   help="scheduled-sampling max prob (0=teacher forcing); fights count collapse")
    p.add_argument("--ss_warmup_start", type=int, default=None)
    p.add_argument("--ss_warmup_end", type=int, default=None)
    # data-path overrides (e.g. point at the densified *_big dataset)
    p.add_argument("--data_lookup", type=str, default=None)
    p.add_argument("--train_idx", type=str, default=None)
    p.add_argument("--val_idx", type=str, default=None)
    p.add_argument("--norm_stats", type=str, default=None)
    p.add_argument("--context_mode", type=str, default=None, choices=["random", "local"],
                   help="local = nearest-(rho,mu) context (stronger prior than random)")
    p.add_argument("--smoke", type=int, default=0, help="if >0, run this many train steps then stop")
    return p.parse_args()


def setup_ddp():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl")
        lr = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr)
        return lr, True
    return 0, False


def is_rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def main():
    args = parse_args()
    local_rank, ddp = setup_ddp()
    C = Config()
    bs = args.batch_size or C.batch_size
    epochs = args.num_epochs or C.num_epochs
    ga = args.grad_accum or C.gradient_accumulation_steps
    if args.lr_projector: C.lr_projector = args.lr_projector
    if args.lr_lora: C.lr_lora = args.lr_lora
    if args.unet_checkpoint: C.unet_checkpoint_path = args.unet_checkpoint
    if args.output_dir: C.output_dir = args.output_dir; os.makedirs(C.output_dir, exist_ok=True)
    if args.ss_prob is not None: C.ss_prob = args.ss_prob
    if args.ss_warmup_start is not None: C.ss_warmup_start = args.ss_warmup_start
    if args.ss_warmup_end is not None: C.ss_warmup_end = args.ss_warmup_end
    if args.data_lookup: C.data_lookup_path = args.data_lookup
    if args.train_idx: C.train_p_idx_path = args.train_idx
    if args.val_idx: C.val_p_idx_path = args.val_idx
    if args.norm_stats: C.norm_stats_path = args.norm_stats
    if args.context_mode: C.context_mode = args.context_mode

    lookup = torch.load(C.data_lookup_path, weights_only=False)
    train_idx = torch.load(C.train_p_idx_path); val_idx = torch.load(C.val_p_idx_path)
    ns = torch.load(C.norm_stats_path)
    p_mean, p_std = ns["p_mean"], ns["p_std"]

    if not os.path.exists(C.unet_checkpoint_path):
        raise FileNotFoundError(f"AE checkpoint missing: {C.unet_checkpoint_path} (run train_ae.py first)")

    nc_prob = max(0.0, min(1.0, args.no_context_prob))
    val_nc_prob = nc_prob if args.val_no_context_prob is None else max(0.0, min(1.0, args.val_no_context_prob))

    if args.resume_from:
        if is_rank0():
            print(f"resuming from {args.resume_from}  (no_context_prob={nc_prob})")
        model = GSPDEModel.from_pretrained(args.resume_from, C, local_rank=local_rank,
                                           p_mean=p_mean, p_std=p_std)
    else:
        if is_rank0():
            print(f"fresh model  (no_context_prob={nc_prob})")
        model = GSPDEModel(C, unet_checkpoint=C.unet_checkpoint_path, local_rank=local_rank,
                           p_mean=p_mean, p_std=p_std)
    dev = model.component_device

    seq_builder = SequenceBuilder(model.autoencoder, model.input_proj, model.special_tok,
                                  C.latent_dim, C.qwen_hidden_dim, param_dim=C.param_dim,
                                  max_seq_len=C.max_seq_len, max_solutions_per_p=C.max_solutions_per_p,
                                  device=dev, p_mean=p_mean, p_std=p_std, fixed_k=C.fixed_k)
    collate = make_collate_fn(seq_builder, in_ch=C.in_ch, img=C.img_size)

    # local-mode context library is always the TRAIN set (the "known solved" params):
    #   train target -> nearest OTHER train params; val target -> nearest train params.
    train_ds = PDEContextDataset(lookup, train_idx, max_context_p=C.max_context_p,
                                 seed=C.sampler_seed, no_context_prob=nc_prob,
                                 context_mode=C.context_mode, context_pool_idx=train_idx)
    val_ds = PDEContextDataset(lookup, val_idx, max_context_p=C.max_context_p,
                               seed=C.sampler_seed + 1, no_context_prob=val_nc_prob,
                               context_mode=C.context_mode, context_pool_idx=train_idx)
    train_sampler = EpochShuffleSampler(train_idx, seed=C.sampler_seed,
                                        repeat_factor=C.target_repeat_per_epoch)
    val_sampler = EpochShuffleSampler(val_idx, seed=C.sampler_seed + 1, repeat_factor=1)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_sampler, collate_fn=collate, num_workers=0)

    proj_params = (list(model.special_tok.parameters()) + list(model.input_proj.parameters())
                   + list(model.dual_head.parameters()))   # B2: output_proj removed
    lora_params = [p for p in model.qwen_model.parameters() if p.requires_grad]
    if is_rank0():
        print(f"proj params {sum(p.numel() for p in proj_params):,}  "
              f"lora params {sum(p.numel() for p in lora_params):,}")
    opt_proj = optim.AdamW(proj_params, lr=C.lr_projector, weight_decay=C.weight_decay)
    opt_lora = optim.AdamW(lora_params, lr=C.lr_lora, weight_decay=C.weight_decay)
    sch_proj = optim.lr_scheduler.ReduceLROnPlateau(opt_proj, "min", factor=C.lr_plateau_factor,
                                                    patience=C.lr_plateau_patience,
                                                    min_lr=C.lr_projector * C.lr_min_ratio)
    sch_lora = optim.lr_scheduler.ReduceLROnPlateau(opt_lora, "min", factor=C.lr_plateau_factor,
                                                    patience=C.lr_plateau_patience,
                                                    min_lr=C.lr_lora * C.lr_min_ratio)

    best = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch); train_sampler.set_epoch(epoch)
        model.train(); model.autoencoder.eval()
        rt = rm = rtp = rl = rp = 0.0; nb = 0
        opt_proj.zero_grad(); opt_lora.zero_grad()
        for bi, batch in enumerate(train_loader):
            out = model(batch, current_epoch=epoch)
            (out["total_loss"] / ga).backward()
            if (bi + 1) % ga == 0:
                if ddp:
                    ws = dist.get_world_size()
                    for p in proj_params + lora_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad.div_(ws)
                torch.nn.utils.clip_grad_norm_(proj_params, C.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(lora_params, C.max_grad_norm)
                opt_proj.step(); opt_lora.step(); opt_proj.zero_grad(); opt_lora.zero_grad()
            rt += out["total_loss"].item(); rm += out["mse_loss"].item(); rtp += out["tail_pde"].item()
            rl += out["latent_loss"].item(); rp += out["pde_loss"].item(); nb += 1
            if args.smoke and bi + 1 >= args.smoke:
                if is_rank0():
                    print(f"[smoke] {bi+1} steps OK  total={rt/nb:.3e} mse={rm/nb:.3e} "
                          f"latent={rl/nb:.3e} pde={rp/nb:.3e} tail_pde={rtp/nb:.3e} "
                          f"lam_pde={out['lambda_pde']:.1e} lam_tail={out.get('lambda_pde_tail',0.0):.1e}")
                return
            if is_rank0() and bi % C.logging_steps == 0:
                print(f"  ep{epoch} step{bi}/{len(train_loader)} total={out['total_loss'].item():.3e} "
                      f"mse={out['mse_loss'].item():.3e} latent={out['latent_loss'].item():.3e} "
                      f"pde={out['pde_loss'].item():.3e} tpde={out['tail_pde'].item():.3e} "
                      f"ss={out.get('ss_prob', 0.0):.2f}", flush=True)
        n = max(nb, 1)

        model.eval(); vm = 0.0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                vm += model(batch, current_epoch=epoch)["mse_loss"].item(); nv += 1
        vm /= max(nv, 1)
        if ddp:
            vt = torch.tensor(vm, device=dev); dist.all_reduce(vt, op=dist.ReduceOp.AVG); vm = vt.item()
        sch_proj.step(vm); sch_lora.step(vm)
        if is_rank0():
            print(f"Epoch {epoch}/{epochs} | train total={rt/n:.4e} mse={rm/n:.4e} tpde={rtp/n:.4e} | "
                  f"val_mse={vm:.4e} | lr_p={opt_proj.param_groups[0]['lr']:.2e} "
                  f"lr_l={opt_lora.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s", flush=True)
            if vm < best:
                best = vm; model.save_pretrained(os.path.join(C.output_dir, "best_model"))
                print(f"  -> best (val_mse={best:.4e})")
            if epoch % 20 == 0:
                model.save_pretrained(os.path.join(C.output_dir, f"epoch_{epoch}"))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
