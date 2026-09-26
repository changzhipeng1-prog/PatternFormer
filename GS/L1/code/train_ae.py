"""Pretrain the conv autoencoder on Gray-Scott solutions (single GPU).

Loss: per-channel normalized MSE (balances A and S). Metric: physical rel_l2.
D4 augmentation (rot/flip) exploits the unit-square symmetry. Saves the best
checkpoint (with norm buffers baked in) to checkpoints/autoencoder2d.pt.
"""
import argparse, os, time
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model.autoencoder2d import SolutionAutoencoder2D

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.normpath(f"{HERE}/../data")   # dataset lives in GS/L1/data/ (code/ is this dir)


def load_split(idx_file, lookup):
    idx = torch.load(f"{D}/{idx_file}").tolist()
    return torch.cat([lookup["solutions_by_p"][i] for i in idx], 0)   # (n,2,128,128)


def d4(x, g):
    """random D4 transform applied to whole batch (x: B,2,128,128)."""
    k = int(torch.randint(0, 4, (1,), generator=g).item())
    x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1, generator=g).item() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


def rel_l2(pred, tgt):
    # per-channel ||.||2 ratio, averaged over channels and batch
    num = (pred - tgt).flatten(2).norm(dim=2)        # (B,2)
    den = tgt.flatten(2).norm(dim=2).clamp_min(1e-8)
    return (num / den).mean().item()


def main(a):
    dev = "cuda"
    lookup = torch.load(f"{D}/gs_lookup_big.pt")
    stats = torch.load(f"{D}/norm_stats_big.pt")
    Xtr = load_split("train_p_idx_big.pt", lookup).to(dev)
    Xva = load_split("val_p_idx_big.pt", lookup).to(dev)
    print(f"train {Xtr.shape} val {Xva.shape}")

    inv_std = (1.0 / stats["std"].view(1, 2, 1, 1)).to(dev)        # for normalized MSE
    model = SolutionAutoencoder2D(latent_dim=a.latent, mean=stats["mean"], std=stats["std"]).to(dev)
    print(f"AE params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.02)
    g = torch.Generator().manual_seed(0)

    ntr = Xtr.shape[0]
    best = 1e9
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(ntr, device=dev)
        tot = 0.0
        for s in range(0, ntr, a.bs):
            xb = Xtr[perm[s:s + a.bs]]
            xb = d4(xb, g)
            z = model(xb, "encode")
            rec = model(z, "decode")
            loss = (((rec - xb) * inv_std) ** 2).mean()      # normalized MSE
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.shape[0]
        sched.step()

        if ep % a.eval_every == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                rv = []
                for s in range(0, Xva.shape[0], a.bs):
                    xb = Xva[s:s + a.bs]
                    rec = model(model(xb, "encode"), "decode")
                    rv.append(rel_l2(rec, xb))
                vr = sum(rv) / len(rv)
            print(f"ep {ep:4d}  train_nmse {tot/ntr:.4e}  val_relL2 {vr:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
            if vr < best:
                best = vr
                os.makedirs(a.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(a.output_dir, "autoencoder2d.pt"))
    print(f"best val rel_l2 = {best:.4f} -> {os.path.join(a.output_dir, 'autoencoder2d.pt')}")

    # reconstruction viz on a few val solutions
    model.load_state_dict(torch.load(os.path.join(a.output_dir, "autoencoder2d.pt")))
    model.eval()
    with torch.no_grad():
        xb = Xva[torch.randperm(Xva.shape[0], generator=torch.Generator().manual_seed(1))[:6]]
        rec = model(model(xb, "encode"), "decode")
    xb, rec = xb.cpu(), rec.cpu()
    fig, ax = plt.subplots(4, 6, figsize=(12, 8))
    for j in range(6):
        ax[0, j].imshow(xb[j, 0], cmap="viridis"); ax[0, j].set_title(f"A in", fontsize=8)
        ax[1, j].imshow(rec[j, 0], cmap="viridis"); ax[1, j].set_title("A rec", fontsize=8)
        ax[2, j].imshow(xb[j, 1], cmap="magma"); ax[2, j].set_title("S in", fontsize=8)
        ax[3, j].imshow(rec[j, 1], cmap="magma"); ax[3, j].set_title("S rec", fontsize=8)
        for r in range(4): ax[r, j].axis("off")
    fig.suptitle(f"AE reconstruction (val), rel_l2={best:.4f}")
    os.makedirs(os.path.join(a.output_dir, "figures"), exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(a.output_dir, "figures", "ae_recon.png"), dpi=110)
    print("wrote", os.path.join(a.output_dir, "figures", "ae_recon.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=10)
    main(p.parse_args())
