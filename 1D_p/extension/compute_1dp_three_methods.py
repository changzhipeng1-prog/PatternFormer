"""算 1D_p 三方法外延 coverage 并存数据(供 1x3 总图用)-> three_methods_1dp.pt"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "data_gen", "cbmfem"))
from gen_gt_extension import march, ZERO_NORM
from model.traditional_refine import refine_with_newton_trace

P_POINTS = [18.01, 18.05, 18.1, 18.5, 19.0, 20.0]
DP_CONT, DP_GT = 1.0, 0.25
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def dedup(sols):
    out = []
    for u in sols:
        if u is not None and not any(rl2(u, v) < DEDUP for v in out):
            out.append(u)
    return out


def coverage(distinct, gt):
    return sum(1 for g in gt if any(rl2(d, g) < COVER for d in distinct))


def snapshots(u0, checkpoints, dp):
    res, u, p = {}, np.asarray(u0, np.float64).copy(), 18.0
    for cp in sorted(checkpoints):
        u2 = march(u, p, cp, dp0=dp)
        if u2 is None:
            break
        res[cp] = u2.copy(); u, p = u2, cp
    return res


def refine_one(u, p):
    ru, ok, _, tr = refine_with_newton_trace(torch.tensor(u), float(p), tol=1e-9, max_iter=30)
    res = float(tr[-1]["residual"]) if tr else float("inf")
    return (np.asarray(ru.detach().cpu(), np.float64).reshape(-1) if (ok and res < REFINE_OK) else None)


def main():
    S = np.load(os.path.join(HERE, "..", "data_gen", "cbmfem", "initial_S2.npy"))
    true18 = [np.real(S[:, b]).astype(np.float64) for b in range(S.shape[1])
              if np.linalg.norm(np.real(S[:, b])) > ZERO_NORM]
    gen18 = torch.load(os.path.join(HERE, "p18seed_result.pt"),
                       map_location="cpu", weights_only=False)["gen18"]
    direct = {float(r["p"]): r["generated"].numpy().astype(np.float64)
              for r in torch.load(os.path.join(HERE, "generated_extension.pt"),
                                  map_location="cpu", weights_only=False)}
    direct.update({k: v.numpy().astype(np.float64)
                   for k, v in torch.load(os.path.join(HERE, "far_direct.pt"),
                                          map_location="cpu", weights_only=False).items()})

    gt_snap = [snapshots(u, P_POINTS, DP_GT) for u in true18]
    m3_snap = [snapshots(u, P_POINTS, DP_CONT) for u in gen18]
    n_exist, m1, m2, m3 = [], [], [], []
    for p in P_POINTS:
        gt = dedup([s[p] for s in gt_snap if p in s]); n_exist.append(len(gt))
        m1.append(coverage(dedup([refine_one(u, p) for u in direct[p]]), gt))
        m2.append(coverage(dedup([refine_one(u, p) for u in gen18]), gt))
        m3.append(coverage(dedup([s[p] for s in m3_snap if p in s]), gt))
        print(f"p={p:>6}: exist={n_exist[-1]} M1={m1[-1]} M2={m2[-1]} M3={m3[-1]}", flush=True)
    torch.save({"P": P_POINTS, "n_exist": n_exist, "m1": m1, "m2": m2, "m3": m3},
               os.path.join(HERE, "three_methods_1dp.pt"))
    print("saved three_methods_1dp.pt")


if __name__ == "__main__":
    main()
