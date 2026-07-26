"""1D_p row2 解数据: p=20 处 GT 7 分支 + 模型直接(M1)恢复的解 -> sols_row2.pt"""
import os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "data_gen", "cbmfem"))
from gen_gt_extension import march, ZERO_NORM
from model.traditional_refine import refine_with_newton_trace

P = 20.0
COVER, DEDUP, REFINE_OK = 1e-2, 1e-3, 1e-6


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def dedup(s):
    o = []
    for u in s:
        if u is not None and not any(rl2(u, v) < DEDUP for v in o):
            o.append(u)
    return o


def main():
    S = np.load(os.path.join(HERE, "..", "data_gen", "cbmfem", "initial_S2.npy"))
    true18 = [np.real(S[:, b]).astype(np.float64) for b in range(S.shape[1])
              if np.linalg.norm(np.real(S[:, b])) > ZERO_NORM]
    far = torch.load(os.path.join(HERE, "far_direct.pt"), map_location="cpu", weights_only=False)
    gen = far[P].numpy().astype(np.float64)                                 # 模型在 p=20 直接输出 (M1)
    gen18 = torch.load(os.path.join(HERE, "p18seed_result.pt"),
                       map_location="cpu", weights_only=False)["gen18"]      # 模型 p=18 输出 (M2 种子)
    gt = dedup([march(u, 18.0, P, dp0=0.25) for u in true18])

    def refine_one(u):
        ru, ok, _, tr = refine_with_newton_trace(torch.tensor(u), P, tol=1e-9, max_iter=30)
        res = float(tr[-1]["residual"]) if tr else float("inf")
        return np.asarray(ru.detach().cpu(), np.float64).reshape(-1) if (ok and res < REFINE_OK) else None

    def cov(sols):
        return sum(1 for g in gt if any(rl2(d, g) < COVER for d in sols))
    m1 = dedup([refine_one(u) for u in gen])                                # 直接输出 refine
    m2 = dedup([refine_one(u) for u in gen18])                              # p=18 种子单步 refine 到 p
    x = np.linspace(0, 1, S.shape[0])
    torch.save({"x": x, "gt": np.stack(gt),
                "m1": np.stack(m1) if m1 else np.zeros((0, S.shape[0])),
                "m2": np.stack(m2) if m2 else np.zeros((0, S.shape[0])),
                "p": P, "cover_m1": cov(m1), "cover_m2": cov(m2)},
               os.path.join(HERE, "sols_row2.pt"))
    print(f"1D_p p={P}: GT={len(gt)}  M1 {len(m1)} (cov {cov(m1)})  M2 {len(m2)} (cov {cov(m2)})")


if __name__ == "__main__":
    main()
