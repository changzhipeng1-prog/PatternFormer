"""a2a4 row2 解数据: 深势阱 cell (a2≈-20.9, a4=0.55) 处 GT + 模型直接(M1)恢复 -> sols_row2.pt"""
import os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_fdm import newton_refine, residual_l2

N = 1024
H = 1.0 / N
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
    G = torch.load(os.path.join(HERE, "grid_direct.pt"), map_location="cpu", weights_only=False)
    GT = torch.load(os.path.join(HERE, "grid_gt_cont.pt"), map_location="cpu", weights_only=False)
    A2 = np.array(G["a2"]); A4 = np.array(G["a4"])
    a2, a4 = float(A2[4]), float(A4[4])                 # 外延、M1 全覆盖: a2≈-16.8, a4=0.55 (k=5)
    gt = np.asarray(GT["gt"][(a2, a4)], np.float64)
    gen = G["gen"][(a2, a4)].numpy().astype(np.float64)

    def refine_one(u):
        ru, ok, _ = newton_refine(u, a4, a2, h=H, max_iter=50, abs_tol=1e-9)
        ru = np.asarray(ru, np.float64).reshape(-1)
        return ru if (ok and residual_l2(ru, a4, a2, H) < REFINE_OK and np.linalg.norm(ru) > 1e-6) else None
    m1 = dedup([refine_one(u) for u in gen])
    cover = sum(1 for g in gt if any(rl2(d, g) < COVER for d in m1))
    x = np.linspace(0, 1, N)
    torch.save({"x": x, "gt": gt, "m1": np.stack(m1) if m1 else np.zeros((0, N)),
                "a2": a2, "a4": a4, "cover": cover}, os.path.join(HERE, "sols_row2.pt"))
    print(f"a2a4 (a2={a2:.2f},a4={a4:.2f}): GT={gt.shape[0]} M1 distinct={len(m1)} cover={cover}/{gt.shape[0]}")


if __name__ == "__main__":
    main()
