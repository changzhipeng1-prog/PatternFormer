"""2D 三方法外延 coverage:
  M1 direct       : refine 模型在 s 的直接输出
  M2 seed+jump    : 从 s=1600 模型输出单步 Newton 到 s
  M3 seed+cont    : 从 s=1600 模型输出分步延拓到 s (Delta s = 100)
GT = 从真实 s=1600 解(lookup)延拓到 s 的存活分支。
输出 -> three_methods_2d.pt + 打印表。
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.newton_refine_2d import newton_refine_2d

C = Config()
COVER, DEDUP, REFINE_OK = 5e-2, 1e-2, 1e-6
DS_CONT, DS_GT = 100.0, 50.0

F = torch.load(os.path.join(HERE, "far_direct_2d.pt"), map_location="cpu", weights_only=False)
COORD, ELEM, FREE = F["coord"].float(), F["elem"].long(), F["free_nodes"].long()
S_SEED, S_POINTS = F["S_SEED"], F["S_POINTS"]
gen1600 = F["gen"][float(S_SEED)].numpy().astype(np.float64)

lk = torch.load(C.data_lookup_path, weights_only=False)
pv = np.array(lk["p_values"]); ib = int(np.argmin(np.abs(pv - S_SEED)))
gt1600 = np.asarray(lk["solutions_by_p"][ib], np.float64)          # 真实 s=1600 解


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


def solve_at(u, s):
    ru, ok, nit, hist = newton_refine_2d(torch.as_tensor(u, dtype=torch.float32), float(s),
                                         COORD, ELEM, FREE, tol=1e-9, max_iter=60)
    if not ok:
        return None
    return np.asarray(ru.detach().cpu(), np.float64).reshape(-1)


def march(u0, s_from, s_to, ds0):
    s, u, ds = s_from, np.asarray(u0, np.float64).copy(), ds0
    while s < s_to - 1e-9:
        step = min(ds, s_to - s)
        un = solve_at(u, s + step)
        if un is None:
            ds *= 0.5
            if ds < 1.0:
                return None
            continue
        u, s = un, s + step
        ds = min(ds0, ds * 1.5)
    return u


def main():
    rows = []
    print(f"{'s':>6} | {'exist':>5} | {'M1 direct':>9} | {'M2 jump':>7} | {'M3 cont':>7}")
    print("-" * 48)
    for s in S_POINTS:
        gt = dedup([march(g, S_SEED, s, DS_GT) for g in gt1600])
        m1 = dedup([solve_at(u, s) for u in F["gen"][float(s)].numpy().astype(np.float64)])
        m2 = dedup([solve_at(u, s) for u in gen1600])
        m3 = dedup([march(u, S_SEED, s, DS_CONT) for u in gen1600])
        r = (s, len(gt), coverage(m1, gt), coverage(m2, gt), coverage(m3, gt))
        rows.append(r)
        print(f"{s:>6.0f} | {r[1]:>5} | {r[2]:>9} | {r[3]:>7} | {r[4]:>7}")
    torch.save(rows, os.path.join(HERE, "three_methods_2d.pt"))
    print("\nsaved three_methods_2d.pt")


if __name__ == "__main__":
    main()
