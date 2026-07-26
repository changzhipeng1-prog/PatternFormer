"""a2a4 网格 ground truth —— 延拓式(与数据集构建一致)。

框内(训练矩形): GT = 数据集中最近参数点的解,在网格点参数处 Newton 精修(连通族,k<=5)。
框外: GT = 从已解的相邻 cell 的解出发,Newton 延拓到该 cell(BFS 向外传播),
       不引入穷举 battery,所以不会凭空多出训练集没有的高阶根。

输出 -> grid_gt_cont.pt  {"a2":[...], "a4":[...], "gt":{(a2,a4): ndarray[k,1024]}}
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_fdm import newton_refine, residual_l2

A2 = np.round(np.linspace(-25.0, -0.5, 13), 4)
A4 = np.round(np.linspace(0.05, 1.30, 11), 4)
N = 1024
H = 1.0 / N
MAXIT = 80
ABS_TOL = 1e-9
RES_OK = 1e-6
DEDUP = 1e-3
# 训练矩形
A2_LO, A2_HI = -14.934, -2.0
A4_LO, A4_HI = 0.301, 0.798


def in_box(a2, a4):
    return (A2_LO <= a2 <= A2_HI) and (A4_LO <= a4 <= A4_HI)


def refine(seed, a4, a2):
    ru, ok, _ = newton_refine(np.asarray(seed, np.float64).reshape(-1), a4, a2,
                              h=H, max_iter=MAXIT, abs_tol=ABS_TOL)
    ru = np.asarray(ru, np.float64).reshape(-1)
    if ok and residual_l2(ru, a4, a2, H) < RES_OK and np.linalg.norm(ru) > 1e-6:
        return ru
    return None


def dedup_into(sols):
    out = []
    for u in sols:
        if u is None:
            continue
        if not any(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-12) < DEDUP for v in out):
            out.append(u)
    return out


def main():
    # ---- 加载数据集解(作为框内 GT 与延拓种子) ----
    rec = []
    for sp in ("train", "val", "test"):
        rec += torch.load(os.path.join(HERE, "..", "data", f"bvp_region_{sp}.pt"),
                          map_location="cpu", weights_only=False)
    d_a4 = np.array([float(r["params"][0]) for r in rec])
    d_a2 = np.array([float(r["params"][1]) for r in rec])
    d_sols = [np.asarray(r["solutions"], np.float64) for r in rec]
    print(f"dataset records: {len(rec)}", flush=True)

    solved = {}                                                   # (i,j) -> list[array]
    # ---- 框内:用最近数据点的解精修到网格点参数 ----
    for i, a2 in enumerate(A2):
        for j, a4 in enumerate(A4):
            if not in_box(float(a2), float(a4)):
                continue
            t = np.argmin((d_a2 - a2) ** 2 + ((d_a4 - a4) * 100) ** 2)  # a4 尺度小,放大
            seeds = d_sols[t]
            if seeds.ndim == 1:
                seeds = seeds[None]
            sols = dedup_into([refine(s, float(a4), float(a2)) for s in seeds])
            solved[(i, j)] = sols
    print(f"in-box cells seeded: {len(solved)}  "
          f"k: {[len(v) for v in solved.values()]}", flush=True)

    # ---- 框外:BFS 从已解邻居延拓 ----
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for rnd in range(40):
        newly = 0
        for i, a2 in enumerate(A2):
            for j, a4 in enumerate(A4):
                if (i, j) in solved:
                    continue
                inits = []
                for di, dj in nbr:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < len(A2) and 0 <= nj < len(A4) and (ni, nj) in solved:
                        inits.extend(solved[(ni, nj)])
                if not inits:
                    continue
                sols = dedup_into([refine(s, float(a4), float(a2)) for s in inits])
                if sols:
                    solved[(i, j)] = sols
                    newly += 1
        print(f"  BFS round {rnd}: +{newly} cells (total {len(solved)}/{len(A2)*len(A4)})", flush=True)
        if newly == 0:
            break

    gt = {}
    for i, a2 in enumerate(A2):
        for j, a4 in enumerate(A4):
            sols = solved.get((i, j), [])
            gt[(float(a2), float(a4))] = np.stack(sols) if sols else np.zeros((0, N))
    torch.save({"a2": A2.tolist(), "a4": A4.tolist(), "gt": gt},
               os.path.join(HERE, "grid_gt_cont.pt"))
    # 打印 k 网格(a4 自上而下递减)
    print("\nk grid (rows = a4 high->low, cols = a2 low->high):", flush=True)
    for j in range(len(A4) - 1, -1, -1):
        row = [gt[(float(A2[i]), float(A4[j]))].shape[0] for i in range(len(A2))]
        tag = "  <box>" if (A4_LO <= A4[j] <= A4_HI) else ""
        print(f"  a4={A4[j]:.3f}: {row}{tag}", flush=True)
    print("saved grid_gt_cont.pt", flush=True)


if __name__ == "__main__":
    main()
