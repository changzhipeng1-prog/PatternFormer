"""2D EXTRAPOLATION ground truth.

Training/test data covers s in [-138, 1600].  Here we obtain the TRUE coexisting
solutions at EXTRAPOLATED s in {1601, 1610, 1650, 1700, 1800} (s > 1600, outside the
training range) by the traditional method: take the known solution branches at the
boundary s=1600 (lookup) and CONTINUE each branch outward in s with damped-Newton on
the FEM residual (newton_refine_2d — the same operator that made the training GT).

A branch that folds / stops converging before a target s genuinely has no solution
there and is simply absent from the GT at that s.

Output -> extension/gt_extension.pt :
    {"s_targets":[...], "gt":{s: ndarray[k,145]}, "provenance":{s:[branch...]},
     "coord","elem","free_nodes","boundary"}
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.newton_refine_2d import newton_refine_2d

C = Config()
S_BOUNDARY = 1600.0
S_TARGETS = [1601.0, 1610.0, 1650.0, 1700.0, 1800.0]
OUT = os.path.join(HERE, "gt_extension.pt")

TOL = 1e-9
MAX_ITER = 60
MAX_DIFF = 6.0             # ||u_new - u_old||_2 jump guard (branch-switch detector)
DEDUP = 1e-3              # rel-L2 distinct-branch threshold


def main():
    lk = torch.load(C.data_lookup_path, map_location="cpu", weights_only=False)
    coord, elem, free = lk["coord"].float(), lk["elem"].long(), lk["free_nodes"].long()
    pv = np.array(lk["p_values"])
    ib = int(np.argmin(np.abs(pv - S_BOUNDARY)))
    seeds = np.asarray(lk["solutions_by_p"][ib], np.float64)   # [k,145]
    print(f"boundary s={pv[ib]:.2f}: {seeds.shape[0]} branches", flush=True)

    def solve_at(u_seed, s):
        ur, ok, nit, hist = newton_refine_2d(
            torch.as_tensor(u_seed, dtype=torch.float32), float(s),
            coord, elem, free, tol=TOL, max_iter=MAX_ITER)
        if not ok:
            return None
        return ur.detach().cpu().numpy().astype(np.float64).reshape(-1)

    def march(u0, s_from, s_to, ds0=5.0):
        s, u, ds = s_from, np.asarray(u0, np.float64).copy(), ds0
        while s < s_to - 1e-9:
            step = min(ds, s_to - s)
            un = solve_at(u, s + step)
            if (un is None) or (np.linalg.norm(un - u) > MAX_DIFF):
                ds *= 0.5
                if ds < 1e-3:
                    return None
                continue
            u, s = un, s + step
            ds = min(ds0, ds * 1.5)
        return u

    targets = sorted(S_TARGETS)
    gt, prov = {s: [] for s in targets}, {s: [] for s in targets}
    for b in range(seeds.shape[0]):
        u, s_cur = seeds[b].copy(), S_BOUNDARY
        for s in targets:
            u_t = march(u, s_cur, s)
            if u_t is None:
                print(f"  branch {b}: folds before s={s}", flush=True)
                break
            gt[s].append(u_t); prov[s].append(b)
            u, s_cur = u_t, s
        else:
            print(f"  branch {b}: survives to s={targets[-1]}", flush=True)

    out_gt, out_prov = {}, {}
    for s in targets:
        keep, keep_b = [], []
        for u, b in zip(gt[s], prov[s]):
            if any(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-12) < DEDUP for v in keep):
                continue
            keep.append(u); keep_b.append(b)
        out_gt[s] = np.stack(keep) if keep else np.zeros((0, seeds.shape[1]))
        out_prov[s] = keep_b
        print(f"  s={s}: GT branches = {len(keep)}  (from {keep_b})", flush=True)

    torch.save({"s_targets": targets, "gt": out_gt, "provenance": out_prov,
                "coord": coord.cpu(), "elem": elem.cpu(), "free_nodes": free.cpu(),
                "boundary": S_BOUNDARY}, OUT)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
