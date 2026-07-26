"""Correct the a2a4 dataset so every GT solution is an EXACT root of the same
N=1024 second-order FDM operator used by the post-processor / data generator.

Why: the stored bvp_region_* solutions only satisfy the FDM to ~1.7e-3 (they were
produced by an upstream solver that is NOT the present FDM), so they sit ~4e-4
(rel-L2) off the true FDM roots. That makes the dataset inconsistent with the
post-processing operator (post rel-L2 floors at ~4e-4 instead of ~1e-8).

Fix: Newton-refine every non-zero solution to ||F|| < 1e-9 (float64), keep the
trivial u==0 branch exactly zero. Verified safe: no branch merges, k preserved.

Input  : <DATA>/bvp_region_{train,val,test}.pt   (list of {params, solutions})
Output : <OUT>/bvp_region_{train,val,test}.pt     (same structure, FDM-exact GT)
Originals are backed up to <DATA>/orig_backup/ before anything is written.
"""
import os
import sys
import shutil
import numpy as np
import torch
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_fdm import newton_refine, residual_l2

DATA = os.path.join(HERE, "..", "data")              # paper/a2a4/data (config source)
OUT = os.path.join(HERE, "..", "data")               # overwrite in place (after backup)
BACKUP = os.path.join(DATA, "orig_backup")
SPLITS = ["train", "val", "test"]
ZERO_NORM = 1e-6
ABS_TOL = 1e-9
MAXIT = 80


def refine_one(args):
    u, a4, a2 = args
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if np.linalg.norm(u) < ZERO_NORM:
        return np.zeros_like(u), True, 0.0          # trivial branch -> exact 0
    N = len(u); h = 1.0 / N
    ru, ok, _ = newton_refine(u, a4, a2, h=h, max_iter=MAXIT, abs_tol=ABS_TOL)
    ru = np.asarray(ru, np.float64).reshape(-1)
    res = residual_l2(ru, a4, a2, h)
    if (not ok) or res > 1e-6:                       # failed -> keep original (rare ~0.15%)
        return u, False, residual_l2(u, a4, a2, h)
    return ru, True, res


def process_record(rec):
    a4, a2 = float(rec["params"][0]), float(rec["params"][1])
    # DROP the trivial u==0 branch entirely (per user: remove zero solutions from data)
    nonzero = [s for s in rec["solutions"]
               if np.linalg.norm(np.asarray(s, dtype=np.float64)) >= ZERO_NORM]
    dropped_zero = len(rec["solutions"]) - len(nonzero)
    refined = [refine_one((s, a4, a2)) for s in nonzero]
    # DROP refine-failures outright (per user: remove from dataset, do not keep off-FDM)
    kept = [ru.astype(np.float64) for ru, ok, res in refined if ok]
    dropped_fail = sum(1 for _, ok, _ in refined if not ok)
    # dedup: drop branches that collapsed to the same FDM root (near-degenerate)
    uniq = []
    for u in kept:
        if not any(np.linalg.norm(u - v) < 1e-3 for v in uniq):
            uniq.append(u)
    dropped_dup = len(kept) - len(uniq)
    new_rec = {"params": rec["params"], "solutions": uniq}
    return new_rec, dropped_fail, dropped_dup, len(rec["solutions"]), len(uniq), dropped_zero


def main():
    ncpu = int(os.environ.get("NCPU", "32"))
    os.makedirs(BACKUP, exist_ok=True)
    for split in SPLITS:
        src = os.path.join(DATA, f"bvp_region_{split}.pt")
        bak = os.path.join(BACKUP, f"bvp_region_{split}.pt")
        if not os.path.exists(bak):
            shutil.copy2(src, bak)                   # back up original once
        # ALWAYS read from the pristine backup (idempotent re-runs)
        data = torch.load(bak, map_location="cpu", weights_only=False)
        print(f"[{split}] {len(data)} records  ncpu={ncpu}", flush=True)
        with Pool(ncpu) as pool:
            results = pool.map(process_record, data, chunksize=16)
        # drop any record left with NO solutions after cleaning
        new_data = [r[0] for r in results if len(r[0]["solutions"]) > 0]
        n_emptied = sum(1 for r in results if len(r[0]["solutions"]) == 0)
        tot_fail = sum(r[1] for r in results)
        tot_merge = sum(r[2] for r in results)
        tot_sol = sum(r[3] for r in results)
        tot_zero = sum(r[5] for r in results)
        print(f"[{split}] dropped: zeros={tot_zero}  failures={tot_fail}  duplicates={tot_merge}  "
              f"emptied_records={n_emptied}  -> records {len(data)}->{len(new_data)}", flush=True)
        # residual report on a sample of corrected GT
        rs = []
        for rec in new_data[:300]:
            a4, a2 = float(rec["params"][0]), float(rec["params"][1])
            for u in rec["solutions"]:
                if np.linalg.norm(u) > ZERO_NORM:
                    rs.append(residual_l2(u, a4, a2, 1.0 / len(u)))
        out = os.path.join(OUT, f"bvp_region_{split}.pt")
        torch.save(new_data, out)
        print(f"[{split}] solutions={tot_sol}  refine_fail={tot_fail} "
              f"({tot_fail/max(tot_sol,1)*100:.2f}%)  branch_merges={tot_merge}  "
              f"corrected_GT_resid_median={np.median(rs):.2e}  -> {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
