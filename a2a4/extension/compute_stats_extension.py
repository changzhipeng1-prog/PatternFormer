"""a2a4 EXTRAPOLATION statistics + results table  (COVERAGE-based, multi-direction).

Same coverage characterisation as 1D_p (see that file): refine every generated
output (FDM Newton at (a4,a2)), keep genuine solutions, dedup to distinct solutions,
measure GT coverage / missing / novel, report accuracy only over covered branches.

Writes extension/stats_extension.csv (per covered branch) and
extension/results_table_extension.csv (per (dir,a4,a2)), + markdown table.
"""
import os, sys, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_fdm import newton_refine, residual_l2

GEN = os.path.join(HERE, "generated_extension.pt")
CSV = os.path.join(HERE, "stats_extension.csv")
TAB = os.path.join(HERE, "results_table_extension.csv")
N = 1024
H = 1.0 / N
REFINE_OK = 1e-6
COVER = 1e-2
DEDUP = 1e-3


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def refine_one(u, a4, a2):
    ru, ok, nit = newton_refine(u, a4, a2, h=H, max_iter=50, abs_tol=1e-9)
    ru = np.asarray(ru, np.float64).reshape(-1)
    res = residual_l2(ru, a4, a2, H)
    st = int(nit) if ok else -1
    return ru, bool(ok), st, res


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def analyze(gt, gen, a4, a2):
    sols = []
    for u in gen:
        ru, ok, st, res = refine_one(u, a4, a2)
        if ok and res < REFINE_OK and np.linalg.norm(ru) > 1e-6:
            sols.append({"ru": ru, "raw": np.asarray(u, np.float64), "st": st, "res": res})
    distinct = []
    for sd in sols:
        if not any(rl2(sd["ru"], d["ru"]) < DEDUP for d in distinct):
            distinct.append(sd)
    covered = []
    for j, g in enumerate(gt):
        if not distinct:
            continue
        best = min(distinct, key=lambda d: rl2(d["ru"], g))
        if rl2(best["ru"], g) < COVER:
            covered.append({"j": j, "direct": rl2(best["raw"], g), "post": rl2(best["ru"], g),
                            "steps": best["st"], "res": best["res"]})
    novel = [d for d in distinct if all(rl2(d["ru"], g) > COVER for g in gt)]
    return distinct, covered, novel


def main():
    data = torch.load(GEN, map_location="cpu", weights_only=False)
    rows, table = [], []
    for rec in data:
        a4, a2, tag = float(rec["a4"]), float(rec["a2"]), rec["dir"]
        gt = rec["gt"].numpy().astype(np.float64)
        gt = gt[np.linalg.norm(gt, axis=1) > 1e-6] if gt.size else gt
        gen = rec["generated"].numpy().astype(np.float64)
        distinct, covered, novel = analyze(gt, gen, a4, a2)
        for c in covered:
            rows.append([tag, a4, a2, c["j"], c["direct"], c["post"], c["res"], c["steps"]])
        table.append({"dir": tag, "a4": a4, "a2": a2, "n_gt": gt.shape[0], "n_gen": gen.shape[0],
                      "n_distinct": len(distinct), "coverage": len(covered),
                      "missing": gt.shape[0] - len(covered), "novel": len(novel),
                      "direct_med": med([c["direct"] for c in covered]),
                      "post_med": med([c["post"] for c in covered]),
                      "steps_med": med([c["steps"] for c in covered]),
                      "resid_med": med([c["res"] for c in covered])})

    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dir", "a4", "a2", "gt_branch", "direct_rel_l2", "post_rel_l2",
                    "post_residual", "newton_steps"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", int(r[3]), f"{r[4]:.6e}",
                        f"{r[5]:.6e}", f"{r[6]:.6e}", int(r[7])])

    cols = ["dir", "a4", "a2", "n_gt", "n_gen", "n_distinct", "coverage", "missing", "novel",
            "direct_med", "post_med", "steps_med", "resid_med"]
    with open(TAB, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for t in table:
            w.writerow([t["dir"], f"{t['a4']:.4f}", f"{t['a2']:.4f}", t["n_gt"], t["n_gen"],
                        t["n_distinct"], t["coverage"], t["missing"], t["novel"],
                        f"{t['direct_med']:.3e}", f"{t['post_med']:.3e}",
                        f"{t['steps_med']:.1f}", f"{t['resid_med']:.2e}"])

    print("\n### a2a4 extrapolation (coverage-based; GT = traditional FDM multi-start)\n")
    hdr = ["dir", "a4", "a2", "n_gt", "n_gen", "distinct", "coverage", "missing", "novel",
           "rel-L2 direct", "rel-L2 post", "steps", "residual"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for t in table:
        print(f"| {t['dir']} | {t['a4']:.2f} | {t['a2']:+.1f} | {t['n_gt']} | {t['n_gen']} | "
              f"{t['n_distinct']} | {t['coverage']} | {t['missing']} | {t['novel']} | "
              f"{t['direct_med']:.2e} | {t['post_med']:.2e} | {t['steps_med']:.0f} | "
              f"{t['resid_med']:.1e} |")
    print(f"\nsaved {CSV}\nsaved {TAB}")


if __name__ == "__main__":
    main()
