"""1D_p EXTRAPOLATION statistics + results table  (COVERAGE-based).

The model can lose solution DIVERSITY out-of-distribution: several of its outputs
refine to the SAME true branch (mode collapse).  A forced 1-1 Hungarian match then
mis-reports such duplicates as huge rel-L2.  The honest characterisation is:

  - refine every generated output (cbmfem Newton at p); keep the ones that converge
    to a genuine solution (residual < 1e-6);
  - DEDUP them -> the distinct solutions the model actually produces;
  - COVERAGE: how many of the true continued-GT branches are recovered
    (rel-L2 < 1e-2 to some distinct solution)  -> missing = n_gt - coverage;
  - NOVEL: distinct valid solutions that are NOT any GT branch (spurious / extra real
    solutions the model invents);  for 1D_p this is the "finds a different solution?" test;
  - accuracy (direct / post rel-L2, Newton steps, residual) is reported ONLY over the
    covered branches (the duplicates and non-converged junk are not charged as rel-L2≈1).

Writes:
  extension/stats_extension.csv          one row per covered GT branch
  extension/results_table_extension.csv  one row per extrapolated p (aggregated)
  + prints a markdown table.
"""
import os, sys, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.traditional_refine import refine_with_newton_trace

GEN = os.path.join(HERE, "generated_extension.pt")
CSV = os.path.join(HERE, "stats_extension.csv")
TAB = os.path.join(HERE, "results_table_extension.csv")
REFINE_OK = 1e-6      # residual below which a refined output is a genuine solution
COVER = 1e-2          # rel-L2 below which a solution counts as a given GT branch
DEDUP = 1e-3          # distinct-solution dedup threshold


def rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def refine_one(u, p):
    """-> (refined_np, ok, steps, residual)."""
    ru, ok, _, tr = refine_with_newton_trace(torch.tensor(u), float(p), tol=1e-9, max_iter=30)
    res = float(tr[-1]["residual"]) if tr else float("inf")
    st = int(tr[-1]["iter"]) if (ok and tr) else -1
    return np.asarray(ru.detach().cpu(), np.float64).reshape(-1), bool(ok), st, res


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def analyze(gt, gen, p):
    # refine all generated outputs, keep genuine solutions
    sols = []
    for u in gen:
        ru, ok, st, res = refine_one(u, p)
        if ok and res < REFINE_OK:
            sols.append({"ru": ru, "raw": np.asarray(u, np.float64), "st": st, "res": res})
    # dedup -> distinct solutions
    distinct = []
    for s in sols:
        if not any(rl2(s["ru"], d["ru"]) < DEDUP for d in distinct):
            distinct.append(s)
    # coverage of GT + per-covered accuracy
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
        p = float(rec["p"])
        gt = rec["gt"].numpy().astype(np.float64)
        gen = rec["generated"].numpy().astype(np.float64)
        distinct, covered, novel = analyze(gt, gen, p)
        for c in covered:
            rows.append([p, c["j"], c["direct"], c["post"], c["res"], c["steps"]])
        table.append({
            "p": p, "n_gt": gt.shape[0], "n_gen": gen.shape[0],
            "n_distinct": len(distinct), "coverage": len(covered),
            "missing": gt.shape[0] - len(covered), "novel": len(novel),
            "direct_med": med([c["direct"] for c in covered]),
            "post_med": med([c["post"] for c in covered]),
            "steps_med": med([c["steps"] for c in covered]),
            "resid_med": med([c["res"] for c in covered]),
        })

    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "gt_branch", "direct_rel_l2", "post_rel_l2", "post_residual", "newton_steps"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}", int(r[1]), f"{r[2]:.6e}", f"{r[3]:.6e}", f"{r[4]:.6e}", int(r[5])])

    cols = ["p", "n_gt", "n_gen", "n_distinct", "coverage", "missing", "novel",
            "direct_med", "post_med", "steps_med", "resid_med"]
    with open(TAB, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for t in table:
            w.writerow([t["p"], t["n_gt"], t["n_gen"], t["n_distinct"], t["coverage"],
                        t["missing"], t["novel"], f"{t['direct_med']:.3e}",
                        f"{t['post_med']:.3e}", f"{t['steps_med']:.1f}", f"{t['resid_med']:.2e}"])

    print("\n### 1D_p extrapolation (coverage-based; GT = continuation from p=18)\n")
    hdr = ["p", "n_gt", "n_gen", "distinct", "coverage", "missing", "novel",
           "rel-L2 direct", "rel-L2 post", "steps", "residual"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for t in table:
        print(f"| {t['p']:.2f} | {t['n_gt']} | {t['n_gen']} | {t['n_distinct']} | "
              f"{t['coverage']} | {t['missing']} | {t['novel']} | {t['direct_med']:.2e} | "
              f"{t['post_med']:.2e} | {t['steps_med']:.0f} | {t['resid_med']:.1e} |")
    print(f"\nsaved {CSV}\nsaved {TAB}")


if __name__ == "__main__":
    main()
