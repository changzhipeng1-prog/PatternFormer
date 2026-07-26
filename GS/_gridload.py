"""Shared loader for the combined GS figures (fig2 / fig3). Reads per-param solution
counts directly from the .pt / .json artifacts (NO config import, so L1 and L2 can be
loaded in the same process). All counts = # distinct generated solutions.
  load_m(setup, method, TE)  -> {ti: {ti,rho,mu,total,matched,beyond}}  (TEST-filtered)
"""
import os, glob, json
import torch
HERE = os.path.dirname(os.path.abspath(__file__))

CFG = {"L1": dict(te="L1/data/test_p_idx_big.pt", look="L1/data/gs_lookup_big.pt"),
       "L2": dict(te="L2/data/test_p_idx_L2.pt", look="L2/data/gs_lookup_L2.pt")}
# diffusion-coefficient value strings (LaTeX) per setup; L2 = L1/4 (2x-larger domain)
DIFF = {"L1": (r"2.5{\times}10^{-4}", r"5{\times}10^{-4}"),
        "L2": (r"6.25{\times}10^{-5}", r"1.25{\times}10^{-4}")}
METHODS = [("random", "traditional\n(random init)"), ("stop", "STOP"),
           ("det", "fixed-K"), ("noise", "fixed-K + noise")]
# the max-distinct TEST param shown in Fig1 / Fig4; its noise count is taken from the
# fixed-seed example run so Fig1/Fig4 and Fig2/Fig3 report the SAME number for it.
EXAMPLE_TI = {"L1": 576, "L2": 595}


def test_set(setup):
    return set(torch.load(f"{HERE}/{CFG[setup]['te']}").tolist())


def load_m(setup, m, TE=None):
    """det -> from the existing full_chunks eval (rho/mu from lookup); others -> grid shards."""
    base = f"{HERE}/{setup}"
    if m == "det":
        fc = sorted(glob.glob(f"{base}/results/full_chunks_*"))
        P = torch.load(f"{HERE}/{CFG[setup]['look']}", weights_only=False)["p_values"]
        rs = {}
        for f in glob.glob(f"{fc[0]}/chunk_*.json"):
            for r in json.load(open(f)).get("recs", []):
                ti, Kt, di, cov = r["ti"], r["Kt"], r["distinct"], r.get("cov", 0.0)
                mt = round(cov * Kt)
                rs[ti] = dict(ti=ti, rho=float(P[ti, 0]), mu=float(P[ti, 1]),
                              total=di, matched=mt, beyond=max(di - mt, 0))
    else:
        rs = {}
        for f in glob.glob(f"{base}/try/results_grid/{m}_shard*.json"):
            for r in json.load(open(f)).get("recs", []):
                rs[r["ti"]] = r
    if m == "noise":   # override the example param's count with the fixed-seed Fig1 run
        ex_ti = EXAMPLE_TI.get(setup)
        ex_path = f"{HERE}/{setup}/results/fig1_{setup}_ti{ex_ti}_noise.pt"
        if ex_ti in rs and os.path.exists(ex_path):
            d = torch.load(ex_path, weights_only=False)
            tot = int(len(d["sols"])); mt = int(d["is_gt"].sum())
            rs[ex_ti] = {**rs[ex_ti], "total": tot, "matched": mt, "beyond": tot - mt}
    if TE is not None:
        rs = {t: r for t, r in rs.items() if t in TE}
    return rs
