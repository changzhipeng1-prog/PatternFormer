"""Dump standardized data for the cross-project "who solves?" figure (Fig. 5).
Picks the richest example, Newton-refines the direct outputs, and stores
GT / direct / refined fields + the tolerance dial from stats.csv -> who_npz.npz.
Run from this dir (uses paper/1D_p/code refine).
"""
import os, sys, csv, numpy as np, torch
import scipy.optimize as sciopt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.traditional_refine import refine_with_newton


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def match(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


ex = torch.load(os.path.join(HERE, "examples_solutions.pt"), weights_only=False)
rec = max(ex, key=lambda r: r["gt"].shape[0])          # richest (k=7) example
p = float(rec["p"]); gt = rec["gt"].numpy().astype(np.float64)
gen = rec["generated"].numpy().astype(np.float64)
gtm, direct, refined, dre, pre = [], [], [], [], []
for pi, gj in match(gen, gt):
    ru, ok, _ = refine_with_newton(torch.tensor(gen[pi]), p, tol=1e-9, max_iter=30)
    ru = ru.detach().cpu().numpy().reshape(-1).astype(np.float64)
    gtm.append(gt[gj]); direct.append(gen[pi]); refined.append(ru)
    dre.append(rel_l2(gen[pi], gt[gj])); pre.append(rel_l2(ru, gt[gj]))
o = np.argsort([g.sum() for g in gtm])                 # order branches by signed integral
gtm = np.array(gtm)[o]; direct = np.array(direct)[o]; refined = np.array(refined)[o]
dre = np.array(dre)[o]; pre = np.array(pre)[o]

rows = list(csv.DictReader(open(os.path.join(HERE, "stats.csv"))))
d = np.array([float(r["direct_rel_l2"]) for r in rows])
po = np.array([float(r["post_rel_l2"]) for r in rows])
st = np.array([float(r["newton_steps"]) for r in rows])
cv = np.array([int(r["converged"]) for r in rows])
tol = np.logspace(np.log10(0.5), -3, 40)
frac = np.array([np.mean(d > t) for t in tol])

np.savez(os.path.join(HERE, "who_npz.npz"),
         kind="1d", label="1D single-parameter", pde=r"$-u''+u^2(u^2-p)=0$",
         param_str=f"$p={p:.0f}$ ($k={gt.shape[0]}$)", x=np.linspace(0, 1, gt.shape[1]),
         gt=gtm, direct=direct, refined=refined, direct_rel=dre, post_rel=pre,
         tol=tol, frac=frac, all_direct=d, all_post=po,
         med_direct=float(np.median(d)), med_post=float(np.median(po)),
         med_steps=float(np.median(st[cv > 0])), conv=float(cv.mean() * 100))
print(f"1D_p who_npz: k={gt.shape[0]} direct_rel={np.round(dre,3)} post_rel max={pre.max():.1e}")
