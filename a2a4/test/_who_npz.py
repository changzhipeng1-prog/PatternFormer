"""Dump standardized data for the cross-project "who solves?" figure (Fig. 5).
a2a4 (1D two-parameter): -u'' + a4 u^4 + a2 u^2 = 0. Run from this dir.
"""
import os, sys, csv, numpy as np, torch
import scipy.optimize as sciopt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_fdm import newton_refine


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def match(preds, gt):
    cost = np.linalg.norm(preds[:, None, :] - gt[None, :, :], axis=2)
    r, c = sciopt.linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


data = torch.load(os.path.join(HERE, "generated_solutions.pt"), weights_only=False)
# richest example (k=5) with a well-behaved (near-median) direct error
cands = [r for r in data if r["gt"].shape[0] >= 5 and r["generated"].shape[0] >= 5]
def med_err(r):
    g = r["gt"].numpy().astype(np.float64); ge = r["generated"].numpy().astype(np.float64)
    return np.median([rel_l2(ge[pi], g[gj]) for pi, gj in match(ge, g)])
rec = min(cands, key=med_err)
a4, a2 = float(rec["params"][0]), float(rec["params"][1])
gt = rec["gt"].numpy().astype(np.float64); gen = rec["generated"].numpy().astype(np.float64)
h = 1.0 / gt.shape[1]
gtm, direct, refined, dre, pre = [], [], [], [], []
for pi, gj in match(gen, gt):
    ru, ok, _ = newton_refine(torch.tensor(gen[pi]), a4, a2, h=h, max_iter=50, abs_tol=1e-9)
    ru = (ru.detach().cpu().numpy() if torch.is_tensor(ru) else np.asarray(ru)).reshape(-1).astype(np.float64)
    gtm.append(gt[gj]); direct.append(gen[pi]); refined.append(ru)
    dre.append(rel_l2(gen[pi], gt[gj])); pre.append(rel_l2(ru, gt[gj]))
o = np.argsort([g.sum() for g in gtm])
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
         kind="1d", label="1D two-parameter", pde=r"$-u''+a_4u^4+a_2u^2=0$",
         param_str=f"$a_4={a4:.2f},\\ a_2={a2:.1f}$ ($k={gt.shape[0]}$)",
         x=np.linspace(0, 1, gt.shape[1]),
         gt=gtm, direct=direct, refined=refined, direct_rel=dre, post_rel=pre,
         tol=tol, frac=frac, all_direct=d, all_post=po,
         med_direct=float(np.median(d)), med_post=float(np.median(po)),
         med_steps=float(np.median(st[cv > 0])), conv=float(cv.mean() * 100))
print(f"a2a4 who_npz: k={gt.shape[0]} direct_rel={np.round(dre,3)} post_rel max={pre.max():.1e}")
