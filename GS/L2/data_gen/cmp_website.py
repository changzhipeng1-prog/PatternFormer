"""Step-1 figure: our L=2 GT atlas beside the xmorphia reference, same (F,k) window.

Builds a (F rows x k cols) canvas from the data_L2 cells (richest-texture rep per
cell, border colour = #solutions), crops the reference map to exactly our (F,k)
range, and places them side by side -> qwen_L2/slides/figs/L2_vs_website.png.
"""
import glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.image import imread

import os; FDM = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FDM, "outputs")
REF = f"{OUT}/xmorphia_ref.jpg"

cells = []
for f in glob.glob(f"{FDM}/L2_gen/data_L2/cell_*.npz"):
    z = np.load(f)
    if z["A"].shape[0] == 0:
        continue
    A = z["A"]
    rep = max(A, key=lambda a: a.max()-a.min())
    cells.append((float(z["rho"]), float(z["mu"]), A.shape[0], rep))   # F,k,count,rep
Fs = sorted(set(round(c[0], 5) for c in cells))
ks = sorted(set(round(c[1], 5) for c in cells))
print(f"{len(cells)} nonempty cells, F {min(Fs):.3f}-{max(Fs):.3f}, k {min(ks):.3f}-{max(ks):.3f}")
Fmin, Fmax, kmin, kmax = min(Fs), max(Fs), min(ks), max(ks)

NR, NK = len(Fs), len(ks)
TH = 56
Fi = {f: i for i, f in enumerate(Fs)}; ki = {k: j for j, k in enumerate(ks)}
cmax = max(c[2] for c in cells)
cmap = cm.get_cmap("rainbow"); norm = Normalize(0, cmax); vir = cm.get_cmap("viridis")
canvas = np.ones((NR*TH, NK*TH, 3))
for (F, k, cnt, rep) in cells:
    i = Fi[round(F, 5)]; j = ki[round(k, 5)]
    a = (rep - rep.min())/(rep.ptp()+1e-9)
    yi = np.linspace(0, a.shape[0]-1, TH-6).astype(int); xi = np.linspace(0, a.shape[1]-1, TH-6).astype(int)
    r0 = (NR-1-i)*TH; c0 = j*TH
    canvas[r0:r0+TH, c0:c0+TH] = cmap(norm(cnt))[:3]
    canvas[r0+3:r0+TH-3, c0+3:c0+TH-3] = vir(a)[..., :3][np.ix_(yi, xi)]

# ---- website crop to same (F,k) window ----
refimg = imread(REF); Wr = refimg.shape[1]; S = Wr/640.0
def _kx(k): return S*(151 + (k-0.031)/(0.073-0.031)*(623-151))
def _Fy(F): return S*(600 + (F-0.006)/(0.110-0.006)*(16-600))
x0i, x1i = int(_kx(kmin)), int(_kx(kmax))
y0i, y1i = int(_Fy(Fmax)), int(_Fy(Fmin))
crop = refimg[max(0, y0i):y1i, x0i:x1i]
ext = [kmin, kmax, Fmin, Fmax]

fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 8.4))
b1.imshow(crop, extent=ext, aspect="auto", origin="upper")
b1.set_title("xmorphia reference (same $(F,k)$ window)\nlong-time Gray-Scott patterns", fontsize=12)
b2.imshow(canvas, extent=ext, aspect="auto", origin="upper")
b2.set_title("Our $L=2$ traditional-solver GT\n(border colour = #steady states)", fontsize=12)
for b in (b1, b2):
    b.set_xlabel(r"$\mu=k$ (kill)", fontsize=12); b.set_ylabel(r"$\rho=F$ (feed)", fontsize=12)
fig.suptitle("Step 1: $L=2$ GT vs reference atlas (same parameter window)", fontsize=14, y=1.0)
fig.savefig(f"{OUT}/L2_vs_website.png", dpi=140, bbox_inches="tight")
print("wrote L2_vs_website.png")
