"""a2a4 EXTRAPOLATION ground truth (multiple directions).

Training rectangle:  a4 in [0.301, 0.798],  a2 in [-14.934, -2.0].
We probe FOUR extrapolation directions OUT of that rectangle, each at several
distances:
   a2-  : a2 more negative (deeper well), a4 fixed at mid 0.55
   a2+  : a2 toward 0       (shallower),  a4 fixed at mid 0.55
   a4+  : a4 above 0.798,                 a2 fixed at mid -8.0
   a4-  : a4 below 0.301,                 a2 fixed at mid -8.0

Ground truth = the TRADITIONAL classical solver that generated the training data:
a parameter-agnostic battery of half-cosine initial guesses + damped FDM-Newton,
collecting all distinct non-trivial roots (model.newton_fdm; identical residual
convention).  This finds ALL coexisting branches at the (out-of-distribution)
parameter without any learned warm start.

Output -> extension/gt_extension.pt :
   {"params":[{dir, a4, a2}], "gt":{(a4,a2): ndarray[k,1024]}}
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from model.newton_fdm import newton_refine, residual_l2

OUT = os.path.join(HERE, "gt_extension.pt")

N = 1024
H = 1.0 / N
X = np.linspace(0.0, 1.0, N)
NEWTON_TOL = 1e-9
NEWTON_MAX = 80
DEDUP = 1e-3
N_MODES = 8
A_MAX = 12.0

# ---- extrapolation grid: (direction tag, a4, a2) ; boundaries 0.301/0.798, -2/-14.934
A4_MID, A2_MID = 0.55, -8.0
PARAMS = (
    [("a2-", A4_MID, a2) for a2 in (-15.5, -17.0, -20.0, -25.0)] +
    [("a2+", A4_MID, a2) for a2 in (-1.8, -1.4, -1.0, -0.5)] +
    [("a4+", a4, A2_MID) for a4 in (0.82, 0.90, 1.00, 1.20)] +
    [("a4-", a4, A2_MID) for a4 in (0.28, 0.24, 0.18, 0.10)]
)


def make_battery():
    A = np.concatenate([np.linspace(0.05, 1.5, 20), np.linspace(1.6, A_MAX, 34)])
    inits = []
    for j in range(1, N_MODES + 1):
        phi = np.cos((2 * j - 1) * np.pi * X / 2.0)
        for a in A:
            inits.append(a * phi); inits.append(-a * phi)
    return inits


BATTERY = make_battery()


def multistart(a4, a2):
    sols = []
    for u0 in BATTERY:
        ru, ok, _ = newton_refine(u0, a4, a2, h=H, max_iter=NEWTON_MAX, abs_tol=NEWTON_TOL)
        if ok and np.linalg.norm(ru) > 1e-6 and residual_l2(ru, a4, a2, H) < 1e-6:
            ru = np.asarray(ru, np.float64).reshape(-1)
            if not any(np.linalg.norm(ru - v) / (np.linalg.norm(v) + 1e-12) < DEDUP for v in sols):
                sols.append(ru)
    return sols


def main():
    print(f"battery = {len(BATTERY)} inits ({N_MODES} modes)", flush=True)
    gt, meta = {}, []
    for tag, a4, a2 in PARAMS:
        sols = multistart(a4, a2)
        gt[(round(a4, 4), round(a2, 4))] = np.stack(sols) if sols else np.zeros((0, N))
        meta.append({"dir": tag, "a4": float(a4), "a2": float(a2), "k": len(sols)})
        print(f"  [{tag}] a4={a4:.3f} a2={a2:+.2f}  ->  k={len(sols)} branches", flush=True)
    torch.save({"params": meta, "gt": gt}, OUT)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
