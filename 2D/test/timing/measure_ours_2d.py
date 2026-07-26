"""OUR-method 2D timing: Qwen forward (generate all coexisting solutions for a target
s in one shot) + per-solution FEM Newton refine to residual < 1e-9 (same operator as
the data generator). Mirrors the 1D_p timing protocol.

For each target test-s:
  forward_s  = mean wall-time of model.generate (5 reps after 1 warmup)
  refine_s   = wall-time to Newton-refine ALL generated solutions
  newton_steps per solution + total; residual reported
Output: ours_2d_timing.csv.  Needs 1 GPU.
"""
import os, sys, time, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from config import Config
from model.v3_model import V3PDEModel
from model.newton_refine_2d import newton_refine_2d

C = Config()
CKPT = os.path.join(HERE, "..", "..", "best_ckpt", "best_model")
TARGETS = [1189, 869, 650, 304, -101]
N_REP = 5


def main():
    lk = torch.load(C.data_lookup_path, weights_only=False)
    coord, elem, free = lk["coord"].float(), lk["elem"].long(), lk["free_nodes"].long()
    model = V3PDEModel.from_pretrained(CKPT, C, mesh_coord=coord, mesh_elem=elem,
                                       mesh_free_nodes=free)
    model.eval()

    rows = []
    for s in TARGETS:
        s = float(s)
        # ---- forward: warmup + N_REP timed ----
        with torch.no_grad():
            _ = model.generate([], [], target_p_val=s, max_solutions=C.max_solutions_per_p)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ts = []
            preds = None
            for _r in range(N_REP):
                t0 = time.time()
                with torch.no_grad():
                    preds = model.generate([], [], target_p_val=s,
                                           max_solutions=C.max_solutions_per_p)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                ts.append(time.time() - t0)
        fwd = float(np.mean(ts))
        gen = [pp.cpu().float().reshape(-1) for pp in preds]
        kg = len(gen)

        # ---- refine each generated solution (FEM Newton, residual<1e-9) ----
        t0 = time.time(); steps = []; resids = []; convs = []
        for u in gen:
            ru, ok, nit, hist = newton_refine_2d(u, s, coord, elem, free, tol=1e-9, max_iter=30)
            steps.append(int(nit)); convs.append(bool(ok))
            resids.append(float(hist[-1]) if len(hist) else float("inf"))
        ref = time.time() - t0

        rows.append({"s": int(s), "k_gen": kg,
                     "forward_s": round(fwd, 4), "refine_s": round(ref, 4),
                     "total_s": round(fwd + ref, 4),
                     "newton_steps_per_sol": "|".join(map(str, steps)),
                     "newton_steps_total": int(sum(steps)),
                     "max_residual": max(resids) if resids else -1,
                     "converged": int(all(convs))})
        print(f"s={int(s):5d}: k_gen={kg} fwd={fwd:.3f}s refine={ref:.3f}s "
              f"total={fwd+ref:.3f}s steps={sum(steps)} maxres={max(resids):.1e}", flush=True)

    with open(os.path.join(HERE, "ours_2d_timing.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print("wrote ours_2d_timing.csv", flush=True)


if __name__ == "__main__":
    main()
