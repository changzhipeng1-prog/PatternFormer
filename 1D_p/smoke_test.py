"""Smoke test for the 1D_p pipeline: data-gen -> preprocess -> post-process.

Run:  cd paper/1D_p && python smoke_test.py
Exercises every non-training stage on tiny / real-GT inputs (no GPU, no Qwen).
Exits non-zero on any failure.
"""
import os
import sys
import tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATAGEN = os.path.join(HERE, "data_gen", "cbmfem")
PREP = os.path.join(HERE, "preprocess")
MODEL = os.path.join(HERE, "code", "model")
DATA = os.path.join(HERE, "data")


def stage_datagen():
    sys.path.insert(0, DATAGEN)
    from Sexample1 import faSh1
    from Newton import Newton
    N = 16
    # trivial solution u=0 is exact: F(0)=0
    x0 = np.zeros((N, 1), dtype=complex)
    F, dF = faSh1(x0, 5.0)
    r0 = float(np.linalg.norm(F(x0)))
    assert r0 < 1e-10, f"u=0 should be exact, got residual {r0}"
    # Newton from a small perturbation must converge back
    xp = (0.01 * np.random.RandomState(0).randn(N, 1)).astype(complex)
    F, dF = faSh1(xp, 5.0)
    out = Newton(F, dF, xp.copy(), 1e-10, 50)
    assert out is not None, "Newton returned None (non-convergence)"
    x_ref = out[:N]
    r = float(np.linalg.norm(F(x_ref)))
    assert r < 1e-8, f"Newton did not converge: residual {r}"
    print(f"  [datagen] faSh1+Newton OK  (N={N}, refined residual {r:.2e})")


def stage_preprocess():
    sys.path.insert(0, PREP)
    import build_lookup
    N = 16
    tmp_in = tempfile.mkdtemp()
    tmp_out = tempfile.mkdtemp()
    rng = np.random.RandomState(1)
    for b in range(2):
        # 3 continuation steps; include one zero column (trivial sol -> dropped)
        S = rng.randn(N, 3).astype(np.float64)
        S[:, 0] = 0.0
        p = np.array([1.0 + b, 2.0 + b, 3.0 + b])
        np.save(os.path.join(tmp_in, f"branch_{b}_p_new.npy"), p)
        np.save(os.path.join(tmp_in, f"branch_{b}_S_new.npy"), S)
    lookup, meta = build_lookup.build(tmp_in, tmp_out, num_branches=2, solution_dim=N)
    assert os.path.exists(os.path.join(tmp_out, "p_solutions_lookup.pt"))
    P = len(lookup["p_values"])
    tr = torch.load(os.path.join(tmp_out, "train_p_idx.pt"))
    va = torch.load(os.path.join(tmp_out, "val_p_idx.pt"))
    te = torch.load(os.path.join(tmp_out, "test_p_idx.pt"))
    assert len(tr) + len(va) + len(te) == P, "split sizes must cover all p"
    assert meta["num_solutions_total"] == sum(s.shape[0] for s in lookup["solutions_by_p"])
    print(f"  [preprocess] build_lookup OK  ({P} unique p, splits {len(tr)}/{len(va)}/{len(te)})")


def stage_postprocess():
    sys.path.insert(0, MODEL)
    import traditional_refine as tr
    assert tr._HAS_TRAD_SOLVER, "cbmfem solver not importable from data_gen/cbmfem"
    lk = torch.load(os.path.join(DATA, "p_solutions_lookup.pt"), map_location="cpu", weights_only=False)
    # pick a p with a nontrivial solution
    i = next(j for j, s in enumerate(lk["solutions_by_p"]) if s is not None and len(s)
             and float(torch.as_tensor(s[0]).norm()) > 1.0)
    p = float(lk["p_values"][i])
    u_gt = torch.as_tensor(lk["solutions_by_p"][i][0], dtype=torch.float64).reshape(-1)
    u_noisy = u_gt + 0.01 * torch.randn_like(u_gt)
    refined, ok, reason, trace = tr.refine_with_newton_trace(u_noisy, p, tol=1e-9, max_iter=30)
    assert ok, f"Newton refine failed: {reason}"
    r0, rF = trace[0]["residual"], trace[-1]["residual"]
    assert rF < r0, f"residual did not decrease ({r0:.2e} -> {rF:.2e})"
    print(f"  [postprocess] refine_with_newton OK  (p={p:.3f}, residual {r0:.2e} -> {rF:.2e}, {trace[-1]['iter']} iters)")


def main():
    print("1D_p smoke test")
    stage_datagen()
    stage_preprocess()
    stage_postprocess()
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
