"""Smoke test for the 2D pipeline: data-gen -> preprocess -> post-process.

Run:  cd paper/2D && python smoke_test.py
Exercises every non-training stage on tiny / real-GT inputs (no GPU, no Qwen).
Exits non-zero on any failure.
"""
import os
import sys
import py_compile
import tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATAGEN = os.path.join(HERE, "data_gen")
PREP = os.path.join(HERE, "preprocess")
MODEL = os.path.join(HERE, "code", "model")
DATA = os.path.join(HERE, "data")
LOOKUP = os.path.join(DATA, "p_solutions_lookup_2d_filtered.pt")


def stage_datagen():
    for f in ["generate_dataset_2d.py", "filter_d4_dataset.py"]:
        py_compile.compile(os.path.join(DATAGEN, f), doraise=True)
    # base mesh files must be loadable as numeric tables
    for f in ["coordinates.dat", "elements.dat", "dirichlet.dat"]:
        arr = np.loadtxt(os.path.join(DATAGEN, "Companion_Method", f))
        assert arr.size > 0, f"empty mesh file {f}"
    print("  [datagen] generators compile + base mesh loads OK")


def stage_preprocess():
    sys.path.insert(0, PREP)
    import make_splits
    tmp = tempfile.mkdtemp()
    make_splits.make_splits(LOOKUP, tmp, seed=42)
    n = {k: len(torch.load(os.path.join(tmp, f"{k}_p_idx.pt"), weights_only=False))
         for k in ["train", "val", "test"]}
    N = len(torch.load(LOOKUP, weights_only=False)["p_values"])
    assert sum(n.values()) == N, f"splits must cover all s-values: {n} vs {N}"
    print(f"  [preprocess] make_splits OK  ({N} s-values -> {n})")


def stage_postprocess():
    sys.path.insert(0, MODEL)
    import newton_refine_2d as nr
    lk = torch.load(LOOKUP, weights_only=False)
    coord, elem, free_nodes = lk["coord"].float(), lk["elem"].long(), lk["free_nodes"].long()
    pv, sols = lk["p_values"], lk["solutions_by_p"]
    i = next(j for j, s in enumerate(sols) if s is not None and len(s)
             and float(torch.as_tensor(s[0]).norm()) > 1.0)
    s_val = float(pv[i])
    u_gt = torch.as_tensor(sols[i][0], dtype=torch.float32).reshape(-1)
    u_noisy = u_gt + 0.01 * torch.randn_like(u_gt)
    refined, conv, nit, rh = nr.newton_refine_2d(u_noisy, s_val, coord, elem, free_nodes,
                                                 tol=1e-9, max_iter=30)
    assert conv, f"Newton did not converge in {nit} iters (residual {rh[-1]:.2e})"
    assert rh[-1] < rh[0], f"residual did not decrease ({rh[0]:.2e} -> {rh[-1]:.2e})"
    print(f"  [postprocess] newton_refine_2d OK  (s={s_val:.2f}, residual {rh[0]:.2e} -> {rh[-1]:.2e}, {nit} iters)")


def main():
    print("2D smoke test")
    stage_datagen()
    stage_preprocess()
    stage_postprocess()
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
