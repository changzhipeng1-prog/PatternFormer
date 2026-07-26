"""Smoke test for the a2a4 pipeline: data-gen -> preprocess -> post-process.

Run:  cd paper/a2a4 && python smoke_test.py
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


def stage_datagen():
    # generator scripts are multi-GPU / large-seed; verify they compile cleanly,
    # then exercise the SAME discrete operator via the post-processing module.
    for f in ["generate_fullspace_gpu.py", "simplify_dataset.py"]:
        py_compile.compile(os.path.join(DATAGEN, f), doraise=True)
    sys.path.insert(0, MODEL)
    import newton_fdm
    N = 64
    h = 1.0 / N
    a4, a2 = 1.0, -5.0
    # u=0 is an exact discrete solution (f(0)=0)
    r0 = newton_fdm.residual_l2(np.zeros(N), a4, a2, h)
    assert r0 < 1e-10, f"u=0 should be exact, residual {r0}"
    print(f"  [datagen] generators compile + fd_F operator OK  (N={N})")


def stage_preprocess():
    sys.path.insert(0, PREP)
    import split_dataset
    tmp = tempfile.mkdtemp()
    region = os.path.join(tmp, "bvp_region.pt")
    data = [{"params": (1.0, -float(i)), "solutions": [np.random.randn(1024).astype(np.float32)]}
            for i in range(20)]
    torch.save(data, region)
    split_dataset.split(region, tmp, seed=42)
    counts = {n: len(torch.load(os.path.join(tmp, f"bvp_region_{n}.pt"), weights_only=False))
              for n in ["train", "val", "test"]}
    assert sum(counts.values()) == 20, f"splits must cover all: {counts}"
    print(f"  [preprocess] split_dataset OK  (20 -> {counts})")


def stage_postprocess():
    sys.path.insert(0, MODEL)
    import newton_fdm
    test = torch.load(os.path.join(DATA, "bvp_region_test.pt"), map_location="cpu", weights_only=False)
    # find a sample with a nontrivial solution
    e = next(x for x in test if any(float(np.linalg.norm(np.asarray(s))) > 1.0 for s in x["solutions"]))
    a4, a2 = float(e["params"][0]), float(e["params"][1])
    u_gt = np.asarray(next(s for s in e["solutions"]
                           if float(np.linalg.norm(np.asarray(s))) > 1.0), dtype=np.float64).reshape(-1)
    h = 1.0 / len(u_gt)
    u_noisy = u_gt + 0.01 * np.random.RandomState(0).randn(len(u_gt))
    r0 = newton_fdm.residual_l2(u_noisy, a4, a2, h)
    refined, conv, nit = newton_fdm.newton_refine(u_noisy, a4, a2, h, tol=1e-6, max_iter=30)
    rF = newton_fdm.residual_l2(refined, a4, a2, h)
    assert conv, f"Newton did not converge in {nit} iters (residual {rF:.2e})"
    assert rF < r0, f"residual did not decrease ({r0:.2e} -> {rF:.2e})"
    print(f"  [postprocess] newton_refine OK  (a4={a4}, a2={a2}, residual {r0:.2e} -> {rF:.2e}, {nit} iters)")


def main():
    print("a2a4 smoke test")
    stage_datagen()
    stage_preprocess()
    stage_postprocess()
    print("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
