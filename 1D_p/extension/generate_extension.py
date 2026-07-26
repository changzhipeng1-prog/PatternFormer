"""1D_p EXTRAPOLATION generation.

Identical to test/generate_test.py but the parameters are the EXTRAPOLATED p values
(p > 18, outside the training range) instead of the test-set parameters.  Ground
truth comes from extension/gt_extension.pt (traditional continuation from p=18).

  model.generate([], [], target_p_val=p, max_solutions=8)   # no-context setting

Output -> extension/generated_extension.pt
  list of dict {p, n_gt, gt [k,1024], generated [n_gen,1024]}
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v2_model import V2PDEModel

C = Config()
CKPT = os.path.join(HERE, "..", "best_ckpt", "best_model")
GT = os.path.join(HERE, "gt_extension.pt")
OUT = os.path.join(HERE, "generated_extension.pt")


def main():
    g = torch.load(GT, map_location="cpu", weights_only=False)
    p_targets, gt_by_p = g["p_targets"], g["gt"]

    print(f"Loading {CKPT} ...", flush=True)
    model = V2PDEModel.from_pretrained(CKPT, C)
    model.eval()

    results = []
    for p in p_targets:
        gt = torch.as_tensor(gt_by_p[p], dtype=torch.float32)
        if gt.dim() == 1:
            gt = gt.unsqueeze(0)
        with torch.no_grad():
            preds = model.generate([], [], target_p_val=float(p),
                                   max_solutions=C.max_solutions_per_p)
        gen = (torch.stack([pp.cpu().float().reshape(-1) for pp in preds])
               if len(preds) else torch.zeros(0, C.solution_dim))
        results.append({"p": float(p), "n_gt": int(gt.shape[0]),
                        "gt": gt, "generated": gen})
        print(f"  p={p:<7} n_gt={gt.shape[0]}  n_gen={gen.shape[0]}", flush=True)

    torch.save(results, OUT)
    print(f"saved {len(results)} extrapolated params -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
