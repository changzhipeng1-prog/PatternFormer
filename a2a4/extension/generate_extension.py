"""a2a4 EXTRAPOLATION generation (multiple directions).

Identical to test/generate_test.py but the parameters are the EXTRAPOLATED (a4,a2)
points (outside the training rectangle) from extension/gt_extension.pt.

  model.generate([], [], target_params=(a4, a2), max_solutions=...)   # no-context

Output -> extension/generated_extension.pt
  list of dict {dir, a4, a2, n_gt, gt[k,1024], generated[n_gen,1024]}
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
    meta, gt_map = g["params"], g["gt"]

    print(f"Loading {CKPT} ...", flush=True)
    model = V2PDEModel.from_pretrained(CKPT, C)
    model.eval()

    results = []
    for m in meta:
        a4, a2 = float(m["a4"]), float(m["a2"])
        gt = torch.as_tensor(gt_map[(round(a4, 4), round(a2, 4))], dtype=torch.float32)
        if gt.dim() == 1:
            gt = gt.unsqueeze(0)
        with torch.no_grad():
            preds = model.generate([], [], target_params=(a4, a2),
                                   max_solutions=C.max_solutions_per_p)
        gen = (torch.stack([pp.cpu().float().reshape(-1) for pp in preds])
               if len(preds) else torch.zeros(0, C.solution_dim))
        results.append({"dir": m["dir"], "a4": a4, "a2": a2,
                        "n_gt": int(gt.shape[0]), "gt": gt, "generated": gen})
        print(f"  [{m['dir']}] a4={a4:.3f} a2={a2:+.2f}  n_gt={gt.shape[0]}  n_gen={gen.shape[0]}",
              flush=True)

    torch.save(results, OUT)
    print(f"saved {len(results)} extrapolated params -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
