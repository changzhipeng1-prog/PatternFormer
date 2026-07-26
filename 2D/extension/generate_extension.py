"""2D EXTRAPOLATION generation.

Identical to test/generate_test.py but the parameters are the EXTRAPOLATED s values
(s > 1600, outside training) from extension/gt_extension.pt.

  model.generate([], [], target_p_val=s, max_solutions=...)   # no-context setting

Output -> extension/generated_extension.pt
  {"results":[{s, n_gt, gt[k,145], generated[n_gen,145]}], coord, elem, free_nodes}
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v3_model import V3PDEModel

C = Config()
CKPT = os.path.join(HERE, "..", "best_ckpt", "best_model")
GT = os.path.join(HERE, "gt_extension.pt")
OUT = os.path.join(HERE, "generated_extension.pt")


def main():
    g = torch.load(GT, map_location="cpu", weights_only=False)
    s_targets, gt_by_s = g["s_targets"], g["gt"]
    coord, elem, free = g["coord"].float(), g["elem"].long(), g["free_nodes"].long()

    print(f"Loading {CKPT} ...", flush=True)
    model = V3PDEModel.from_pretrained(CKPT, C, mesh_coord=coord,
                                       mesh_elem=elem, mesh_free_nodes=free)
    model.eval()

    results = []
    for s in s_targets:
        gt = torch.as_tensor(gt_by_s[s], dtype=torch.float32)
        if gt.dim() == 1:
            gt = gt.unsqueeze(0)
        with torch.no_grad():
            preds = model.generate([], [], target_p_val=float(s),
                                   max_solutions=C.max_solutions_per_p)
        gen = (torch.stack([pp.cpu().float().reshape(-1) for pp in preds])
               if len(preds) else torch.zeros(0, C.solution_dim))
        results.append({"s": float(s), "n_gt": int(gt.shape[0]), "gt": gt, "generated": gen})
        print(f"  s={s:<8} n_gt={gt.shape[0]}  n_gen={gen.shape[0]}", flush=True)

    torch.save({"results": results, "coord": coord.cpu(), "elem": elem.cpu(),
                "free_nodes": free.cpu()}, OUT)
    print(f"saved {len(results)} extrapolated params -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
