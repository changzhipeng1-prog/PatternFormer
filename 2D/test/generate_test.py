"""Run the final 2D model over every TEST-set parameter s and save the
generated solutions (no in-context examples = the nocontext deliverable setting).

PDE: -Delta u - u^2 = -s*sin(pi x)sin(pi y) on the ell=3 FEM mesh (145 nodes).
Output -> test/generated_solutions.pt
  dict {results: list of {idx, s, n_gt, gt [k,145], generated [n_gen,145]},
        coord [145,2], elem [256,3], free_nodes [113]}  (mesh saved for viz/metrics)
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v3_model import V3PDEModel

C = Config()
# CKPT_DIR / GEN_OUT override so an ablation checkpoint can be evaluated without
# clobbering the shipped warm-start baseline artifacts (defaults = ship paths).
CKPT = os.environ.get("CKPT_DIR", os.path.join(HERE, "..", "best_ckpt", "best_model"))
OUT = os.environ.get("GEN_OUT", os.path.join(HERE, "generated_solutions.pt"))


def main(limit=0):
    lk = torch.load(C.data_lookup_path, weights_only=False)
    coord, elem, free_nodes = lk["coord"].float(), lk["elem"].long(), lk["free_nodes"].long()
    pv, sols = lk["p_values"], lk["solutions_by_p"]

    print(f"Loading {CKPT} ...", flush=True)
    model = V3PDEModel.from_pretrained(CKPT, C, mesh_coord=coord,
                                       mesh_elem=elem, mesh_free_nodes=free_nodes)
    model.eval()

    test_idx = torch.load(C.test_p_idx_path, weights_only=False).tolist()
    print(f"test params: {len(test_idx)}", flush=True)

    results = []
    for n, i in enumerate(test_idx):
        if limit and n >= limit:
            break
        s = sols[i]
        if s is None or len(s) == 0:
            continue
        gt = torch.as_tensor(s, dtype=torch.float32)
        if gt.dim() == 1:
            gt = gt.unsqueeze(0)
        s_val = float(pv[i].item())
        with torch.no_grad():
            preds = model.generate([], [], target_p_val=s_val, max_solutions=C.max_solutions_per_p)
        gen = (torch.stack([pp.cpu().float().reshape(-1) for pp in preds])
               if len(preds) else torch.zeros(0, C.solution_dim))
        results.append({"idx": int(i), "s": s_val, "n_gt": int(gt.shape[0]),
                        "gt": gt, "generated": gen})
        if n % 50 == 0:
            print(f"  {n}/{len(test_idx)}  s={s_val:.2f}  n_gt={gt.shape[0]}  n_gen={gen.shape[0]}", flush=True)

    torch.save({"results": results, "coord": coord.cpu(),
                "elem": elem.cpu(), "free_nodes": free_nodes.cpu()}, OUT)
    print(f"saved {len(results)} params -> {OUT}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="First N test entries; 0 uses the full test list")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    main(args.limit)
