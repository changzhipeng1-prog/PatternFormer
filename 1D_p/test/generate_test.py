"""Run the final 1D_p model over every TEST-set parameter p and save the
generated solutions (no in-context examples = the nocontext deliverable setting).

PDE: -u'' + u^2(u^2 - p) = 0.  Output -> test/generated_solutions.pt
  list of dict: {idx, p, n_gt, gt [k,1024], generated [n_gen,1024]}
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v2_model import V2PDEModel

C = Config()
CKPT = os.environ.get("CKPT_DIR", os.path.join(HERE, "..", "best_ckpt", "best_model"))
OUT = os.environ.get("GEN_OUT", os.path.join(HERE, "generated_solutions.pt"))


def main(limit=0):
    print(f"Loading {CKPT} ...", flush=True)
    model = V2PDEModel.from_pretrained(CKPT, C)
    model.eval()

    lk = torch.load(C.data_lookup_path, map_location="cpu", weights_only=False)
    test_idx = torch.load(C.test_p_idx_path, map_location="cpu", weights_only=False).tolist()
    pv, sols = lk["p_values"], lk["solutions_by_p"]
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
        p = float(pv[i])
        with torch.no_grad():
            preds = model.generate([], [], target_p_val=p, max_solutions=C.max_solutions_per_p)
        gen = (torch.stack([pp.cpu().float().reshape(-1) for pp in preds])
               if len(preds) else torch.zeros(0, C.solution_dim))
        results.append({"idx": int(i), "p": p, "n_gt": int(gt.shape[0]),
                        "gt": gt, "generated": gen})
        if n % 200 == 0:
            print(f"  {n}/{len(test_idx)}  p={p:.3f}  n_gt={gt.shape[0]}  n_gen={gen.shape[0]}", flush=True)

    torch.save(results, OUT)
    print(f"saved {len(results)} params -> {OUT}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="First N test entries; 0 uses the full test list")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    main(args.limit)
