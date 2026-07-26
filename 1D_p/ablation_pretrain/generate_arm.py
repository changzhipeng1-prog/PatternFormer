"""Run ONE ablation arm's final (stage-2) checkpoint over every TEST-set p and save
the generated solutions, for the pretrained-vs-random comparison (DIRECT predictions;
no post-processing).  Mirrors test/generate_test.py but parameterised by env vars:

    ARM_CKPT   path to <arm>/best_model
    ARM_OUT    output .pt path
    ARM_RANDOM 1 -> random-init backbone (seed 1234), 0 -> pretrained backbone

Output -> ARM_OUT  (list of dict: {idx, p, n_gt, gt [k,1024], generated [n_gen,1024]})
"""
import os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v2_model import V2PDEModel

C = Config()
C.random_init_backbone = bool(int(os.environ.get("ARM_RANDOM", "0")))
C.backbone_init_seed = 1234                       # must match the arm's training seed
CKPT = os.environ["ARM_CKPT"]
OUT = os.environ["ARM_OUT"]


def main():
    print(f"Loading {CKPT}  (random_init_backbone={C.random_init_backbone}) ...", flush=True)
    model = V2PDEModel.from_pretrained(CKPT, C)
    model.eval()

    lk = torch.load(C.data_lookup_path, map_location="cpu", weights_only=False)
    test_idx = torch.load(C.test_p_idx_path, map_location="cpu", weights_only=False).tolist()
    pv, sols = lk["p_values"], lk["solutions_by_p"]
    print(f"test params: {len(test_idx)}", flush=True)

    results = []
    for n, i in enumerate(test_idx):
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
    main()
