"""Generate the model's DIRECT output at the far-extrapolated p = 20, 25 -> far_direct.pt."""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v2_model import V2PDEModel

C = Config()
CKPT = os.path.join(HERE, "..", "best_ckpt", "best_model")
TARGETS = [20.0, 25.0]


def main():
    model = V2PDEModel.from_pretrained(CKPT, C); model.eval()
    out = {}
    for p in TARGETS:
        with torch.no_grad():
            preds = model.generate([], [], target_p_val=float(p), max_solutions=C.max_solutions_per_p)
        gen = (torch.stack([pp.cpu().float().reshape(-1) for pp in preds])
               if len(preds) else torch.zeros(0, C.solution_dim))
        out[float(p)] = gen
        print(f"  p={p}: n_gen={gen.shape[0]}", flush=True)
    torch.save(out, os.path.join(HERE, "far_direct.pt"))
    print("saved far_direct.pt")


if __name__ == "__main__":
    main()
