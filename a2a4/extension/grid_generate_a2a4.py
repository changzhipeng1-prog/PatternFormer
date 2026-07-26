"""Model DIRECT generation over a 2-D (a2,a4) grid for the a2a4 extrapolation heatmap.
Output -> grid_direct.pt  {"a2":[...], "a4":[...], "gen":{(a2,a4): tensor[n,1024]}}"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v2_model import V2PDEModel

C = Config()
CKPT = os.path.join(HERE, "..", "best_ckpt", "best_model")
A2 = np.round(np.linspace(-25.0, -0.5, 13), 4)
A4 = np.round(np.linspace(0.05, 1.30, 11), 4)


def main():
    model = V2PDEModel.from_pretrained(CKPT, C); model.eval()
    gen = {}
    for i, a2 in enumerate(A2):
        for a4 in A4:
            with torch.no_grad():
                preds = model.generate([], [], target_params=(float(a4), float(a2)),
                                       max_solutions=C.max_solutions_per_p)
            gen[(float(a2), float(a4))] = (torch.stack([p.cpu().float().reshape(-1) for p in preds])
                                           if len(preds) else torch.zeros(0, C.solution_dim))
        print(f"  a2={a2:+.2f} row done ({i+1}/{len(A2)})", flush=True)
    torch.save({"a2": A2.tolist(), "a4": A4.tolist(), "gen": gen},
               os.path.join(HERE, "grid_direct.pt"))
    print("saved grid_direct.pt")


if __name__ == "__main__":
    main()
