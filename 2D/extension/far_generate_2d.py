"""2D: 生成模型在 s=1600(种子)和一组更远 s 的直接输出 -> far_direct_2d.pt"""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from config import Config
from model.v3_model import V3PDEModel

C = Config()
CKPT = os.path.join(HERE, "..", "best_ckpt", "best_model")
S_SEED = 1600.0
S_POINTS = [1650.0, 1700.0, 1800.0, 2000.0, 2300.0, 2700.0, 3200.0]


def main():
    lk = torch.load(C.data_lookup_path, weights_only=False)
    coord, elem, free = lk["coord"].float(), lk["elem"].long(), lk["free_nodes"].long()
    model = V3PDEModel.from_pretrained(CKPT, C, mesh_coord=coord,
                                       mesh_elem=elem, mesh_free_nodes=free)
    model.eval()
    gen = {}
    for s in [S_SEED] + S_POINTS:
        with torch.no_grad():
            preds = model.generate([], [], target_p_val=float(s), max_solutions=C.max_solutions_per_p)
        gen[float(s)] = (torch.stack([p.cpu().float().reshape(-1) for p in preds])
                         if len(preds) else torch.zeros(0, C.solution_dim))
        print(f"  s={s:<7} n_gen={gen[float(s)].shape[0]}", flush=True)
    torch.save({"gen": gen, "coord": coord.cpu(), "elem": elem.cpu(), "free_nodes": free.cpu(),
                "S_SEED": S_SEED, "S_POINTS": S_POINTS}, os.path.join(HERE, "far_direct_2d.pt"))
    print("saved far_direct_2d.pt")


if __name__ == "__main__":
    main()
