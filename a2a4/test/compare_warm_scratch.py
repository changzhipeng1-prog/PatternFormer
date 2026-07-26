"""Ablation comparison for Ex2 (a2a4): WARM-start (initialized from the Ex1 1D_p
checkpoint) vs SCRATCH (fresh LoRA on the pretrained Qwen, --no_warmstart).

Both models were evaluated with the identical no-in-context test protocol
(generate_test.py + compute_stats.py). This script only reads the two stats.pt
files and prints a side-by-side table on the paper's metrics:
  - DIRECT rel-L2   (accuracy of the raw generated field, before refinement)
  - POST   rel-L2   (after the FDM Newton refine == data-gen operator)
  - lost fraction   (post > 1e-2: refine converged to the wrong / adjacent branch)
  - converged %     (Newton reached ||F|| < 1e-9)
  - recovered multiplicity per parameter (# non-zero GT branches with post < 1e-2),
    grouped by the non-zero multiplicity k.
The trivial u==0 branch is already excluded upstream.
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WARM = os.path.join(HERE, "stats.pt")               # shipped Ex1->Ex2 warm baseline
SCR = os.path.join(HERE, "scratch", "stats.pt")     # Ex2 cold-start (--no_warmstart)
LOST_TH = 1e-2


def load(p):
    a = torch.load(p, map_location="cpu", weights_only=False)
    return {k: np.asarray(a[k]) for k in ("a4", "a2", "k", "direct", "post", "steps", "conv")} | {
        "n_params": int(a["n_params"]), "n_solutions": int(a["n_solutions"])}


def summarize(a, tag):
    direct, post, conv = a["direct"].astype(float), a["post"].astype(float), a["conv"].astype(bool)
    lost = post > LOST_TH
    # recovered branches = matched rows that land on the correct branch after refine
    rec = ~lost
    # per-parameter recovered multiplicity
    key = np.stack([a["a4"].round(4), a["a2"].round(4)], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    rec_per_param = np.array([rec[inv == i].sum() for i in range(len(uniq))])
    gt_per_param = np.array([(inv == i).sum() for i in range(len(uniq))])  # matched GT branches present
    print(f"\n===== {tag} =====")
    print(f"  params (with >=1 gen)      : {a['n_params']}")
    print(f"  matched non-zero branches  : {a['n_solutions']}")
    print(f"  DIRECT rel-L2  median      : {np.median(direct):.3e}   mean {direct.mean():.3e}   p90 {np.percentile(direct,90):.3e}")
    print(f"  POST   rel-L2  median      : {np.median(post):.3e}   mean {post.mean():.3e}   p90 {np.percentile(post,90):.3e}")
    print(f"  lost (post>{LOST_TH:.0e})          : {lost.sum()}/{lost.size}  ({lost.mean()*100:.2f}%)")
    print(f"  converged (||F||<1e-9)     : {conv.mean()*100:.2f}%")
    print(f"  recovered mult / param     : mean {rec_per_param.mean():.3f}   total {rec_per_param.sum()}")
    ks = sorted(set(a["k"].astype(int).tolist()))
    for kk in ks:
        m = a["k"].astype(int) == kk
        pl = post[m] > LOST_TH
        print(f"    k={kk:>2}: n={m.sum():>4}  DIRECT med {np.median(direct[m]):.2e}  POST med {np.median(post[m]):.2e}  lost {pl.mean()*100:5.2f}%")
    return {"direct_med": np.median(direct), "post_med": np.median(post),
            "lost_pct": lost.mean()*100, "conv_pct": conv.mean()*100,
            "rec_mean": rec_per_param.mean(), "n_sol": a["n_solutions"]}


def main():
    w = summarize(load(WARM), "WARM  (Ex1 -> Ex2, cross-equation warm-start)")
    if not os.path.exists(SCR):
        print(f"\n[scratch stats not found yet: {SCR}]")
        return
    s = summarize(load(SCR), "SCRATCH (fresh LoRA on pretrained Qwen, --no_warmstart)")
    print("\n================ WARM vs SCRATCH ================")
    print(f"{'metric':30s} {'WARM':>14s} {'SCRATCH':>14s}")
    for name, kk in [("DIRECT rel-L2 median", "direct_med"), ("POST rel-L2 median", "post_med"),
                     ("lost % (post>1e-2)", "lost_pct"), ("converged %", "conv_pct"),
                     ("recovered mult / param", "rec_mean"), ("matched branches (total)", "n_sol")]:
        fw = f"{w[kk]:.3e}" if "med" in kk else (f"{w[kk]:.2f}" if kk != "n_sol" else f"{int(w[kk])}")
        fs = f"{s[kk]:.3e}" if "med" in kk else (f"{s[kk]:.2f}" if kk != "n_sol" else f"{int(s[kk])}")
        print(f"{name:30s} {fw:>14s} {fs:>14s}")


if __name__ == "__main__":
    main()
