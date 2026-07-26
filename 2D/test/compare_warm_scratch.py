"""Ablation comparison for Ex3 (2D forced -Delta u - u^2 = -s sin sin):
WARM-start (initialized from the Ex2 a2a4 checkpoint) vs SCRATCH (fresh LoRA on
the pretrained Qwen, no --resume_from / --init_from_v2).

Both models were evaluated with the identical no-in-context test protocol
(generate_test.py + compute_stats.py); this script reads the two stats.pt files
and prints a side-by-side table on the paper's metrics, grouped by multiplicity k.
NOTE: the 2D POST rel-L2 floors at ~1e-3 (near-degenerate one-to-one matching of a
near-zero-amplitude branch), NOT a refinement failure -- residual reaches 1e-9.
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WARM = os.path.join(HERE, "stats.pt")               # shipped Ex2->Ex3 warm baseline
SCR = os.path.join(HERE, "scratch", "stats.pt")     # Ex3 cold-start
LOST_TH = 1e-2


def load(p):
    a = torch.load(p, map_location="cpu", weights_only=False)
    return {k: np.asarray(a[k]) for k in ("s", "k", "direct", "post", "steps", "conv")} | {
        "n_params": int(a["n_params"]), "n_solutions": int(a["n_solutions"])}


def summarize(a, tag, kkeep=None):
    m = np.isin(a["k"].astype(int), kkeep) if kkeep is not None else np.ones(a["k"].shape, bool)
    direct, post, conv, steps = (a["direct"][m].astype(float), a["post"][m].astype(float),
                                 a["conv"][m].astype(bool), a["steps"][m].astype(float))
    lost = post > LOST_TH
    key = a["s"][m].round(4)
    uniq, inv = np.unique(key, return_inverse=True)
    rec = np.array([(~lost)[inv == i].sum() for i in range(len(uniq))])
    print(f"\n===== {tag} =====")
    print(f"  params / matched branches  : {len(uniq)} / {m.sum()}")
    print(f"  DIRECT rel-L2  median      : {np.median(direct):.3e}   mean {direct.mean():.3e}   p90 {np.percentile(direct,90):.3e}")
    print(f"  POST   rel-L2  median      : {np.median(post):.3e}   mean {post.mean():.3e}   p90 {np.percentile(post,90):.3e}")
    print(f"  lost (post>{LOST_TH:.0e})          : {lost.sum()}/{lost.size}  ({lost.mean()*100:.2f}%)")
    print(f"  converged (||F||<1e-9)     : {conv.mean()*100:.2f}%")
    print(f"  Newton steps (conv)        : median {np.median(steps[conv]):.1f}  mean {steps[conv].mean():.2f}  p90 {np.percentile(steps[conv],90):.0f}")
    print(f"  recovered mult / param     : mean {rec.mean():.3f}")
    for kk in sorted(set(a["k"][m].astype(int).tolist())):
        mk = a["k"][m].astype(int) == kk
        print(f"    k={kk:>2}: n={mk.sum():>4}  DIRECT med {np.median(direct[mk]):.2e}  POST med {np.median(post[mk]):.2e}  lost {(post[mk]>LOST_TH).mean()*100:5.2f}%")
    return {"direct_med": np.median(direct), "post_med": np.median(post),
            "lost_pct": lost.mean()*100, "conv_pct": conv.mean()*100,
            "steps_mean": steps[conv].mean(), "rec_mean": rec.mean(), "n_sol": int(m.sum())}


def main():
    w = load(WARM)
    if not os.path.exists(SCR):
        summarize(w, "WARM (Ex2 -> Ex3)")
        print(f"\n[scratch stats not found yet: {SCR}]")
        return
    s = load(SCR)
    kcommon = sorted(set(w["k"].astype(int).tolist()) & set(s["k"].astype(int).tolist()))
    print(f"common multiplicity set k in {kcommon} (used for the fair table)")
    W = summarize(w, "WARM  (Ex2 -> Ex3, cross-equation warm-start)", kcommon)
    S = summarize(s, "SCRATCH (fresh LoRA on pretrained Qwen)", kcommon)
    print("\n================ WARM vs SCRATCH (common k) ================")
    print(f"{'metric':30s} {'WARM':>14s} {'SCRATCH':>14s}")
    for name, kk, fmt in [("DIRECT rel-L2 median", "direct_med", "e"), ("POST rel-L2 median", "post_med", "e"),
                          ("lost % (post>1e-2)", "lost_pct", "f"), ("converged %", "conv_pct", "f"),
                          ("Newton steps (mean)", "steps_mean", "f"), ("recovered mult / param", "rec_mean", "f"),
                          ("matched branches (total)", "n_sol", "d")]:
        f = (lambda v: f"{v:.3e}") if fmt == "e" else ((lambda v: f"{v:.2f}") if fmt == "f" else (lambda v: f"{int(v)}"))
        print(f"{name:30s} {f(W[kk]):>14s} {f(S[kk]):>14s}")


if __name__ == "__main__":
    main()
