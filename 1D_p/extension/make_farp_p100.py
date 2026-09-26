"""Far extrapolation of Example 1 by seed + stepped march: p=50 and p=100.
Grey = GT continuation; dashed colour = model-seed stepped march. -> fig_farp_continuation.{pdf,png}"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,"."); sys.path.insert(0,"../code"); sys.path.insert(0,"../data_gen/cbmfem")
from gen_gt_extension import march, ZERO_NORM
DEDUP=1e-3
def rl2(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12))
def dedup(s):
    o=[]
    for u in s:
        if u is not None and not any(rl2(u,v)<DEDUP for v in o): o.append(u)
    return o
S=np.load("../data_gen/cbmfem/initial_S2.npy")
true18=[np.real(S[:,b]).astype(np.float64) for b in range(S.shape[1]) if np.linalg.norm(np.real(S[:,b]))>ZERO_NORM]
gen18=np.asarray(torch.load("p18seed_result.pt",map_location="cpu",weights_only=False)["gen18"],np.float64)
data={}
for P in [50.0,100.0]:
    gt=dedup([march(u,18.0,P,dp0=1.0) for u in true18])
    step=dedup([march(np.asarray(u,np.float64),18.0,P,dp0=1.0) for u in gen18])
    data[P]={"gt":gt,"step":step}; print(f"p={P:.0f}: GT={len(gt)} model-seed={len(step)}",flush=True)
torch.save(data, "farp_fields_p100.pt")
x=np.linspace(0,1,gen18.shape[1]); COL=plt.cm.viridis(np.linspace(0.05,0.9,7))
plt.rcParams.update({"font.size":15,"axes.linewidth":1.4})
fig,axs=plt.subplots(1,2,figsize=(11,4.0))
for ax,P in zip(axs,[50.0,100.0]):
    gt,step=data[P]["gt"],data[P]["step"]
    for g in gt: ax.plot(x,g,color="0.6",lw=4.0,zorder=1)
    order=sorted(range(len(step)),key=lambda i:np.linalg.norm(step[i]))
    for c,i in enumerate(order): ax.plot(x,step[i],color=COL[c%7],lw=1.8,ls="--",zorder=3)
    ax.set_title(f"$p={P:.0f}$  ({len(step)}/{len(gt)} branches)",fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_xlim(0,1)
axs[0].set_ylabel("$u(x)$")
axs[1].legend(handles=[Line2D([0],[0],color="0.6",lw=4,label="ground truth (continuation)"),
                       Line2D([0],[0],color=COL[3],lw=1.8,ls="--",label="model seed + stepped march")],
              fontsize=11,loc="upper right",framealpha=1.0)
fig.tight_layout()
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_farp_continuation.pdf")
fig.savefig(OUT,bbox_inches="tight"); fig.savefig(OUT.replace(".pdf",".png"),dpi=160,bbox_inches="tight")
print("saved",OUT)
