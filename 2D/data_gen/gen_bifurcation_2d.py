"""
bifurcation_diagram.py  —  Reproduce Fig. 11 from the paper.

Equation (61):
    -Delta u - u^2 = -s * sin(pi*x) * sin(pi*y),  u=0 on dOmega

Method: Pseudo-arclength (homotopy) continuation.
Mesh:   ell=3 (145 nodes).  Quadrature: 16-point (from quadrature.m).

Seed strategy:
  1. Fast Newton (maxiter=15) over many diverse starting points at s=1600.
     Covers all D4 representation sectors via:
       - sin(m*pi*x)*sin(n*pi*y) combinations (symmetric & asymmetric)
       - B1-type: sin(2x)sin(y) - sin(x)sin(2y)  (anti-diagonal symmetric)
       - B2-type: sin(2x)sin(y) + sin(x)sin(2y)  (diagonal symmetric)
       - Full random search
  2. D4 augmentation on found seeds.
  3. Full Newton (maxiter=50) to refine.

Continuation: backward AND forward from each unique seed at s=1600.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Companion_Method')
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2d_eq61_dataset')

# ── 16-point Gauss quadrature (quadrature.m) ────────────────────────────────────
_GP = np.array([
    [9.703785126946112e-3, 8.602401356562194e-1],
    [4.612207990645205e-2, 8.602401356562194e-1],
    [9.363778443732850e-2, 8.602401356562194e-1],
    [1.300560792168344e-1, 8.602401356562194e-1],
    [2.891208422438901e-2, 5.835904323689168e-1],
    [1.374191041345744e-1, 5.835904323689168e-1],
    [2.789904634965088e-1, 5.835904323689168e-1],
    [3.874974834066942e-1, 5.835904323689168e-1],
    [5.021012321136977e-2, 2.768430136381238e-1],
    [2.386486597314429e-1, 2.768430136381238e-1],
    [4.845083266304333e-1, 2.768430136381238e-1],
    [6.729468631505064e-1, 2.768430136381238e-1],
    [6.546699455501446e-2, 5.710419611451768e-2],
    [3.111645522443570e-1, 5.710419611451768e-2],
    [6.317312516411253e-1, 5.710419611451768e-2],
    [8.774288093304679e-1, 5.710419611451768e-2],
])
_GW = np.array([
    5.423225910525254e-3, 1.016725956447879e-2,
    1.016725956447879e-2, 5.423225910525254e-3,
    2.258404928236993e-2, 4.233972452174629e-2,
    4.233972452174629e-2, 2.258404928236993e-2,
    3.538806789808595e-2, 6.634421610704973e-2,
    6.634421610704973e-2, 3.538806789808595e-2,
    2.356836819338233e-2, 4.418508852236173e-2,
    4.418508852236173e-2, 2.356836819338233e-2,
])
_PHI = np.stack([_GP[:,0], _GP[:,1], 1-_GP[:,0]-_GP[:,1]], axis=1)  # (16,3)

# ── Mesh ─────────────────────────────────────────────────────────────────────────
def load_col(fname, skip_cols=1, dtype=float):
    raw = np.loadtxt(fname)
    if raw.ndim == 1: raw = raw[np.newaxis, :]
    return raw[:, skip_cols:].astype(dtype)

def refine(coord, elem, diri_set):
    n = len(coord); edge_map = {}
    new_coords = list(coord); new_diri = set(diri_set); new_elem = []; nn = n
    for tri in elem:
        v = [int(tri[0]),int(tri[1]),int(tri[2])]; mids = []
        for a,b in [(v[0],v[1]),(v[1],v[2]),(v[0],v[2])]:
            key=(min(a,b),max(a,b))
            if key not in edge_map:
                edge_map[key]=nn; new_coords.append((coord[a]+coord[b])/2)
                if a in new_diri and b in new_diri: new_diri.add(nn)
                nn+=1
            mids.append(edge_map[key])
        m01,m12,m02=mids
        new_elem.extend([[v[0],m01,m02],[m01,v[1],m12],[m02,m12,v[2]],[m01,m12,m02]])
    rows,cols,vals=list(range(n)),list(range(n)),[1.0]*n
    for (a,b),mid in edge_map.items():
        rows+=[mid,mid]; cols+=[a,b]; vals+=[0.5,0.5]
    return (np.array(new_coords,dtype=float), np.array(new_elem,dtype=int),
            np.array(sorted(new_diri),dtype=int),
            sp.csr_matrix((vals,(rows,cols)),shape=(nn,n)))

# ── FEM (16-pt Gauss) ─────────────────────────────────────────────────────────────
def _geom(coord,elem):
    v0,v1,v2=elem[:,0],elem[:,1],elem[:,2]
    B00=coord[v0,0]-coord[v2,0]; B01=coord[v1,0]-coord[v2,0]
    B10=coord[v0,1]-coord[v2,1]; B11=coord[v1,1]-coord[v2,1]
    return v0,v1,v2,np.abs(B00*B11-B01*B10)

def build_stiffness(coord,elem):
    n=len(coord); v0,v1,v2=elem[:,0],elem[:,1],elem[:,2]
    B00=coord[v0,0]-coord[v2,0]; B01=coord[v1,0]-coord[v2,0]
    B10=coord[v0,1]-coord[v2,1]; B11=coord[v1,1]-coord[v2,1]
    detB=np.abs(B00*B11-B01*B10)
    Bi00=B11/detB; Bi01=-B01/detB; Bi10=-B10/detB; Bi11=B00/detB
    gpx=np.stack([Bi00,Bi10,-Bi00-Bi10],axis=1)
    gpy=np.stack([Bi01,Bi11,-Bi01-Bi11],axis=1)
    Aloc=(detB/2)[:,None,None]*(gpx[:,:,None]*gpx[:,None,:]+gpy[:,:,None]*gpy[:,None,:])
    verts=np.stack([v0,v1,v2],axis=1)
    R=np.repeat(verts,3,axis=1).ravel(); C=np.tile(verts,(1,3)).ravel()
    return sp.csr_matrix((Aloc.ravel(),(R,C)),shape=(n,n))

def assemble_F_J(coord,elem,u,s,A):
    n=len(coord); v0,v1,v2,detB=_geom(coord,elem)
    b=s*np.sin(np.pi*coord[:,0])*np.sin(np.pi*coord[:,1])
    ue=np.stack([u[v0],u[v1],u[v2]],axis=1); fe=np.stack([b[v0],b[v1],b[v2]],axis=1)
    uh=ue@_PHI.T; fh=fe@_PHI.T
    wt=detB[:,None]*_GW[None,:]
    intgd=wt*(uh**2-fh)
    ce=(intgd[:,:,None]*_PHI[None,:,:]).sum(axis=1)
    coeff=np.zeros(n)
    np.add.at(coeff,v0,ce[:,0]); np.add.at(coeff,v1,ce[:,1]); np.add.at(coeff,v2,ce[:,2])
    F=A@u-coeff
    ji=wt*uh
    je=-2*np.einsum('eg,gk,gi->eki',ji,_PHI,_PHI)
    verts=np.stack([v0,v1,v2],axis=1)
    R=np.repeat(verts,3,axis=1).ravel(); C=np.tile(verts,(1,3)).ravel()
    J=A+sp.csr_matrix((je.ravel(),(R,C)),shape=(n,n))
    return F,J

def assemble_dFds(coord,elem):
    n=len(coord); v0,v1,v2,detB=_geom(coord,elem)
    fn=np.sin(np.pi*coord[:,0])*np.sin(np.pi*coord[:,1])
    fe=np.stack([fn[v0],fn[v1],fn[v2]],axis=1); fh=fe@_PHI.T
    wt=detB[:,None]*_GW[None,:]
    ce=((wt*fh)[:,:,None]*_PHI[None,:,:]).sum(axis=1)
    dFds=np.zeros(n)
    np.add.at(dFds,v0,ce[:,0]); np.add.at(dFds,v1,ce[:,1]); np.add.at(dFds,v2,ce[:,2])
    return dFds

def newton(coord,elem,u0,s,A,FN,dbc,tol=1e-9,maxiter=50):
    u=u0.copy()
    for _ in range(maxiter):
        F,J=assemble_F_J(coord,elem,u,s,A); F[dbc]=0.0
        if np.max(np.abs(F[FN]))<tol: return u,True
        try: u[FN]-=spla.spsolve(J[np.ix_(FN,FN)].tocsc(),F[FN])
        except: return u,False
    F,_=assemble_F_J(coord,elem,u,s,A); F[dbc]=0.0
    return u,np.max(np.abs(F[FN]))<tol

def newton_fast(coord,elem,u0,s,A,FN,dbc):
    """Fast Newton for seed search: loose tol, few iterations."""
    return newton(coord,elem,u0,s,A,FN,dbc,tol=1e-7,maxiter=15)

# ── Arclength continuation ────────────────────────────────────────────────────────
def get_tangent(J_ff,dFds_f,tu_prev,ts_prev):
    try: tu=spla.spsolve(J_ff.tocsc(),-dFds_f)
    except: return None,None
    nt=np.sqrt(np.dot(tu,tu)+1.0); tu/=nt; ts=1.0/nt
    if np.dot(tu,tu_prev)+ts*ts_prev<0: tu=-tu; ts=-ts
    return tu,ts

def arclength_step(coord,elem,A,FN,dbc,dFds_f,u0,s0,tu0,ts0,dl,tol=1e-8,maxiter=15):
    u=u0.copy(); u[FN]+=dl*tu0; s=float(s0+dl*ts0)
    for _ in range(maxiter):
        F,J=assemble_F_J(coord,elem,u,s,A); F[dbc]=0.0; F_f=F[FN]
        Jfc=J[np.ix_(FN,FN)].tocsc()
        N=float(np.dot(u[FN]-u0[FN],tu0)+(s-s0)*ts0-dl)
        try: p=spla.spsolve(Jfc,-F_f); q=spla.spsolve(Jfc,dFds_f)
        except: return u0.copy(),s0,tu0.copy(),ts0,False
        denom=ts0-float(np.dot(tu0,q))
        if abs(denom)<1e-14: return u0.copy(),s0,tu0.copy(),ts0,False
        ds=(-N-float(np.dot(tu0,p)))/denom
        du=p-q*ds; u[FN]+=du; s=float(s+ds)
        if max(np.max(np.abs(du)),abs(ds))<tol: break
    F_chk,J_chk=assemble_F_J(coord,elem,u,s,A); F_chk[dbc]=0.0
    if np.max(np.abs(F_chk[FN]))>1e-6: return u0.copy(),s0,tu0.copy(),ts0,False
    tu_new,ts_new=get_tangent(J_chk[np.ix_(FN,FN)],dFds_f,tu0,ts0)
    if tu_new is None: return u0.copy(),s0,tu0.copy(),ts0,False
    return u.copy(),s,tu_new,ts_new,True

def trace_branch(coord,elem,A,FN,dbc,dFds_f,u_seed,s_seed,direction,
                 s_min=-400.,s_max=1900.,dl_init=10.,dl_min=0.02,dl_max=40.,
                 max_steps=5000):
    cidx=np.argmin(np.sum((coord-[0.5,0.5])**2,axis=1))
    F0,J0=assemble_F_J(coord,elem,u_seed,s_seed,A)
    try: tu0=spla.spsolve(J0[np.ix_(FN,FN)].tocsc(),-dFds_f)
    except: return np.array([float(s_seed)]),np.array([float(u_seed[cidx])])
    nt=np.sqrt(np.dot(tu0,tu0)+1.0); tu0/=nt; ts0=1.0/nt
    if direction<0: tu0=-tu0; ts0=-ts0
    s_list=[float(s_seed)]; uc_list=[float(u_seed[cidx])]
    u,s,tu,ts=u_seed.copy(),float(s_seed),tu0.copy(),float(ts0)
    dl=dl_init
    for _ in range(max_steps):
        if s<s_min or s>s_max or abs(uc_list[-1])>600: break
        u_n,s_n,tu_n,ts_n,ok=arclength_step(coord,elem,A,FN,dbc,dFds_f,u,s,tu,ts,dl)
        if not ok: dl*=0.5;
        if dl<dl_min: break
        if not ok: continue
        u,s,tu,ts=u_n,s_n,tu_n,ts_n
        s_list.append(s); uc_list.append(float(u[cidx])); dl=min(dl*1.2,dl_max)
    return np.array(s_list),np.array(uc_list)

# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    print("="*65, flush=True)
    print("Bifurcation diagram  Fig. 11  (16-pt Gauss, ell=3)", flush=True)
    print("="*65, flush=True)

    # Build mesh
    print("\n[1] Mesh...", flush=True)
    coord0=load_col(os.path.join(BASE,'coordinates.dat'),skip_cols=1)
    elem0=load_col(os.path.join(BASE,'elements.dat'),skip_cols=1,dtype=int)-1
    diri0=load_col(os.path.join(BASE,'dirichlet.dat'),skip_cols=0,dtype=int).flatten()-1
    d0=set(diri0)
    c1,e1,d1,_=refine(coord0,elem0,d0)
    c2,e2,d2,_=refine(c1,e1,d1)
    c3,e3,d3,_=refine(c2,e2,d2)
    n3=len(c3); FN3=np.array([i for i in range(n3) if i not in set(d3)],dtype=int)
    dbc3=np.array(sorted(d3),dtype=int); A3=build_stiffness(c3,e3)
    dFds=assemble_dFds(c3,e3); dFds_f=dFds[FN3]
    cidx=np.argmin(np.sum((c3-[0.5,0.5])**2,axis=1))
    print(f"   {n3} nodes, {len(FN3)} free, center={c3[cidx]}", flush=True)
    assert n3==145

    rng=np.random.default_rng(42)

    # ── Seed finding at s=1600 ────────────────────────────────────────────────
    print(f"\n[2] Finding seeds at s=1600 (diverse D4-sector coverage)...", flush=True)
    seeds=[]

    def try_fast(u0, s=1600):
        u0=u0.copy(); u0[dbc3]=0.0
        u,conv=newton_fast(c3,e3,u0,s,A3,FN3,dbc3)
        if conv and not any(np.max(np.abs(sv-u))<1e-3 for sv in seeds):
            # Refine with full Newton
            u,conv=newton(c3,e3,u,s,A3,FN3,dbc3)
            if conv and not any(np.max(np.abs(sv-u))<1e-4 for sv in seeds):
                seeds.append(u.copy())

    # Build basis modes
    f11=np.sin(np.pi*c3[:,0])*np.sin(np.pi*c3[:,1])
    f21=np.sin(2*np.pi*c3[:,0])*np.sin(np.pi*c3[:,1])
    f12=np.sin(np.pi*c3[:,0])*np.sin(2*np.pi*c3[:,1])
    f22=np.sin(2*np.pi*c3[:,0])*np.sin(2*np.pi*c3[:,1])
    f31=np.sin(3*np.pi*c3[:,0])*np.sin(np.pi*c3[:,1])
    f13=np.sin(np.pi*c3[:,0])*np.sin(3*np.pi*c3[:,1])
    f32=np.sin(3*np.pi*c3[:,0])*np.sin(2*np.pi*c3[:,1])
    f23=np.sin(2*np.pi*c3[:,0])*np.sin(3*np.pi*c3[:,1])
    f33=np.sin(3*np.pi*c3[:,0])*np.sin(3*np.pi*c3[:,1])

    # A1 sector (D4-symmetric)
    print("   A1 (symmetric)...", flush=True)
    for sc in np.concatenate([np.geomspace(0.5,300,40),-np.geomspace(0.5,300,40)]):
        try_fast(sc*f11)
    for sc in np.concatenate([np.geomspace(1,300,20),-np.geomspace(1,300,20)]):
        try_fast(sc*f22); try_fast(sc*f33)

    # B1 sector: anti-symmetric under diagonal (x<->y) reflection
    # B1 functions: f21 - f12,  f32 - f23, etc.
    print("   B1 (anti-diagonal)...", flush=True)
    b1_base = f21 - f12   # changes sign under (x,y)->(y,x)
    b1_b    = f32 - f23
    for sc in np.concatenate([np.geomspace(1,500,40),-np.geomspace(1,500,40)]):
        try_fast(sc*b1_base)
        try_fast(sc*b1_b)
        try_fast(sc*f11 + 0.5*sc*b1_base)
        try_fast(sc*f11 - 0.5*sc*b1_base)

    # B2 sector: symmetric under diagonal, anti under axis
    # B2 functions: f21 + f12
    print("   B2 (diagonal-symmetric)...", flush=True)
    b2_base = f21 + f12
    b2_b    = f32 + f23
    for sc in np.concatenate([np.geomspace(1,500,40),-np.geomspace(1,500,40)]):
        try_fast(sc*b2_base)
        try_fast(sc*b2_b)
        try_fast(sc*f11 + 0.5*sc*b2_base)
        try_fast(sc*f11 - 0.5*sc*b2_base)

    # E sector: pure x or y asymmetry
    print("   E (one-axis asymmetry)...", flush=True)
    for sc in np.concatenate([np.geomspace(1,500,40),-np.geomspace(1,500,40)]):
        try_fast(sc*f21)
        try_fast(sc*f12)
        try_fast(sc*f31); try_fast(sc*f13)
        try_fast(sc*f11 + sc*f21)
        try_fast(sc*f11 + sc*f12)
        try_fast(sc*f11 - sc*f21)
        try_fast(sc*f11 - sc*f12)

    print(f"   After targeted starts: {len(seeds)} solutions", flush=True)

    # Mixed 2D combinations for remaining branches
    print("   2D grid search...", flush=True)
    all_modes = [f11, f21, f12, f22, f31, f13, f32, f23, f33]
    for c1v in np.linspace(-200, 200, 15):
        for c2v in np.linspace(-200, 200, 15):
            u0 = c1v*f21 + c2v*f12
            try_fast(u0)
    for c1v in np.linspace(-200, 200, 10):
        for c2v in np.linspace(-200, 200, 10):
            u0 = c1v*f11 + c2v*b1_base
            try_fast(u0)
            u0 = c1v*f11 + c2v*b2_base
            try_fast(u0)

    print(f"   After 2D grid: {len(seeds)} solutions", flush=True)

    # Random search
    print("   Random search...", flush=True)
    for _ in range(500):
        coeffs=rng.uniform(-200,200,len(all_modes))
        u0=sum(c*f for c,f in zip(coeffs,all_modes))
        try_fast(u0)

    print(f"   After random: {len(seeds)} solutions", flush=True)

    # D4 augmentation
    print("   D4 augmentation...", flush=True)
    tree=cKDTree(c3)
    d4_ops=[
        lambda c: np.stack([c[:,0],c[:,1]],axis=1),
        lambda c: np.stack([1-c[:,0],c[:,1]],axis=1),
        lambda c: np.stack([c[:,0],1-c[:,1]],axis=1),
        lambda c: np.stack([c[:,1],c[:,0]],axis=1),
        lambda c: np.stack([1-c[:,0],1-c[:,1]],axis=1),
        lambda c: np.stack([1-c[:,1],1-c[:,0]],axis=1),
        lambda c: np.stack([c[:,1],1-c[:,0]],axis=1),
        lambda c: np.stack([1-c[:,1],c[:,0]],axis=1),
    ]
    for T in d4_ops:
        _,perm=tree.query(T(c3))
        for uk in list(seeds):
            u_rot=np.empty(n3); u_rot[perm]=uk
            try_fast(u_rot)

    n_seeds=len(seeds)
    uc_vals=sorted([float(u[cidx]) for u in seeds])
    print(f"\n   Total seeds: {n_seeds}", flush=True)
    print(f"   u_center values at s=1600:", flush=True)
    for v in uc_vals:
        print(f"     {v:+.4f}", flush=True)

    if n_seeds==0:
        print("ERROR: no seeds!", flush=True); sys.exit(1)

    # Deduplicate by u_center (D4 related → same u_center → same branch)
    unique_seeds = []
    seen_uc = []
    for u in seeds:
        uc = float(u[cidx])
        if not any(abs(uc-v)<0.2 for v in seen_uc):
            seen_uc.append(uc)
            unique_seeds.append(u)

    print(f"   Unique branches (by u_center): {len(unique_seeds)}", flush=True)
    for v in sorted(seen_uc):
        print(f"     u_center = {v:+.4f}", flush=True)

    # ── Arclength continuation (both directions) ──────────────────────────────
    print(f"\n[3] Pseudo-arclength continuation...", flush=True)
    print(f"    {'#':>3}  {'dir':>4}  {'pts':>5}  {'s_min':>8}  {'s_max':>8}",
          flush=True)
    print("    "+"-"*38, flush=True)

    branch_data = []   # list of (label, s_arr, uc_arr)
    t0=time.time()

    for i, u_seed in enumerate(unique_seeds):
        uc0=float(u_seed[cidx])
        for direction, dname in [(-1,'bwd'),(+1,'fwd')]:
            s_arr,uc_arr=trace_branch(
                c3,e3,A3,FN3,dbc3,dFds_f, u_seed,1600.0,
                direction=direction,
                s_min=-400.,s_max=1900.,
                dl_init=10.,dl_min=0.02,dl_max=40.,max_steps=5000)
            if len(s_arr)>3:
                branch_data.append((f'B{i+1} {dname} uc={uc0:.1f}',s_arr,uc_arr,i))
                print(f"    {i+1:>3}  {dname:>4}  {len(s_arr):>5}"
                      f"  {s_arr.min():>8.1f}  {s_arr.max():>8.1f}", flush=True)

    print(f"\n   Done in {time.time()-t0:.1f}s.  {len(branch_data)} segments.",
          flush=True)

    # ── Save full branch data for new-style replot ─────────────────────────────
    import torch as _t
    PAPER_OUT = os.path.dirname(os.path.abspath(__file__))
    _t.save([{'label': l, 's': _t.tensor(np.asarray(sa, float)),
              'uc': _t.tensor(np.asarray(ua, float)), 'bi': int(bi)}
             for (l, sa, ua, bi) in branch_data],
            os.path.join(PAPER_OUT, 'bifurcation_branches_full.pt'))
    print(f"   Saved full branch data -> {PAPER_OUT}/bifurcation_branches_full.pt", flush=True)

    # ── Plot ─────────────────────────────────────────────────────────────────
    print(f"\n[4] Plotting...", flush=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728',
            '#9467bd','#8c564b','#e377c2','#bcbd22']
    drawn=set()
    for label,s_arr,uc_arr,bi in branch_data:
        color=colors[bi%len(colors)]
        lbl=f'Branch {bi+1}  (u_c={seen_uc[bi]:+.1f})' if bi not in drawn else None
        drawn.add(bi)
        ax.plot(s_arr,uc_arr,color=color,linewidth=1.8,label=lbl,alpha=0.9)

    ax.set_xlabel('$s$',fontsize=14)
    ax.set_ylabel('$u(1/2,\\ 1/2)$',fontsize=14)
    ax.set_title('Bifurcation diagram of Eq.(61)  (Fig. 11 reproduction)',fontsize=12)
    ax.set_xlim(-420,1920); ax.axvline(0,color='gray',ls=':',lw=0.7,alpha=0.5)
    ax.axhline(0,color='gray',ls=':',lw=0.7,alpha=0.5)
    ax.axvline(635,color='silver',ls='--',lw=0.8,alpha=0.5,label='bifurc. s≈635')
    ax.grid(True,alpha=0.2); ax.legend(loc='upper left',fontsize=8,framealpha=0.8)
    plt.tight_layout()
    fig_path=os.path.join(OUT,'fig11_bifurcation.png')
    plt.savefig(fig_path,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"   Saved: {fig_path}", flush=True)
    print(f"\nDone. {len(unique_seeds)} branches.", flush=True)

if __name__=='__main__':
    main()
