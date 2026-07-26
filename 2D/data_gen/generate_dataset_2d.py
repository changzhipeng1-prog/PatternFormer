"""
generate_dataset_2d.py
======================
Generate the 2D PDE dataset for Eq.(61):
    -Delta u - u^2 = -s * sin(pi*x) * sin(pi*y),  u=0 on boundary

Strategy:
  1. Find all solutions at s=1600 (seed search + D4 augmentation → 10 solutions)
  2. March downward: s=1599, 1598, ..., using previous s solutions as Newton init
  3. Verify each solution: residual ||F(u,s)||_inf < 1e-9
  4. Save as p_solutions_lookup.pt  (compatible with v2 dataset format)

Output format:
  {
    "p_values":       FloatTensor [N_s]          # s values with at least 1 solution
    "solutions_by_p": list of FloatTensor[K_i, 145]  # raw FEM node values
    "coord":          FloatTensor [145, 2]       # mesh node coordinates
    "elem":           LongTensor  [256, 3]       # triangle connectivity
    "s_min": float, "s_max": float,
    "n_total_solutions": int,
    "note": str,
  }

Node ordering: ell=3 mesh (145 nodes, 256 triangles).
Solution vector: u[i] = u(coord[i]) for i=0..144.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import torch
import os, sys, time

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Companion_Method')
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2d_eq61_dataset')

# ── 16-point Gauss quadrature (quadrature.m) ─────────────────────────────────
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

# ── Mesh utilities ────────────────────────────────────────────────────────────
def load_col(fname, skip_cols=1, dtype=float):
    raw = np.loadtxt(fname)
    if raw.ndim == 1: raw = raw[np.newaxis,:]
    return raw[:,skip_cols:].astype(dtype)

def refine(coord, elem, diri_set):
    n = len(coord); edge_map = {}
    new_coords = list(coord); new_diri = set(diri_set); new_elem = []; nn = n
    for tri in elem:
        v = [int(tri[0]),int(tri[1]),int(tri[2])]; mids = []
        for a,b in [(v[0],v[1]),(v[1],v[2]),(v[0],v[2])]:
            key = (min(a,b),max(a,b))
            if key not in edge_map:
                edge_map[key] = nn
                new_coords.append((coord[a]+coord[b])/2)
                if a in new_diri and b in new_diri: new_diri.add(nn)
                nn += 1
            mids.append(edge_map[key])
        m01,m12,m02 = mids
        new_elem.extend([[v[0],m01,m02],[m01,v[1],m12],
                         [m02,m12,v[2]], [m01,m12,m02]])
    rows,cols,vals = list(range(n)),list(range(n)),[1.0]*n
    for (a,b),mid in edge_map.items():
        rows += [mid,mid]; cols += [a,b]; vals += [0.5,0.5]
    return (np.array(new_coords,dtype=float), np.array(new_elem,dtype=int),
            np.array(sorted(new_diri),dtype=int),
            sp.csr_matrix((vals,(rows,cols)),shape=(nn,n)))

# ── FEM assembly ──────────────────────────────────────────────────────────────
def _geom(coord,elem):
    v0,v1,v2 = elem[:,0],elem[:,1],elem[:,2]
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
    Aloc=(detB/2)[:,None,None]*(gpx[:,:,None]*gpx[:,None,:]+
                                 gpy[:,:,None]*gpy[:,None,:])
    verts=np.stack([v0,v1,v2],axis=1)
    R=np.repeat(verts,3,axis=1).ravel(); C=np.tile(verts,(1,3)).ravel()
    return sp.csr_matrix((Aloc.ravel(),(R,C)),shape=(n,n))

def assemble_F_J(coord,elem,u,s,A):
    n=len(coord); v0,v1,v2,detB=_geom(coord,elem)
    b=s*np.sin(np.pi*coord[:,0])*np.sin(np.pi*coord[:,1])
    ue=np.stack([u[v0],u[v1],u[v2]],axis=1)
    fe=np.stack([b[v0],b[v1],b[v2]],axis=1)
    uh=ue@_PHI.T; fh=fe@_PHI.T
    wt=detB[:,None]*_GW[None,:]
    intgd=wt*(uh**2-fh)
    ce=(intgd[:,:,None]*_PHI[None,:,:]).sum(axis=1)
    coeff=np.zeros(n)
    np.add.at(coeff,v0,ce[:,0])
    np.add.at(coeff,v1,ce[:,1])
    np.add.at(coeff,v2,ce[:,2])
    F=A@u-coeff
    ji=wt*uh
    je=-2*np.einsum('eg,gk,gi->eki',ji,_PHI,_PHI)
    verts=np.stack([v0,v1,v2],axis=1)
    R=np.repeat(verts,3,axis=1).ravel(); C=np.tile(verts,(1,3)).ravel()
    J=A+sp.csr_matrix((je.ravel(),(R,C)),shape=(n,n))
    return F,J

# ── Newton solver ─────────────────────────────────────────────────────────────
NEWTON_TOL = 1e-9

def newton(coord,elem,u0,s,A,FN,dbc,tol=NEWTON_TOL,maxiter=50):
    u=u0.copy()
    for _ in range(maxiter):
        F,J=assemble_F_J(coord,elem,u,s,A)
        F[dbc]=0.0
        if np.max(np.abs(F[FN]))<tol: return u,True
        try:
            u[FN] -= spla.spsolve(J[np.ix_(FN,FN)].tocsc(), F[FN])
        except Exception:
            return u,False
    F,_=assemble_F_J(coord,elem,u,s,A); F[dbc]=0.0
    return u, np.max(np.abs(F[FN]))<tol

def verify(coord,elem,u,s,A,FN, tol=NEWTON_TOL):
    """Return residual; True if < tol."""
    F,_=assemble_F_J(coord,elem,u,s,A); F[dbc3]=0.0
    res=float(np.max(np.abs(F[FN])))
    return res, res<tol

def dedup(sols, tol=1e-4):
    out=[]
    for u in sols:
        if not any(np.max(np.abs(u-v))<tol for v in out):
            out.append(u)
    return out

# ── D4 augmentation ───────────────────────────────────────────────────────────
def d4_augment(seeds, coord, elem, A, FN, dbc, s):
    """Apply all 8 D4 ops to seeds, Newton-refine each, deduplicate."""
    tree=cKDTree(coord); n=len(coord)
    ops=[
        lambda c: np.stack([c[:,0],   c[:,1]  ],axis=1),  # identity
        lambda c: np.stack([1-c[:,0], c[:,1]  ],axis=1),  # reflect x
        lambda c: np.stack([c[:,0],   1-c[:,1]],axis=1),  # reflect y
        lambda c: np.stack([c[:,1],   c[:,0]  ],axis=1),  # swap x,y
        lambda c: np.stack([1-c[:,0], 1-c[:,1]],axis=1),  # 180° rot
        lambda c: np.stack([1-c[:,1], 1-c[:,0]],axis=1),  # anti-diag reflect
        lambda c: np.stack([c[:,1],   1-c[:,0]],axis=1),  # 90° rot
        lambda c: np.stack([1-c[:,1], c[:,0]  ],axis=1),  # 270° rot
    ]
    all_sols=list(seeds)
    for T in ops:
        _,perm=tree.query(T(coord))
        for uk in list(seeds):
            u_rot=np.empty(n); u_rot[perm]=uk; u_rot[dbc]=0.0
            u,conv=newton(coord,elem,u_rot,s,A,FN,dbc)
            if conv:
                all_sols.append(u)
    return dedup(all_sols)

# ── Canonicalize: sort by (u_center, x-asymmetry) ────────────────────────────
def canonicalize(sols, cidx, coord):
    """Sort solutions: primary by u_center asc, secondary by x-asymmetry."""
    if not sols: return sols
    def key(u):
        uc = float(u[cidx])
        # x-asymmetry: integral of u(x,y)*sign(x-0.5) ≈ mean of left vs right
        x_asym = float(np.mean(u[coord[:,0]<0.5]) - np.mean(u[coord[:,0]>0.5]))
        return (round(uc, 1), x_asym)   # round u_center to group D4 families
    return sorted(sols, key=key)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start=time.time()
    print("="*65)
    print("2D PDE Dataset Generation — Eq.(61)")
    print("="*65)

    # ── Build mesh ─────────────────────────────────────────────────────────
    print("\n[1] Building ell=3 mesh...")
    coord0=load_col(os.path.join(BASE,'coordinates.dat'),skip_cols=1)
    elem0 =load_col(os.path.join(BASE,'elements.dat'),skip_cols=1,dtype=int)-1
    diri0 =load_col(os.path.join(BASE,'dirichlet.dat'),skip_cols=0,dtype=int).flatten()-1
    d0=set(diri0)
    c1,e1,d1,_=refine(coord0,elem0,d0)
    c2,e2,d2,_=refine(c1,e1,d1)
    global coord, elem, dbc3, FN3, A3, cidx
    coord,elem,d3,_=refine(c2,e2,d2)
    n=len(coord)
    dbc3=np.array(sorted(d3),dtype=int)
    FN3=np.array([i for i in range(n) if i not in d3],dtype=int)
    A3=build_stiffness(coord,elem)
    cidx=np.argmin(np.sum((coord-[0.5,0.5])**2,axis=1))
    print(f"   {n} nodes, {len(FN3)} free, center={coord[cidx]}")
    assert n==145

    # ── Find all solutions at s=1600 ───────────────────────────────────────
    print("\n[2] Finding all solutions at s=1600...")
    seeds_1600=[]

    def try_seed(u0, s=1600, tol_fast=1e-7):
        u0=u0.copy(); u0[dbc3]=0.0
        u,conv=newton(coord,elem,u0,s,A3,FN3,dbc3,tol=tol_fast,maxiter=15)
        if conv and not any(np.max(np.abs(sv-u))<1e-3 for sv in seeds_1600):
            u,conv=newton(coord,elem,u,s,A3,FN3,dbc3)
            if conv and not any(np.max(np.abs(sv-u))<1e-4 for sv in seeds_1600):
                seeds_1600.append(u.copy())

    f11=np.sin(np.pi*coord[:,0])*np.sin(np.pi*coord[:,1])
    f21=np.sin(2*np.pi*coord[:,0])*np.sin(np.pi*coord[:,1])
    f12=np.sin(np.pi*coord[:,0])*np.sin(2*np.pi*coord[:,1])
    f22=np.sin(2*np.pi*coord[:,0])*np.sin(2*np.pi*coord[:,1])
    f32=np.sin(3*np.pi*coord[:,0])*np.sin(2*np.pi*coord[:,1])
    f23=np.sin(2*np.pi*coord[:,0])*np.sin(3*np.pi*coord[:,1])
    b1=f21-f12; b2=f21+f12

    rng=np.random.default_rng(42)
    for sc in np.concatenate([np.geomspace(0.5,300,40),-np.geomspace(0.5,300,40)]):
        try_seed(sc*f11); try_seed(sc*f22)
    for sc in np.concatenate([np.geomspace(1,500,40),-np.geomspace(1,500,40)]):
        try_seed(sc*b1); try_seed(sc*b2)
        try_seed(sc*f21); try_seed(sc*f12)
        try_seed(sc*f11+0.5*sc*b1); try_seed(sc*f11-0.5*sc*b1)
        try_seed(sc*f11+0.5*sc*b2); try_seed(sc*f11-0.5*sc*b2)
        try_seed(sc*(f32-f23)); try_seed(sc*(f32+f23))
    for _ in range(300):
        coeffs=rng.uniform(-200,200,6)
        try_seed(coeffs[0]*f11+coeffs[1]*f21+coeffs[2]*f12+
                 coeffs[3]*f22+coeffs[4]*f32+coeffs[5]*f23)

    print(f"   Canonical seeds: {len(seeds_1600)}")

    # D4 augment → get full orbit
    all_1600=d4_augment(seeds_1600,coord,elem,A3,FN3,dbc3,1600)
    all_1600=canonicalize(all_1600,cidx,coord)
    print(f"   After D4 augment: {len(all_1600)} solutions")
    print(f"   u_center values: {[round(u[cidx],3) for u in all_1600]}")

    # Verify all s=1600 solutions
    for u in all_1600:
        res,ok=verify(coord,elem,u,1600,A3,FN3)
        assert ok, f"s=1600 solution FAILED verification: res={res:.2e}"
    print(f"   All s=1600 solutions verified (residual < {NEWTON_TOL:.0e})")

    # ── March downward ─────────────────────────────────────────────────────
    print(f"\n[3] Marching from s=1599 down to s=-200...")
    print(f"    {'s':>6}  {'#sols':>6}  {'res_max':>10}  {'time':>6}")
    print("    "+"-"*34)

    solutions_by_s = {1600: all_1600}
    empty_streak = 0  # consecutive s values with 0 solutions

    for s in range(1599, -201, -1):
        t0=time.time()
        prev=solutions_by_s[s+1]
        if not prev:
            empty_streak+=1
            if empty_streak>50:  # stop if no solutions for 50 consecutive steps
                print(f"    No solutions for 50 consecutive s values; stopping at s={s+1}.")
                break
            solutions_by_s[s]=[]
            continue
        empty_streak=0

        curr=[]; res_vals=[]
        for u_prev in prev:
            u,conv=newton(coord,elem,u_prev,s,A3,FN3,dbc3)
            if conv:
                # Verify residual
                res,ok=verify(coord,elem,u,s,A3,FN3)
                if ok:
                    curr.append(u)
                    res_vals.append(res)
                else:
                    print(f"    WARNING: s={s} solution converged but verification "
                          f"failed (res={res:.2e}), discarded.")

        curr=dedup(curr)
        curr=canonicalize(curr,cidx,coord)
        solutions_by_s[s]=curr

        # Print progress every 50 steps or when solution count changes
        prev_count=len(solutions_by_s.get(s+1,[]))
        if s%50==0 or len(curr)!=prev_count or s>1590:
            res_max=max(res_vals) if res_vals else 0.0
            print(f"    {s:>6}  {len(curr):>6}  {res_max:>10.2e}  {time.time()-t0:.2f}s",
                  flush=True)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n[4] Summary:")
    s_vals_with_sols=[s for s,sols in solutions_by_s.items() if sols]
    s_min_data=min(s_vals_with_sols); s_max_data=max(s_vals_with_sols)
    total_sols=sum(len(v) for v in solutions_by_s.values())
    print(f"   s range with solutions: [{s_min_data}, {s_max_data}]")
    print(f"   Total (s, solution) pairs: {total_sols}")
    print(f"   Total s values with ≥1 solution: {len(s_vals_with_sols)}")

    # Count distribution of solution counts
    from collections import Counter
    cnt=Counter(len(v) for v in solutions_by_s.values() if v)
    for k in sorted(cnt): print(f"     {k} solutions: {cnt[k]} s-values")

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\n[5] Saving dataset...")

    # Build lookup in v2 format
    p_values_list=[]
    solutions_by_p_list=[]
    for s in sorted(s for s in solutions_by_s if solutions_by_s[s]):
        sols=solutions_by_s[s]
        p_values_list.append(float(s))
        # Stack solutions: [K, 145]
        sol_tensor=torch.tensor(np.stack(sols,axis=0),dtype=torch.float64)
        solutions_by_p_list.append(sol_tensor)

    lookup={
        "p_values":       torch.tensor(p_values_list,dtype=torch.float32),
        "solutions_by_p": solutions_by_p_list,
        "coord":          torch.tensor(coord,dtype=torch.float64),  # [145,2]
        "elem":           torch.tensor(elem, dtype=torch.long),     # [256,3]
        "dbc":            torch.tensor(dbc3, dtype=torch.long),
        "free_nodes":     torch.tensor(FN3,  dtype=torch.long),
        "s_min":          float(s_min_data),
        "s_max":          float(s_max_data),
        "n_total_solutions": total_sols,
        "solution_dim":   145,
        "note": ("2D Eq.(61) dataset. March method from s=1600. "
                 "solutions_by_p[i] is [K_i, 145] float64 FEM node values "
                 "sorted by (u_center, x_asymmetry). "
                 "All solutions verified in float64: residual < 1e-9."),
    }

    out_path=os.path.join(OUT,'p_solutions_lookup_2d.pt')
    torch.save(lookup, out_path)
    print(f"   Saved: {out_path}")

    # Quick sanity check on saved file
    loaded=torch.load(out_path,weights_only=False)
    assert len(loaded["p_values"])==len(loaded["solutions_by_p"])
    print(f"   Sanity check passed: {len(loaded['p_values'])} s-values loaded.")

    # Print a few sample entries
    print(f"\n   Sample entries:")
    for i in [0, len(p_values_list)//4, len(p_values_list)//2,
              3*len(p_values_list)//4, -1]:
        s_=int(p_values_list[i]); k_=solutions_by_p_list[i].shape[0]
        ucs=[round(float(solutions_by_p_list[i][j,cidx]),3) for j in range(k_)]
        print(f"     s={s_:>5}: {k_} solutions, u_centers={ucs}")

    print(f"\n   Total time: {time.time()-t_start:.1f}s")
    print("Done.")

if __name__=='__main__':
    main()
