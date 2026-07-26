"""Traditional 2D solver timing == EXACTLY the dataset-generation method
(generate_dataset_2d.py): companion/continuation in the parameter s.

Pipeline (faithful copy of the data-gen code):
  (A) COLD START at s=1600: seed search (Fourier + random seeds, Newton) + D4
      augmentation -> all coexisting solutions at the anchor s=1600.
  (B) MARCH downward s=1599,1598,...: previous-s solutions are the Newton init for
      the current s; dedup + canonicalize; verify residual < 1e-9.

To obtain ALL coexisting solutions at a target test-s the traditional method must
pay (A) once + (B) from 1600 down to that s. We run ONE march from 1600 to the
lowest target and checkpoint the cumulative wall-time and cumulative Newton
iterations each time we pass a selected target s.

Mesh / assembly / Newton are copied verbatim from generate_dataset_2d.py (ell=3,
145 nodes). Output: trad_2d_timing.csv, trad_2d_steps.csv. CPU only.
"""
import os, sys, time, csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "data_gen", "Companion_Method")   # mesh .dat files
TARGETS = [1189, 869, 650, 304, -101]     # selected test-set s (span march distance)
NEWTON_TOL = 1e-9

# ── 16-point Gauss quadrature (verbatim) ─────────────────────────────────────
_GP = np.array([
    [9.703785126946112e-3,8.602401356562194e-1],[4.612207990645205e-2,8.602401356562194e-1],
    [9.363778443732850e-2,8.602401356562194e-1],[1.300560792168344e-1,8.602401356562194e-1],
    [2.891208422438901e-2,5.835904323689168e-1],[1.374191041345744e-1,5.835904323689168e-1],
    [2.789904634965088e-1,5.835904323689168e-1],[3.874974834066942e-1,5.835904323689168e-1],
    [5.021012321136977e-2,2.768430136381238e-1],[2.386486597314429e-1,2.768430136381238e-1],
    [4.845083266304333e-1,2.768430136381238e-1],[6.729468631505064e-1,2.768430136381238e-1],
    [6.546699455501446e-2,5.710419611451768e-2],[3.111645522443570e-1,5.710419611451768e-2],
    [6.317312516411253e-1,5.710419611451768e-2],[8.774288093304679e-1,5.710419611451768e-2]])
_GW = np.array([5.423225910525254e-3,1.016725956447879e-2,1.016725956447879e-2,5.423225910525254e-3,
    2.258404928236993e-2,4.233972452174629e-2,4.233972452174629e-2,2.258404928236993e-2,
    3.538806789808595e-2,6.634421610704973e-2,6.634421610704973e-2,3.538806789808595e-2,
    2.356836819338233e-2,4.418508852236173e-2,4.418508852236173e-2,2.356836819338233e-2])
_PHI = np.stack([_GP[:,0],_GP[:,1],1-_GP[:,0]-_GP[:,1]],axis=1)


def load_col(fname, skip_cols=1, dtype=float):
    raw = np.loadtxt(fname)
    if raw.ndim == 1: raw = raw[np.newaxis,:]
    return raw[:,skip_cols:].astype(dtype)

def refine(coord, elem, diri_set):
    n=len(coord); edge_map={}; new_coords=list(coord); new_diri=set(diri_set); new_elem=[]; nn=n
    for tri in elem:
        v=[int(tri[0]),int(tri[1]),int(tri[2])]; mids=[]
        for a,b in [(v[0],v[1]),(v[1],v[2]),(v[0],v[2])]:
            key=(min(a,b),max(a,b))
            if key not in edge_map:
                edge_map[key]=nn; new_coords.append((coord[a]+coord[b])/2)
                if a in new_diri and b in new_diri: new_diri.add(nn)
                nn+=1
            mids.append(edge_map[key])
        m01,m12,m02=mids
        new_elem.extend([[v[0],m01,m02],[m01,v[1],m12],[m02,m12,v[2]],[m01,m12,m02]])
    return (np.array(new_coords,dtype=float),np.array(new_elem,dtype=int),np.array(sorted(new_diri),dtype=int))

def _geom(coord,elem):
    v0,v1,v2=elem[:,0],elem[:,1],elem[:,2]
    B00=coord[v0,0]-coord[v2,0];B01=coord[v1,0]-coord[v2,0]
    B10=coord[v0,1]-coord[v2,1];B11=coord[v1,1]-coord[v2,1]
    return v0,v1,v2,np.abs(B00*B11-B01*B10)

def build_stiffness(coord,elem):
    n=len(coord);v0,v1,v2=elem[:,0],elem[:,1],elem[:,2]
    B00=coord[v0,0]-coord[v2,0];B01=coord[v1,0]-coord[v2,0]
    B10=coord[v0,1]-coord[v2,1];B11=coord[v1,1]-coord[v2,1]
    detB=np.abs(B00*B11-B01*B10)
    Bi00=B11/detB;Bi01=-B01/detB;Bi10=-B10/detB;Bi11=B00/detB
    gpx=np.stack([Bi00,Bi10,-Bi00-Bi10],axis=1);gpy=np.stack([Bi01,Bi11,-Bi01-Bi11],axis=1)
    Aloc=(detB/2)[:,None,None]*(gpx[:,:,None]*gpx[:,None,:]+gpy[:,:,None]*gpy[:,None,:])
    verts=np.stack([v0,v1,v2],axis=1)
    R=np.repeat(verts,3,axis=1).ravel();C=np.tile(verts,(1,3)).ravel()
    return sp.csr_matrix((Aloc.ravel(),(R,C)),shape=(n,n))

def assemble_F_J(coord,elem,u,s,A):
    n=len(coord);v0,v1,v2,detB=_geom(coord,elem)
    b=s*np.sin(np.pi*coord[:,0])*np.sin(np.pi*coord[:,1])
    ue=np.stack([u[v0],u[v1],u[v2]],axis=1);fe=np.stack([b[v0],b[v1],b[v2]],axis=1)
    uh=ue@_PHI.T;fh=fe@_PHI.T;wt=detB[:,None]*_GW[None,:]
    intgd=wt*(uh**2-fh);ce=(intgd[:,:,None]*_PHI[None,:,:]).sum(axis=1)
    coeff=np.zeros(n);np.add.at(coeff,v0,ce[:,0]);np.add.at(coeff,v1,ce[:,1]);np.add.at(coeff,v2,ce[:,2])
    F=A@u-coeff
    ji=wt*uh;je=-2*np.einsum('eg,gk,gi->eki',ji,_PHI,_PHI)
    verts=np.stack([v0,v1,v2],axis=1)
    R=np.repeat(verts,3,axis=1).ravel();C=np.tile(verts,(1,3)).ravel()
    J=A+sp.csr_matrix((je.ravel(),(R,C)),shape=(n,n))
    return F,J

def newton(coord,elem,u0,s,A,FN,dbc,tol=NEWTON_TOL,maxiter=50):
    """Returns (u, converged, n_iters)."""
    u=u0.copy()
    for it in range(maxiter):
        F,J=assemble_F_J(coord,elem,u,s,A); F[dbc]=0.0
        if np.max(np.abs(F[FN]))<tol: return u,True,it
        try:
            u[FN]-=spla.spsolve(J[np.ix_(FN,FN)].tocsc(),F[FN])
        except Exception:
            return u,False,it+1
    F,_=assemble_F_J(coord,elem,u,s,A);F[dbc]=0.0
    return u,np.max(np.abs(F[FN]))<tol,maxiter

def verify(coord,elem,u,s,A,FN,dbc):
    F,_=assemble_F_J(coord,elem,u,s,A);F[dbc]=0.0
    return float(np.max(np.abs(F[FN])))

def dedup(sols,tol=1e-4):
    out=[]
    for u in sols:
        if not any(np.max(np.abs(u-v))<tol for v in out): out.append(u)
    return out

def d4_augment(seeds,coord,elem,A,FN,dbc,s):
    tree=cKDTree(coord);n=len(coord);it_tot=0
    ops=[lambda c:np.stack([c[:,0],c[:,1]],axis=1),lambda c:np.stack([1-c[:,0],c[:,1]],axis=1),
         lambda c:np.stack([c[:,0],1-c[:,1]],axis=1),lambda c:np.stack([c[:,1],c[:,0]],axis=1),
         lambda c:np.stack([1-c[:,0],1-c[:,1]],axis=1),lambda c:np.stack([1-c[:,1],1-c[:,0]],axis=1),
         lambda c:np.stack([c[:,1],1-c[:,0]],axis=1),lambda c:np.stack([1-c[:,1],c[:,0]],axis=1)]
    all_sols=list(seeds)
    for T in ops:
        _,perm=tree.query(T(coord))
        for uk in list(seeds):
            u_rot=np.empty(n);u_rot[perm]=uk;u_rot[dbc]=0.0
            u,conv,it=newton(coord,elem,u_rot,s,A,FN,dbc);it_tot+=it
            if conv: all_sols.append(u)
    return dedup(all_sols),it_tot

def canonicalize(sols,cidx,coord):
    if not sols: return sols
    def key(u):
        uc=float(u[cidx]); x_asym=float(np.mean(u[coord[:,0]<0.5])-np.mean(u[coord[:,0]>0.5]))
        return (round(uc,1),x_asym)
    return sorted(sols,key=key)


def setup_mesh():
    coord0=load_col(os.path.join(BASE,'coordinates.dat'),skip_cols=1)
    elem0 =load_col(os.path.join(BASE,'elements.dat'),skip_cols=1,dtype=int)-1
    diri0 =load_col(os.path.join(BASE,'dirichlet.dat'),skip_cols=0,dtype=int).flatten()-1
    c1,e1,d1=refine(coord0,elem0,set(diri0)); c2,e2,d2=refine(c1,e1,set(d1))
    coord,elem,d3=refine(c2,e2,set(d2)); n=len(coord)
    dbc=np.array(sorted(d3),dtype=int); FN=np.array([i for i in range(n) if i not in set(d3)],dtype=int)
    A=build_stiffness(coord,elem); cidx=int(np.argmin(np.sum((coord-[0.5,0.5])**2,axis=1)))
    assert n==145, n
    return coord,elem,A,FN,dbc,cidx


def seed_search_1600(coord,elem,A,FN,dbc,cidx):
    """(A) cold start: replicate generate_dataset_2d block [2]. Returns (sols,t,iters)."""
    t0=time.time(); it_tot=0; seeds=[]
    def try_seed(u0,s=1600,tol_fast=1e-7):
        nonlocal it_tot
        u0=u0.copy();u0[dbc]=0.0
        u,conv,it=newton(coord,elem,u0,s,A,FN,dbc,tol=tol_fast,maxiter=15);it_tot+=it
        if conv and not any(np.max(np.abs(sv-u))<1e-3 for sv in seeds):
            u,conv,it=newton(coord,elem,u,s,A,FN,dbc);it_tot+=it
            if conv and not any(np.max(np.abs(sv-u))<1e-4 for sv in seeds): seeds.append(u.copy())
    f11=np.sin(np.pi*coord[:,0])*np.sin(np.pi*coord[:,1]);f21=np.sin(2*np.pi*coord[:,0])*np.sin(np.pi*coord[:,1])
    f12=np.sin(np.pi*coord[:,0])*np.sin(2*np.pi*coord[:,1]);f22=np.sin(2*np.pi*coord[:,0])*np.sin(2*np.pi*coord[:,1])
    f32=np.sin(3*np.pi*coord[:,0])*np.sin(2*np.pi*coord[:,1]);f23=np.sin(2*np.pi*coord[:,0])*np.sin(3*np.pi*coord[:,1])
    b1=f21-f12;b2=f21+f12;rng=np.random.default_rng(42)
    for sc in np.concatenate([np.geomspace(0.5,300,40),-np.geomspace(0.5,300,40)]):
        try_seed(sc*f11);try_seed(sc*f22)
    for sc in np.concatenate([np.geomspace(1,500,40),-np.geomspace(1,500,40)]):
        try_seed(sc*b1);try_seed(sc*b2);try_seed(sc*f21);try_seed(sc*f12)
        try_seed(sc*f11+0.5*sc*b1);try_seed(sc*f11-0.5*sc*b1)
        try_seed(sc*f11+0.5*sc*b2);try_seed(sc*f11-0.5*sc*b2)
        try_seed(sc*(f32-f23));try_seed(sc*(f32+f23))
    for _ in range(300):
        c=rng.uniform(-200,200,6)
        try_seed(c[0]*f11+c[1]*f21+c[2]*f12+c[3]*f22+c[4]*f32+c[5]*f23)
    alls,it_d4=d4_augment(seeds,coord,elem,A,FN,dbc,1600); it_tot+=it_d4
    alls=canonicalize(alls,cidx,coord)
    return alls,time.time()-t0,it_tot


def main():
    coord,elem,A,FN,dbc,cidx=setup_mesh()
    print(f"mesh: {len(coord)} nodes, {len(FN)} free", flush=True)

    # (A) COLD START
    print("seed search @ s=1600 ...", flush=True)
    sols,t_seed,it_seed=seed_search_1600(coord,elem,A,FN,dbc,cidx)
    res0=max(verify(coord,elem,u,1600,A,FN,dbc) for u in sols)
    print(f"  s=1600: {len(sols)} solutions, max-res={res0:.2e}, seed_time={t_seed:.2f}s, seed_iters={it_seed}", flush=True)

    # (B) MARCH 1600 -> lowest target, checkpoint at each target
    lo=min(TARGETS); targets=set(TARGETS)
    rows=[]; cum_t=0.0; cum_it=0
    cur=sols
    for s in range(1599, lo-1, -1):
        t0=time.time(); nxt=[]; step_it=0
        for u_prev in cur:
            u,conv,it=newton(coord,elem,u_prev,s,A,FN,dbc); step_it+=it
            if conv and verify(coord,elem,u,s,A,FN,dbc)<NEWTON_TOL: nxt.append(u)
        nxt=canonicalize(dedup(nxt),cidx,coord); cur=nxt
        cum_t+=time.time()-t0; cum_it+=step_it
        if s in targets:
            rows.append({"s":s,"k":len(cur),"march_steps":1600-s,
                         "march_time_s":round(cum_t,3),"march_newton_iters":cum_it,
                         "seed_time_s":round(t_seed,3),"seed_newton_iters":it_seed,
                         "total_time_s":round(t_seed+cum_t,3),"total_newton_iters":it_seed+cum_it})
            print(f"  reached s={s}: k={len(cur)} march_steps={1600-s} "
                  f"march_t={cum_t:.1f}s cumIters={cum_it} total_t={t_seed+cum_t:.1f}s", flush=True)

    with open(os.path.join(HERE,"trad_2d_timing.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
        for r in sorted(rows,key=lambda d:-d["s"]): w.writerow(r)
    print("wrote trad_2d_timing.csv", flush=True)


if __name__=="__main__":
    main()
