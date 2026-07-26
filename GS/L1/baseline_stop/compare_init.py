"""Random initial guess vs Qwen initial guess for the classical FDM solver.

The data was never cold-started from random fields (it bootstrapped from 8 seeds
+ continuation, because cold random starts are hard). Here we test, per held-out
test parameter, giving the SAME quasi-Newton FDM solver (fdm/gs_torch) the SAME
budget of M initial guesses from two sources:
  RANDOM : degree-<=4 random polynomial fields (fdm poly_perturb, spot-amplitude)
  QWEN   : M generations from the model (drop STOP, force M, decoded to (A,S))
and compare, on EXACT (post-processed) outputs:
  - convergence rate (frac of guesses reaching a nontrivial solution)
  - # DISTINCT solutions found  and  coverage of the true set
  - iterations-to-converge  (compute proxy)  and  wall-clock solve time
This is the user's comparison (2026-06): does a learned initial guess beat random?
"""
import os, sys, argparse, pickle, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, scipy.io as sio
from config import Config, P_START_ID, SOL_ID, STOP_ID
from model.v3_model import GSPDEModel
from model.canonicalize import canonicalize_solutions
from gs_torch import GSOperator
from gs_continuation import poly_perturb
from eval import d4_rel_l2
DEV = torch.device("cuda:0"); DT = torch.bfloat16


def solve_track(op, rho, mu, A0, S0, tol, maxiter, early_iter, early_tol, stepsize=0.1):
    """solve_batch + per-candidate convergence iteration (compute proxy)."""
    dt = op.dtype
    A = torch.as_tensor(A0, dtype=dt, device=DEV).clone(); B, n, _ = A.shape
    rho = torch.as_tensor(rho, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    mu = torch.as_tensor(mu, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    S = torch.as_tensor(S0, dtype=dt, device=DEV).clone()
    active = torch.ones(B, dtype=torch.bool, device=DEV)
    crit = torch.full((B,), float("inf"), dtype=dt, device=DEV)
    citer = torch.full((B,), -1, dtype=torch.long, device=DEV)
    DA, DS, Lam = op.DA, op.DS, op.Lam
    for it in range(1, maxiter + 1):
        rA = DA * op.lap(A) + (-S * A * A + (mu + rho) * A)
        rS = DS * op.lap(S) + (S * A * A - rho * (1.0 - S))
        c = torch.maximum(rA.abs().amax((1, 2)), rS.abs().amax((1, 2)))
        crit = torch.where(active, c, crit)
        newly = active & (c < tol); citer = torch.where(newly, torch.full_like(citer, it), citer)
        diverged = (~torch.isfinite(c)) | (c > 1000.0)
        active = active & ~diverged & ~(c < tol)
        if it == early_iter: active = active & ~(c > early_tol)
        if not active.any(): break
        m1U = (mu + rho) - 2*A*S; m2V = rho + A*A; m1V = -A*A; m2U = 2*A*S
        disc = torch.sqrt(torch.clamp((m1U-m2V)**2 + 4*m1V*m2U, min=0.0)); half = (m1U+m2V)/2
        beta = ((half+disc/2).amax((1,2)) + (half-disc/2).amin((1,2))).reshape(B,1,1)/2
        dA = torch.nan_to_num(op.inv_shift(rA, DA*Lam+beta)); dS = torch.nan_to_num(op.inv_shift(rS, DS*Lam+beta))
        mm = active.reshape(B,1,1)
        A = torch.where(mm, A-stepsize*dA, A); S = torch.where(mm, S-stepsize*dS, S)
    conv = torch.isfinite(crit) & (crit < tol)
    rngA = A.reshape(B,-1).amax(1) - A.reshape(B,-1).amin(1)
    return dict(A=A.cpu().numpy(), S=S.cpu().numpy(), res=crit.cpu().numpy(),
                conv=(conv & (rngA > 0.05)).cpu().numpy(), citer=citer.cpu().numpy(), iters=it)


def main(a):
    C = Config()
    lookup = torch.load("./data/gs_lookup_big.pt", weights_only=False)
    test_idx = torch.load("./data/test_p_idx_big.pt").tolist()
    train_idx = torch.load("./data/train_p_idx_big.pt").tolist()
    ns = torch.load("./data/norm_stats_big.pt")
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    model = GSPDEModel.from_pretrained("./ss_checkpoints_big_local/best_model", C, local_rank=0,
                                       p_mean=ns["p_mean"], p_std=ns["p_std"]); model.eval()
    op = GSOperator(f"{HERE}/op_n128.mat", device="cuda", dtype=torch.float64)
    xg = op.Tx.new_tensor(sio.loadmat(f"{HERE}/op_n128.mat")["x"].ravel())
    gen_t = torch.Generator(device="cuda").manual_seed(0)
    tr_p = params[torch.as_tensor(train_idx)].float(); psc = tr_p.std(0).clamp_min(1e-8)

    ctx_cache = {}
    @torch.no_grad()
    def ctx_block(i):
        if i in ctx_cache: return ctx_cache[i]
        p_v = params[i]; e = [model._special_emb(P_START_ID), model._input_proj(torch.zeros(C.latent_dim), p_v)]
        S = canonicalize_solutions(sols[i].to(DEV, DT))
        for k in range(S.shape[0]):
            z = model.autoencoder(S[k:k+1], "encode").float().squeeze(0)
            e += [model._special_emb(SOL_ID), model._input_proj(z, p_v)]
        e.append(model._special_emb(STOP_ID)); ctx_cache[i] = torch.stack(e, 0); return ctx_cache[i]
    def det_ctx(ti):
        d = ((tr_p - params[ti].float())/psc).pow(2).sum(1).sqrt()
        return [train_idx[j] for j in torch.argsort(d).tolist() if train_idx[j] != ti][:C.max_context_p]
    near = lambda ti: [train_idx[j] for j in torch.argsort(((tr_p-params[ti].float())/psc).pow(2).sum(1).sqrt()).tolist() if train_idx[j]!=ti]
    def samp_ctx(ti, rng): nr = near(ti)[:2*C.max_context_p]; return rng.choice(nr, size=min(C.max_context_p,len(nr)), replace=False).tolist()

    @torch.no_grad()
    def qwen_pool(ti, M):
        rng = np.random.RandomState(1234+ti); out = []; runs = -(-M // a.per_run)
        for m in range(runs):
            ctx = det_ctx(ti) if m == 0 else samp_ctx(ti, rng); acc = []
            for _ in range(a.per_run):
                tp = params[ti]; rows = [ctx_block(i) for i in ctx]
                tb = [model._special_emb(P_START_ID), model._input_proj(torch.zeros(C.latent_dim), tp)]
                for z in acc: tb += [model._special_emb(SOL_ID), model._input_proj(z, tp)]
                tb.append(model._special_emb(SOL_ID))
                seq = torch.cat(rows+[torch.stack(tb,0)],0).unsqueeze(0).to(DEV, DT)
                am = torch.ones(1, seq.shape[1], dtype=torch.long, device=DEV)
                h = model._get_transformer()(inputs_embeds=seq, attention_mask=am, return_dict=True).last_hidden_state[:,-1]
                _, reg = model.dual_head(h); z = reg.float().squeeze(0); acc.append(z)
                f = model.autoencoder(z.to(DEV,DT).unsqueeze(0), "decode").float().squeeze(0)
                out.append(f.cpu().numpy())
                if len(out) >= M: break
            if len(out) >= M: break
        return np.stack(out[:M])

    def rand_pool(M):
        A0 = poly_perturb(xg, a.rscale, M, gen_t, "cuda", op.dtype).cpu().numpy()
        S0 = poly_perturb(xg, a.rscale, M, gen_t, "cuda", op.dtype).cpu().numpy()
        return np.stack([A0, S0], 1)                  # [M,2,n,n]

    def evaluate(pool, ti):
        rho, mu = float(params[ti,0]), float(params[ti,1])
        t0 = time.time()
        o = solve_track(op, rho, mu, pool[:,0], pool[:,1], tol=a.tol, maxiter=a.maxiter,
                        early_iter=a.early_iter, early_tol=1e-2)
        wall = time.time() - t0
        true = sols[ti]; Kt = true.shape[0]; kept = []; matched = set()
        for k in np.where(o["conv"])[0]:
            f = torch.tensor(np.stack([o["A"][k], o["S"][k]]), dtype=torch.float32)
            if all(d4_rel_l2(f, g) > a.dedup for g in kept):
                kept.append(f)
                ds = [d4_rel_l2(f, true[j]) for j in range(Kt)]; j = int(np.argmin(ds))
                if ds[j] < 0.15: matched.add(j)
        ci = o["citer"][o["conv"]]
        return dict(conv=int(o["conv"].sum()), distinct=len(kept), cov=len(matched)/Kt,
                    med_iter=float(np.median(ci)) if len(ci) else float('nan'), wall=wall,
                    A=o["A"], conv_mask=o["conv"])

    recs = []; ex = {}
    targets = [int(x) for x in a.only.split(",")] if a.only else (test_idx if a.n <= 0 else test_idx[:a.n])
    for n, ti in enumerate(targets):
        qp = qwen_pool(ti, a.M); rp = rand_pool(a.M)
        rq = evaluate(qp, ti); rr = evaluate(rp, ti)
        recs.append(dict(ti=ti, rho=float(params[ti,0]), mu=float(params[ti,1]), Kt=sols[ti].shape[0],
                         q=rq, r=rr))
        if ti in (1064, 70, 446):
            ex[ti] = dict(true=sols[ti][:,0].numpy(), q=qp, r=rp,
                          qA=rq["A"], qconv=rq["conv_mask"], rA=rr["A"], rconv=rr["conv_mask"],
                          rho=float(params[ti,0]), mu=float(params[ti,1]))
        if n % 10 == 0:
            print(f"...{n} p{ti} Kt={sols[ti].shape[0]} | QWEN conv={rq['conv']} dist={rq['distinct']} cov={rq['cov']:.2f} it={rq['med_iter']:.0f} | RAND conv={rr['conv']} dist={rr['distinct']} cov={rr['cov']:.2f} it={rr['med_iter']:.0f}", flush=True)
    pickle.dump(dict(recs=recs, ex=ex), open(a.out, "wb"))
    def agg(side):
        cov = np.array([r[side]["cov"] for r in recs]); dist = np.array([r[side]["distinct"] for r in recs])
        conv = np.array([r[side]["conv"] for r in recs]); wall = np.array([r[side]["wall"] for r in recs])
        it = np.array([r[side]["med_iter"] for r in recs])
        return dict(coverage=float(np.nanmean(cov)), distinct=float(dist.mean()),
                    conv_rate=float(conv.mean()/a.M), med_iter=float(np.nanmedian(it)),
                    wall_per_param=float(wall.mean()))
    summary = dict(M=a.M, rscale=a.rscale, maxiter=a.maxiter, n_params=len(recs),
                   Ktrue_mean=float(np.mean([r["Kt"] for r in recs])),
                   QWEN=agg("q"), RANDOM=agg("r"))
    print("\n=== random-init vs Qwen-init ==="); print(json.dumps(summary, indent=2))
    json.dump(summary, open(a.out.replace(".pkl",".json"), "w"), indent=2); print("DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--M", type=int, default=12, help="# initial guesses per source per param")
    p.add_argument("--per_run", type=int, default=12)
    p.add_argument("--rscale", type=float, default=0.5, help="random poly init amplitude")
    p.add_argument("--maxiter", type=int, default=4000)
    p.add_argument("--early_iter", type=int, default=1500)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--dedup", type=float, default=0.15)
    p.add_argument("--only", default="", help="comma-separated param indices to run")
    p.add_argument("--out", default="/tmp/compare_init.pkl")
    main(p.parse_args())
