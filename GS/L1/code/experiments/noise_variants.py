"""Noise x autoregression DESIGN-SPACE experiment (L=1; reuse for L=2 with --L2).

Compares, per test param, several ways of injecting noise into the K-step ordered AR:

  det            sigma=0, 1 pass                      (24 candidates)   -- headline baseline
  base_1         current scheme, sigma>0, 1 pass      (24)              -- noised output = noised AR input
  A_1            variant A, 1 pass                     (24)              -- noised OUTPUT, CLEAN z_k fed back
  B_novel_1      variant B, 1 pass, M cand/step        (24*M)           -- pick most-novel cand to continue
  B_resid_1      variant B, 1 pass, M cand/step        (24*M)           -- pick lowest-residual cand to continue
  base_Npass     current scheme, sigma>0, N passes UNION(24*N)          -- matched budget to B (N=M)
  A_Npass        variant A, N passes UNION             (24*N)

Each scheme's candidate fields -> FDM refine -> D4-dedup -> # distinct + GT coverage.
Standalone: rebuilds the AR loop from model internals; does NOT touch model/v3_model.py.

Run (L=1):  CUDA_VISIBLE_DEVICES=0 <torch124> experiments/noise_variants.py --out experiments/nv_L1.json
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> code/
sys.path.insert(0, ROOT)
import numpy as np, torch
from config import Config, P_START_ID, SOL_ID
from model.v3_model import GSPDEModel
from gs_torch import GSOperator
DEV = torch.device("cuda:0")
DT = torch.bfloat16


def d4_rel_l2(a, b):
    g = [torch.rot90(b, k, (-2, -1)) for k in range(4)]
    bt = b.transpose(-1, -2); g += [torch.rot90(bt, k, (-2, -1)) for k in range(4)]
    gb = torch.stack(g)
    return ((a[None]-gb).flatten(2).norm(dim=2)/b.flatten(1).norm(dim=1).clamp_min(1e-8)).mean(1).min().item()


def solve_track(op, rho, mu, A0, S0, maxiter, early_iter, early_tol=0.05, tol=1e-9, stepsize=0.1):
    dt = op.dtype
    A = torch.as_tensor(A0, dtype=dt, device=DEV).clone(); B, n, _ = A.shape
    rho = torch.as_tensor(rho, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    mu = torch.as_tensor(mu, dtype=dt, device=DEV).expand(B).reshape(B, 1, 1)
    S = torch.as_tensor(S0, dtype=dt, device=DEV).clone()
    active = torch.ones(B, dtype=torch.bool, device=DEV); crit = torch.full((B,), float("inf"), dtype=dt, device=DEV)
    DA, DS, Lam = op.DA, op.DS, op.Lam
    for it in range(1, maxiter+1):
        rA = DA*op.lap(A) + (-S*A*A + (mu+rho)*A); rS = DS*op.lap(S) + (S*A*A - rho*(1.0-S))
        c = torch.maximum(rA.abs().amax((1, 2)), rS.abs().amax((1, 2))); crit = torch.where(active, c, crit)
        active = active & ~((~torch.isfinite(c)) | (c > 1000.0)) & ~(c < tol)
        if early_iter and it == early_iter: active = active & ~(c > early_tol)
        if not active.any(): break
        m1U = (mu+rho)-2*A*S; m2V = rho+A*A; m1V = -A*A; m2U = 2*A*S
        disc = torch.sqrt(torch.clamp((m1U-m2V)**2 + 4*m1V*m2U, min=0.0)); half = (m1U+m2V)/2
        beta = ((half+disc/2).amax((1, 2)) + (half-disc/2).amin((1, 2))).reshape(B, 1, 1)/2
        dA = torch.nan_to_num(op.inv_shift(rA, DA*Lam+beta)); dS = torch.nan_to_num(op.inv_shift(rS, DS*Lam+beta))
        mm = active.reshape(B, 1, 1); A = torch.where(mm, A-stepsize*dA, A); S = torch.where(mm, S-stepsize*dS, S)
    conv = torch.isfinite(crit) & (crit < tol); rngA = A.reshape(B, -1).amax(1) - A.reshape(B, -1).amin(1)
    return A.cpu().numpy(), S.cpu().numpy(), (conv & (rngA > 0.05)).cpu().numpy()


def residual_maxabs(op, rho, mu, fields):
    dt = op.dtype
    A = torch.as_tensor(fields[:, 0], dtype=dt, device=DEV); S = torch.as_tensor(fields[:, 1], dtype=dt, device=DEV)
    B = A.shape[0]; rho = torch.tensor(rho, dtype=dt, device=DEV); mu = torch.tensor(mu, dtype=dt, device=DEV)
    rA = op.DA*op.lap(A) + (-S*A*A + (mu+rho)*A); rS = op.DS*op.lap(S) + (S*A*A - rho*(1.0-S))
    return torch.maximum(rA.abs().amax((1, 2)), rS.abs().amax((1, 2))).cpu().numpy()


class Gen:
    """Re-implements the K-step ordered AR loop with pluggable noise schemes."""
    def __init__(self, m):
        self.m = m; self.T = m._get_transformer(); self.dev = m.component_device

    def _clen(self, c):
        if c is None: return 0
        if hasattr(c, "get_seq_length"):
            try: return int(c.get_seq_length())
            except TypeError: return int(c.get_seq_length(0))
        return int(c[0][0].shape[2])

    def _fwd(self, emb, past):
        L = self._clen(past)
        return self.T(inputs_embeds=emb, attention_mask=torch.ones(1, L + emb.shape[1], dtype=torch.long, device=self.dev),
                      past_key_values=past, use_cache=True, return_dict=True)

    def run(self, p_value, K, sigma, mode, M=1, select="novel", op=None):
        m = self.m; dev = self.dev
        tp = torch.as_tensor(p_value, dtype=torch.float32)
        rho, mu = float(p_value[0]), float(p_value[1])
        seq = torch.stack([m._special_emb(P_START_ID),
                           m._input_proj(torch.zeros(m.config.latent_dim), tp)], 0).unsqueeze(0).to(dev, DT)
        sol_emb = m._special_emb(SOL_ID).unsqueeze(0).unsqueeze(0).to(dev, DT)
        past = None; outputs = []; chosen_A = []
        for k in range(K):
            emb = torch.cat([seq, sol_emb], 1) if past is None else sol_emb
            o = self._fwd(emb, past); past = o.past_key_values
            _, reg = m.dual_head(o.last_hidden_state[:, -1, :])          # [1,256]
            if mode == "B":
                cregs = [reg + sigma*torch.randn_like(reg) for _ in range(M)]
                csol = [m.autoencoder(z.to(DT), "decode") for z in cregs]
                for s in csol: outputs.append(s.float().squeeze(0).cpu().numpy())
                cA = [s.float().squeeze(0)[0] for s in csol]
                if select == "resid":
                    fields = np.stack([s.float().squeeze(0).cpu().numpy() for s in csol])
                    pick = int(residual_maxabs(op, rho, mu, fields).argmin())
                elif select == "novel" and chosen_A:
                    ref = torch.stack(chosen_A)
                    score = [min(d4_rel_l2(a.cpu(), r.cpu()) for r in ref) for a in cA]
                    pick = int(np.argmax(score))                          # most different from chosen path
                else:
                    pick = int(torch.randint(0, M, (1,)).item())
                chosen_A.append(cA[pick].detach())
                feed = csol[pick]
            else:
                zc = reg + sigma*torch.randn_like(reg) if sigma > 0 else reg
                sol = m.autoencoder(zc.to(DT), "decode")
                outputs.append(sol.float().squeeze(0).cpu().numpy())
                feed = sol if mode == "base" else m.autoencoder(reg.to(DT), "decode")  # A: feed CLEAN
            z = m.autoencoder(feed, "encode").squeeze(0).float()
            nxt = m._input_proj(z, tp).unsqueeze(0).unsqueeze(0).to(dev, DT)
            o2 = self._fwd(nxt, past); past = o2.past_key_values
        return np.stack(outputs)


def distinct_cov(op, rho, mu, fields, true, maxiter, early_iter):
    A, S, conv = solve_track(op, rho, mu, fields[:, 0], fields[:, 1], maxiter, early_iter)
    Kt = true.shape[0]; kept = []; matched = set()
    for k in np.where(conv)[0]:
        fk = torch.tensor(np.stack([A[k], S[k]]), dtype=torch.float32)
        if all(d4_rel_l2(fk, g) > 0.15 for g in kept):
            kept.append(fk)
            ds = [d4_rel_l2(fk, true[j]) for j in range(Kt)]; j = int(np.argmin(ds))
            if ds[j] < 0.15: matched.add(j)
    return len(kept), len(matched)/Kt


def union_passes(gen, op, p, true, K, sigma, mode, N, maxiter, early_iter, select="novel", M=1):
    rho, mu = float(p[0]), float(p[1]); kept = []; matched = set(); Kt = true.shape[0]
    for _ in range(N):
        fields = gen.run(p, K, sigma, mode, M=M, select=select, op=op)
        A, S, conv = solve_track(op, rho, mu, fields[:, 0], fields[:, 1], maxiter, early_iter)
        for k in np.where(conv)[0]:
            fk = torch.tensor(np.stack([A[k], S[k]]), dtype=torch.float32)
            if all(d4_rel_l2(fk, g) > 0.15 for g in kept):
                kept.append(fk)
                ds = [d4_rel_l2(fk, true[j]) for j in range(Kt)]; j = int(np.argmin(ds))
                if ds[j] < 0.15: matched.add(j)
    return len(kept), len(matched)/Kt


@torch.no_grad()
def main(a):
    C = Config()
    lookup = torch.load(C.data_lookup_path, weights_only=False); ns = torch.load(C.norm_stats_path)
    params = lookup["p_values"]; sols = lookup["solutions_by_p"]
    test_idx = torch.load(C.test_p_idx_path).tolist()
    m = GSPDEModel.from_pretrained(a.ckpt, C, local_rank=0, p_mean=ns["p_mean"], p_std=ns["p_std"]); m.eval()
    op = GSOperator(C.op_path, device="cuda", dtype=torch.float64)
    if a.L2: op.DA /= 4.0; op.DS /= 4.0
    gen = Gen(m)
    K = C.fixed_k; sig = a.sigma; M = a.M; N = a.M  # matched budget: N passes == M cand/step
    # pick high-Kt test params
    cand = sorted(test_idx, key=lambda ti: -sols[ti].shape[0])[:a.n]
    print(f"=== noise variants: {len(cand)} params, sigma={sig}, M=N={M}, FDM {a.maxiter}/{a.early_iter} L2={a.L2} ===", flush=True)
    rows = []
    for n, ti in enumerate(cand):
        p = params[ti]; true = sols[ti]; rho, mu = float(p[0]), float(p[1])
        r = {"ti": int(ti), "Kt": int(true.shape[0])}
        # single-pass schemes
        for tag, mode, sel in [("det", "base", None), ("base_1", "base", None), ("A_1", "A", None),
                               ("B_novel_1", "B", "novel"), ("B_resid_1", "B", "resid")]:
            s = 0.0 if tag == "det" else sig
            fields = gen.run(p, K, s, mode, M=(M if mode == "B" else 1), select=sel or "novel", op=op)
            d, cov = distinct_cov(op, rho, mu, fields, true, a.maxiter, a.early_iter)
            r[tag] = d; r[tag+"_cov"] = round(cov, 3); r[tag+"_ncand"] = int(fields.shape[0])
        # matched-budget multipass-union baselines
        for tag, mode in [("base_Npass", "base"), ("A_Npass", "A")]:
            d, cov = union_passes(gen, op, p, true, K, sig, mode, N, a.maxiter, a.early_iter)
            r[tag] = d; r[tag+"_cov"] = round(cov, 3)
        rows.append(r)
        print(f"  {n+1}/{len(cand)} p{ti} Kt={r['Kt']} | det {r['det']} base1 {r['base_1']} A1 {r['A_1']} "
              f"Bnov {r['B_novel_1']} Bres {r['B_resid_1']} | base{N}x {r['base_Npass']} A{N}x {r['A_Npass']}", flush=True)
    # aggregate
    tags = ["det", "base_1", "A_1", "B_novel_1", "B_resid_1", "base_Npass", "A_Npass"]
    agg = {t: float(np.mean([r[t] for r in rows])) for t in tags}
    aggc = {t: float(np.mean([r[t+"_cov"] for r in rows])) for t in tags}
    print("\n=== MEAN distinct ===")
    for t in tags: print(f"  {t:12s} distinct={agg[t]:.2f}  cov={aggc[t]:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"sigma": sig, "M": M, "agg_distinct": agg, "agg_cov": aggc, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="./checkpoints_final/epoch_60")
    p.add_argument("--n", type=int, default=10, help="# highest-Kt test params")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--M", type=int, default=4, help="B candidates/step == N union passes (matched budget)")
    p.add_argument("--maxiter", type=int, default=8000)
    p.add_argument("--early_iter", type=int, default=3000)
    p.add_argument("--L2", action="store_true")
    p.add_argument("--out", default="./experiments/nv_L1.json")
    main(p.parse_args())
