# Extrapolation (out-of-distribution parameter) test — results

Same evaluation pipeline as the in-distribution test set, with the **parameters
replaced by extrapolated values outside the training range**. Ground truth at each
extrapolated parameter is produced by the **traditional classical solver** (no learned
warm start). Pipeline per project: `extension/gen_gt_extension.py` (traditional GT) →
`extension/generate_extension.py` (Qwen generation) →
`extension/compute_stats_extension.py` (coverage metrics + table). Raw data:
`gt_extension.pt`, `generated_extension.pt`; tables: `results_table_extension.csv`
(per parameter) and `stats_extension.csv` (per covered branch).

## Metric (coverage-based)

A multi-solution model can fail out-of-distribution in two distinct ways, and a forced
1-1 Hungarian match conflates them. We separate them:

1. refine **every** generated output with the traditional Newton solver at the
   extrapolated parameter; keep the ones that converge to a genuine solution
   (residual below the data-gen tolerance);
2. **dedup** them → the *distinct* solutions the model actually produces;
3. **coverage** = how many true (continued/multistart) GT branches are recovered
   (rel-L2 < 1e-2, 2D < 5e-2). `missing = n_gt − coverage`;
4. **novel** = distinct *valid* solutions that match **no** GT branch — i.e. spurious /
   extra real solutions the model invents (this is the literal "does it land on a
   *different* solution?" test);
5. accuracy (direct / post rel-L2, Newton steps, residual) is reported **only over the
   covered branches** — duplicates and non-converged junk are not charged as rel-L2≈1.

> **Key cross-project finding.** `novel ≈ 0` everywhere: the model essentially never
> produces a spurious/different solution. Every branch it *covers* is recovered to
> machine precision after refine (post rel-L2 ≈ 2.5e-8 for 1D_p / a2a4; ≈1.6e-3 for 2D,
> which is the GT's own residual-1e-9 accuracy floor). Extrapolation degradation is
> therefore **loss of solution diversity (mode collapse → dropping `coverage`)**, not
> wrong solutions. Even far out-of-distribution, the raw output can have direct rel-L2 ≫ 1
> yet still sit in a true branch's Newton basin, so refine pulls it back exactly.

---

## 1D_p   ( −u″ + u²(u²−p) = 0 )   — GT = continuation of the 7 p=18 branches

| p | n_gt | n_gen | distinct | coverage | missing | novel | rel-L2 direct | rel-L2 post | steps | residual |
|---|---|---|---|---|---|---|---|---|---|---|
| 18.01 | 7 | 8 | 7 | **7** | 0 | 0 | 1.81e-02 | 2.63e-08 | 4 | 1.2e-11 |
| 18.05 | 7 | 8 | 7 | **7** | 0 | 0 | 1.91e-02 | 2.65e-08 | 4 | 1.1e-11 |
| 18.10 | 7 | 8 | 4 | 4 | 3 | 0 | 1.15e-01 | 2.76e-08 | 6 | 9.5e-12 |
| 18.50 | 7 | 8 | 2 | 2 | 5 | 0 | 4.35e+00 | 2.71e-08 | 9 | 5.5e-12 |
| 19.00 | 7 | 8 | 2 | 2 | 5 | 0 | 8.86e+00 | 2.68e-08 | 8 | 5.2e-12 |

**Narrowest margin (direct OOD generation).** Full coverage (7/7) holds to p=18.05, then
collapses 7→4→2→2. No spurious solutions, and each covered branch is exact.

### Post-processing cannot fix this, but p=18-seeding can

A tol/iteration sweep at p=19 shows the collapse is **not** a post-processing setting:
coverage stays 2/7 for tol∈[1e-11,1e-6] and max_iter∈[30,300] (a very loose tol=1e-3
only fabricates non-converged "distinct" outputs, coverage 1/7). Newton converges by
**basin of attraction**, not proximity/iterations — an oracle that refines the best raw
output per branch with 200 iters still gets only 2/7, because the model's direct p=19
outputs occupy just 2 basins.

The fix is to seed Newton with the model's reliable **in-distribution p=18 output** and
let one Newton solve carry each branch to p (model-seeded classical continuation):

| p | direct coverage | **p18-seed coverage** |
|---|---|---|
| 18.01 | 7 | 7 |
| 18.05 | 7 | 7 |
| 18.10 | 4 | **7** |
| 18.50 | 2 | **7** |
| 19.00 | 2 | **7** |

The model's value out-of-distribution is therefore *giving all coexisting seeds at once at
a reachable parameter*; classical Newton then continues them outward to full coverage
(`extension/experiment_p18seed.py`, `fig_p18seed_compare`). Direct OOD generation collapses;
in-distribution generation + Newton does not.

**Far extrapolation (p=20, 25).** All 7 branches still *exist* out to p=25 (they do not fold).
A single Newton jump from the p=18 seed reaches p=20 (7/7) but a Δp=7 jump to p=25 starts to
fall out of basin (5/7); stepped model-seeded continuation recovers all 7 at both
(`extension/experiment_far_p.py`):

| p | branches exist | model direct | p18-seed (1 Newton jump) | p18-seed (stepped continuation) |
|---|---|---|---|---|
| 20 | 7 | 1/7 | **7/7** | **7/7** |
| 25 | 7 | 1/7 | 5/7 | **7/7** |

So: direct OOD generation is useless far out (1/7), but the model's p=18 multi-solution
output + classical continuation recovers every branch arbitrarily far (as long as the branch
exists) — a single Newton jump suffices for moderate distance, stepping for large distance.

## 2D   ( −Δu − u² = −s·sin(πx)sin(πy) )   — GT = continuation of the 4 s=1600 branches

| s | n_gt | n_gen | distinct | coverage | missing | novel | rel-L2 direct | rel-L2 post | steps | residual |
|---|---|---|---|---|---|---|---|---|---|---|
| 1601 | 4 | 4 | 4 | **4** | 0 | 0 | 4.28e-03 | 1.60e-03 | 4 | 9.4e-14 |
| 1610 | 4 | 4 | 4 | **4** | 0 | 0 | 6.59e-03 | 1.36e-03 | 4 | 1.1e-13 |
| 1650 | 4 | 4 | 4 | **4** | 0 | 0 | 8.23e-03 | 1.41e-03 | 4 | 1.3e-13 |
| 1700 | 4 | 4 | 4 | **4** | 0 | 0 | 2.25e-02 | 1.50e-03 | 4 | 5.0e-12 |
| 1800 | 4 | 4 | 4 | **4** | 0 | 0 | 4.07e-02 | 1.66e-03 | 4 | 2.9e-10 |

**Most robust.** Full coverage (4/4) maintained to s=1800 (+12.5% beyond boundary), no
missing, no spurious; direct rel-L2 grows gracefully (4e-3→4e-2).

**Far extrapolation (direct generation).** The table above (rel-L2 detail) stops at s=1800,
but direct generation holds **4/4** much further. `extension/experiment_three_methods_2d.py`
(→ `three_methods_2d.pt`, from `far_direct_2d.pt`) evaluates s ∈ {1650,1700,1800,2000,2300,
2700,**3200**}: at every s the true count is 4 and direct generation recovers all 4 (M1=4/4),
so coverage stays full out to **s=3200 (2× the upper training value 1600)** — this is the
figure the manuscript cites. (Seed+jump and seed+continuation also give 4/4 throughout.)

## a2a4   ( −u″ + a₄u⁴ + a₂u² = 0 )   — 4 directions, GT = traditional FDM multi-start

Training rectangle: a₄∈[0.301, 0.798], a₂∈[−14.93, −2.0]. (n_gen is capped at the
model's max=6, so it over-emits where few branches exist; coverage handles this.)

| dir | a4 | a2 | n_gt | n_gen | distinct | coverage | missing | novel | rel-L2 direct | rel-L2 post | steps | residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2− | 0.55 | −15.5 | 6 | 6 | 5 | 4 | 2 | 1 | 3.10e-02 | 2.61e-08 | 4 | 3.0e-11 |
| a2− | 0.55 | −17.0 | 6 | 6 | 5 | 5 | 1 | 0 | 7.85e-02 | 2.53e-08 | 4 | 5.6e-11 |
| a2− | 0.55 | −20.0 | 6 | 6 | 2 | 2 | 4 | 0 | 1.37e+00 | 2.57e-08 | 7 | 2.3e-11 |
| a2− | 0.55 | −25.0 | 6 | 6 | 2 | 2 | 4 | 0 | 1.92e+01 | 2.28e-08 | 16 | 1.1e-11 |
| a2+ | 0.55 | −1.8 | 1 | 6 | 1 | **1** | 0 | 0 | 2.93e-02 | 2.63e-08 | 3 | 3.1e-10 |
| a2+ | 0.55 | −1.4 | 1 | 6 | 1 | **1** | 0 | 0 | 4.69e-02 | 2.75e-08 | 4 | 6.6e-12 |
| a2+ | 0.55 | −1.0 | 1 | 6 | 1 | **1** | 0 | 0 | 1.24e-01 | 2.76e-08 | 5 | 6.3e-12 |
| a2+ | 0.55 | −0.5 | 1 | 6 | 1 | **1** | 0 | 0 | 3.12e-01 | 2.66e-08 | 10 | 6.1e-12 |
| a4+ | 0.82 | −8.0 | 3 | 6 | 3 | **3** | 0 | 0 | 3.02e-02 | 2.49e-08 | 3 | 9.1e-12 |
| a4+ | 0.90 | −8.0 | 3 | 6 | 3 | **3** | 0 | 0 | 2.23e-02 | 2.49e-08 | 3 | 3.2e-11 |
| a4+ | 1.00 | −8.0 | 3 | 6 | 3 | **3** | 0 | 0 | 3.69e-02 | 2.54e-08 | 3 | 6.7e-11 |
| a4+ | 1.20 | −8.0 | 3 | 6 | 3 | **3** | 0 | 0 | 5.38e-02 | 2.84e-08 | 3 | 2.3e-10 |
| a4− | 0.28 | −8.0 | 3 | 6 | 3 | **3** | 0 | 0 | 4.42e-02 | 2.55e-08 | 4 | 1.8e-11 |
| a4− | 0.24 | −8.0 | 3 | 6 | 3 | **3** | 0 | 0 | 6.23e-02 | 2.46e-08 | 4 | 1.1e-11 |
| a4− | 0.18 | −8.0 | 3 | 6 | 3 | **3** | 0 | 0 | 1.67e-01 | 2.52e-08 | 6 | 1.7e-11 |
| a4− | 0.10 | −8.0 | 7 | 6 | 1 | 1 | 6 | 0 | 4.01e-01 | 2.60e-08 | 4 | 6.9e-12 |

**Direction-dependent.**
- **a4+ (quartic stronger) and a2+ (well shallower):** full coverage in every step, post
  rel-L2 ≈ 2.5e-8 — strong extrapolation far out (a₄ up to 1.2, a₂ up to −0.5).
- **a4− (quartic weaker):** full coverage to a₄≈0.18; at a₄=0.10 the true count jumps to 7
  and coverage collapses to 1/7.
- **a2− (well deeper):** coverage 5/6 to a₂=−17, then collapses (2/6) for a₂≤−20.
- One spurious solution appears once (a₂=−15.5, novel=1); otherwise the model invents none.

---

## Summary

| project | boundary | behaviour out-of-distribution |
|---|---|---|
| 1D_p | p=18 | direct OOD generation narrow: 7/7 to p=18.05, collapses 7→2 by p=19. But p=18-seeded Newton recovers 7/7 to p=19 |
| 2D   | s=1600 | robust: full 4/4 coverage to s=1800 (+12.5%) |
| a2a4 | rect.  | direction-dependent: full coverage for a₄↑ / a₂→0; collapses when the well deepens (a₂≤−20) or a₄→0 (k jumps to 7) |

Failure mode is uniform: **mode collapse (coverage loss), not wrong solutions** —
`novel≈0`, and covered branches are always recovered to solver precision.
