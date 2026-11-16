# roe — dam break with the classic Roe solver

`HydroEXA.model = "Roe"` — the classic Roe flux solver. This folder's runs are
the **baseline dynamics battery**: they validate that the AMR machinery
(level creation, ghost fill, reflux, subcycling, regridding) stays correct
when a strong wavefront crosses coarse-fine boundaries, using the solver with
no hydrostatic reconstruction.

## Why this test matters for the solver

1. **Conservation through shocks with AMR.** The dam break produces a genuine
   shock; conservative update + reflux must keep mass/momentum exact across
   coarse-fine faces where fine fluxes don't identically match coarse ones.
   A non-conservative AMR discretization shifts shock speeds — this test
   catches that against the SWASHES analytical position.
2. **Dynamic ghost fill.** Unlike the lake-at-rest battery (static state,
   roundoff-tight), here the coarse-fine ghost region carries a *moving,
   steep* state every step. Interpolation errors there become visible
   oscillations, not roundoff.
3. **Solver regression baseline.** Every change to flux kernels, CFL logic,
   or the AMR state machinery must keep this battery's behavior; a delta
   here is a red flag even when the lake tests stay green.

## Variants

| file | config | purpose |
|---|---|---|
| `inputs_amr` | reflux=1, subcycle=1, regrid=50 | stock full-machinery run |
| `inputs_amr_noreflux` | reflux=0 | bisect: flux-register correction |
| `inputs_amr_fastregrid` | regrid_int=5 (vs 50) | bisect: front-tracking cadence — recovers 101 % of `stable` |
| `inputs_amr_buf` | n_error_buf=4 (vs 1) | bisect: patch placed further ahead of the front — recovers 95 % |
| `inputs_amr_nosubcycle` | subcycle=0 | bisect: lockstep advance |
| `inputs_single_level` | max_level=0, 512×64 (`dx0`) | **base** yardstick: what AMR must beat |
| `inputs_single_level_fine` | max_level=0, 1024×128 (`dx0/2`) | **stable** yardstick: uniform solution at the AMR level-1 cell size |

Run from `tests/dambreak/`:

```bash
bash run.sh roe amr ../../install_<target>/hydroexa_DOUBLE/bin/HydroEXA
bash run.sh roe amr_noreflux  ...   # etc.
```

Artifacts land in `outputs/roe/<variant>__<tag>/` (plotfiles, log.txt, and
auto-generated `analysis_t*.png` / `probe_t*.png`).

## Findings log

### Battery n1 (2026-11-15) — plateau jitter retired, norms adopted

**The metric changed; earlier numbers are not comparable.** Plateau jitter
(peak-to-peak of `h` on the post-rarefaction plateau) is retired. It was
measured, but the CFL sweep exposed it as actively misleading:

| run | jitter (p2p) | L1 vs exact |
|---|---|---|
| `amr` (baseline) | 4.44e-5 | 1.39e-5 |
| `cfl=0.25` | 1.24e-5 | 1.40e-5 |
| `cfl=0.125` | **3.48e-6** | **1.72e-5** |

`cfl=0.125` had **12.7× less jitter yet 23 % worse L1**. Minimising the
smoothness proxy was not minimising error — the metric was rewarding a
quieter-looking plateau while the solution drifted further from the exact
one. (L1 above is the old unweighted-mean definition; see below.)

**What replaced it.** `analyze.py` now computes length-weighted L1, L2 and
Linf of the depth error against three yardsticks:

- the SWASHES exact solution → `L1`, `L2`, `Linf`
- the two uniform runs the AMR grid straddles → `stable_*` (1024×128, i.e. the
  level-1 cell size) and `base_*` (512×64, the level-0 cell size), each with

  ```
  gain_<slot>_p = (that single-level run's error vs exact)_p / (this run's error)_p
  ```

Length weighting is required because the AMR-composite centreline is
non-uniform: fine cells sit where the wave is, so an unweighted point average
silently rewards refinement. On a uniform grid the weights reduce to the
textbook mean / RMS / max, which was verified numerically (`L2 = 1/√2` exactly
for `sin 2πx`; a grid twice as dense in the right half reproduces the uniform
answer where the naive mean is biased by 0.5 %).

**Acceptance test:** `gain_base_* > 1` (refinement must beat the unrefined
grid) and `gain_stable_*` near 1 (it should recover the uniform fine-grid
accuracy). In words: *one added level at `r = 2` should buy up to 2× in L1;
buying ~1× means the refinement is decoration, and falling short of the fine
uniform run means the coupling is losing the resolution the fine cells paid
for.*

**Consequences for the r29 conclusions below** — they were ranked by jitter:

- `amr` vs `amr_noreflux`: jitter said "identical → reflux irrelevant". The
  norms agree on L1 (1.39e-5 vs 1.63e-5) but disagree on Linf
  (5.98e-4 vs 1.26e-3): reflux off is **worse on the shock**, which the
  p2p metric could not see.
- `amr_nosubcycle`: better jitter (1.44e-5 vs 4.44e-5) but its `gain_Linf` is
  0.79 — it improves the plateau by degrading the shock. A trade, not a fix.
- `hydrostatic_roe/amr` wins L1 (8.94e-6) while being the *roughest* on the
  plateau, so "plateau smoothness" was never a proxy for accuracy in either
  direction.

### Baseline: AMR vs the two uniform yardsticks (tag `n1`)

Both single levels were run so the AMR result can be judged against the
resolution it actually has: `single_level_fine` is a uniform 1024×128 run,
i.e. the level-1 cell size, and `single_level` is the 512×64 level-0 run.
L1 of the depth error against the exact solution (V100, `np=1`, CFL 0.5):

| t | base (dx0) | `amr` | stable (dx0/2) | gain_base | stable recovered |
|---|---|---|---|---|---|
| 1.0 | 1.03e-5 | 6.11e-6 | 6.12e-6 | 1.69 | 100 % |
| 2.0 | 1.23e-5 | 8.76e-6 | 7.12e-6 | 1.40 | 81 % |
| 5.0 | 1.49e-5 | 9.38e-6 | 8.61e-6 | 1.59 | 92 % |
| 10.0 | 1.72e-5 | 1.12e-5 | 9.73e-6 | 1.54 | 87 % |
| 15.0 | 1.85e-5 | 1.30e-5 | 1.07e-5 | 1.43 | 82 % |
| 20.0 | 1.94e-5 | 1.38e-5 | 1.11e-5 | 1.41 | 80 % |
| 25.0 | 1.01e-5 | 8.23e-6 | 5.86e-6 | 1.23 | 71 % |
| 30.0 | 9.82e-6 | 7.93e-6 | 5.85e-6 | 1.24 | 74 % |

What this pins down:

- **AMR always beats no-AMR** (1.23–1.69×), so the machinery is not harmful.
- **It never reaches the uniform fine grid** (71–100 %), and the gap grows with
  time. Since the fine cells solve the same equations as `single_level_fine`,
  the missing accuracy is in the AMR coupling — level thresholds, ghost fill,
  subcycled time interpolation, average-down — not in the flux scheme. That is
  where the next work should go, and the deficit is now measurable per frame.
- **One doubling buys only 1.7× in L1, not 2×**, even for the perfectly uniform
  pair (1.85e-5 → 1.07e-5). So ~2× is an upper bound that this scheme does not
  reach; AMR's 1.4× should be read against 1.7×, not 2×.
- **`Linf` is misleading at a shock:** `gain_stable_Linf` can exceed 100 %
  because the finer grid resolves a sharper front whose position error is a
  larger absolute spike. L1/L2 are the accuracy norms; Linf measures shock
  sharpness.

### Where the AMR error comes from — the flux is exonerated

The conclusion the two yardsticks support: **the Roe flux scheme is not the
problem.**

1. **The same flux converges.** Uniform 512×64 → 1024×128 improves L1 by 1.7×
   (1.85e-5 → 1.07e-5 at t=15). The discretisation is consistent and behaves as
   a first-order method; nothing about the Riemann solver is misfiring.
2. **AMR loses resolution it has already paid for.** With the same flux and the
   same finest `dx`, AMR lands at 1.30e-5 — between the two uniform runs, and
   20–30 % short of the fine one, with the shortfall growing over time.
3. **The extra error is spectrally distinct.** On the shelf at t=15, the AMR
   error against the exact solution is dominated by 0.27–0.50 m wavelengths at
   amplitudes 1.6–3.5e-6. The uniform 1024×128 run on the same shelf has *no*
   such content: its spectrum is 1.2–3.5 m at 1–3e-7, an order of magnitude
   smaller. A flux scheme cannot produce structure at wavelengths between one
   fine cell (0.0098 m) and the patch width while doing nothing of the kind
   when run uniformly.

So the defect lives in the level coupling, not in the scheme — and the bisection
(tag `n1`, one switch per run, all scored against `stable`) now says **which**
coupling term it is:

| variant | change | L1 @t=15 | gain_base | stable recovered |
|---|---|---|---|---|
| `amr` | stock: regrid 50, subcycled, reflux | 1.30e-5 | 1.43 | 82 % |
| `amr_noreflux` | reflux off | 1.64e-5 | 1.13 | 65 % |
| `amr_nosubcycle` | lockstep | 1.11e-5 | 1.67 | 96 % |
| `amr_buf` | `amr.n_error_buf = 4` | 1.13e-5 | 1.64 | 95 % |
| `amr_fastregrid` | `amr.regrid_int = 5` | **1.06e-5** | **1.75** | **101 %** |

- **Reflux is load-bearing.** Turning it off is the only change that makes
  things worse (65 %, shock error +0.031 vs −0.008), so the dual effective
  fluxes and the side-selected `D±` in the flux registers are consistent.
- **Subcycling costs ~14 points**, but it is not the main term.
- **Regrid cadence is the main term, and it is a configuration defect, not a
  numerical one.** `n_error_buf` defaults to 1 cell, and `regrid_int = 50`
  lets the shock travel ~0.23 m (twelve coarse cells) *out of* the fine patch
  between regrids. The front is therefore captured at level-0 resolution for
  most of each interval, and because the plateau behind it is a steady state,
  the resulting error is frozen into the fluid — it reappears as 0.27–0.35 m
  "teeth" in `AMR − exact` *inside* level 1, which is what made the noise look
  like an intra-level instability. With `regrid_int = 5` the teeth vanish
  (`AMR − stable` on the shelf: 3.1e-6 → 7.2e-7), shock position error drops to
  the uniform fine run's value, and AMR reaches 101–105 % of `stable` for no
  measurable runtime cost. The `n_error_buf = 4` run recovers most of the same
  ground (95 %) by placing the patch further ahead of the front, which is the
  same mechanism seen from the other side.


### Battery r29 (2026-11-14, nukwa-02, V100 DOUBLE, np=1, unsplit-CFL dt, plot_freq=1)

> **Historical — scored by p2p jitter, which is retired.** Kept for the record;
> see the n1 section above for how the norm-based ranking differs.

All runs: exit 0, t=30 complete, 31 plotfiles each, all 31 figures auto-rendered
by `run.sh`. **Metric bug corrected:** the earlier `plateau_band()` selected the
flat run with the highest h — i.e. the LEFT RESERVOIR (h = h_L), not the
post-rarefaction plateau. Every "plateau jitter" number before r29 was therefore
a lake-at-rest measurement. Both bands are now reported:

| run | shock err (%L) | **plateau** h p2p | plateau hu p2p | **lake** h p2p | lake level offset |
|---|---|---|---|---|---|
| `amr` | 0.08% | 4.43e-05 | 1.35e-06 | 5.02e-05 | 5.3e-06 |
| `single_level` | 0.12% | **6.89e-06** | 3.07e-07 | 7.56e-05 | 1.0e-05 |
| `amr_noreflux` | 0.31% | 4.40e-05 | 1.35e-06 | 5.26e-05 | 8.9e-06 |
| `amr_nosubcycle` | 0.08% | 1.45e-05 | 4.69e-07 | 5.31e-05 | 7.5e-06 |
| `hydrostatic_roe/amr` | 0.09% | **7.37e-05** | **1.43e-05** | **1.07e-15** | **0** |

**What this actually says (supersedes the r27 conclusions):**

- **Plateau (intermediate state):** the fixed-grid control is the smoothest
  (6.9e-6) and full AMR is ~6× rougher (4.4e-5) → **AMR interpolation does add
  noise on the plateau**. `amr_noreflux` is identical to `amr` (4.40 vs 4.43e-5)
  → flux registers are NOT the source. `amr_nosubcycle` drops to 1.45e-5 →
  **subcycling / coarse-level time interpolation is the dominant contributor
  (~3×)**, making CC-3 (time interpolation) the concrete next target.
- **Lake at rest:** classic Roe is not well-balanced — ~5e-5 p2p and a surface
  0.1% low (h = 4.9947e-3 vs 5.0000e-3), in the fixed grid and AMR alike.
  HydrostaticRoe holds it at 1e-15 with zero offset. That is the correct
  well-balancedness discriminator between the solvers.
- **HydrostaticRoe is the noisiest on the plateau** (h 7.4e-5, hu 1.4e-5 ≈ 10×
  classic Roe) — the hu sawtooth visible in its figures. It trades plateau
  smoothness for exact lake preservation; it is not a "clean reference".
- `amr_noreflux` remains the worst on shock position (0.31% vs 0.08% for `amr`)
  → reflux is the momentum/shock-speed accuracy knob; keep it ON.

### dt formula history (2026-11-13/14)

`Roe::compute_dt_Impl` now uses the **unsplit 2D CFL**
`1/((|u|+c)/dx + (|v|+c)/dy)` — identical to HydrostaticRoe — so the input
`cfl = 0.5` means an actual unsplit CFL of 0.5. The earlier split wavespeed
form `min(dx/(|u|+c), dy/(|v|+c))` was 2× less restrictive on square cells
(unsplit CFL = 2·cfl = 1.0). The jitter metric proved insensitive to the
formula (5–9e-5 lake / 4.4e-5 plateau, both CFL 0.5 and 1.0), but the
unsplit form was adopted so that "CFL" has the same meaning in both
solvers. It did NOT fix the plateau noise because the noise is not
CFL-marginality.
