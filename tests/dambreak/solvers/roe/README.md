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
| `inputs_amr_nosubcycle` | subcycle=0 | bisect: lockstep advance |
| `inputs_single_level` | max_level=0 | fixed-grid scheme control |

Run from `tests/dambreak/`:

```bash
bash run.sh roe amr ../../install_<target>/hydroexa_DOUBLE/bin/HydroEXA
bash run.sh roe amr_noreflux  ...   # etc.
```

Artifacts land in `outputs/roe/<variant>__<tag>/` (plotfiles, log.txt, and
auto-generated `analysis_t*.png` / `probe_t*.png`).

## Findings log

### Battery r27 (2026-11-14, nukwa-02, V100 DOUBLE, np=1, unsplit-CFL dt)

All runs: exit 0, no aborts, t=30 complete. Metrics at t=15 from
`analyze.py` (shock position vs SWASHES; plateau jitter = detrended p2p of
centerline h on the post-rarefaction plateau):

| run | shock error (% of L) | plateau jitter |
|---|---|---|
| `amr` | 0.08% | 5.85e-05 |
| `single_level` | 0.12% | 9.39e-05 |
| `amr_noreflux` | 0.31% | 6.36e-05 |
| `amr_nosubcycle` | 0.08% | 6.17e-05 |
| `hydrostatic_roe/amr` (reference) | 0.09% | **1.3e-15** |

**Conclusions (closes the 2026-11-13 jitter investigation):**

- The post-shock plateau wiggle (±~3e-5 visually) is a **classic-Roe
  flux/limiter characteristic at this resolution**, present in the
  fixed-grid control. It is NOT: the AMR machinery (single_level rings the
  most), the flux registers (noreflux unchanged), subcycling (nosubcycle
  unchanged), or CFL margin (see below). The well-balanced HydrostaticRoe
  run is jitter-free at machine precision on the same AMR config — a useful
  cross-solver discriminator for future regressions.
- Acceptance threshold for CI: plateau jitter ≤ 2e-4 on the plateau.
- Shock position agrees with the analytical solution to ≤0.31% of the
  domain length in all configurations.

### dt formula history (2026-11-13/14)

`Roe::compute_dt_Impl` now uses the **unsplit 2D CFL**
`1/((|u|+c)/dx + (|v|+c)/dy)` — identical to HydrostaticRoe — so the input
`cfl = 0.5` means an actual unsplit CFL of 0.5. The earlier split wavespeed
form `min(dx/(|u|+c), dy/(|v|+c))` was 2× less restrictive on square cells
(unsplit CFL = 2·cfl = 1.0). The jitter metric proved insensitive to the
formula (5–9e-5 at both CFL 0.5 and 1.0), but the unsplit form was adopted
so that "CFL" has the same meaning in both solvers.
