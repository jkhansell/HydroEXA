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

### Battery r29 (2026-11-14, nukwa-02, V100 DOUBLE, np=1, unsplit-CFL dt, plot_freq=1)

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
