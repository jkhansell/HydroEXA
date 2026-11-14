# hydrostatic_roe — dam break with the well-balanced Roe solver

`HydroEXA.model = "HydrostaticRoe"` — the hydrostatic-reconstruction Roe
solver (the well-balanced variant). This folder runs the **same dam-break
battery** on the well-balanced solver to verify that hydrostatic
reconstruction does not degrade the dynamics that the classic Roe battery
validates: conservation through the shock, CFL control, and AMR machinery
under a moving wavefront.

## Why this test matters for the solver

1. **Well-balanced must not mean less-robust.** Hydrostatic reconstruction
   modifies the pressure fluxes to preserve lake-at-rest exactly (validated
   in `tests/LakeAtRestHydrostaticRoe/`). This test proves the modification
   survives genuine dynamics: the same shock, rarefaction, and AMR crossings
   as the classic solver, with conservative behavior intact.
2. **Cross-solver consistency.** On a *flat bed* both solvers should produce
   nearly identical profiles (the hydrostatic correction vanishes for
   constant η up to reconstruction details). Divergence between
   `roe/amr` and `hydrostatic_roe/amr` on this flat-bed case isolates
   solver-specific bugs from AMR-machinery bugs, which both share.
3. **Unsplit-CFL dt.** This solver's dt uses the unsplit 2D form
   `1/((|u|+c)/dx + (|v|+c)/dy)` — the strictest of the two formulas. Its dt
   trace doubles as a CFL-reference for the Roe folder's split-form dt.

## Variants

| file | config | purpose |
|---|---|---|
| `inputs_amr` | reflux=1, subcycle=1, regrid=50 | stock full-machinery run |

(Bisect variants can be added mirroring `../roe/` when needed.)

Run from `tests/dambreak/`:

```bash
bash run.sh hydrostatic_roe amr ../../install_<target>/hydroexa_DOUBLE/bin/HydroEXA
```

## Findings log

- Runs complete t=30 with no aborts; dMass = 0 (2026-11-13).
- Shows the same post-shock plateau wiggle as the classic Roe AMR runs
  (±~3e-5) — consistent with a shared AMR-machinery cause rather than a
  solver-specific one (both solvers' ghost fill was η-routed in R2-3).
- Flat-bed profile at t=15 matches the Roe run's plateau/shock positions.
