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

## Findings log (dam-break jitter investigation, 2026-11-13/14)

Centerline probe (`profile_probe.py`) at t=15, y = mid-domain. All runs
complete t=30 with no aborts; dMass = 0.

| run | config | plateau wiggle (± on h≈0.0025) |
|---|---|---|
| `single_level` (dbL0) | max_level=0, unsplit-CFL dt | **none — smooth** |
| `amr` (dbRoe) | split-CFL dt, subcycle+reflux | yes, whole L1 strip |
| `amr` + unsplit dt (dbRoe3) | CFL unified with HydrostaticRoe | **yes, unchanged** |
| `amr_noreflux` (dbRoeNR) | reflux=0 | yes, unchanged |
| `amr_nosubcycle` (dbRoeNS) | subcycle=0 | yes, unchanged |

Eliminated as causes: **flux registers** (wrong fingerprint too — wiggle spans
the L1 interior, not just C-F faces), **CFL formula** (unsplit build
unchanged; `Roe.cpp` dt reverted to the split wavespeed form per that
result), **subcycling alone**. The scheme itself is clean at fixed grid.

Remaining suspects: the runtime coarse-fine ghost fill under *dynamic* flow
(works at roundoff for the static lake, may not be consistent for a moving
shock), the initial-condition prolongation, and regrid-time level recreation
(regrid_int=50). Next bisect: `single_level` + regrid=0 vs regrid runs, and
ghost-fill-time consistency between substeps.

**dt formula note:** `Roe::compute_dt_Impl` uses the split wavespeed form
`min(dx/(|u|+c), dy/(|v|+c))` — on square cells this is 2× less restrictive
than the unsplit form HydrostaticRoe uses (i.e. unsplit CFL = 2·cfl). It was
temporarily unified to the unsplit form to test the jitter hypothesis and
reverted when falsified. Revisit only with a dedicated CFL-marginality test.
