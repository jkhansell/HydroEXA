# dambreak — End-to-End (E2E) solver tests

This directory is the home of the **rectangular dam break** end-to-end test
battery. These are the tests intended for the future CI/CD pipeline: each run
exercises the *full* solver path (input HDF5 → AMR mesh → fluxes → plotfile),
as opposed to unit-level checks.

> **Why end-to-end and not unit tests (yet):** the solver is a tightly coupled
> stack (AMReX AMR machinery + CRTP solvers + GPU kernels). Unit tests of
> individual kernels exist implicitly (`tests/*/make_test_h5.py` + small
> configs), but the failure modes we actually hit in development (level-time
> desynchronization, h-interpolation across coarse-fine faces, CFL formula
> inconsistencies) only manifest through the full pipeline. E2E tests with
> tight acceptance criteria are the highest-value automation target; kernel
> unit tests can be evaluated later once the E2E battery is stable.

## The analytical benchmark (SWASHES)

The reference solution is the **exact self-similar Riemann solution** of the
1D shallow water equations for a dam break over a flat bed (`z = 0`), the same
problem as the `dam-break` case in the
[SWASHES](https://swashes.github.io/) analytic-solution library (Delestre et
al. 2013, *Physics/1/dambreak_sd*).

Initial condition (built by `make_test_h5.py` with its defaults):

| quantity | value |
|----------|-------|
| upstream depth `h_L` | 0.005 |
| downstream depth `h_R` | 0.001 |
| dam position `x0` | 5.0 |
| domain | 10 × 0.5 (square cells) |
| bed `z` | flat, 0.0 |

Structure of the exact solution at time `t` (self-similar in `x/t`):

```
h_L                         rarefaction fan                h_R
──── plateau ────────────╱                              ── plateau ────
                        ╱ smooth h decrease
   u = 0              ╱  u increases                  post-fan state ── shock ── u=0
```

- a **left-going rarefaction** (smooth fan) connecting `h_L` to the
  intermediate state,
- a **right-going shock** connecting the intermediate state to `h_R`.

`analyze.py` overlays this exact solution on a plotfile's y-averaged (or
centerline) `h` profile — shock position, rarefaction shape, and plateau
level are all quantitative acceptance checks. Because the bed is flat, this
test does *not* exercise well-balancing (that is the lake-at-rest battery's
job); it exercises **dynamics**: conservation through shocks, CFL control,
and AMR machinery under a moving wavefront.

## Directory layout

```
dambreak/
├── README.md            ← this file
├── RectangularDambreak.h5   ← shared input state (single copy, all tests)
├── make_test_h5.py      ← regenerates the HDF5 (uv run python make_test_h5.py)
├── analyze.py           ← profile vs. exact SWASHES solution → PNG
├── analytical.py        ← exact solution helper
├── plot_output.py       ← frame/GIF rendering
├── profile_probe.py     ← yt centerline probe (levels, C-F edges, jump table)
├── run.sh               ← generic runner: bash run.sh <solver> <variant> <exe>
├── outputs/             ← ALL run artifacts, per solver (not committed)
│   ├── roe/
│   └── hydrostatic_roe/
└── solvers/
    ├── roe/             ← classic Roe: inputs variants + README
    └── hydrostatic_roe/ ← hydrostatic (well-balanced) Roe: inputs + README
```

- **Input `.h5` lives once at the parent** — every solver/variant consumes the
  same initial state, so cross-solver differences are pure solver+AMR effects.
- **Analysis scripts live at the parent** and write to
  `outputs/<solver>/*.png`.
- Solver folders hold only what differs between solvers: input variants and
  their test documentation.

## Test matrix

| solver folder | variant | config | what it validates |
|---|---|---|---|
| `roe` | `amr` | reflux=1, subcycle=1, regrid=50, max_level=1 | full AMR machinery under dynamic flow |
| `roe` | `amr_noreflux` | reflux=0 | isolates the flux-register correction |
| `roe` | `amr_nosubcycle` | subcycle=0 (lockstep dt) | isolates subcycled time interpolation |
| `roe` | `single_level` | max_level=0 | fixed-grid scheme baseline (no AMR) |
| `hydrostatic_roe` | `amr` | reflux=1, subcycle=1, regrid=50 | well-balanced solver under dynamics |

The bisection principle: any AMR-only defect must vanish in `single_level`;
any reflux-only defect must vanish in `amr_noreflux`; any subcycle-only
defect must vanish in `amr_nosubcycle`. Run the stock variant first, then
bisect with the single-off variants.

## Running

From the cluster, source the machine profile first (`source machines/<target>`)
so MPI/compilers are in the environment, then:

```bash
cd tests/dambreak
bash run.sh roe amr ../../install_<target>/hydroexa_DOUBLE/bin/HydroEXA [np]
bash run.sh hydrostatic_roe amr ../../install_<target>/hydroexa_DOUBLE/bin/HydroEXA [np]
```

Artifacts land in `outputs/<solver>/`:

- `plt_<variant>####/` — plotfiles (every 5 time units)
- `log_<variant>.txt` — full stdout
- `<variant>_t<time>.png` — profile-vs-analytical overlay
- `<variant>_probe_t<time>.png` — centerline probe

## Acceptance criteria (CI)

1. **No abort** — exit status 0; the dt-collapse `Abort` must never fire.
2. **Conservation** — `[Conservation]` lines: `dMass` within roundoff;
   total momentum smooth (shock speed consistent with the plateau state).
3. **Analytical agreement** — shock position within 0.5% of the domain
   length; profile L1 error of the same order as the `single_level`
   control.
4. **Plateau smoothness** — detrended plateau jitter ≤ 2e-4 (current
   classic-Roe baseline is 5–9e-5; HydrostaticRoe achieves 1e-15).

## Status / known issues

- **Jitter / accuracy (metric corrected 2026-11-14):** two bands are scored —
  the intermediate plateau (post-rarefaction) and the undisturbed left
  reservoir (well-balancedness). On the plateau, AMR adds ~6× noise over the
  fixed-grid control (4.4e-5 vs 6.9e-6 p2p) and subcycling is the dominant
  contributor (`amr_nosubcycle` 1.45e-5); flux registers are not involved
  (`amr_noreflux` 4.40e-5). On the lake, classic Roe is ~5e-5 with the surface
  0.1% low, HydrostaticRoe is 1e-15 with zero offset. Earlier claims that the
  wiggle was "classic-Roe flux dispersion, AMR-exonerated" were based on the
  band-selection bug (the old code scored the lake) and are withdrawn — see
  `solvers/roe/README.md`. CI thresholds: plateau jitter ≤ 2e-4, lake jitter
  ≤ 2e-4, shock position ≤ 0.5% of L.
- `tests/dambreakNoAMR/` (legacy) is a **different problem** (51.2 × 1.6
  domain) — not a valid control for this battery; `solvers/*/inputs_single_level`
  replaces it as the fixed-grid reference.
