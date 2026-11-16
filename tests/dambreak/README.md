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
├── analyze.py           ← readers + norm maths (L1/L2/Linf, gain) — source of truth
├── analysis.py          ← one-shot: metrics for a plotfile + rendered figure
├── plot.py              ← 6-row renderer (map, resolution, h, errors, hu, metrics)
├── metrics_table.py     ← battery scoreboard → CSV (one row per run, score time)
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
| `roe` | `single_level_fine` | max_level=0, 1024×128 (`dx0/2`) | **stable** yardstick: uniform solution at the AMR level-1 cell size |
| `roe` | `single_level` | max_level=0, 512×64 (`dx0`) | **base** yardstick: fixed-grid scheme baseline (no AMR) |
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
source machines/<target>          # MPI/compilers/CUDA
bash run.sh roe single_level_fine ../../install_<target>/hydroexa_DOUBLE/bin/HydroEXA [np] [tag]
bash run.sh roe single_level      ... [np] [tag]   # same tag
bash run.sh roe amr               ... [np] [tag]   # same tag → both references
bash run.sh hydrostatic_roe amr   ... [np] [tag]
```

**Both single levels must run first with the same tag**, because every other
variant auto-resolves `outputs/<solver>/single_level_fine__<tag>` (stable) and
`outputs/<solver>/single_level__<tag>` (base) as its yardsticks.

A bisect sweep should not pay for figures: prefix `HYDROEXA_SKIP_PLOTS=1` to
solve only, then score the whole sweep in one pass with
`uv run python metrics_table.py --times 1,2,5,10,15,20 -o metrics.csv`.

Artifacts land in `outputs/<solver>/<variant>__<tag>/`:

- `plt_<variant>####/` — native plotfiles (`plot_freq` is a physical time
  interval, not a step count)
- `log.txt` — full stdout
- `analysis_t<time>.png` — 6-row figure: 2D map **with the level-1 box
  outlines drawn** (so a wiggle that lines up with a patch boundary is
  distinguishable from one generated inside a grid), resolution strip, `h`
  overlay (exact + both single levels + this run), error row (this run minus
  exact and minus each single level, full-domain symlog plus shelf→elbow zoom),
  `hu` overlay, metrics box
- `probe_t<time>.png` — centerline probe

## The two single-level yardsticks

Accuracy alone is not the acceptance test — refinement has to earn its keep,
and "keep" is defined by two uniform-grid runs at the two cell sizes an AMR run
straddles (domain 10 × 1.25, so `dx0 = 0.0195`):

| variant | grid | role |
|---|---|---|
| `single_level_fine` | 1024 × 128 (`dx0/2` = the AMR **level-1** cell size) | **stable** solution: what AMR with one added level should reproduce |
| `single_level` | 512 × 64 (`dx0` = the AMR **level-0** cell size) | **base** solution: the unrefined run AMR must beat |

`single_level_fine` uses `RectangularDambreakFine.h5`, generated at level-0
1024 × 128; everything else is identical to `inputs_single_level` (`max_level=0`).

```
gain_<slot>_p = (that single-level run's error vs exact)_p / (this run's error)_p
```

Read them differently, because they answer different questions:

- `gain_base > 1` — the extra level beats no-AMR. `≈ 1` means the refinement is
  decoration; `< 1` means it is actively harmful.
- `gain_stable ≈ 1` — AMR recovers the uniform fine-grid accuracy. Reported in
  the summary as "accuracy recovered … %", where 100 % is parity. Well below
  100 % means a level's worth of resolution is being lost in the AMR coupling
  (ghost fill, reflux, subcycled time interpolation) rather than in the flux
  scheme, since the fine cells solve the same equations as `single_level_fine`.

## Acceptance criteria (CI)

1. **No abort** — exit status 0; the dt-collapse `Abort` must never fire.
2. **Conservation** — `[Conservation]` lines: `dMass` within roundoff;
   total momentum smooth (shock speed consistent with the plateau state).
3. **Analytical agreement** — shock position within 0.5% of the domain
   length, and the three norms of the depth error against the exact solution
   (`L1`, `L2`, `Linf`) no worse than the `base` single-level run.
4. **Refinement must earn its keep** — `gain_base_L1`, `gain_base_L2`,
   `gain_base_Linf` > 1, and `gain_stable_*` close to 1. With `r = 2` and a
   first-order scheme, one added level should buy roughly 2× in L1 against the
   base run. A run that refines without improving is a defect regardless of how
   smooth its profile looks.

   `gain_stable_* ≥ 0.95` is **achievable on this case** (`amr_fastregrid`,
   `amr_nosubcycle`, `amr_buf` all reach 95–101 % at t=15). A configuration
   sitting at 80 % is therefore a front-tracking defect, not an accuracy limit
   of the scheme: check `amr.regrid_int` / `amr.n_error_buf` before suspecting
   the flux or the reflux machinery. The general rule for sizing those two
   knobs on any new case is **`docs/PIPELINE.md` §8a** — in short, the shock
   must not outrun the patch between regrids:
   `regrid_int * cfl * (U_front / lambda_max) < n_error_buf`.

### How the norms are computed

All three norms are true integrals on the (non-uniform) AMR-composite
centreline, so they are independent of how refinement clustered the samples:

```
L1   = sum(w |e|)      / sum(w)          mean absolute error
L2   = sqrt(sum(w e^2) / sum(w))         RMS error
Linf = max(|e|)                          worst point
```

`w` are trapezoidal cell widths from the composite x coordinates; on a
uniform grid this reduces exactly to the textbook mean / RMS / max. An
unweighted point average would silently reward refinement, because fine cells
are concentrated where the error is.

**The scored band is the whole shelf.** `plateau_band()` covers the flat
intermediate state from the rarefaction tail to the **analytical shock elbow**
(right edge pinned to the exact shock position, left edge where the exact
solution first reaches the shelf level) — deliberately untrimmed, so the error
build-up toward the front is part of the number instead of a 5 % margin nobody
looks at. Consequences:

- `plateau_Linf` now includes the shock-adjacent cell and is therefore of the
  same order as the global `Linf`; it measures the front, not the shelf.
- `plateau_L1` / `plateau_L2` remain shelf diagnostics, with the shock foot
  contributing only through its one or two cells.
- The zoom panel in the figures is symlog for the same reason: the band spans
  errors from O(1e-7) on the shelf to O(1e-3) at the elbow.
- The left reservoir (`lake_range`, `lake_level_err`) is a **separate** band and
  a different diagnostic — well-balancedness. It only exists while the
  rarefaction is still inside the domain (`t ≲ 22.6` here).

**Plateau jitter is no longer a metric.** It was measured as peak-to-peak of
`h` on the interior of the post-rarefaction plateau and abandoned after the
CFL sweep showed it to be actively misleading: `cfl=0.125` had 12.7× less
jitter than the baseline yet 23 % *worse* L1. Both bands (plateau and left
reservoir) are still reported as band-restricted L1/L2/Linf and as
`lake_range` / `lake_level_err`.

### Two intra-level hypotheses that were tested and refuted

The AMR error at `regrid_int = 50` looks like an instability *inside* level 1:
0.27–0.35 m oscillations on the shelf, 10–50× larger than the uniform fine run
over the same region. Two explanations were checked and are **not** the cause;
both are cheap to re-run if the symptom returns.

1. **Patch boundaries.** With the level-1 boxes drawn on the 2D map it is
   visible that they are 96–128 fine cells wide, not 32, and binning
   `|AMR − exact|` by distance to the nearest box face gives near-face/mid-box
   = **1.18×** — no concentration. The ringing is generated inside grids, and
   the +1-cell boundary factor is entirely explained by the box faces also
   being where the patch's *leading edge* sits.
2. **Level churn** (refine → coarsen → refine, each hop re-injecting level-0
   error through `FillCoarsePatch`). The plateau is untagged, so this was the
   leading candidate. Counting level transitions per column over all 31 frames
   gives churn ≤ 1 everywhere (the patch grows monotonically) and
   corr(churn, mean |e|) = **+0.13** — refuted.

What it actually is: **regrid cadence.** See the bisect in *Status / known
issues* — with `regrid_int = 5` the oscillations disappear completely.


### Scoreboard

`metrics_table.py` writes one CSV row per (run, score time):

```bash
uv run python metrics_table.py --times 1,2,5,15,30 -o metrics.csv
uv run python metrics_table.py --times 15 outputs/roe/amr__n1   # one run
```

Each run is auto-scored against the sibling `single_level_fine__<tag>` and
`single_level__<tag>` runs, so **run both single levels first, with the same
tag**. Columns include `L1`, `L2`, `Linf` (vs exact), `plateau_*`, `lake_*`,
`stable_*` / `base_*` (profile difference) and `gain_stable_*` /
`gain_base_*` (vs exact, ratio to this run).

## Status / known issues

- **Metrics are now norms, not jitter (2026-11-15).** L1/L2/Linf against the
  exact solution and against both single-level yardsticks, plus the two `gain_*`
  families. The pre-clean scoreboard (jitter-era columns) is preserved as
  `metrics_preclean.csv`, but its `L1` used a naive unweighted mean and it had
  no fine yardstick, so it is not comparable with current numbers.
- **First clean baseline, tag `n1`** (V100, `np=1`, CFL 0.5, shallow IC),
  L1 of the depth error vs exact:

  | t | `single_level` (dx0) | `amr` | `single_level_fine` (dx0/2) | `gain_base` | stable recovered |
  |---|---|---|---|---|---|
  | 1 | 1.03e-5 | 6.11e-6 | 6.12e-6 | 1.69 | 100 % |
  | 2 | 1.23e-5 | 8.76e-6 | 7.12e-6 | 1.40 | 81 % |
  | 5 | 1.49e-5 | 9.38e-6 | 8.61e-6 | 1.59 | 92 % |
  | 10 | 1.72e-5 | 1.12e-5 | 9.73e-6 | 1.54 | 87 % |
  | 15 | 1.85e-5 | 1.30e-5 | 1.07e-5 | 1.43 | 82 % |
  | 20 | 1.94e-5 | 1.38e-5 | 1.11e-5 | 1.41 | 80 % |
  | 25 | 1.01e-5 | 8.23e-6 | 5.86e-6 | 1.23 | 71 % |
  | 30 | 9.82e-6 | 7.93e-6 | 5.85e-6 | 1.24 | 74 % |

  So AMR beats the unrefined run everywhere (`gain_base` 1.23–1.69) but
  recovers only **71–100 %** of the uniform fine-grid accuracy, and the deficit
  widens with time. One added level should be worth ~2× in L1; the fine run
  itself only achieves 1.7× per doubling, and AMR gets ~1.4×. The deficit is
  *not* a flux or coupling defect — see the bisect below: it is regrid cadence,
  and `regrid_int = 5` restores parity with the uniform fine run.
- **Linf behaves perversely at the shock.** `gain_stable_Linf` can exceed 100 %
  (AMR "better" than the fine run): a sharper uniform grid places the shock
  within a smaller fraction of `dx` of the exact discontinuity and pays a
  larger pointwise spike, while AMR's smeared front looks better in the max
  norm. Treat `Linf` as a shock-capturing diagnostic, not an accuracy score;
  L1/L2 are the accuracy norms.
- **Same-config A/B verdict (2026-11-14, withdrawn claim):** an earlier
  "AMR exonerated" conclusion compared against `tests/dambreakNoAMR`, which is
  a **different problem** (51.2×1.6 domain) and therefore not a valid control.
  With AMR and single level run at the *same* 512×64 config, AMR stalls at a
  floor ~4.5e-5 on the plateau while the single level decays to 3.2e-6
  (6.6× worse at t=15, 9.3× at t=30).
- **Reflux, subcycling and regridding, split apart (tag `n1`).** Same problem,
  same binary, one switch changed per run; L1 vs exact and the fraction of the
  uniform fine-grid accuracy recovered:

  | variant | what changed | L1 @t=15 | gain_base | stable recovered @15 | @20 |
  |---|---|---|---|---|---|
  | `amr` | stock: regrid 50, subcycled, reflux | 1.30e-5 | 1.43 | 82 % | 80 % |
  | `amr_noreflux` | reflux off | 1.64e-5 | 1.13 | 65 % | 61 % |
  | `amr_nosubcycle` | lockstep | 1.11e-5 | 1.67 | 96 % | 96 % |
  | `amr_buf` | `amr.n_error_buf = 4` | 1.13e-5 | 1.64 | 95 % | 98 % |
  | `amr_fastregrid` | `amr.regrid_int = 5` | **1.06e-5** | **1.75** | **101 %** | **105 %** |

  So: reflux is load-bearing (removing it costs 17 points and triples the shock
  position error), subcycled time interpolation costs ~14 points, and **the
  dominant term is regrid cadence**: with `regrid_int = 5` AMR *matches* the
  uniform fine run (101–105 % of it, i.e. within noise of parity) and beats
  no-AMR by 1.75–1.85×.

  Mechanism: `n_error_buf` defaults to **1 cell**, so the level-1 patch is
  placed with ~one coarse cell of margin ahead of the front, and
  `regrid_int = 50` means the shock then travels up to 50·`dt0`·0.21 m/s ≈
  0.23 m — about twelve coarse cells — *out of the fine region* before the
  patch is re-cut. For most of each interval the front is captured at level-0
  resolution, and the plateau behind it is a steady state, so the error the
  shock leaves behind is frozen into the fluid instead of decaying. That is
  why the damage shows up as a spatial structure *inside* level 1 (0.27–0.35 m
  "teeth" in `AMR − exact` on the shelf) even though nothing is wrong at the
  patch boundary: the patch boundary is only where the error is *imported*.
  Regridding every 5 steps removes the teeth entirely (`AMR − stable` on the
  shelf drops from 3.1e-6 to 7.2e-7) at no measurable runtime cost, and the
  shock position error falls from −0.0080 to +0.0018, i.e. to the uniform
  fine run's value.
- **Score between t ≈ 2 and t ≈ 24 only.** The analytical shock reaches the
  downwind wall at `t ≈ 23.8 s` (`x_s = 5 + 0.21 t`), after which every run's
  L1 drops sharply (7.9e-6 at t=30 vs 1.4e-5 at t=20) because the wave is
  leaving the domain, not because accuracy improved. The apparent "deficit
  widening" at t=25/30 in the baseline table is partly this regime change.

- **Classic Roe is not well-balanced:** left reservoir `lake_range` ~5e-5 with
  the surface ~0.1 % low. `HydrostaticRoe` holds `1e-15` with zero offset —
  that is the correct discriminator between the solvers, and it is why
  `hydrostatic_roe` wins L1 despite being the noisiest on the plateau.
- `tests/dambreakNoAMR/` (legacy) is a **different problem** — not a valid
  control for this battery; `solvers/*/inputs_single_level` is the fixed-grid
  reference.
- `plateau_band()` is only meaningful before the reflected wave returns (first
  wall arrival, `t ≈ 22.6` for the shallow IC); scoring past that is
  meaningless.
- `[Initial] mass` is grid-dependent (0.005469 at 512×64 vs 0.001660 at
  1024×128), so `dMass` needs care as a conservation gate.
