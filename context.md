# HydroEXA — Working Context (C-Property Restoration)

Running state for agents. Companions: `AGENTS.md` (architecture/build),
`docs/PIPELINE.md` (pipeline), **`plans/IdealAMR.md` (§9 = compliance matrix for
R1–R9 and CC-1..CC-5 — the live status table)**. Updated 2026-11-14.

## Workflow rules (user-mandated)

1. **One conceptual change per test.** Passing a test proves only that its one
   change didn't regress the rest state — never that the sim works.
2. **Incremental rebuilds:** `source machines/<target>; cmake --build
   build_<t>/hydroexa_<P> -j 16 && cmake --install ...`. No `rm -rf`, no full
   build script, for edit-compile loops.
3. **Never gate on `pgrep`** (self-matches, stale wrappers). Read the log:
   `grep 'exit status'`, `grep finalized`, `[Conservation]`/`[Step]` lines.
4. **Per-run isolation** — every run writes its own folder; stale frames from
   another case have already cost a full misdiagnosis round.
5. **No Jacobi iteration.** NODATA = mask guards (`== -9999 → dry`).

## Problem & root cause

Lake-at-rest breaks **only with AMR** (commit `709937a`); the Riemann kernels are
fine. With `η = h + z` and a lake at rest (constant η, zero momentum), any
**interpolation of `h`** across a resolution change is inconsistent with
`h = η − z` on a sloping bed → spurious coarse-fine fluxes. Interpolating **η**
is constant-preserving for *any* interpolator, hence rule R1: interpolate η (+
momenta), never h; rebuild `h = max(0, η − z)` from the target level's own
terrain.

## Plan (details and status in `plans/IdealAMR.md`)

| Step | Change | Test |
|------|--------|------|
| CC-1 | η-routing at level creation + solver ghost reconstruction + mask guards | T1a |
| CC-2 | side-selected HydrostaticRoe flux registers (mirror `Roe.cpp` L420–565) | T1b |
| CC-3 | time-interp-tolerant `FillCoarsePatchEta/Momentum` | T2 |
| CC-4 | η-routed `RemakeLevel` | T3 |
| CC-5 | dry-shoreline battery | T4 |
| — | dambreak AMR regression vs fixed grid | T5 |

Acceptance: `[Conservation]` per-level `|px|/|py|` at roundoff all run
(1e-13 DOUBLE; ~1e-6 SINGLE noise floor). DOUBLE on V100 = acceptance numbers,
SINGLE on L40S = fast functional runs.

## T1a: solved (2026-11-13) — the four fixes that got there

Symptom: perfect momentum + `dMass` roundoff, yet dt collapsed exponentially and
t froze (3.39 → 20.86 s across attempts), ending in single-cell velocity dipoles
locked to the AMR band. Global sums are blind to it (dipole = +/− pairs).

| Fix | Effect |
|---|---|
| R2-1 scratch η synthesis in `GetEtaData`, `Eta_old` deleted | tree runnable; t=0 dry band (194 cells) reproduced |
| R2-2 η-route the three creation sites | **t=0 defect gone** (eta_f = 0.01 exact at ratio 2), killed a −0.18% mass leak; collapse 3.39 → 20.86 |
| R2-3 solver ghost fill via η (`Eta_ext` + momenta-only fill + h rebuild) | collapse unchanged → ghost h was not the driver |
| **dt lockstep fix** in `AmrMeshState::ComputeDt` | root cause: with `do_subcycle=0` each level still advanced `dt[0]/r^lev` once ⇒ level times desynchronized ⇒ ghost FillPatch time factor far outside [0,1] ⇒ ×1.019/step growth |

Green result (T1a_r24, V100 DOUBLE np=1, 7698 steps to t=30): `dMass =
0.000000%`, px = py = KE = 0, **max|vel| = 9.95e-16 over the whole run**.
`LakeAtRestRoe` (classic Roe, same config) also green: 9.99e-15 — the η-routing
and dt lockstep are solver-agnostic.

## Dam-break battery (`tests/dambreak/`, r29) — CI baseline

SWASHES benchmark: hL=0.005, hR=0.001, x0=5, L=10, flat bed, max_time=30,
`io.plot_freq=1` (31 frames/run). All five runs exit 0, all frames rendered.

| run | shock err (%L) | **plateau** h p2p | plateau hu p2p | **lake** h p2p | lake offset |
|---|---|---|---|---|---|
| `roe/amr` | 0.08% | 4.43e-5 | 1.35e-6 | 5.02e-5 | 5.3e-6 |
| `roe/single_level` | 0.12% | **6.89e-6** | 3.07e-7 | 7.56e-5 | 1.0e-5 |
| `roe/amr_noreflux` | 0.31% | 4.40e-5 | 1.35e-6 | 5.26e-5 | 8.9e-6 |
| `roe/amr_nosubcycle` | 0.08% | 1.45e-5 | 4.69e-7 | 5.31e-5 | 7.5e-6 |
| `hydrostatic_roe/amr` | 0.09% | 7.37e-5 | 1.43e-5 | **1.07e-15** | **0** |

- **Metric bug found 2026-11-14:** the old `plateau_band()` selected the flat run
  with the highest h = the **left reservoir**. Every pre-r29 "plateau jitter"
  number (5–9e-5) was a lake measurement; conclusions drawn from it are
  withdrawn. Both bands are now reported (`plateau_band`, `lake_band`).
- Plateau: fixed grid is smoothest (6.9e-6) → **AMR adds ~6× noise**;
  `noreflux` unchanged (flux registers innocent); `nosubcycle` 1.45e-5 ⇒
  **subcycling / coarse-level time interpolation dominates (~3×)** → CC-3 is
  the next target after R2-4.
- Lake: classic Roe ~5e-5 p2p with the surface 0.1% low (4.9947e-3) — not
  well-balanced; HydrostaticRoe exact (1e-15, offset 0) but **noisiest on the
  plateau** (the hu sawtooth), so it is not a "clean reference".
- `noreflux` also has the worst shock position (0.31% vs 0.08%) → reflux is the
  momentum/shock-speed knob; keep it ON.
- **Score at t=15.** The analytical shock exits at t≈24.9, so later
  shock-position numbers are boundary artifacts.
- CI thresholds: plateau jitter ≤ 2e-4, lake jitter ≤ 2e-4, shock ≤ 0.5% of L.

## Tooling

- `tests/dambreak/analysis.py` — single entry point: `analyze.py` (yt readers
  re-exported from `plot_output.py` + CI metrics) and `plot.py` (plot_output-style
  5-panel figure, metrics embedded). PNGs named by **actual simulation time**.
- `run.sh <solver> <variant> <exe> [np] [tag] [machine]` — sources the machine
  profile itself (default `machines/kabreV100.gpu`), writes
  `outputs/<solver>/<variant>__<tag>/`, and renders **every** plotfile
  (`plt####.old.NNN` rotation files filtered). Each run folder is self-contained.
- `tests/LakeAtRestHydrostaticRoe/run_t.sh` — T-runner
  (`<exe> <tag> [np] [reflux] [subcycle] [regrid]`); override `mass_diag_freq`,
  it is in time units and never fires otherwise. Profiles replace modules.
- `tests/LakeAtRestHydrostaticRoe/dipole_probe.py` — level-wise max|u|/|η−H|,
  dry-cell census, 5×5 dumps around the worst cell.

## Next moves

1. Commit CC-1 (uncommitted in tree) with its T1a evidence.
2. **R2-4**: delete `AmrMeshState::Eta`, `ComputeEta()` (5 call sites), `ctx.Eta`;
   Roe uses a local η scratch via `ctx.FillPatchEta` (R2-3 pattern). Behavior
   preserving — T1a must stay green. Bundle with **R5 for Roe**: it rebuilds h
   from η but never rescales momentum (velocity jumps) while HydrostaticRoe
   preserves velocity.
3. **CC-3** (time interpolation) — the measured plateau-noise target: move
   `nosubcycle`'s 1.45e-5 toward the 6.9e-6 fixed-grid floor.
4. Then CC-2 → T1b, CC-4 → T3, CC-5 → T4; plus diagnostics work (per-level
   conservation lines, zero-reference `dPx=-inf%` fix in
   `AmrMeshState_Utils.cpp`).

## Key files

| File | Role |
|------|------|
| `src/state/AmrMeshState_AMR.cpp` | CC-1 η-routing; `ComputeEta` sites; `RemakeLevel` (CC-4) |
| `src/state/AmrMeshState_Physics.cpp` | subcycling loop, dt, remaining `ComputeEta` calls (R2-4) |
| `src/state/AmrMeshState_Utils.cpp` | `GetSolverContext`/`GetEtaData` (R2-4 target), diagnostics |
| `src/solvers/HydrostaticRoe.cpp` | `Eta_ext` ghost reconstruction; registers ~L562+ = CC-2 target |
| `src/solvers/Roe.cpp` | `ctx.Eta` consumer (R2-4), side-selected registers L420–565 = CC-2 reference |
| `include/solvers/SolverContext.H` | `Eta` member to delete (R2-4) |
| `tpl/amrex/.../AMReX_FillPatchUtil_I.H` | `InterpFromCoarseLevel` (old t=0 defect mechanism) |
| `machines/kabreV100.gpu` | module profile incl. `cmake/4.1.1`, parallel HDF5 |

## Environment quirks

- **nvcc:** extended `__device__` lambdas cannot be *defined* inside
  private/protected members — kernels live in public members or file-scope
  helpers (`fill_eta_from_state`, `rebuild_h_from_eta` in `AmrMeshState_AMR.cpp`).
- Non-interactive ssh has no `mpirun` until `source machines/<target>`; note the
  profile returns nonzero, so **use `;` not `&&`** after sourcing.
- Kabré `/opt/tools` cmake ≥ 3.31.8 is broken (`libssl.so.10`); profiles pin
  `cmake/4.1.1`. Installed `AMReXConfig.cmake` requires the cmake that built AMReX.
- Interactive allocation moves between `nukwa-0N` nodes; node-local `/tmp`
  dies with it. Check `ssh kabre squeue -u jovillalobos`.
- sshfs mount: avoid whole-tree grep/find; use targeted paths.
- `plans/` is **gitignored** — `context.md` and the plan docs are local-only.
