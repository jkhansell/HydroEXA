# HydroEXA Simulation Pipeline

High-level overview of the HydroEXA simulation pipeline — from input loading through time-stepping to output. Use this to verify that each stage matches the intended physics and numerical methods.

---

## 1. Entry Point & Initialization

**File:** `src/HydroEXADriver.cpp`

```
amrex::Initialize(argc, argv)
  → HydroEXA hydroexa              // Constructor reads parameters from inputs file
  → hydroexa.Initialize()          // Sets up mesh, geometry, ICs, solver
  → hydroexa.Compute()             // Main time-stepping loop
  → hydroexa.Finalize()            // Cleanup
amrex::Finalize()
```

---

## 2. Parameter Reading

**File:** `src/HydroEXA.cpp` → `HydroEXA::ReadParameters()`

Reads from the inputs file via `amrex::ParmParse`:

| Section | Parameter | Meaning |
|---------|-----------|---------|
| `HydroEXA` | `max_time` | Simulation stop time |
| `HydroEXA` | `model` | Solver type (`"Roe"`, etc.) |
| `HydroEXA` | `cfl` | CFL number for timestep |
| `HydroEXA` | `mass_diag_freq` | Conservation diagnostic interval (physical time, -1 = disabled) |
| `HydroEXA` | `h_grad_thresh` | Array of depth-gradient tagging thresholds per AMR level |
| `HydroEXA` | `umag_grad_thresh` | Array of velocity-magnitude-gradient tagging thresholds |
| `HydroEXA` | `z_grad_thresh` | Array of bathymetry-gradient tagging thresholds |
| `HydroEXA` | `lo_bc`, `hi_bc` | Physical boundary condition types (0-5) per direction |
| `amr` | `max_level` | Maximum AMR refinement level |
| `amr` | `regrid_int` | Regridding frequency (iterations) |
| `amr` | `do_reflux` | Enable coarse-fine flux correction |
| `amr` | `do_subcycle` | Enable temporal subcycling at fine levels |
| `amr` | `terrain_ref_lev` | AMR level at which terrain HDF5 data is loaded |
| `io` | `plot_freq`, `chk_freq` | Plotfile/checkpoint output intervals |
| `io` | `chk_type` | Checkpoint format (`"HDF5"` or `"Native"`) |
| `io` | `input_file` | HDF5 file with initial terrain + fluid ICs |

---

## 3. Geometry & Mesh Setup

**File:** `src/HydroEXA.cpp` → `HydroEXA::Initialize()`

1. **Read HDF5 metadata** from `input_file` (terrain dataset):
   - `IO->ReadHDF5Metadata("terrain", metadata)` extracts `global_nx`, `global_ny`, `dx`, `dy`, `prob_lo_x`, `prob_lo_y`, `prob_hi_x`, `prob_hi_y`.

2. **Compute Level-0 resolution** from HDF5 data:
   - `base_nx_raw = global_nx / refinement_scale` where `refinement_scale = 2^terrain_ref_lev`
   - Pad to blocking factor (default 16): `base_nx = ceil(base_nx_raw / bf) * bf`
   - Scale cell spacing up: `level0_dx = dx * refinement_scale`
   - Compute extended physical domain bounds from padded cell count × cell spacing.

3. **Construct AmrMeshState** (inherits `amrex::AmrCore`):
   - Passes `RealBox`, `max_level`, `n_cell`, `coord_sys`, `ref_ratios`.
   - Inside `AmrMeshState` constructor: allocates `U_new`, `U_old`, `DynamicTerrain` vectors per level; sets up boundary condition `BCRec`s.

---

## 4. AmrMeshState Construction & Setup

**File:** `src/state/AmrMeshState_Init.cpp` → `AmrMeshState::AmrMeshState()`

1. **Resize containers** for `max_level + 1` levels.
2. **Set subcycling**: `nsubsteps[lev] = MaxRefRatio(lev-1)` if `do_subcycle` is enabled.
3. **Build BCRecs**: Maps physical BC types (0-5) to mathematical BCRecs per component:
   - Component 0 (h): scalar BC
   - Component 1 (hu): x-velocity BC
   - Component 2 (hv): y-velocity BC

---

## 5. Initialization Flow

**File:** `src/state/AmrMeshState_Init.cpp` → `AmrMeshState::Initialize()`

### 5a. InitializeSolver()
- Creates the solver variant: `solver = Roe{}` (currently only Roe is supported).

### 5b. InitializeTerrainFluid()
**File:** `src/state/AmrMeshState_Init.cpp` → `AmrMeshState::InitializeTerrainFluid()`

1. Defines the static domain from HDF5 extents.
2. Allocates `StaticTerrain` (2 components: bathymetry `z` + mask `n`) and `StaticFluid` (3 components: `h`, `hu`, `hv`).
3. **Parallel hyperslab reads** from HDF5:
   - `IO->ReadHDF5HyperslabComponents(z_arr, bx, "terrain", 0, 2, -9999.0)` — reads terrain dataset, fills NODATA cells with -9999.0.
   - `IO->ReadHDF5HyperslabComponents(u_arr, bx, "fluid", 0, 3, 0.0)` — reads fluid initial conditions.
4. Fills boundaries for periodicity.

### 5c. InitFromScratch() (AMReX AmrCore override)
- Calls `PostProcessBaseGrids()` to prune blocks that are 100% NODATA.
- Creates Level 0 MultiFabs via `MakeNewLevelFromScratch()`.
- Copies static terrain/fluid data to dynamic MultiFabs via `TerrainMapStaticToDynamic()` and `FluidMapStaticToDynamic()`.
- Creates finer levels via `MakeNewLevelFromCoarse()` if `max_level > 0`.

### 5d. StoreInitialMassMomentum()
**File:** `src/state/AmrMeshState_Utils.cpp` → `StoreInitialMassMomentum()`

Computes and stores baseline conserved quantities across all AMR levels:
- **Mass** = `∫ h · dA`
- **Momentum_x** = `∫ hu · dA`
- **Momentum_y** = `∫ hv · dA`
- **Kinetic Energy** = `∫ 0.5·(hu²+hv²)/h · dA`

Uses GPU-compatible `amrex::ReduceOps` over valid fluid cells only (excludes NODATA cells where terrain mask == -9999, and dry cells where h ≤ 0). MPI-reduces per-rank sums, then broadcasts to all ranks.

### 5e. Write Initial Plotfile
- Writes iteration 0 plotfile with `U_new` and `DynamicTerrain`.

---

## 6. Main Time-Stepping Loop

**File:** `src/HydroEXA.cpp` → `HydroEXA::Compute()`

```
plot_time_next = plot_freq, chk_time_next = chk_freq, mass_time_next = mass_diag_freq

while (etime < max_time):

    // 1. Compute timestep (CFL condition)
    MeshState->ComputeDt()

    // 2. Advance all levels (with subcycling)
    MeshState->TimeStepWithSubcycling(0, etime, iteration)
    etime = t_new[0]
    iteration++

    // 3. Conservation diagnostics (if mass_diag_freq > 0)
    if etime >= mass_time_next:
        MeshState->ComputeMassDiagnostics(etime, iteration)
        mass_time_next += mass_diag_freq

    // 4. Print step info (if Verbose)

    // 5. Write plotfile (if plot_freq > 0)
    if etime >= plot_time_next:
        MeshState->WritePlotfile(plot_iter, etime)
        plot_time_next += plot_freq

    // 6. Write checkpoint (if chk_freq > 0)
    if etime >= chk_time_next:
        MeshState->WriteCheckpoint()
        chk_time_next += chk_freq
```

---

## 7. Timestep Computation

**File:** `src/state/AmrMeshState_Physics.cpp` → `AmrMeshState::ComputeDt()`
**File:** `src/solvers/Roe.cpp` → `Roe::compute_dt_Impl()`

1. **Per-level CFL computation** (GPU kernel):
   - For each cell: `c = sqrt(g * h)`, `dt_cell = min(dx/|u|+c, dy/|v|+c)`
   - Excludes NODATA cells (terrain == -9999) and dry cells (h < 1e-12).
   - Takes minimum across all cells on the level.
2. **Global minimum** across all MPI ranks.
3. **Backward propagation**: `dt[lev] = min(dt[lev+1] * ref_ratio, dt_tmp[lev])` — ensures coarser levels don't violate their own CFL constraint.

---

## 8. AMR Time Stepping with Subcycling

**File:** `src/state/AmrMeshState_Physics.cpp` → `AmrMeshState::TimeStepWithSubcycling()`

```
TimeStepWithSubcycling(lev, time, iteration):

    // Regrid (if needed)
    if lev < max_level and istep[lev] % regrid_int == 0:
        regrid(lev, time)
        tag cells based on gradients (see §9)
        prune NODATA blocks via PostProcessBaseGrids()

    // Advance this level
    t_old[lev] = t_new[lev]
    t_new[lev] += dt[lev]
    AdvanceLevel(lev, t_old[lev], dt[lev])
    istep[lev]++

    // Recurse into finer levels (subcycle)
    if lev < finest_level:
        for i = 1 to nsubsteps[lev+1]:
            TimeStepWithSubcycling(lev+1, time + (i-1)*dt[lev+1], i)

        // Coarse-fine conservation correction
        if do_reflux:
            flux_reg[lev+1]->Reflux(U_new[lev])

        // Downward interpolation (non-conservative)
        AverageDownTo(lev, U_new)
```

---

## 9. AMR Refinement (Tagging)

**File:** `src/solvers/Roe.cpp` → `Roe::tag_cells_Impl()`

**At t = 0** (initial tagging):
- Compute bathymetry gradient: `|∇z| = sqrt((dz/dx)² + (dz/dy)²)`
- Tag if `|∇z| > z_grad_thresh[lev]`

**At t > 0** (dynamic tagging):
- Compute depth gradient: `|∇h| = sqrt((dh/dx)² + (dh/dy)²)`
- Tag if `|∇h| > h_grad_thresh[lev]`
- Else compute velocity magnitude gradient: `|∇|u||`
- Tag if `|∇|u|| > umag_grad_thresh[lev]`

Uses central differences. Skips cells adjacent to NODATA regions.

---

## 10. Single-Level Advance (Flux Computation + Update)

**File:** `src/state/AmrMeshState_Physics.cpp` → `AmrMeshState::AdvanceLevel()`
**File:** `src/solvers/Roe.cpp` → `Roe::compute_fluxes_Impl()`

### Step 1: Swap old/new
```cpp
std::swap(U_new[lev], U_old[lev]);
// U_o = old state, U_n = destination for new state
```

### Step 2: Allocate dual effective flux arrays
- `D_minus_mf[AMREX_SPACEDIM]` and `D_plus_mf[AMREX_SPACEDIM]`
- Defined on nodal grids (surroundingNodes) for each direction.

### Step 3: Fill ghost cells
- `U_old` and `Terrain` fill interior boundaries (periodicity).
- Apply physical boundary conditions via `BCFill` (SlipWall, Outflow, etc.).

### Step 4: Compute directional dual effective fluxes
**File:** `src/solvers/Roe.cpp` → `Roe::compute_amrex_effective_fluxes()`

For each face in each direction, calls `roeSolver()` with left and right cell states.

### Step 5: Roe Riemann Solver Kernel
**File:** `include/solvers/kernels/Roe.H` → `roeSolver()`

For each face (left state i, right state j):

1. **Dry handling**: `h = max(h, 0)` with tolerance `1e-6`.
2. **Rotate** to normal direction: `u_hat = u·n`, `v_hat = u·t`.
3. **Roe averages**: `ũ = (u_hat_i·√h_i + u_hat_j·√h_j) / (√h_i + √h_j)`.
4. **Wave speeds**: `λ₀ = ũ - c`, `λ₁ = ũ`, `λ₂ = ũ + c` where `c = √(g·h̃)`.
5. **Entropy fix**: Adjusts `λ₀` and `λ₂` near sonic points.
6. **Wave strengths**: `α₀`, `α₁`, `α₂` from jumps in conserved variables.
7. **Source terms** (bathymetry + roughness):
   - Bathymetry: `thrust` from depth difference `Δz` with upwinding.
   - Roughness: `τ` from Manning's n coefficient.
8. **Source splitting**: `β₀`, `β₁`, `β₂` distributed to `S_minus` and `S_plus`.
9. **Assemble dual effective fluxes**:
   - `D_minus = F - S_minus` (flux minus left-distributed sources)
   - `D_plus = F + S_plus` (flux plus right-distributed sources)
10. **Rotate back** to global coordinates.

### Step 6: Conservative cell update (Exact Fluctuation Differencing)
```cpp
U_n(i,j,k,n) = U_o(i,j,k,n)
             - (dt/dx) * (D_minus_x(i+1,j,k,n) - D_plus_x(i,j,k,n))
             - (dt/dy) * (D_minus_y(i,j+1,k,n) - D_plus_y(i,j,k,n))
```

### Step 7: Fill ghost cells of new state + apply physical BCs

### Step 8: Flux register synchronization (if reflux enabled)
- For coarse-fine interfaces, map `D_minus`/`D_plus` to `FluxRegister`:
  - `CrseInit()`: coarse cell receives flux from its face adjacent to fine cell.
  - `FineAdd()`: fine cell receives flux from its face adjacent to coarse cell.
- `Reflux()` is called later in `TimeStepWithSubcycling()` to correct `U_new[lev]`.

---

## 11. Conservation Diagnostics

**File:** `src/state/AmrMeshState_Utils.cpp` → `ComputeMassDiagnostics()`

Called at intervals specified by `mass_diag_freq` (if > 0).

1. Iterates over all AMR levels and all `MFIter` tiles.
2. GPU `ReduceOps` computes per-cell contributions:
   - `mass = h · dA`
   - `mom_x = hu · dA`
   - `mom_y = hv · dA`
   - `KE = 0.5·(hu²+hv²)/h · dA`
   - `min_h` (minimum water depth)
3. Excludes NODATA cells (terrain mask == -9999) and dry cells (h ≤ 0).
4. MPI-reduce sums across ranks.
5. IO processor prints totals and conservation error percentages:
   ```
   [Conservation] t=X  iter=Y  mass=M  px=Px  py=Py  KE=KE  min_h=min_h
   [Conservation error] dMass=e%  dPx=e%  dPy=e%  dKE=e%
   ```

---

## 12. Output

### Plotfiles (AMReX format)
**File:** `src/state/AmrMeshState_Utils.cpp` → `WritePlotfile()`
**Delegates to:** `IO->WritePlotfile()`

Writes `U_new` and `DynamicTerrain` per level. State variables: `h_fluid`, `hu_momentum`, `hv_momentum`. Terrain: bathymetry + mask.

### Checkpoints
**File:** `src/io/Checkpointer.cpp` → `Checkpointer::Write()`

Writes `U_new`, `U_old`, `DynamicTerrain` per level in HDF5 or AMReX native format. Does NOT write conservation totals.

---

## Data Flow Summary

```
HDF5 Input (terrain + fluid)
    │
    ▼
StaticTerrain / StaticFluid (per-MPI-rank hyperslab)
    │
    ▼
MakeNewLevelFromScratch / MakeNewLevelFromCoarse
    │  (copy static → dynamic, prune NODATA)
    ▼
DynamicTerrain / U_new (AMR levels, initialized)
    │
    ▼
StoreInitialMassMomentum() ← baseline for conservation error
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  TIME-STEPPING LOOP                  │
│                                                      │
│  ComputeDt()                                         │
│    └── Roe::compute_dt_Impl()  ← CFL per cell, min  │
│                                                      │
│  TimeStepWithSubcycling()                            │
│    ├── Regrid + TagCells()  ← gradient-based AMR    │
│    ├── AdvanceLevel()                                 │
│    │   ├── BCFill (ghost cells + physical BCs)      │
│    │   ├── Roe::compute_fluxes_Impl()               │
│    │   │   ├── roeSolver() per face (Riemann +      │
│    │   │   │   source terms + dual effective fluxes) │
│    │   │   ├── Conservative update (fluctuation     │
│    │   │   │   differencing)                         │
│    │   │   └── FluxRegister sync (coarse-fine)      │
│    │   └── BCFill (new state ghost cells)           │
│    ├── Reflux()  ← coarse-fine conservation corr.   │
│    └── AverageDownTo()  ← non-conservative interp.  │
│                                                      │
│  ComputeMassDiagnostics()  ← conservation check      │
│  WritePlotfile()         ← visualization output      │
│  WriteCheckpoint()       ← restart capability        │
└─────────────────────────────────────────────────────┘
```

---

## Conserved Variables Layout

| Component | Index | Physical Meaning |
|-----------|-------|-----------------|
| `U[lev]` | 0 | Water depth `h` |
| `U[lev]` | 1 | x-momentum `hu` |
| `U[lev]` | 2 | y-momentum `hv` |
| `Terrain[lev]` | 0 | Bathymetry `z` |
| `Terrain[lev]` | 1 | Mask `n` (1 = fluid, -9999 = NODATA) |

---

## Key Physics Constants

| Constant | Value | Source |
|----------|-------|--------|
| Gravity `g` | 9.81 m/s² | `PhysConstants::GRAV()` |
| Depth tolerance | 1e-4 | `Tolerances::DEPTH()` |
| Flux tolerance | 1e-7 | `Tolerances::FLUX()` |
| Dry depth threshold | 1e-6 | `Tolerances::TOLDRY()` |
| Small epsilon | 1e-12 | `Tolerances::TOL12()` |

---

## What's Implemented vs. What's Missing

### Implemented
- [x] HDF5 terrain + fluid IC loading
- [x] AMR mesh setup with NODATA pruning
- [x] Roe Riemann solver with bathymetry source terms
- [x] Dual effective flux formulation (FCT-style)
- [x] CFL-based timestep with backward propagation
- [x] AMR subcycling + reflux conservation correction
- [x] Gradient-based AMR tagging (bathymetry, depth, velocity)
- [x] Mass/momentum/KE conservation diagnostics (stdout)
- [x] Initial mass/momentum storage for error tracking
- [x] Plotfile output (AMReX native + HDF5)
- [x] Checkpoint/restart
- [x] Physical boundary conditions (SlipWall, Outflow, etc.)

### Missing / Partial
- [ ] `mass_diag_freq` not set in test inputs (diagnostics disabled by default)
- [ ] No file-based conservation output (CSV or separate diagnostic file)
- [ ] No energy conservation tracking (potential + kinetic)
- [ ] No reflux conservation verification (relies on mass/momentum diagnostics only)
- [ ] Only Roe solver implemented (HLLC, Lax-Wendroff, RoeExner listed but not coded)
- [ ] `ComputeMassDiagnostics` uses `==` comparison for NODATA mask (`-9999.0`); should use tolerance-based comparison for floating-point safety
- [ ] Per-level conservation breakdown printed (currently only totals)
- [ ] Final conservation report at simulation end (last step only)
