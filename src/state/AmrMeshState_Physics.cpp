#include <AMReX_FillPatchUtil.H>


#include <state/AmrMeshState.H>
#include <utils/Logging.H>
#include <solvers/SolverVariant.H>
#include <solvers/kernels/Roe.H>

/* ------------------------------------------------------------------ */
/*  ComputeDt — CFL timestep at finest level, scaled for coarser levels */
/*  dt[lev] = dt[finest] * ref_ratio[lev] * ref_ratio[lev+1] * ...    */
/* ------------------------------------------------------------------ */
void AmrMeshState::ComputeDt()
{
    BL_PROFILE("AmrMeshState::ComputeDt");

    // Delegate to BaseSolver::compute_dt — it handles finest-level CFL
    // computation and correct backward propagation to coarser levels.
    solver.compute_dt(dt, U_new, DynamicTerrain, Geom(), refRatio(), finestLevel(), physics_p.cfl);

    // CFL invariant (both stepping modes):
    //
    //   dx[lev] = dx_0 / r^lev   (AMReX ref_ratio), so the per-level CFL dt is
    //   dt_CFL[lev] = dt_CFL[0] / r^lev.
    //
    // Subcycled mode (do_subcycle=1): lev takes r^lev substeps of dt[lev] per
    // outer step, so every level runs at the SAME cfl and all clocks sync.
    // dt[lev] = dt_CFL[lev] from compute_dt is exactly right.
    //
    // Lockstep mode (do_subcycle=0): every level takes exactly ONE step per
    // outer iteration, so a shared dt is mandatory.  Choosing dt = dt[finest]
    // = dt_0 / r^n keeps the finest level at its cfl and every coarser level
    // correspondingly safer (cfl_lev = cfl_fine / r^(n-lev)) — the only
    // choice that satisfies every level's CFL while keeping clocks synced.
    // (dt[lev] from compute_dt is already descending in lev, so min = dt[finest].)
    if (!amr_p.do_subcycle) {
        amrex::Real dt_min = dt[0];
        for (int lev = 1; lev <= finestLevel(); ++lev) {
            dt_min = std::min(dt_min, dt[lev]);
        }
        for (int lev = 0; lev <= finestLevel(); ++lev) {
            dt[lev] = dt_min;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  TimeStepWithSubcycling — recursive AMR time advance               */
/* ------------------------------------------------------------------ */
void AmrMeshState::TimeStepWithSubcycling(int lev, amrex::Real time, int iteration)
{
    BL_PROFILE("AmrMeshState::TimeStepWithSubcycling");

    // --- Regrid logic ---------------------------------------------------
    //
    // Cadence matters physically, not just as bookkeeping: regridding is what
    // keeps a moving front INSIDE the fine patch.  The patch is grown by
    // n_error_buf level-0 cells beyond the tagged region, and between regrids
    // the front advances dt[0] * U_front per step, so the requirement is
    //
    //     regrid_int * dt[0] * U_front  <  n_error_buf * dx[0]
    // <=> regrid_int * cfl * (U_front / lambda_max) < n_error_buf
    //
    // (n_error_buf defaults to 1 cell, AMReX AmrMesh).  Violating it lets the
    // front fall back to level-0 resolution for part of every interval; on a
    // steady plateau that error is frozen into the fluid and never decays, so
    // it reads as a 20-30 % accuracy loss vs a uniform fine run (or, worse, as
    // "instability inside level 1") with no crash and no conservation failure.
    // On tests/dambreak the stock regrid_int=50 put the shock outside level 1
    // in 25 % of frames (up to 0.23 m past the patch edge); regrid_int=5 fixed
    // it at no measurable runtime cost.  See docs/PIPELINE.md §8a before
    // raising regrid_int for a new case with a shock/bore/wet-dry front.
    //
    static amrex::Vector<int> last_regrid_step(max_level + 1, 0);

    if (amr_p.regrid_int > 0 && lev < max_level && istep[lev] > last_regrid_step[lev]) {
        if (istep[lev] % amr_p.regrid_int == 0) {

            int old_finest = finestLevel();
            {
                BL_PROFILE("AmrMeshState::TimeStepWithSubcycling::Regrid");
                regrid(lev, time);
            }

            for (int k = lev; k <= finestLevel(); ++k) {
                last_regrid_step[k] = istep[k];
            }

            for (int k = old_finest + 1; k <= finestLevel(); ++k) {
                // A level born in this regrid must join the CURRENT step clock.
                // dt[k] = dt[k-1] / (number of substeps k takes per step of k-1):
                //   subcycling: nsubsteps[k] = r  -> exact geometric ratio 1/r
                //   lockstep:   nsubsteps[k] = 1  -> dt[k] = dt[k-1] (shared dt)
                // Dividing by MaxRefRatio unconditionally (previous code) gave
                // the new level dt[k-1]/r in lockstep too, where it takes only
                // ONE substep: its clock landed dt/2 behind its parent and the
                // offset was permanent (every later fill then time-blends).
                // This is AMReX Amr::timeStep: dt_level[k] = dt_level[k-1]/n_cycle[k].
                dt[k] = dt[k - 1] / static_cast<amrex::Real>(nsubsteps[k]);
            }
        }
    }

    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    // Advance this level: fill BCs, compute fluxes, update conservative vars
    AdvanceLevel(lev, t_old[lev], dt[lev]);

    ++istep[lev];

    // --- Recurse into finer levels (subcycle) ---------------------------
    if (lev < finestLevel()) {
        // The time handed to a substep is its START time.  "time" is the start
        // time of THIS level's step (the caller passes it and nothing below
        // modifies it), so substep i of the child starts at the parent start
        // plus (i-1) child steps:
        //
        //     timeStepWithSubcycling(lev+1, time + (i-1)*dt[lev+1], i);
        //
        // which is the AMReX reference form verbatim (Tests/Amr/Advection_AmrCore
        // Source/AmrCoreAdv.cpp, timeStepWithSubcycling recursion).
        //
        // Passing t_old[lev+1] here instead was stale by exactly one child step:
        // after substep i-1 completed, t_old[lev+1] = t_{i-1} whereas
        // t_new[lev+1] = t_i, so regrid(lev+1, time) -- and the FillCoarsePatch
        // or RemakeLevel it drives -- ran one dt[lev+1] in the past, rewinding
        // the remade or newly created level.
        for (int i = 1; i <= nsubsteps[lev+1]; ++i) {
            TimeStepWithSubcycling(lev + 1, time + (i - 1) * dt[lev + 1], i);
        }

        // Reflux: correct lev based on coarse-fine flux mismatch
        if (amr_p.do_reflux && flux_reg[lev + 1] != nullptr) {
            BL_PROFILE("AmrMeshState::TimeStepWithSubcycling::Reflux");
            flux_reg[lev + 1]->Reflux(U_new[lev], 1.0, 0, 0, U_new[lev].nComp(), Geom()[lev]);
        }

        // Average down covered coarse cells from fine grid
        AverageDownTo(lev, U_new);

        // Reflux and averaging down both modify the coarse state, so the free
        // surface must be rebuilt before this level is used as a coarse source
        // for the next subcycled step.  No swap happened, so read from U_new.
        // ComputeEta(lev, false);
    }
}

/* ------------------------------------------------------------------ */
/*  AdvanceLevel — single-level time advance                            */
/*  Fill BCs -> compute fluxes -> conservative update                 */
/* ------------------------------------------------------------------ */
void AmrMeshState::AdvanceLevel(int lev, amrex::Real time, amrex::Real dt)
{
    BL_PROFILE("AmrMeshState::AdvanceLevel");

    SolverContext ctx = GetSolverContext();

    solver.compute_fluxes(ctx, lev, dt, time);

    // Eta = h + z is a live state variable.  Refresh it from the updated
    // conserved state so that finer levels (which FillPatch it) and the
    // eta-based depth reconstruction always see the current free surface.
    // compute_fluxes_Impl swaps U_new/U_old by reference, so after the call
    // U_new holds the updated state (t^{n+1}).
    //
    // R2 note (plans/IdealAMR.md): the FillPatch machinery no longer reads
    // this stored Eta — GetEtaData synthesizes η from U_old/U_new.  This
    // refresh exists only to keep ctx.Eta fresh for the Roe solver, whose
    // kernels read it directly.  Removing it (and ctx.Eta) is a later step.
    ComputeEta(lev, false);
}
