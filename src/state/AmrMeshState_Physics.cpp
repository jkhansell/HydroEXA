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
    solver.compute_dt(dt, U_new, DynamicTerrain, geom, refRatio(), finest_level, physics_p.cfl);
}

/* ------------------------------------------------------------------ */
/*  TimeStepWithSubcycling — recursive AMR time advance               */
/*  Pattern from AMReX tutorials:                                      */
/*  https://github.com/AMReX-Codes/amrex-tutorials/tree/main/         */
/*  ExampleCodes/Amr/Advection_AmrCore                                */
/* ------------------------------------------------------------------ */
void AmrMeshState::TimeStepWithSubcycling(int lev, amrex::Real time, int iteration)
{
    BL_PROFILE("AmrMeshState::TimeStepWithSubcycling");

    // --- Regrid logic ---------------------------------------------------
    static amrex::Vector<int> last_regrid_step(max_level + 1, 0);

    if (amr_p.regrid_int > 0 && lev < max_level && istep[lev] > last_regrid_step[lev]) {
        if (istep[lev] % amr_p.regrid_int == 0) {
            regrid(lev, time);

            for (int k = lev; k <= finest_level; ++k) {
                last_regrid_step[k] = istep[k];
            }
        }
    }

    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    // Advance this level: fill BCs, compute fluxes, update conservative vars
    AdvanceLevel(lev, t_old[lev], dt[lev]);

    ++istep[lev];

    // --- Recurse into finer levels (subcycle) ---------------------------
    if (lev < finest_level) {
        for (int i = 1; i <= nsubsteps[lev+1]; ++i) {
            TimeStepWithSubcycling(lev + 1, time + (i-1) * dt[lev + 1], i);
        }

        // Reflux: correct lev based on coarse-fine flux mismatch
        if (amr_p.do_reflux && flux_reg[lev + 1] != nullptr) {
            flux_reg[lev + 1]->Reflux(U_new[lev], 1.0, 0, 0, U_new[lev].nComp(), geom[lev]);
        }

        // Note: Each AMR level maintains its own U_new independently.
        // Reflux ensures conservation at coarse-fine interfaces.
        // AverageDown is NOT called on U_new — it would overwrite the
        // reflux-corrected coarse solution with non-conservative interpolation.
        AverageDownTo(lev, U_new);
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
}
