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
/* ------------------------------------------------------------------ */
void AmrMeshState::TimeStepWithSubcycling(int lev, amrex::Real time, int iteration)
{
    BL_PROFILE("AmrMeshState::TimeStepWithSubcycling");

    // --- Regrid logic ---------------------------------------------------
    static amrex::Vector<int> last_regrid_step(max_level + 1, 0);

    if (amr_p.regrid_int > 0 && lev < max_level && istep[lev] > last_regrid_step[lev]) {
        if (istep[lev] % amr_p.regrid_int == 0) {
            int old_finest = finest_level;
            regrid(lev, time);

            for (int k = lev; k <= finest_level; ++k) {
                last_regrid_step[k] = istep[k];
            }

            for (int k = old_finest + 1; k <= finest_level; ++k) {
                // Inherit dt using exact refinement ratio
                dt[k] = dt[k - 1] / MaxRefRatio(k - 1);

                // Compute CFL requirement for the newly created grid level
                amrex::Real dt_cfl = solver.compute_dt_level(U_new, DynamicTerrain, geom, k, physics_p.cfl);
                amrex::ParallelDescriptor::ReduceRealMin(&dt_cfl, 1);

                // Only perform the multi-level backward propagation if the inherited dt violates CFL
                if (dt[k] > dt_cfl) {
                    amrex::Print() << "WARNING: new level " << k
                                << " inherited dt=" << dt[k]
                                << " but CFL requires dt=" << dt_cfl 
                                << ". Recalculating dt across all levels.\n";

                    // 1. Set the newly restricted fine level dt
                    dt[k] = dt_cfl;

                    // 2. Propagate backward to update all coarser levels (level k-1 down to 0)
                    for (int p = k - 1; p >= 0; --p) {
                        dt[p] = dt[p + 1] * MaxRefRatio(p);
                    }
                }
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
            TimeStepWithSubcycling(lev + 1, t_old[lev + 1], i);
        }

        // Reflux: correct lev based on coarse-fine flux mismatch
        if (amr_p.do_reflux && flux_reg[lev + 1] != nullptr) {
            flux_reg[lev + 1]->Reflux(U_new[lev], 1.0, 0, 0, U_new[lev].nComp(), geom[lev]);
        }

        // Average down covered coarse cells from fine grid
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
