#include <state/AmrMeshState.H>
#include <io/CheckpointerContext.H>
#include <utils/Logging.H>
#include <limits>


void AmrMeshState::GetTerrainData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime) {
    data.clear();
    datatime.clear();
    data.push_back(&DynamicTerrain[lev]);
    datatime.push_back(t_new[lev]);
}

void AmrMeshState::GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime) {
    data.clear();
    datatime.clear();

    // Always return both states so FillPatch can interpolate correctly.
    // This is robust against any internal swaps of U_old / U_new because
    // the timestamps t_old / t_new always describe the logical time of the
    // old and new states respectively, regardless of which MultiFab holds them.
    data.push_back(&U_old[lev]);
    data.push_back(&U_new[lev]);
    datatime.push_back(t_old[lev]);
    datatime.push_back(t_new[lev]);
}

void AmrMeshState::WritePlotfile(int iteration, amrex::Real time) {
    IO->WritePlotfile(
        U_new, DynamicTerrain,
        iteration, time,
        Geom(), refRatio(),
        finest_level
    );
}

void AmrMeshState::WriteCheckpoint() {
    IO->WriteCheckpoint(GetCheckpointerContext(), istep[0]);
}


SolverContext AmrMeshState::GetSolverContext()
{
    return
    {

        //==========================================================================
        // Simulation state
        //==========================================================================
        
        U_new,
        U_old,
        DynamicTerrain,

        //==========================================================================
        // Time stamps
        //==========================================================================

        t_new,
        t_old,

        //==========================================================================
        // Mesh hierarchy
        //==========================================================================

        Geom(),
        grids,
        dmap,
        [this](int lev) { return refRatio(lev); },

        //==========================================================================
        // Boundary conditions
        //==========================================================================

        U_bcs,
        Terrain_bcs,

        //==========================================================================
        // Refluxing
        //==========================================================================

        flux_reg,

        static_cast<bool>(amr_p.do_reflux),
        finest_level,

        //==========================================================================
        // Mesh services
        //==========================================================================

        [this](
            int lev,
            amrex::Real time,
            amrex::MultiFab& mf,
            const amrex::Vector<amrex::BCRec>& bc,
            int icomp,
            int ncomp)
        {
            FillPatch(lev,time,mf,bc,icomp,ncomp);
        },

        [](const amrex::MultiFab& mf,
           const amrex::BoxArray& ba,
           const amrex::IntVect& rr,
           int ngrow,
           int blocking)
        {
            return makeFineMask(mf, ba, rr, ngrow, blocking);
        }
    };
}

CheckpointerContext AmrMeshState::GetCheckpointerContext()
{
    return {
        U_new, U_old, DynamicTerrain,
        grids, dmap,
        dt, t_new, t_old,
        finest_level, istep[0], istep
    };
}

/* ------------------------------------------------------------------ */
/*  ComputeMassDiagnostics — mass & momentum conservation check across */
/*  all AMR levels. Outputs to stdout via LOG(INFO).                   */
/*  Each rank computes its local contribution; MPI-reduce gives global.*/
/*                                                                     */
/*  Conserved quantities (over valid fluid cells only):                */
/*    Mass        = integral(h * dA)                                   */
/*    Momentum_x  = integral(hu * dA)                                  */
/*    Momentum_y  = integral(hv * dA)                                  */
/*    Kinetic_E   = integral(0.5*(hu^2+hv^2)/h * dA)                  */
/*  (NODATA cells where terrain mask == -9999 are excluded).           */
/* ------------------------------------------------------------------ */
void AmrMeshState::ComputeMassDiagnostics(amrex::Real time, int iteration)
{
    BL_PROFILE("AmrMeshState::ComputeMassDiagnostics");

    constexpr amrex::Real nodata_val = -9999.0;

    // Per-level local accumulators
    struct LevelStats {
        amrex::Real mass = 0.0;
        amrex::Real mom_x = 0.0;
        amrex::Real mom_y = 0.0;
        amrex::Real ke = 0.0;
        amrex::Real min_h = std::numeric_limits<amrex::Real>::max();
    };
    amrex::Vector<LevelStats> local_stats(finest_level + 1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Skip levels that haven't been allocated yet
        if (U_new[lev].boxArray().empty() || DynamicTerrain[lev].boxArray().empty()) {
            continue;
        }

        const amrex::Geometry& g = geom[lev];
        const amrex::Real dx = g.CellSize(0);
        const amrex::Real dy = g.CellSize(1);
        const amrex::Real dA = dx * dy;

        // Accumulate over all boxes at this level
        for (amrex::MFIter mfi(U_new[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& vb = mfi.validbox();
            const auto u_arr        = U_new[lev].array(mfi);
            const auto terrain_arr  = DynamicTerrain[lev].array(mfi);

            using ReduceTuple = amrex::GpuTuple<
                amrex::Real,   // mass
                amrex::Real,   // mom_x
                amrex::Real,   // mom_y
                amrex::Real,   // kinetic energy
                amrex::Real    // min_h
            >;

            amrex::ReduceOps<
                amrex::ReduceOpSum,   // mass
                amrex::ReduceOpSum,   // mom_x
                amrex::ReduceOpSum,   // mom_y
                amrex::ReduceOpSum,   // kinetic energy
                amrex::ReduceOpMin    // min_h
            > reduce_op;
            amrex::ReduceData<
                amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real
            > reduce_data(reduce_op);

            reduce_op.eval(vb, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> ReduceTuple
            {
                amrex::Real h  = u_arr(i, j, k, 0);          // water depth
                amrex::Real hu = u_arr(i, j, k, 1);          // x-momentum
                amrex::Real hv = u_arr(i, j, k, 2);          // y-momentum
                int is_nodata = (terrain_arr(i, j, k, 1) == nodata_val) ? 1 : 0;

                amrex::Real s_mass = 0.0;
                amrex::Real s_mx   = 0.0;
                amrex::Real s_my   = 0.0;
                amrex::Real s_ke   = 0.0;
                amrex::Real min_h  = std::numeric_limits<amrex::Real>::max();

                if (is_nodata == 0 && h > 0.0) {
                    s_mass = h * dA;
                    s_mx   = hu * dA;
                    s_my   = hv * dA;
                    s_ke   = 0.5 * (hu * hu + hv * hv) / h * dA;
                    min_h  = h;
                }

                return {s_mass, s_mx, s_my, s_ke, min_h};
            });

            auto result = reduce_data.value();
            local_stats[lev].mass     = amrex::get<0>(result);
            local_stats[lev].mom_x    = amrex::get<1>(result);
            local_stats[lev].mom_y    = amrex::get<2>(result);
            local_stats[lev].ke       = amrex::get<3>(result);
            local_stats[lev].min_h    = amrex::min(local_stats[lev].min_h, amrex::get<4>(result));
        }
    }

    // MPI reduce across all ranks
    for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].mass,     1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].mom_x,    1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].mom_y,    1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].ke,       1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealMin(&local_stats[lev].min_h,    1, amrex::ParallelDescriptor::IOProcessorNumber());
    }

    // IO processor prints diagnostics
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        amrex::Real tot_mass = 0.0, tot_mx = 0.0, tot_my = 0.0, tot_ke = 0.0;
        amrex::Real g_min_h = std::numeric_limits<amrex::Real>::max();
        for (int lev = 0; lev <= finest_level; ++lev) {
            tot_mass  += local_stats[lev].mass;
            tot_mx    += local_stats[lev].mom_x;
            tot_my    += local_stats[lev].mom_y;
            tot_ke    += local_stats[lev].ke;
            g_min_h   = amrex::min(g_min_h, local_stats[lev].min_h);
        }

        LOG(INFO,
            "[Conservation] t=" + std::to_string(time)
            + "  iter=" + std::to_string(iteration)
            + "  mass=" + std::to_string(tot_mass)
            + "  px=" + std::to_string(tot_mx)
            + "  py=" + std::to_string(tot_my)
            + "  KE=" + std::to_string(tot_ke)
            + "  min_h=" + std::to_string(g_min_h)
        );

        // Report conservation error if initial values are stored
        if (initial_mass_stored && tot_mass > 0.0) {
            amrex::Real err_mass = (tot_mass  - initial_mass)  / initial_mass  * 100.0;
            amrex::Real err_mx   = (tot_mx   - initial_mom_x) / initial_mom_x * 100.0;
            amrex::Real err_my   = (tot_my   - initial_mom_y) / initial_mom_y * 100.0;
            amrex::Real err_ke   = (tot_ke   - initial_ke)    / initial_ke    * 100.0;
            LOG(INFO,
                "[Conservation error] dMass=" + std::to_string(err_mass) + "%"
                + "  dPx=" + std::to_string(err_mx) + "%"
                + "  dPy=" + std::to_string(err_my) + "%"
                + "  dKE=" + std::to_string(err_ke) + "%"
            );
        }
    }
}

/* ------------------------------------------------------------------ */
/*  StoreInitialMassMomentum — compute and store baseline totals at   */
/*  t=0 so that ComputeMassDiagnostics can report conservation error. */
/* ------------------------------------------------------------------ */
void AmrMeshState::StoreInitialMassMomentum()
{
    BL_PROFILE("AmrMeshState::StoreInitialMassMomentum");

    constexpr amrex::Real nodata_val = -9999.0;

    struct LevelStats {
        amrex::Real mass = 0.0;
        amrex::Real mom_x = 0.0;
        amrex::Real mom_y = 0.0;
        amrex::Real ke = 0.0;
    };
    amrex::Vector<LevelStats> local_stats(finest_level + 1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (U_new[lev].boxArray().empty() || DynamicTerrain[lev].boxArray().empty()) {
            continue;
        }

        const amrex::Geometry& g = geom[lev];
        const amrex::Real dx = g.CellSize(0);
        const amrex::Real dy = g.CellSize(1);
        const amrex::Real dA = dx * dy;

        for (amrex::MFIter mfi(U_new[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& vb = mfi.validbox();
            const auto u_arr     = U_new[lev].array(mfi);
            const auto terrain_arr = DynamicTerrain[lev].array(mfi);

            using ReduceTuple = amrex::GpuTuple<
                amrex::Real, amrex::Real, amrex::Real, amrex::Real
            >;

            amrex::ReduceOps<
                amrex::ReduceOpSum, amrex::ReduceOpSum,
                amrex::ReduceOpSum, amrex::ReduceOpSum
            > reduce_op;
            amrex::ReduceData<
                amrex::Real, amrex::Real, amrex::Real, amrex::Real
            > reduce_data(reduce_op);

            reduce_op.eval(vb, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> ReduceTuple
            {
                amrex::Real h  = u_arr(i, j, k, 0);
                amrex::Real hu = u_arr(i, j, k, 1);
                amrex::Real hv = u_arr(i, j, k, 2);
                int is_nodata = (terrain_arr(i, j, k, 1) == nodata_val) ? 1 : 0;

                amrex::Real s_mass = 0.0;
                amrex::Real s_mx   = 0.0;
                amrex::Real s_my   = 0.0;
                amrex::Real s_ke   = 0.0;

                if (is_nodata == 0 && h > 0.0) {
                    s_mass = h * dA;
                    s_mx   = hu * dA;
                    s_my   = hv * dA;
                    s_ke   = 0.5 * (hu * hu + hv * hv) / h * dA;
                }

                return {s_mass, s_mx, s_my, s_ke};
            });

            auto result = reduce_data.value();
            local_stats[lev].mass = amrex::get<0>(result);
            local_stats[lev].mom_x = amrex::get<1>(result);
            local_stats[lev].mom_y = amrex::get<2>(result);
            local_stats[lev].ke   = amrex::get<3>(result);
        }
    }

    // MPI reduce
    for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].mass, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].mom_x, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].mom_y, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealSum(&local_stats[lev].ke,   1, amrex::ParallelDescriptor::IOProcessorNumber());
    }

    // IO processor accumulates global totals and stores them
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        amrex::Real tot_mass = 0.0, tot_mx = 0.0, tot_my = 0.0, tot_ke = 0.0;
        for (int lev = 0; lev <= finest_level; ++lev) {
            tot_mass += local_stats[lev].mass;
            tot_mx   += local_stats[lev].mom_x;
            tot_my   += local_stats[lev].mom_y;
            tot_ke   += local_stats[lev].ke;
        }

        initial_mass  = tot_mass;
        initial_mom_x = tot_mx;
        initial_mom_y = tot_my;
        initial_ke    = tot_ke;
        initial_mass_stored = true;

        LOG(INFO,
            "[Initial] mass=" + std::to_string(initial_mass)
            + "  px=" + std::to_string(initial_mom_x)
            + "  py=" + std::to_string(initial_mom_y)
            + "  KE=" + std::to_string(initial_ke)
        );
    }

    // Broadcast stored values to all ranks so every rank has the baseline
    amrex::ParallelDescriptor::Bcast(&initial_mass,  1, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::Bcast(&initial_mom_x, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::Bcast(&initial_mom_y, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::Bcast(&initial_ke,    1, amrex::ParallelDescriptor::IOProcessorNumber());
    initial_mass_stored = true;  // every rank reaches here after the function runs
}