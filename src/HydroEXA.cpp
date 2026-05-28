
#include <hydroexa/HydroEXA.H>
#include <hydroexa/HydroEXA_prob_parm.H>

#include <AMReX_Gpu.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

#include <climits>

using namespace amrex;

constexpr int HydroEXA::NUM_GROW;

// Enforce problem defaults
BCRec     HydroEXA::phys_bc;
int       HydroEXA::verbose = 0;
IntVect   HydroEXA::hydro_tile_size {AMREX_D_DECL(16,16,1)};
Real      HydroEXA::cfl       = 0.5;
int       HydroEXA::do_reflux = 1;

int  HydroEXA::refine_max_h_grad_lev    = 5;
Real HydroEXA::h_grad_threshold         = 1e-2;

int  HydroEXA::refine_max_umag_grad_lev = 5;
Real HydroEXA::umag_grad_threshold      = 1e-2;

HydroEXA::HydroEXA ()
{}

HydroEXA::HydroEXA (Amr&            papa,
          int             lev,
          const Geometry& level_geom,
          const BoxArray& bl,
          const DistributionMapping& dm,
          Real            time)
    : AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    if (do_reflux && level > 0) {
        flux_reg.reset(new FluxRegister(grids,dmap,crse_ratio,level,NUM_STATE));
    }

    buildMetrics();
}

HydroEXA::~HydroEXA ()
{}

void
HydroEXA::init (AmrLevel& old)
{
    auto& oldlev = dynamic_cast<HydroEXA&>(old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev.state[State_Type].curTime();
    Real prev_time = oldlev.state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    MultiFab& S_new = get_new_data(State_Type);
    FillPatch(old,S_new,0,cur_time,State_Type,0,NUM_STATE);

    MultiFab& T_new = get_new_data(Terrain_Type);
    FillPatch(old,T_new,0,cur_time,Terrain_Type,0,NUM_TERRAIN);
}

void
HydroEXA::init ()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/static_cast<Real>(parent->MaxRefRatio(level-1));
    setTimeLevel(cur_time,dt_old,dt);

    MultiFab& S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NUM_STATE);

    MultiFab& T_new = get_new_data(Terrain_Type);
    FillCoarsePatch(T_new, 0, cur_time, Terrain_Type, 0, NUM_TERRAIN);
}

void
HydroEXA::initData ()
{
    BL_PROFILE("HydroEXA::initData()");

    const auto geomdata = geom.data();
    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& T_new = get_new_data(Terrain_Type);


#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.validbox();
        auto sfab = S_new.array(mfi);
        auto tfab = T_new.array(mfi);

        amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            initdata(i, j, k, sfab, tfab, geomdata);
        });
    }
}

void
HydroEXA::computeInitialDt (int                    finest_level,
                       int                    /*sub_cycle*/,
                       Vector<int>&           n_cycle,
                       const Vector<IntVect>& /*ref_ratio*/,
                       Vector<Real>&          dt_level,
                       Real                   stop_time)
{
    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0) {
        return;
    }

    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor   *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_level[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
HydroEXA::computeNewDt (int                    finest_level,
                   int                    /*sub_cycle*/,
                   Vector<int>&           n_cycle,
                   const Vector<IntVect>& /*ref_ratio*/,
                   Vector<Real>&          dt_min,
                   Vector<Real>&          dt_level,
                   Real                   stop_time,
                   int                    post_regrid_flag)
{
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0) {
        return;
    }

    for (int i = 0; i <= finest_level; i++)
    {
        dt_min[i] = getLevel(i).estTimeStep();
    }

    if (post_regrid_flag == 1)
    {
        //
        // Limit dt's by pre-regrid dt
        //
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],dt_level[i]);
        }
    }
    else
    {
        //
        // Limit dt's by change_max * old dt
        //
        static Real change_max = 1.1;
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
        }
    }

    //
    // Find the minimum over all levels
    //
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
HydroEXA::post_regrid (int /*lbase*/, int /*new_finest*/)
{
}

void
HydroEXA::post_timestep (int /*iteration*/)
{
    BL_PROFILE("post_timestep");

    if (do_reflux && level < parent->finestLevel()) {
        MultiFab& S = get_new_data(State_Type);
        HydroEXA& fine_level = getLevel(level+1);
        fine_level.flux_reg->Reflux(S, Real(1.0), 0, 0, NUM_STATE, geom);
    }

    if (level < parent->finestLevel()) {
        avgDown();
    }
}

void
HydroEXA::postCoarseTimeStep (Real /*time*/)
{
    BL_PROFILE("postCoarseTimeStep()");

    // This only computes sum on level 0
    if (verbose >= 2) {
        printTotal();
    }
}

void
HydroEXA::printTotal () const
{
    const MultiFab& S_new = get_new_data(State_Type);
    std::array<Real,3> tot;
    for (int comp = 0; comp < 3; ++comp) {
        tot[comp] = S_new.sum(comp,true) * geom.ProbSize();
    }
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
            ParallelDescriptor::ReduceRealSum(tot.data(), 3, ParallelDescriptor::IOProcessorNumber());
            amrex::Print().SetPrecision(17) << "\n[HydroEXA] Total mass       is " << tot[0] << "\n"
                                             <<   "      Total x-momentum is " << tot[1] << "\n"
                                             <<   "      Total y-momentum is " << tot[2] << "\n";
#ifdef BL_LAZY
        });
#endif
}

void
HydroEXA::post_init (Real)
{
    if (level > 0) return;
    for (int k = parent->finestLevel()-1; k >= 0; --k) {
        getLevel(k).avgDown();
    }

    if (verbose >= 2) {
        printTotal();
    }
}

void
HydroEXA::post_restart ()
{
}

void
HydroEXA::errorEst (TagBoxArray& tags, int, int, Real /*time*/, int, int)
{
    BL_PROFILE("HydroEXA::errorEst()");

    if (level < refine_max_h_grad_lev || level < refine_max_umag_grad_lev) {
        const MultiFab& StateNew = get_new_data(State_Type);
        const Real cur_time = state[State_Type].curTime();
        
        MultiFab StateBorder(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());
        FillPatch(*this, StateBorder, NUM_GROW, cur_time, State_Type, 0, NUM_STATE);

        const char   tagval = TagBox::SET;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(StateNew,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            auto const& state = StateBorder.array(mfi);
            auto const& tag = tags.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                hydroexa_tagging(i, j, k, tag, state, h_grad_threshold, umag_grad_threshold, tagval);
            });
        }
    }
}

void
HydroEXA::read_params ()
{
    ParmParse pp("HydroEXA");

    pp.query("v", verbose);

    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
        for (int i=0; i<AMREX_SPACEDIM; i++) hydro_tile_size[i] = tilesize[i];
    }

    pp.query("cfl", cfl);

    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);

    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    pp.query("do_reflux", do_reflux);

    pp.query("refine_max_h_grad_lev", refine_max_h_grad_lev);
    pp.query("h_grad_threshold", h_grad_threshold);
    pp.query("refine_max_umag_grad_lev", refine_max_umag_grad_lev);
    pp.query("umag_grad_threshold", umag_grad_threshold);
}

void
HydroEXA::avgDown ()
{
    BL_PROFILE("HydroEXA::avgDown()");

    if (level == parent->finestLevel()) return;

    auto& fine_lev = getLevel(level+1);

    MultiFab& S_crse =          get_new_data(State_Type);
    MultiFab& S_fine = fine_lev.get_new_data(State_Type);

    amrex::average_down(S_fine, S_crse, fine_lev.geom, geom,
                        0, S_fine.nComp(), parent->refRatio(level));

    const int nghost = 0;
}

void
HydroEXA::buildMetrics ()
{
    // make sure dx == dy == dz
    const Real* dx = geom.CellSize();
    if (std::abs(dx[0]-dx[1]) > Real(1.e-12)*dx[0]) {
        amrex::Abort("HydroEXA: must have dx == dy\n");
    }
}



Real
HydroEXA::estTimeStep ()
{
    BL_PROFILE("HydroEXA::estTimeStep()");

    const auto dx = geom.CellSizeArray();
    const MultiFab& S = get_new_data(State_Type);

    amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);

    using ReduceTuple = decltype(reduce_data)::Type;

    for (amrex::MFIter mfi(S, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& U = S.const_array(mfi);
        const amrex::Real dxx = dx[0];
        const amrex::Real dyy = dx[1];

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            amrex::Real h = U(i,j,k,0);
            amrex::Real dt_cell = 1.e6; 

            if (h > 1.e-12) { 
                amrex::Real u = U(i,j,k,1) / h;
                amrex::Real v = U(i,j,k,2) / h;
                amrex::Real c = std::sqrt(9.81 * h); 

                amrex::Real dt_x = dxx / (std::abs(u) + c);
                amrex::Real dt_y = dyy / (std::abs(v) + c);

                dt_cell = amrex::min(dt_x, dt_y);
            }
            return dt_cell;
        });
    }

    // Extract the LOCAL minimum from the GPU to the CPU
    ReduceTuple local_dt = reduce_data.value();
    amrex::Real dt_est = amrex::get<0>(local_dt);

    // Apply CFL multiplier to the local estimate
    dt_est *= cfl;

    ParallelDescriptor::ReduceRealMin(dt_est);

    return dt_est;

}

Real
HydroEXA::initialTimeStep ()
{
    return estTimeStep();
}


