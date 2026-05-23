#include <AMReX_TagBox.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuMemory.H>

#include "hydroexa/HydroLevel.hpp"
#include "hydroexa/KernelsSWE.hpp"
#include "hydroexa/KernelsExner.hpp"

HydroLevel::HydroLevel(amrex::Amr& papa, int lev, const amrex::Geometry& level_geom,
                       const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                       amrex::Real time)
    : amrex::AmrLevel(papa, lev, level_geom, ba, dm, time)
{
    amrex::ParmParse pp;
    use_exner = false;
    pp.query("use_exner", use_exner);
}

void HydroLevel::variableSetUp() {
    bool use_exner = false;
    amrex::ParmParse pp;
    pp.query("use_exner", use_exner);
    int ncomp = (use_exner ? 5 : 4);
    
    // Default amrex interpolation scheme
    amrex::Interpolater* interp = &amrex::cell_cons_interp;
    
    amrex::AmrLevel::desc_lst.addDescriptor(0, amrex::IndexType::TheCellType(), amrex::StateDescriptor::Point, 2, ncomp, interp);
}

void HydroLevel::variableCleanUp() {
    amrex::AmrLevel::desc_lst.clear();
}

void HydroLevel::computeInitialDt(int finest_level, int sub_cycle, amrex::Vector<int>& n_cycle, const amrex::Vector<amrex::IntVect>& ref_ratio, amrex::Vector<amrex::Real>& dt_level, amrex::Real stop_time) {
    amrex::Real dt = 0.01;
    amrex::ParmParse pp; pp.query("dt", dt);
    for (int i = 0; i <= finest_level; ++i) dt_level[i] = dt;
}

void HydroLevel::computeNewDt(int finest_level, int sub_cycle, amrex::Vector<int>& n_cycle, const amrex::Vector<amrex::IntVect>& ref_ratio, amrex::Vector<amrex::Real>& dt_min, amrex::Vector<amrex::Real>& dt_level, amrex::Real stop_time, int post_regrid_flag) {
    amrex::Real dt = 0.01;
    amrex::ParmParse pp; pp.query("dt", dt);
    for (int i = 0; i <= finest_level; ++i) dt_level[i] = dt;
}

amrex::Real HydroLevel::advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle) {
    auto& state_new = get_new_data(0);
    auto& state_old = get_old_data(0);
    
    // Explicit forward euler / simple dispatch for now over state 0
    amrex::MultiFab::Copy(state_new, state_old, 0, 0, state_new.nComp(), state_new.nGrow());
    
    auto const& dx = geom.CellSizeArray();
    
    amrex::MultiFab rhs_mf(grids, dmap, state_new.nComp(), 0);
    rhs_mf.setVal(0.0);
    
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(state_old, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& rhs_arr = rhs_mf.array(mfi);
        auto const& state_arr = state_old.const_array(mfi);
        
        if (use_exner) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                HydroEXA::SWEExner::compute_fluxes(i, j, k, rhs_arr, state_arr, dx);
            });
        } else {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                HydroEXA::SWE::compute_fluxes(i, j, k, rhs_arr, state_arr, dx);
            });
        }
    }
    
    // state_new += dt * rhs // standard euler for now instead of amrex timeframe
    // amrex::MultiFab::Saxpy(state_new, dt, rhs_mf, 0, 0, state_new.nComp(), 0);
    
    return dt;
}

void HydroLevel::post_regrid(int lbase, int new_finest) {}
void HydroLevel::post_init(amrex::Real stop_time) {}

void HydroLevel::initData() {
    auto& S_new = get_new_data(0);
    S_new.setVal(0.0);
}

void HydroLevel::init(amrex::AmrLevel& old) {}
void HydroLevel::init() {}

void HydroLevel::errorEst(amrex::TagBoxArray& tb, int clearval, int tagval, amrex::Real time, int n_error_buf, int ngrow) {
    // TODO: implement momentum gradient tagging here for dynamic grid refinement!
}
