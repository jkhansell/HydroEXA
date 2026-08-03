#ifndef BASE_SOLVER_H_
#define BASE_SOLVER_H_

#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_FluxRegister.H>

#include <algorithm>

#include <solvers/SolverContext.H>

template <typename Derived>
struct BaseSolver {
    
    void compute_fluxes(
        SolverContext ctx, int lev, amrex::Real dt, amrex::Real time
    ) {
        static_cast<Derived*>(this)->compute_fluxes_Impl(ctx, lev, dt, time);
    }

    amrex::Real compute_dt_level(const amrex::Vector<amrex::MultiFab>& U,
                                 const amrex::Vector<amrex::MultiFab>& terrain,
                                 const amrex::Vector<amrex::Geometry>& geom,
                                 int lev, amrex::Real cfl)
    {
        return static_cast<Derived*>(this)->compute_dt_Impl(U, terrain, geom, lev, cfl);
    }

    void compute_dt(amrex::Vector<amrex::Real>& dt,
                    const amrex::Vector<amrex::MultiFab>& U,
                    const amrex::Vector<amrex::MultiFab>& terrain,
                    const amrex::Vector<amrex::Geometry>& geom,
                    const amrex::Vector<amrex::IntVect>& ref_ratio,
                    int finest_level, amrex::Real cfl) 
    {
        amrex::Vector<amrex::Real> dt_tmp(finest_level + 1);
        for (int lev = 0; lev <= finest_level; ++lev) {
            dt_tmp[lev] = compute_dt_level(U, terrain, geom, lev, cfl);
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Computed dt per level:\n";
            for (int lev = 0; lev <= finest_level; ++lev) {
                amrex::Print()
                    << "  Level " << lev
                    << " : " << dt_tmp[lev] << "\n";
            }
        }

        // 1. Parallel reduction to get the true minimums across all MPI ranks
        amrex::ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

        // 2. Start with the dt calculated for the finest level
        int max_lev = finest_level;
        dt[max_lev] = dt_tmp[max_lev];

        // 3. Work backwards from the finest level to the coarsest level
        //    dt_coarse = dt_fine * ref_ratioEx
        for (int lev = max_lev - 1; lev >= 0; --lev) {
            // ref_ratio[lev][0] is the refinement ratio between 'lev' and 'lev+1'
            dt[lev] = dt[lev + 1] * ref_ratio[lev][0];
            
            // Safety check: Ensure the scaled dt doesn't accidentally 
            // violate the original physics constraints of this coarser level
            dt[lev] = std::min(dt[lev], dt_tmp[lev]);
        }
    
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Final dt per level:\n";
            for (int lev = 0; lev <= finest_level; ++lev) {
                amrex::Print()
                    << "  Level " << lev
                    << " : " << dt[lev] << "\n";
            }
        }    
    }

    void tag_cells(amrex::TagBoxArray& tags,
                   const amrex::MultiFab& U,
                   const amrex::MultiFab& terrain,
                   const amrex::Geometry& geom,
                   int lev,
                   amrex::Real time,
                   const PhysicsParameters& phys_params) {
        static_cast<Derived*>(this)->tag_cells_Impl(tags, U, terrain, geom, lev, time, phys_params);
    }

};

#endif // BASE_SOLVER_H_