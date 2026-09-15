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

        // 1. Reduce minimums across ranks FIRST before printing or processing
        amrex::ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Computed dt per level:\n";
            for (int lev = 0; lev <= finest_level; ++lev) {
                amrex::Print() << "  Level " << lev << " : " << dt_tmp[lev] << "\n";
            }
        }

        // 2. Project all level constraints up to Level 0 to find the safest global coarse dt
        amrex::Real dt_0 = dt_tmp[0];
        amrex::Real factor = 1.0;
        for (int lev = 1; lev <= finest_level; ++lev) {
            factor *= static_cast<amrex::Real>(ref_ratio[lev - 1][0]);
            // Scale the fine level constraint up to Level 0 equivalent and cap dt_0
            dt_0 = std::min(dt_0, dt_tmp[lev] * factor);
        }

        // 3. Assign Level 0
        dt[0] = dt_0;

        // 4. Downscale to fine levels using STRICT integer division
        for (int lev = 1; lev <= finest_level; ++lev) {
            dt[lev] = dt[lev - 1] / static_cast<amrex::Real>(ref_ratio[lev - 1][0]);
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Final dt per level:\n";
            for (int lev = 0; lev <= finest_level; ++lev) {
                amrex::Print() << "  Level " << lev << " : " << dt[lev] << "\n";
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