#ifndef ROE_H
#define ROE_H

#include <AMReX_REAL.H>
#include <AMReX_Math.H>
#include <cmath>

#include <utils/Constants.H>
#include <solvers/BaseSolver.H>
#include <solvers/kernels/Roe.H>

struct Roe : public BaseSolver<Roe> {
    void compute_fluxes_Impl(amrex::Vector<amrex::MultiFab>& U,
                             amrex::Vector<amrex::MultiFab>& terrain,
                             const amrex::Vector<amrex::Geometry>& geom,
                             int lev);

    amrex::Real compute_dt_Impl(const amrex::Vector<amrex::MultiFab>& U,
                                const amrex::Vector<amrex::MultiFab>& terrain,
                                const amrex::Vector<amrex::Geometry>& geom,
                                int lev, amrex::Real cfl );
    
};


#endif // ROE_H