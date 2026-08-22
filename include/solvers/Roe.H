#ifndef ROE_H
#define ROE_H

#include <AMReX_REAL.H>
#include <AMReX_Math.H>
#include <AMReX_TagBox.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_FluxRegister.H>

#include <cmath>

#include <utils/Constants.H>
#include <utils/Params.H>
#include <solvers/BaseSolver.H>

struct Roe : public BaseSolver<Roe> {
    void compute_fluxes_Impl(
        SolverContext ctx, int lev, amrex::Real dt, amrex::Real time
    );

    void compute_amrex_effective_fluxes(
        const amrex::Box& face_box,
        amrex::Array4<amrex::Real const> const& U,
        amrex::Array4<amrex::Real const> const& terrain,
        amrex::Array4<amrex::Real> const& D_minus_mf,
        amrex::Array4<amrex::Real> const& D_plus_mf,
        const amrex::Real dt, const amrex::Real dx, const int dir
    );

    amrex::Real compute_dt_Impl(const amrex::Vector<amrex::MultiFab>& U,
                                const amrex::Vector<amrex::MultiFab>& terrain,
                                const amrex::Vector<amrex::Geometry>& geom,
                                int lev, amrex::Real cfl);

    void tag_cells_Impl(amrex::TagBoxArray& tags,
                        const amrex::MultiFab& U,
                        const amrex::MultiFab& terrain,
                        const amrex::Geometry& geom,
                        int lev,
                        amrex::Real time,
                        const PhysicsParameters& phys_params);
};

#endif // ROE_H