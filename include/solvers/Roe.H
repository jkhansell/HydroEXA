#ifndef ROE_H
#define ROE_H

#include <AMReX_REAL.H>
#include <AMReX_Math.H>
#include <cmath>

#include <utils/Constants.H>

#include <state/AmrMeshState.H>
#include <solvers/BaseSolver.H>
#include <solvers/kernels/Roe.H>

using namespace amrex;

struct Roe : public BaseSolver<Roe> {
    int get_ncomp_Impl() const { return 3; }          // h, hu, hv
    int get_ncomp_terrain_Impl() const { return 1; }  // Zb
    void compute_fluxes_Impl(AmrMeshState& mesh_state, int lev);
    Real compute_dt_Impl(AmrMeshState& mesh_state, int lev);
};

#include <solvers/Roe.tpp>
#endif