#ifndef _HYDROEXA_KERNELS_SWE_H_
#define _HYDROEXA_KERNELS_SWE_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include "hydroexa/StateSWE.H"

namespace HydroEXA::SWE {

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void compute_fluxes(int i, int j, int k,
                        amrex::Array4<amrex::Real> const& rhs,
                        amrex::Array4<amrex::Real const> const& state,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx)
    {
        // TODO: Implement advanced flux solvers (e.g., Roe solver)
        // For now, place 0.0 to satisfy compilation and basic dispatch structure.
        
        rhs(i, j, k, h) = 0.0;
        rhs(i, j, k, uh) = 0.0;
        rhs(i, j, k, vh) = 0.0;
        rhs(i, j, k, zb) = 0.0;
    }
}

#endif
