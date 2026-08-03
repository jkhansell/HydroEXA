#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <AMReX_REAL.H>
#include <AMReX_GpuControl.H>

namespace PhysConstants {
    AMREX_GPU_DEVICE constexpr amrex::Real GRAV() { return 9.81; }
}

namespace Tolerances {
    AMREX_GPU_DEVICE constexpr amrex::Real DEPTH()  { return 1.0e-4; }
    AMREX_GPU_DEVICE constexpr amrex::Real FLUX()   { return 1.0e-7; }
    AMREX_GPU_DEVICE constexpr amrex::Real TOLDRY() { return 1e-6; }
    AMREX_GPU_DEVICE constexpr amrex::Real TOL12()  { return 1e-12; }
    AMREX_GPU_DEVICE constexpr amrex::Real TOL10()  { return 1e-10; }
}

#endif