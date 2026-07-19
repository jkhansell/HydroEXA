#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <AMReX_REAL.H>

namespace PhysConstants {
    const amrex::Real GRAV   = 9.81;
}

namespace Tolerances { 
    const amrex::Real DEPTH = 1.0e-4;
    const amrex::Real FLUX = 1.0e-7;
    const amrex::Real TOLDRY = 1e-6;
    const amrex::Real TOL12 = 1e-12; 
}

#endif