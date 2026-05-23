#ifndef HYDROEXA_STATE_SWE_HPP
#define HYDROEXA_STATE_SWE_HPP

namespace HydroEXA::SWE {
    // Indices for state components in the amrex::MultiFab
    constexpr int h  = 0; // water height
    constexpr int uh = 1; // x-momentum
    constexpr int vh = 2; // y-momentum
    constexpr int zb = 3; // bottom elevation
    
    constexpr int NCOMP = 4;
}

#endif
