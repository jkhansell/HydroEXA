#ifndef HYDROEXA_STATE_EXNER_HPP
#define HYDROEXA_STATE_EXNER_HPP

namespace HydroEXA::SWEExner {
    // Indices for state components in the amrex::MultiFab
    constexpr int h    = 0; // water height
    constexpr int uh   = 1; // x-momentum
    constexpr int vh   = 2; // y-momentum
    constexpr int zb   = 3; // bottom elevation
    constexpr int sed  = 4; // sediment concentration
    
    constexpr int NCOMP = 5;
}

#endif
