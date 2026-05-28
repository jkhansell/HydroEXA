#ifndef TAGGING_H
#define TAGGING_H

#include <hydroexa/HydroEXA.H>
#include <AMReX_Array4.H>
#include <cmath>

// Helper to safely calculate velocity magnitude, avoiding dry-cell division by zero
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
get_umag(int i, int j, int k, amrex::Array4<amrex::Real const> const& state)
{
    amrex::Real h = state(i,j,k,0);
    if (h > 1.e-6) { // Dry bed threshold
        amrex::Real u = state(i,j,k,1) / h;
        amrex::Real v = state(i,j,k,2) / h;
        return std::sqrt(u*u + v*v);
    }
    return 0.0;
}

AMREX_GPU_HOST_DEVICE
void
hydroexa_tagging (int i, int j, int k,
             amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             amrex::Real h_grad_thresh, 
             amrex::Real u_grad_thresh, 
             char tagval)
{
    // 1. Calculate h gradient (Central Difference)
    amrex::Real h_r = state(i+1, j,   k, 0); // west 
    amrex::Real h_l = state(i-1, j,   k, 0); // east
    amrex::Real h_t = state(i,   j+1, k, 0); // north
    amrex::Real h_b = state(i,   j-1, k, 0); // south

    // Using undivided differences (change across 2 cells). 
    // This is preferred for AMR tagging over true gradients because it scales nicely with grid levels.
    amrex::Real grad_h = std::sqrt((h_r - h_l)*(h_r - h_l) + (h_t - h_b)*(h_t - h_b));

    // 2. Calculate |u| gradient
    amrex::Real u_r = get_umag(i+1, j,   k, state); // west
    amrex::Real u_l = get_umag(i-1, j,   k, state); // east
    amrex::Real u_t = get_umag(i,   j+1, k, state); // north
    amrex::Real u_b = get_umag(i,   j-1, k, state); // south

    amrex::Real grad_u = std::sqrt((u_r - u_l)*(u_r - u_l) + (u_t - u_b)*(u_t - u_b));

    // 3. Tag the cell if either gradient exceeds its threshold
    if (grad_h > h_grad_thresh || grad_u > u_grad_thresh) {
        tag(i,j,k) = tagval;
    }
}

#endif