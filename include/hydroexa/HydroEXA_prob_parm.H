#ifndef HYDROEXA_PROB_PARM_H_
#define HYDROEXA_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>

using namespace amrex::literals;


AMREX_GPU_DEVICE
inline
void
initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state, amrex::Array4<amrex::Real> const& terrain,
          amrex::GeometryData const& geomdata)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    // 1. Calculate the length of the physical domain axes
    Real Lx = prob_hi[0] - prob_lo[0];
    Real Ly = prob_hi[1] - prob_lo[1];

    // 2. Find L_min (the smallest side of the domain)
    Real L_min = (Lx < Ly) ? Lx : Ly;

    // 3. Define the cylinder radius: L_min / 10
    Real dam_radius = L_min / 10.0;

    // 4. Calculate the coordinates for the exact physical center of the domain
    Real center_x = prob_lo[0] + 0.5 * Lx;
    Real center_y = prob_lo[1] + 0.5 * Ly;

    // 5. Calculate the physical (x, y) coordinates of the current cell center
    // i + 0.5 and j + 0.5 maps to the cell-centered finite volume locations
    Real x = prob_lo[0] + (static_cast<Real>(i) + 0.5) * dx[0];
    Real y = prob_lo[1] + (static_cast<Real>(j) + 0.5) * dx[1];

    // 6. Calculate the radial distance from the domain center to this cell center
    Real r = std::sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y));

    // 7. Assign Shallow Water states based on the partition radius
    // Assuming component mapping: H_COMP=0, HU_COMP=1, HV_COMP=2
    if (r <= dam_radius) {
        state(i, j, k, 0) = 4.0; // High state inside the circular dam
    } else {
        state(i, j, k, 0) = 1.0; // Low state outside
    }

    // Initialize momentum components to zero (stationary fluid initially)
    state(i, j, k, 1) = 0.0; // hu
    state(i, j, k, 2) = 0.0; // hv

    terrain(i, j, k, 0) = 0.1; // Flat terrain
    terrain(i, j, k, 1) = 0.0; // No Friction
}


#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
    }
}


#endif // HYDROEXA_PROB_PARM_H_
