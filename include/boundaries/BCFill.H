#ifndef BCFILL_H
#define BCFILL_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_BCRec.H>

// ============================================================================
// Physical BC type indices (user-facing, read from inputs file)
// ============================================================================
// 0 = Interior (periodic/interior)    3 = Symmetry
// 1 = Inflow                         4 = SlipWall
// 2 = Outflow                        5 = NoSlipWall

// ============================================================================
// Mathematical BC type lookup tables
// ============================================================================
// Maps physical BC type (index) -> mathematical BCType for each state component.
// Components: scalar (h), normal velocity (hu in x-dir, hv in y-dir),
//            tangential velocity (v-component in x-dir, u-component in y-dir).

static int const scalar_bc[] =
{
    amrex::BCType::int_dir, amrex::BCType::ext_dir, amrex::BCType::foextrap,
    amrex::BCType::reflect_even, amrex::BCType::reflect_even, amrex::BCType::reflect_even
};

static int const norm_vel_bc[] =
{
    amrex::BCType::int_dir, amrex::BCType::ext_dir, amrex::BCType::foextrap,
    amrex::BCType::reflect_odd,  amrex::BCType::reflect_odd,  amrex::BCType::reflect_odd
};

static int const tang_vel_bc[] =
{
    amrex::BCType::int_dir, amrex::BCType::ext_dir, amrex::BCType::foextrap,
    amrex::BCType::reflect_even, amrex::BCType::reflect_even, amrex::BCType::reflect_odd
};

// ============================================================================
// Per-component BCRec construction helpers
// ============================================================================

inline
void
set_scalar_bc (amrex::BCRec& bc, const amrex::BCRec& phys_bc)
{
    auto const* lo_bc = phys_bc.lo();
    auto const* hi_bc = phys_bc.hi();
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        bc.setLo(i, scalar_bc[lo_bc[i]]);
        bc.setHi(i, scalar_bc[hi_bc[i]]);
    }
}

inline
void
set_x_vel_bc(amrex::BCRec& bc, const amrex::BCRec& phys_bc)
{
    auto const* lo_bc = phys_bc.lo();
    auto const* hi_bc = phys_bc.hi();
    bc.setLo(0, norm_vel_bc[lo_bc[0]]);
    bc.setHi(0, norm_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, tang_vel_bc[lo_bc[1]]);
    bc.setHi(1, tang_vel_bc[hi_bc[1]]);
#endif
}

inline
void
set_y_vel_bc(amrex::BCRec& bc, const amrex::BCRec& phys_bc)
{
    auto const* lo_bc = phys_bc.lo();
    auto const* hi_bc = phys_bc.hi();
    bc.setLo(0, tang_vel_bc[lo_bc[0]]);
    bc.setHi(0, tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, norm_vel_bc[lo_bc[1]]);
    bc.setHi(1, norm_vel_bc[hi_bc[1]]);
#endif
}

// ============================================================================
// ext_dir fill functor
// ============================================================================
// Called by PhysBCFunct for ghost cells where BCType::ext_dir is set.
// Extrapolates values from the interior boundary cell.

struct HydroEXAFill
{
    AMREX_GPU_DEVICE
    void operator()(
        const amrex::IntVect& iv,
        amrex::Array4<amrex::Real> const& dest,
        const int dcomp,
        const int numcomp,
        amrex::GeometryData const& geom,
        const amrex::Real /*time*/,
        const amrex::BCRec* bcr,
        const int bcomp,
        const int /*orig_comp*/) const
    {
        const amrex::Box& domain = geom.Domain();
        const int i = iv[0];
        const int j = iv[1];

        // --------------------------------------------------------------------
        // X-LOW Domain Boundary (i < domain.smallEnd(0))
        // --------------------------------------------------------------------
        if (i < domain.smallEnd(0)) {
            int ilast = domain.smallEnd(0);
            for (int n = 0; n < numcomp; ++n) {
                int bc = bcr[bcomp + n].lo(0);
                
                if (bc == amrex::BCType::foextrap) {
                    // Outflow / Zero-Gradient Extrapolation
                    dest(i, j, 0, dcomp + n) = dest(ilast, j, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_even) {
                    // Reflective Wall (Symmetric)
                    int i_sym = 2 * ilast - 1 - i;
                    dest(i, j, 0, dcomp + n) = dest(i_sym, j, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_odd) {
                    // Reflective Wall (Anti-symmetric for normal momentum hu)
                    int i_sym = 2 * ilast - 1 - i;
                    dest(i, j, 0, dcomp + n) = -dest(i_sym, j, 0, dcomp + n);
                }
            }
        }

        // --------------------------------------------------------------------
        // X-HIGH Domain Boundary (i > domain.bigEnd(0))
        // --------------------------------------------------------------------
        if (i > domain.bigEnd(0)) {
            int ilast = domain.bigEnd(0);
            for (int n = 0; n < numcomp; ++n) {
                int bc = bcr[bcomp + n].hi(0);

                if (bc == amrex::BCType::foextrap) {
                    dest(i, j, 0, dcomp + n) = dest(ilast, j, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_even) {
                    int i_sym = 2 * ilast + 1 - i;
                    dest(i, j, 0, dcomp + n) = dest(i_sym, j, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_odd) {
                    int i_sym = 2 * ilast + 1 - i;
                    dest(i, j, 0, dcomp + n) = -dest(i_sym, j, 0, dcomp + n);
                }
            }
        }

        // --------------------------------------------------------------------
        // Y-LOW Domain Boundary (j < domain.smallEnd(1))
        // --------------------------------------------------------------------
        if (j < domain.smallEnd(1)) {
            int jlast = domain.smallEnd(1);
            for (int n = 0; n < numcomp; ++n) {
                int bc = bcr[bcomp + n].lo(1);

                if (bc == amrex::BCType::foextrap) {
                    dest(i, j, 0, dcomp + n) = dest(i, jlast, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_even) {
                    int j_sym = 2 * jlast - 1 - j;
                    dest(i, j, 0, dcomp + n) = dest(i, j_sym, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_odd) {
                    int j_sym = 2 * jlast - 1 - j;
                    dest(i, j, 0, dcomp + n) = -dest(i, j_sym, 0, dcomp + n);
                }
            }
        }

        // --------------------------------------------------------------------
        // Y-HIGH Domain Boundary (j > domain.bigEnd(1))
        // --------------------------------------------------------------------
        if (j > domain.bigEnd(1)) {
            int jlast = domain.bigEnd(1);
            for (int n = 0; n < numcomp; ++n) {
                int bc = bcr[bcomp + n].hi(1);

                if (bc == amrex::BCType::foextrap) {
                    dest(i, j, 0, dcomp + n) = dest(i, jlast, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_even) {
                    int j_sym = 2 * jlast + 1 - j;
                    dest(i, j, 0, dcomp + n) = dest(i, j_sym, 0, dcomp + n);
                } 
                else if (bc == amrex::BCType::reflect_odd) {
                    int j_sym = 2 * jlast + 1 - j;
                    dest(i, j, 0, dcomp + n) = -dest(i, j_sym, 0, dcomp + n);
                }
            }
        }
    }
};
#endif
