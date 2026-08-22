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
        const int /*bcomp*/,
        const int /*orig_comp*/) const
    {
        const amrex::Box domain = geom.Domain();

        for (int n = 0; n < numcomp; ++n)
        {
            const int comp = dcomp + n;

            // X-low
            if (iv[0] < domain.smallEnd(0) &&
                bcr->lo(0) == amrex::BCType::ext_dir)
            {
                amrex::IntVect src = iv;
                src[0] = domain.smallEnd(0);
                dest(iv, comp) = dest(src, comp);
            }

            // X-high
            else if (iv[0] > domain.bigEnd(0) &&
                     bcr->hi(0) == amrex::BCType::ext_dir)
            {
                amrex::IntVect src = iv;
                src[0] = domain.bigEnd(0);
                dest(iv, comp) = dest(src, comp);
            }

#if (AMREX_SPACEDIM >= 2)

            // Y-low
            if (iv[1] < domain.smallEnd(1) &&
                bcr->lo(1) == amrex::BCType::ext_dir)
            {
                amrex::IntVect src = iv;
                src[1] = domain.smallEnd(1);
                dest(iv, comp) = dest(src, comp);
            }

            // Y-high
            else if (iv[1] > domain.bigEnd(1) &&
                     bcr->hi(1) == amrex::BCType::ext_dir)
            {
                amrex::IntVect src = iv;
                src[1] = domain.bigEnd(1);
                dest(iv, comp) = dest(src, comp);
            }

#endif
        }
    }
};
#endif
