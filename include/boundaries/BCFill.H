#ifndef BCFILL_H
#define BCFILL_H

#include <AMReX_BCRec.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include <utils/Params.H>

/**
 * \file BCFill.H
 * \brief Map input-file boundary codes to AMReX ``BCType`` codes, per variable.
 *
 * The inputs file describes a boundary by what it *is* -- a wall or an open
 * boundary -- not by what AMReX must do to a ghost cell.  This header is the
 * translation layer, and it exists because the translation is not one mapping
 * but three: the same wall reflects depth evenly, normal momentum oddly (so
 * the velocity reverses) and tangential momentum evenly (so it does not).
 *
 * Only the domain extents get a BC here.  Interior features are not boundaries
 * in this sense: a NODATA cell is invalid *data*, not a face, so it is not
 * given a ``BCType``.
 */

/**
 * \brief Boundary kinds the inputs file can ask for.
 *
 * The numeric values are the codes written in ``phys.bc_*`` and are what
 * ``ScalarBC``, ``NormalBC`` and ``TangentialBC`` interpret.
 */
enum class ExternalBC : int
{
    Reflective,     ///< Wall: no normal flux.
    Transmissive    ///< Open: zero-gradient outflow (``foextrap``).
};

/**
 * \brief ``BCType`` for a scalar (depth, eta) at a face.
 * \param bc Input-file BC code (0 reflective, 1 transmissive).
 * \return ``reflect_even`` or ``foextrap``.
 *
 * ``reflect_even`` mirrors the ghost value, which leaves depth unchanged
 * across a wall -- the wall neither adds nor removes water.
 */
inline int ScalarBC(int bc)
{
    return static_cast<ExternalBC>(bc) == ExternalBC::Reflective ? amrex::BCType::reflect_even : amrex::BCType::foextrap;
}

/**
 * \brief ``BCType`` for the momentum component normal to the face.
 * \param bc Input-file BC code (0 reflective, 1 transmissive).
 * \return ``reflect_odd`` or ``foextrap``.
 *
 * ``reflect_odd`` mirrors with a sign flip: a ghost cell with the opposite
 * normal velocity is what makes the reconstructed face velocity zero, i.e. a
 * wall.  Using ``reflect_even`` here instead is the classic way to get a
 * boundary that looks like a wall in the depth field and leaks momentum.
 */
inline int NormalBC(int bc)
{
    return static_cast<ExternalBC>(bc) == ExternalBC::Reflective ? amrex::BCType::reflect_odd : amrex::BCType::foextrap;
}

/**
 * \brief ``BCType`` for the momentum component tangential to the face.
 * \param bc Input-file BC code (0 reflective, 1 transmissive).
 * \return ``reflect_even`` or ``foextrap``.
 *
 * Tangential momentum is mirrored without a sign flip, so a wall is free-slip
 * (the opposite of a no-slip wall, which would be ``reflect_odd``).  Free-slip
 * is the only wall this solver offers: there is no bed-friction model to make a
 * no-slip wall meaningful, so the tangential ghost value stays as it was.
 */
inline int TangentialBC(int bc)
{
    return static_cast<ExternalBC>(bc) == ExternalBC::Reflective ? amrex::BCType::reflect_even : amrex::BCType::foextrap;
}

/**
 * \brief Fill the ``BCRec`` for a scalar variable from the run's BC codes.
 * \param bc        Destination record for one component.
 * \param bc_params Per-face codes from ``phys.bc_*``.
 */
inline void SetScalarBC(amrex::BCRec& bc, BCParams& bc_params)
{
    bc.setLo(0, ScalarBC(bc_params.lo_bc[0]));
    bc.setHi(0, ScalarBC(bc_params.hi_bc[0]));
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, ScalarBC(bc_params.lo_bc[1]));
    bc.setHi(1, ScalarBC(bc_params.hi_bc[1]));
#endif
}

/**
 * \brief Fill the ``BCRec`` for x-momentum.
 * \param bc        Destination record.
 * \param bc_params Per-face codes from ``phys.bc_*``.
 *
 * ``hu`` is normal to the x faces and tangential to the y faces, so the two
 * pairs of faces need different codes from the same input setting.
 */
inline void SetXVelocityBC(amrex::BCRec& bc, BCParams& bc_params)
{
    bc.setLo(0, NormalBC(bc_params.lo_bc[0]));
    bc.setHi(0, NormalBC(bc_params.hi_bc[0]));
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, TangentialBC(bc_params.lo_bc[1]));
    bc.setHi(1, TangentialBC(bc_params.hi_bc[1]));
#endif
}

/**
 * \brief Fill the ``BCRec`` for y-momentum.
 * \param bc        Destination record.
 * \param bc_params Per-face codes from ``phys.bc_*``.
 *
 * The mirror image of ``SetXVelocityBC``: tangential in x, normal in y.
 */
inline void SetYVelocityBC(amrex::BCRec& bc, BCParams& bc_params)
{
    bc.setLo(0, TangentialBC(bc_params.lo_bc[0]));
    bc.setHi(0, TangentialBC(bc_params.hi_bc[0]));
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, NormalBC(bc_params.lo_bc[1]));
    bc.setHi(1, NormalBC(bc_params.hi_bc[1]));
#endif
}

/**
 * \brief Do nothing, on purpose: the ``BCRec`` already says what to do.
 *
 * AMReX's ``FillDomainBoundary`` fills ``foextrap``, ``hoextrap``,
 * ``reflect_even`` and ``reflect_odd`` ghost cells itself from the ``BCRec``
 * codes; a boundary functor is only consulted for ``ext_dir``, i.e. for a
 * Dirichlet value that no code can express.  This solver never uses
 * ``ext_dir``, so the operator is an empty body that satisfies the interface.
 *
 * That is worth stating explicitly, because an empty function reads like an
 * unfinished one.  It is not: replacing it with a hand-written reflection
 * loop would duplicate ``FillDomainBoundary`` and would have to get the
 * staggered-index cases right again.  The real BC logic is in
 * ``ScalarBC``/``NormalBC``/``TangentialBC`` above.
 *
 * \param iv    Ghost-cell index to fill.
 * \param a     Destination array.
 * \param dcomp First destination component.
 * \param ncomp Number of components.
 * \param geom  Domain geometry.
 * \param bcr   Per-component BC record; unused, because the ``BCType`` codes
 *              in it are what AMReX acts on.
 * \param bcomp First component in \p bcr.
 */
struct ExternalBCFill
{
    AMREX_GPU_DEVICE
    void operator()(const amrex::IntVect& iv,
                    amrex::Array4<amrex::Real> const& a,
                    int dcomp, int ncomp,
                    amrex::GeometryData const& geom,
                    amrex::Real,
                    const amrex::BCRec* bcr,
                    int bcomp, int) const
    {

    }
};


#endif
