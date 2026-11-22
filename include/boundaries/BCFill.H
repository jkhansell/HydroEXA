#ifndef BCFILL_H
#define BCFILL_H

#include <AMReX_BCRec.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FilCC_C.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>

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

namespace hydro_bc
{

/**
 * \brief Fill the ghosts of one FAB that lie outside \p gdomain.
 *
 * ``amrex::FilccCell`` applies the ``foextrap`` / ``hoextrap`` /
 * ``reflect_even`` / ``reflect_odd`` rules to a single cell and reads only
 * cells that are already valid *inside* the domain in every direction it does
 * not itself extrapolate along.  A corner cell therefore has to be filled
 * after the faces: it reads its neighbour on the face, which is itself a ghost
 * cell.  Splitting the launch by the number of directions a cell lies outside
 * in reproduces that ordering without building AMReX's face/edge/corner box
 * lists, and each launch is still one kernel.
 *
 * \param bx      Grown valid box of the FAB; the whole box is scanned, and
 *                cells inside \p gdomain return immediately.
 * \param fab     Destination array.
 * \param gdomain Domain grown by \c nghost in the periodic directions, i.e.
 *                the region that is already filled by ``FillBoundary``.
 * \param d_bc    Device copy of the per-component ``BCRec`` set, already
 *                offset by \c bccomp.
 * \param dcomp   First destination component.
 * \param ncomp   Number of components to fill.
 */
inline void fill_fab_ghosts (amrex::Box const& bx,
                             amrex::Array4<amrex::Real> const& fab,
                             amrex::Box const& gdomain,
                             amrex::BCRec const* d_bc,
                             int dcomp, int ncomp)
{
    for (int nout = 1; nout <= AMREX_SPACEDIM; ++nout)
    {
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            int outside = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (iv[d] < gdomain.smallEnd(d) || iv[d] > gdomain.bigEnd(d)) {
                    ++outside;
                }
            }
            if (outside == nout) {
                amrex::FilccCell{}(iv, fab, dcomp, ncomp, gdomain, d_bc, 0);
            }
        });
    }
}

} // namespace hydro_bc

/**
 * \brief Cached, per-level physical boundary filler.
 *
 * One instance per AMR level and per variable group (conserved state, free
 * surface, terrain), built when the level is created and reused for every
 * ``FillPatch`` afterwards.  It plays the role of AMReX's
 * ``PhysBCFunct<GpuBndryFuncFab<...>>``, which ``FillPatchSingleLevel`` /
 * ``FillPatchTwoLevels`` / ``InterpFromCoarseLevel`` call as
 * ``bc(mf, dcomp, ncomp, nghost, time, bccomp)``.
 *
 * Reimplementing that one entry point is what makes the caching worthwhile.
 * The AMReX functor is only a shell around a per-FAB device call, and that
 * call copies the ``BCRec`` set to the GPU through an ``AsyncArray`` *for
 * every FAB of every fill*, then launches a kernel whose user hook exists
 * solely to fill ``ext_dir`` cells.  This solver never uses ``ext_dir`` -- the
 * whole physical BC is the ``BCRec`` codes produced by ``ScalarBC`` /
 * ``NormalBC`` / ``TangentialBC`` -- so both costs are pure overhead:
 *
 * - the ``BCRec`` set is copied to the device **once**, in the constructor;
 * - the fill is ``amrex::FilccCell`` alone, with no ``ext_dir`` hook;
 * - a fill allocates nothing and touches no host-to-device path.
 *
 * Only the domain extent is handled here.  Interior features are not
 * boundaries: a NODATA cell is invalid *data*, not a face, so it gets no
 * ``BCType`` (see the file comment).
 */
class HydroPhysBC
{
public:
    HydroPhysBC() = default;

    /**
     * \brief Build the cached filler for one level.
     * \param geom Geometry of the level the fill applies to.
     * \param bcs  Per-component ``BCRec`` set for this variable group.
     */
    HydroPhysBC(const amrex::Geometry& geom, const amrex::Vector<amrex::BCRec>& bcs)
        : m_geom(geom)
    {
        copy_bcs(bcs);
    }

    /// Rebuild in place; used when a level is remade at a new resolution.
    void define(const amrex::Geometry& geom, const amrex::Vector<amrex::BCRec>& bcs)
    {
        m_geom = geom;
        copy_bcs(bcs);
    }

    /**
     * \brief Fill the physical-boundary ghosts of \p mf.
     *
     * Signature dictated by ``amrex::FillPatchSingleLevel`` and friends.
     *
     * \param mf     Destination MultiFab; only its own grown boxes are read.
     * \param dcomp  First destination component.
     * \param ncomp  Number of components to fill.
     * \param nghost Ghost width to fill; must not exceed \p mf's own ghosts.
     * \param time   Ignored: the BCs here are all stationary.
     * \param bccomp First entry of the cached ``BCRec`` set to use.
     */
    void operator()(amrex::MultiFab& mf, int dcomp, int ncomp,
                    const amrex::IntVect& nghost, amrex::Real time, int bccomp)
    {
        if (m_d_bcs.empty() || m_geom.isAllPeriodic()) { return; }

        BL_PROFILE("HydroPhysBC::()");

        AMREX_ASSERT(nghost.allLE(mf.nGrowVect()));

        // No early return for nghost == 0: InterpFromCoarseLevel builds its
        // coarse patch with no ghosts and leaves the cells that stick out of
        // the domain as NaN, expecting exactly this call to fill them.
        const amrex::Box& domain = amrex::convert(m_geom.Domain(), mf.boxArray().ixType());
        amrex::Box gdomain = domain;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (m_geom.isPeriodic(d)) { gdomain.grow(d, nghost[d]); }
        }

        amrex::BCRec const* d_bc = m_d_bcs.data() + bccomp;

        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = amrex::grow(mfi.validbox(), nghost);
            if (gdomain.contains(bx)) { continue; }

            amrex::FArrayBox& dest = mf[mfi];
            hydro_bc::fill_fab_ghosts(bx, dest.array(), gdomain, d_bc, dcomp, ncomp);
        }
    }

    /// Whether a geometry and a ``BCRec`` set have been installed.
    [[nodiscard]] bool is_defined() const noexcept { return !m_d_bcs.empty(); }

private:
    /**
     * \brief Upload \p bcs to the cached device copy.
     *
     * Must be a memcpy, not ``PODVector::assign``: with an arena allocator
     * ``assign`` launches a device kernel doing ``dst[i] = src[i]`` and reads
     * \p bcs through a raw host pointer, which is an illegal access.  This runs
     * once per level, so a blocking copy is the right price.
     */
    void copy_bcs (const amrex::Vector<amrex::BCRec>& bcs)
    {
        m_d_bcs.resize(bcs.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, bcs.begin(), bcs.end(), m_d_bcs.begin());
    }

    amrex::Geometry m_geom;                        ///< Level this filler belongs to.
    amrex::Gpu::DeviceVector<amrex::BCRec> m_d_bcs; ///< Device copy, written once.
};

#endif