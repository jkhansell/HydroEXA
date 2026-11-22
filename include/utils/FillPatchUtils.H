#ifndef FILLPATCHUTILS_H_
#define FILLPATCHUTILS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BCRec.H>
#include <AMReX_FillPatchUtil.H>

#include <boundaries/BCFill.H>

/**
 * \file FillPatchUtils.H
 * \brief FillPatch wrappers that take the data source and cached PhysBCs as arguments.
 *
 * AMReX's own ``FillPatchSingleLevel`` / ``FillPatchTwoLevels`` are member
 * functions of ``AmrLevel``, which ties them to one state class.  The solver
 * keeps several independent state groups (conserved variables, terrain, eta)
 * with the same filling rules and different sources, so the wrappers here take
 * the coarse and fine MultiFabs directly.
 *
 * Pre-constructed ``PhysBC`` objects (such as ``HydroPhysBC``) are passed by
 * reference from the level state rather than built per call, avoiding repeated
 * device allocations, host-to-device ``BCRec`` transfers, and empty ``ext_dir``
 * kernel launches.
 */

namespace amrex {

/**
 * \brief Fill a fine-level MultiFab by interpolating from a coarse-level one.
 *
 * Used when a level is created or regridded: the new level's valid cells come
 * from the coarse data, and its ghost cells get coarse data plus physical BCs.
 * Refinement is conservative by default, so a coarse-to-fine fill does not
 * change the amount of water in the fine cells it covers.
 *
 * \tparam MF     MultiFab / FabArray type.
 * \tparam PhysBC Physical boundary condition functor type (e.g. ``HydroPhysBC``).
 * \tparam Interp Interpolator function type; defaults to
 *                ``decltype(&cell_cons_interp)``.
 *
 * \param mf      Destination fine-level MultiFab; valid and ghost cells are
 *                written.
 * \param time    Time to associate with the filled fine data.
 * \param cmf     Source coarse-level MultiFab.
 * \param scomp   First component to read from \p cmf.
 * \param dcomp   First component to write in \p mf.
 * \param ncomp   Number of components to interpolate.
 * \param cgeom   Geometry of the coarse level.
 * \param fgeom   Geometry of the fine level (defines the destination index space).
 * \param cphysbc Cached physical boundary condition functor for the coarse level.
 * \param fphysbc Cached physical boundary condition functor for the fine level.
 * \param bcs     Per-component physical boundary conditions.
 * \param ratio   Refinement ratio between \p cgeom and \p fgeom.
 * \param mapper  Interpolation operator; ``cell_cons_interp`` (conservative
 *                piecewise-constant) for state data, ``pc_interp`` (piecewise
 *                constant) for terrain, where conservation is meaningless.
 */
template <typename MF, typename PhysBC, typename Interp = decltype(&cell_cons_interp)>
std::enable_if_t<IsFabArray<MF>::value>
FillCoarsePatchFrom (MF& mf, Real time,
                     const MF& cmf, int scomp, int dcomp, int ncomp,
                     const Geometry& cgeom, const Geometry& fgeom,
                     PhysBC& cphysbc, PhysBC& fphysbc,
                     const Vector<BCRec>& bcs,
                     const IntVect& ratio,
                     Interp mapper = &cell_cons_interp)
{
    InterpFromCoarseLevel(mf, time,
                          cmf, scomp, dcomp, ncomp,
                          cgeom, fgeom,
                          cphysbc, 0, fphysbc, 0,
                          ratio,
                          mapper,
                          bcs, 0);
}

/**
 * \brief Fill a MultiFab from its own level, plus the coarser level if any.
 *
 * The one entry point for ghost-cell filling during time advance.  Which
 * AMReX routine runs depends only on whether coarse data is supplied:
 *
 * - \p cmf empty  -> ``FillPatchSingleLevel``: level 0, or any level with no
 *   valid coarse data to consult;
 * - \p cmf present -> ``FillPatchTwoLevels``: interpolate in time between the
 *   coarse states, refine, then overwrite the covered region with the fine
 *   state so the fine data always wins inside its own domain.
 *
 * \tparam MF     MultiFab / FabArray type.
 * \tparam PhysBC Physical boundary condition functor type (e.g. ``HydroPhysBC``).
 * \tparam Interp Interpolator function type; defaults to
 *                ``decltype(&cell_cons_interp)``.
 *
 * \param mf      Destination MultiFab.
 * \param time    Time at which the destination state is wanted; coarse data is
 *                interpolated to it.
 * \param cmf     Coarse-level MultiFabs (usually one per time level).  Empty
 *                selects single-level filling.
 * \param ctime   Times matching \p cmf; must be non-decreasing.
 * \param fmf     Fine-level MultiFabs to copy from; at least one.
 * \param ftime   Times matching \p fmf.
 * \param scomp   First source component.
 * \param dcomp   First destination component.
 * \param ncomp   Number of components to fill.
 * \param cgeom   Coarse geometry; ignored when \p cmf is empty.
 * \param fgeom   Destination-level geometry.
 * \param cphysbc Cached coarse-level physical BC functor; ignored when \p cmf is empty.
 * \param fphysbc Cached destination-level physical BC functor.
 * \param bcs     Per-component physical boundary conditions, applied by
 *                whichever AMReX routine runs.
 * \param ratio   Refinement ratio; ignored when \p cmf is empty.
 * \param mapper  Interpolation operator; see ``FillCoarsePatchFrom``.
 */
template <typename MF, typename PhysBC, typename Interp = decltype(&cell_cons_interp)>
std::enable_if_t<IsFabArray<MF>::value>
FillPatchFrom (MF& mf, Real time,
               const Vector<MF*>& cmf, const Vector<Real>& ctime,
               const Vector<MF*>& fmf, const Vector<Real>& ftime,
               int scomp, int dcomp, int ncomp,
               const Geometry& cgeom, const Geometry& fgeom,
               PhysBC& cphysbc, PhysBC& fphysbc,
               const Vector<BCRec>& bcs,
               const IntVect& ratio,
               Interp mapper = &cell_cons_interp)
{
    if (cmf.empty()) {
        FillPatchSingleLevel(mf, time, fmf, ftime, scomp, dcomp, ncomp,
                             fgeom, fphysbc, 0);
    } else {
        FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                           scomp, dcomp, ncomp,
                           cgeom, fgeom,
                           cphysbc, 0, fphysbc, 0,
                           ratio,
                           mapper,
                           bcs, 0);
    }
}

} // namespace amrex

#endif // FILLPATCHUTILS_H_