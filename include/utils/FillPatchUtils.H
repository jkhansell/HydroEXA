#ifndef FILLPATCHUTILS_H_
#define FILLPATCHUTILS_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>

#include <boundaries/BCFill.H>


namespace amrex {

/**
 * Fill a fine-level MultiFab by interpolating from a coarse-level MultiFab.
 *
 * Generic utility that decouples the data source from the interpolation logic.
 * The caller provides the coarse MultiFab directly, making this reusable
 * across different state classes (AmrMeshState, terrain state, custom fields, etc.).
 *
 * Boundary conditions are applied via ExternalBCFill (reflective/transmissive).
 *
 * \tparam MF          MultiFab/FabArray type
 * \tparam Interp      Spatial interpolator function type (defaults to cell_cons_interp)
 *
 * \param mf       Destination fine-level MultiFab (all ghost cells filled)
 * \param time     Time associated with the fine level
 * \param cmf      Source coarse-level MultiFab
 * \param scomp    Starting component in the coarse MultiFab
 * \param dcomp    Starting component in the fine MultiFab
 * \param ncomp    Number of components to interpolate
 * \param cgeom    Geometry of the coarse level
 * \param fgeom    Geometry of the fine level
 * \param bcs      Boundary conditions (per-component BCRec)
 * \param ratio    Refinement ratio (IntVect)
 * \param mapper   Interpolation function (default: cell_cons_interp)
 */
template <typename MF, typename Interp = decltype(&cell_cons_interp)>
std::enable_if_t<IsFabArray<MF>::value>
FillCoarsePatchFrom (MF& mf, Real time,
                     const MF& cmf, int scomp, int dcomp, int ncomp,
                     const Geometry& cgeom, const Geometry& fgeom,
                     const Vector<BCRec>& bcs,
                     const IntVect& ratio,
                     Interp mapper = &cell_cons_interp)
{
    GpuBndryFuncFab<ExternalBCFill> bndry_func(ExternalBCFill{});
    using BndryPhysBC = PhysBCFunct<GpuBndryFuncFab<ExternalBCFill>>;
    BndryPhysBC cphysbc(cgeom, bcs, bndry_func);
    BndryPhysBC fphysbc(fgeom, bcs, bndry_func);

    InterpFromCoarseLevel(mf, time,
                          cmf, scomp, dcomp, ncomp,
                          cgeom, fgeom,
                          cphysbc, 0, fphysbc, 0,
                          ratio,
                          mapper,
                          bcs, 0);
}

/**
 * Fill a MultiFab from neighboring AMR levels (single-level at lev=0,
 * two-level otherwise).
 *
 * Generic utility that decouples the data source from the patching logic.
 * The caller provides coarse and fine MultiFab arrays; passing an empty
 * coarse array selects single-level patching (lev=0 equivalent).
 *
 * Boundary conditions are applied via ExternalBCFill (reflective/transmissive).
 *
 * \tparam MF          MultiFab/FabArray type
 * \tparam Interp      Spatial interpolator function type (defaults to cell_cons_interp)
 *
 * \param mf       Destination MultiFab
 * \param time     Time associated with the destination
 * \param cmf      Source coarse-level MultiFab array (empty => single-level)
 * \param ctime    Times for coarse data
 * \param fmf      Source fine-level MultiFab array
 * \param ftime    Times for fine data
 * \param scomp    Starting component in source MultiFabs
 * \param dcomp    Starting component in the destination
 * \param ncomp    Number of components to interpolate
 * \param cgeom    Geometry of the coarse level (ignored for single-level)
 * \param fgeom    Geometry of the fine level
 * \param bcs      Boundary conditions (per-component BCRec; ignored for single-level)
 * \param ratio    Refinement ratio (ignored for single-level)
 * \param mapper   Interpolation function (default: cell_cons_interp)
 */
template <typename MF, typename Interp = decltype(&cell_cons_interp)>
std::enable_if_t<IsFabArray<MF>::value>
FillPatchFrom (MF& mf, Real time,
               const Vector<MF*>& cmf, const Vector<Real>& ctime,
               const Vector<MF*>& fmf, const Vector<Real>& ftime,
               int scomp, int dcomp, int ncomp,
               const Geometry& cgeom, const Geometry& fgeom,
               const Vector<BCRec>& bcs,
               const IntVect& ratio,
               Interp mapper = &cell_cons_interp)
{
    if (cmf.empty()) {
        // Single-level patching (lev == 0)
        GpuBndryFuncFab<ExternalBCFill> bndry_func(ExternalBCFill{});
        using BndryPhysBC = PhysBCFunct<GpuBndryFuncFab<ExternalBCFill>>;
        BndryPhysBC physbc(fgeom, bcs, bndry_func);
        FillPatchSingleLevel(mf, time, fmf, ftime, scomp, dcomp, ncomp,
                             fgeom, physbc, 0);
    } else {
        // Two-level patching (lev > 0)
        GpuBndryFuncFab<ExternalBCFill> bndry_func(ExternalBCFill{});
        using BndryPhysBC = PhysBCFunct<GpuBndryFuncFab<ExternalBCFill>>;
        BndryPhysBC cphysbc(cgeom, bcs, bndry_func);
        BndryPhysBC fphysbc(fgeom, bcs, bndry_func);
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
