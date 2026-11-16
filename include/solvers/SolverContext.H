// include/solvers/SolverContext.H
#ifndef SOLVER_CONTEXT_H
#define SOLVER_CONTEXT_H


#include <memory>
#include <functional>

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>

/**
 * \file SolverContext.H
 * \brief Everything a solver may touch, handed over as one object.
 *
 * The contract between ``AmrMeshState`` and the solvers: what a flux
 * computation is allowed to assume, written as a signature.
 */

/**
 * \brief A view of the state, mesh and boundary data a solver may read.
 *
 * Before this existed, the solvers were members of the state class and read its
 * fields directly, which meant a solver could not be understood -- let alone
 * exercised -- without a live AMR hierarchy.  Now the dependency is explicit:
 * what appears here is the whole of what a flux computation may assume.
 *
 * It is a *view*, not an owner.  Scalars are copies, containers are
 * references, and the ``std::function`` members are lambdas that capture
 * ``AmrMeshState``.  A context must not outlive the state that produced it,
 * and holding one across a regrid is a bug even though it compiles.
 *
 * The ``std::function`` indirection is the price of removing the back-pointer:
 * ``FillPatch``, ``makeFineMask`` and ``refRatio`` are ``AmrCore`` member
 * functions, so a solver can only reach them through a callback.  One context
 * is built per level per step (``AmrMeshState::GetSolverContext``), which is
 * cheap enough that the callbacks need no caching.
 */
struct SolverContext
{
    //==========================================================================
    // Simulation state
    //==========================================================================

    amrex::Vector<amrex::MultiFab>& U_new;      ///< Destination state; swapped with ``U_old`` on entry.
    amrex::Vector<amrex::MultiFab>& U_old;      ///< Source state, after the swap.
    amrex::Vector<amrex::MultiFab>& Eta;        ///< Free surface ``h+z``; the quantity that crosses coarse-fine boundaries.
    amrex::Vector<amrex::MultiFab>& Terrain;    ///< Per-level terrain: bed elevation and roughness, NODATA marked -9999.

    amrex::Vector<amrex::Real> t_new;           ///< Time of ``U_new``, per level (a copy).
    amrex::Vector<amrex::Real> t_old;           ///< Time of ``U_old``, per level (a copy).

    //==========================================================================
    // Mesh hierarchy
    //==========================================================================

    // These are required for allocating temporary MultiFabs
    const amrex::Vector<amrex::Geometry>& geom;              ///< Per-level geometry (cell sizes, domain, periodicity).
    const amrex::Vector<amrex::BoxArray>& grids;             ///< Per-level box layout; nodal variants are derived from it.
    const amrex::Vector<amrex::DistributionMapping>& dmap;   ///< Per-level rank assignment; temporaries must reuse it.
    std::function<amrex::IntVect(int)> RefRatio;              ///< Refinement ratio of level ``n`` over ``n-1``.

    //==========================================================================
    // Boundary conditions
    //==========================================================================

    const amrex::Vector<amrex::BCRec>& UBCs;       ///< Per-component state BCs (h scalar, momenta normal/tangential).
    const amrex::Vector<amrex::BCRec>& TerrainBCs; ///< Terrain BCs; always transmissive, a bed has nothing to reflect.
    const amrex::Vector<amrex::BCRec>& EtaBCs;     ///< Free-surface BCs; scalar, so same mapping as depth.

    //==========================================================================
    // Refluxing
    //==========================================================================

    amrex::Vector<std::unique_ptr<amrex::FluxRegister>>& FluxRegisters; ///< ``FluxRegisters[lev]`` covers the ``lev-1``/``lev`` interface.
    bool do_reflux;                                                      ///< False means coarse-fine fluxes are not corrected.
    int finest_level;                                                    ///< Deepest level, so a solver knows where its upper register stops.

    //==========================================================================
    // Mesh services
    //==========================================================================

    std::function<void(
        int,
        amrex::Real,
        amrex::MultiFab&,
        const amrex::Vector<amrex::BCRec>&,
        int,
        int)> FillPatch;    ///< Fill a state MultiFab's ghosts, coarse data included.

    std::function<void(
        int,
        amrex::Real,
        amrex::MultiFab&,
        const amrex::Vector<amrex::BCRec>&,
        int,
        int)> FillPatchTerrain;    ///< Same, for terrain; uses piecewise-constant interpolation.

    std::function<void(
        int,
        amrex::Real,
        amrex::MultiFab&,
        const amrex::Vector<amrex::BCRec>&,
        int,
        int)> FillPatchEta;    ///< Same, for the free surface, which is what may cross a coarse-fine face.

    std::function<amrex::iMultiFab(
        const amrex::MultiFab&,
        const amrex::BoxArray&,
        const amrex::IntVect&,
        int,
        int)> makeFineMask;    ///< Mask of cells covered by a finer level; needed to place reflux corrections.

};

#endif
