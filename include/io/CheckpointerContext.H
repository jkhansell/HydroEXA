#ifndef CHECKPOINTER_CONTEXT_H
#define CHECKPOINTER_CONTEXT_H

#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Vector.H>

/**
 * \file CheckpointerContext.H
 * \brief What a checkpoint writer is allowed to touch.
 *
 * The checkpoint counterpart of ``SolverContext``, and the same idea: a
 * checkpoint is a bag of per-level data with a Header in front of it, so the
 * writer does not need the state class, only the fields.  Every member is a
 * reference to live state, so nothing is copied on the way in and a write is a
 * write of what is in memory.
 *
 * The const-ness is not decoration -- it encodes the direction of data flow:
 *
 * - ``U_new``, ``U_old``, ``Terrain``, ``dt``, ``t_new``, ``t_old`` and
 *   ``grids`` are non-const because **reading** a checkpoint fills them in;
 * - ``dmap`` and ``istep`` are const because they are properties of the run,
 *   not of the file: the rank assignment is decided when the levels are
 *   defined, and a restart reuses it rather than inventing a new one.
 */

/**
 * \brief A view of the state, hierarchy and clock that a checkpoint carries.
 *
 * \note Writing works today.  Reading is implemented in ``Checkpointer`` and
 * ``IOHandler`` but is not wired into ``AmrMeshState::Initialize``, so a run
 * with ``io.restart_chkfile`` set starts from scratch instead of resuming --
 * and ``dmap`` being const is part of why: a restart would have to rebuild the
 * hierarchy before it can read into it.
 */
struct CheckpointerContext
{
    // Evolving fields — non-const so Checkpointer::Read can write into them
    amrex::Vector<amrex::MultiFab>& U_new;      ///< State at the newer time level.
    amrex::Vector<amrex::MultiFab>& U_old;      ///< State at the older time level.
    amrex::Vector<amrex::MultiFab>& Terrain;    ///< Per-level terrain (bed and roughness).

    // AMR hierarchy — grids is non-const because Read populates it from file
    amrex::Vector<amrex::BoxArray>&     grids;  ///< Per-level box layout; read back from the Header.
    const amrex::Vector<amrex::DistributionMapping>& dmap;  ///< Rank assignment; fixed by the run.

    // Time state
    amrex::Vector<amrex::Real>& dt;     ///< Per-level timestep at the checkpoint.
    amrex::Vector<amrex::Real>& t_new;  ///< Per-level newer time.
    amrex::Vector<amrex::Real>& t_old;  ///< Per-level older time.

    // Metadata
    int finest_level;                   ///< Deepest level present in the file.
    int iteration;                      ///< Coarse-level step counter at the checkpoint.
    const amrex::Vector<int>& istep;    ///< Per-level step counters; written, not restored here.
};

#endif
