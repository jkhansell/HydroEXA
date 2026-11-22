#ifndef PARAMS_H
#define PARAMS_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

/**
 * \file Params.H
 * \brief Parameter groups parsed from the inputs file.
 *
 * Each struct mirrors one input-file namespace (``HydroEXA.*``, ``io.*``,
 * ``amr.*``, ``phys.*``), so a reader looking for "where does this keyword
 * land?" has one place to look.  Every field carries its default value as a
 * brace initialiser: an input file that sets nothing produces these numbers,
 * and that is what the test cases compare against.
 *
 * Time-valued frequencies are in *physical* time, not steps; a negative
 * frequency disables that output.
 */

// ============================================================================
// Parameter Groups
// ============================================================================

/**
 * \brief Run-level bookkeeping: where the clock is and when to stop.
 *
 * Parsed from ``HydroEXA.*``.
 */
struct RuntimeParameters
{
    amrex::Real etime{0.0};           ///< Current physical simulation time, s.
    amrex::Real max_time{1.0};        ///< Stop time, physical seconds (``HydroEXA.max_time``).
    amrex::Real mass_diag_freq{-1.0}; ///< Mass-diagnostic interval, s; negative disables.
    int iteration{0};                 ///< Coarse-level step counter.
    int log_progress_every{10};       ///< Print progress every N coarse steps (``HydroEXA.log_progress_every``).
    amrex::Real wall_elapsed{0.0};    ///< Cumulative wall time elapsed so far.
};

/**
 * \brief Output cadence, file names and formats.
 *
 * Parsed from ``io.*``.  ``plot_freq = 0`` suppresses plotfiles entirely,
 * which is what the scoring runs use when only the final state matters.
 */
struct IOParameters
{
    amrex::Real io_freq{1.0};                 ///< Generic I/O interval, s.
    amrex::Real chk_freq{-1.0};               ///< Checkpoint interval, s; negative disables.
    amrex::Real plot_freq{-1.0};              ///< Plotfile interval, s; negative disables.
    bool async_io{false};                     ///< Write from a separate thread while advancing.
    std::string chk_type{"native"};           ///< Checkpoint format: ``native`` or ``hdf5``.
    std::string plot_file{"plt"};             ///< Plotfile prefix.
    std::string plot_type{"native"};          ///< Plotfile format: ``native`` or ``hdf5``.
    std::string chk_file{"chk"};              ///< Checkpoint prefix.
    std::string input_file{"input.h5"};       ///< HDF5 file holding terrain and initial fields.
    std::string restart_chkfile{"restart"};   ///< Checkpoint to restart from; empty means cold start.
};

/**
 * \brief AMR configuration.
 *
 * Parsed from ``amr.*``.  See ``docs/PIPELINE.md`` §8a for the interaction
 * between ``regrid_int`` and ``n_error_buf`` that decides whether a moving
 * front is followed fast enough to stay accurate.
 */
struct AMRParameters
{
    int regrid_int{2};            ///< Regrid interval, in coarse-level steps.
    int do_reflux{1};             ///< Apply flux registers at coarse-fine interfaces.
    int do_subcycle{1};           ///< Subcycle in time; 0 advances every level with the coarse dt.
    int terrain_ref_lev{0};       ///< Level at which terrain is read (and the resolution it implies).
    int max_amr_level{0};         ///< Deepest AMR level requested; -1 means "whatever the inputs imply".
};

/**
 * \brief Physical boundary-condition types, one entry per face.
 *
 * Indices are HydroEXA's own codes, mapped to AMReX ``BCType`` in
 * ``boundaries/BCFill.H``; they are deliberately not the AMReX enum so that
 * the inputs file stays readable.
 *
 * 0 = reflective wall, 1 = transmissive (zero-gradient outflow).
 */
struct BCParams
{
    int lo_bc[AMREX_SPACEDIM] = {0, 0};   ///< Lower-face BC codes; default reflective.
    int hi_bc[AMREX_SPACEDIM] = {0, 0};   ///< Upper-face BC codes; default reflective.
};

/**
 * \brief Physics and solver configuration.
 *
 * Parsed from ``phys.*`` (and ``HydroEXA.model``).  The three gradient
 * thresholds are per-level vectors: refinement is driven by whichever of
 * depth, velocity or bed slope changes fastest in that level's cells.
 *
 * ``cfl`` is the only parameter that can make a run unstable on its own; the
 * timestep collapses and aborts rather than silently reducing it.
 */
struct PhysicsParameters
{
    amrex::Real cfl{0.5};                         ///< CFL number for the timestep (``HydroEXA.cfl``).
    std::string model{"Roe"};                     ///< Solver: ``Roe`` or ``HydrostaticRoe``.
    amrex::Vector<amrex::Real> h_grad_thresh;     ///< Per-level depth-gradient tagging threshold.
    amrex::Vector<amrex::Real> umag_grad_thresh;  ///< Per-level velocity-gradient tagging threshold.
    amrex::Vector<amrex::Real> z_grad_thresh;     ///< Per-level bed-slope tagging threshold.
    BCParams bc_params;                           ///< Physical boundary-condition codes.
};


#endif
