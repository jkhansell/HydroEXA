#ifndef HydroEXA_H_
#define HydroEXA_H_

#include <variant>
#include <memory>


#include <AMReX.H>

#include <state/AmrMeshState.H>
#include <solvers/SolverVariant.H>
#include <io/IOHandler.H>       
#include <utils/Params.H>

// ============================================================================
// Core HydroEXA Driver Class
// ============================================================================

/**
 * \file HydroEXA.H
 * \brief The driver: parameters, one AMR state, one time loop.
 *
 * Four calls, in a fixed order, and nothing else.  The driver owns no physics:
 * it parses the inputs, builds the state object, runs the loop and closes the
 * input file.  Every decision about how to advance lives in ``AmrMeshState``
 * and the solvers.
 *
 * The order matters because each phase depends on the previous one having
 * finished: the parameters are needed to describe the mesh, the mesh is needed
 * to allocate the levels, and the levels are needed before a timestep can be
 * computed.
 */

/**
 * \brief Owns the run configuration, the I/O handler and the AMR state.
 *
 * Instantiated by ``main`` between ``amrex::Initialize`` and
 * ``amrex::Finalize``; the parameters are read from the ``ParmParse`` database
 * that ``amrex::Initialize`` has already populated from the command line and
 * the inputs file.
 */
class HydroEXA
{
    public:
        /// Construct an empty driver; nothing is read or allocated yet.
        HydroEXA();
        /// Releases the state object and, with it, every level's data.
        ~HydroEXA() = default;

        /// Parse every parameter group out of ``ParmParse`` and open the input file.
        void ReadParameters();
        /// Build the AMR state, the terrain database and the solver.
        void Initialize();
        /// Run the subcycled time loop until ``max_time``, writing output as due.
        void Compute();
        /// Close the input file and release the I/O handler.
        void Finalize();

    private:
        // Core Components
        std::shared_ptr<IOHandler> IO;              // Manages all I/O operations
        std::unique_ptr<AmrMeshState> MeshState;    // Encapsulates the AMR mesh, variables, and flux registers

        // Grouped Parameter Contexts
        RuntimeParameters runtime_params;   ///< Clock and stop time.
        IOParameters      io_params;        ///< Output cadence, prefixes, formats.
        AMRParameters     amr_params;       ///< Regridding, refluxing, subcycling.
        PhysicsParameters physics_params;   ///< CFL, model, tagging thresholds, BCs.
};
#endif
