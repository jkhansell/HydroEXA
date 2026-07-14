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

class HydroEXA
{
    public:
        HydroEXA();
        ~HydroEXA() = default;

        void ReadParameters();
        void Initialize();
        void Compute();
        void Finalize();

        bool Verbose() const { return verbose; }

    private:
        // Core Components
        SolverVariant Model;
        std::shared_ptr<IOHandler> IO;              // Manages all I/O operations
        std::unique_ptr<AmrMeshState> MeshState;    // Encapsulates the AMR mesh, variables, and flux registers

        bool verbose{true};

        // Grouped Parameter Contexts
        RuntimeParameters runtime_params;
        IOParameters      io_params;
        AMRParameters     amr_params;
        PhysicsParameters physics_params;
};
#endif