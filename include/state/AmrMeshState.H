// include/core/AmrMeshState.H
#ifndef AMRMESHSTATE_H_
#define AMRMESHSTATE_H_
#include <memory>
#include <vector>
#include <algorithm>

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>

#include <io/IOHandler.H>
#include <io/CheckpointerContext.H>

#include <utils/Params.H>

#include <solvers/SolverVariant.H>
#include <solvers/SolverContext.H>
#include <boundaries/BCFill.H>


class AmrMeshState : public amrex::AmrCore {
    public:

        // Constructor
        AmrMeshState() : amrex::AmrCore() {}
        
        AmrMeshState(std::shared_ptr<IOHandler> io_handler,
                     int terrain_ref_lev, 
                     HDF5SpatialMetadata& meta,
                     std::string input,
                     RuntimeParameters runtime_params,
                     IOParameters io_params,
                     AMRParameters amr_params,
                     PhysicsParameters physics_params,
                     /*AmrCore params*/
                     const amrex::RealBox& rb, 
                     int max_level_in,
                     const amrex::Vector<int>& n_cell_in, 
                     int coord,
                     const amrex::Vector<amrex::IntVect>& ref_ratios);

        void Initialize();

        /*void Initialize(const int ncomp_U = 4, const int ncomp_Terrain = 1) {
            this->ncomp_U = ncomp_U;
            this->ncomp_Terrain = ncomp_Terrain;
        }*/

        void InitializeTerrainFluid();
        void InitializeSolver();

        void Regrid(int lbase, amrex::Real time);

            
        // Destructor
        virtual ~AmrMeshState() = default;

        // Standard 5 AmrCore overrides
        virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        virtual void ClearLevel(int lev) override;
        virtual void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

        int get_ncomp_U() const { return ncomp_U; }
        int get_ncomp_Terrain() const { return ncomp_Terrain; }

        bool is_flux_reg_allocated(int lev) const { return flux_reg[lev] != nullptr; }
        amrex::FluxRegister* get_flux_reg(int lev) { return flux_reg[lev].get(); }

        SolverContext GetSolverContext();
        CheckpointerContext GetCheckpointerContext();

        // Time stepping
        void ComputeDt();
        void TimeStepWithSubcycling(int lev, amrex::Real time, int iteration);
        void AdvanceLevel(int lev, amrex::Real time, amrex::Real dt);

        // Utils
        void ResizeLevels(int nlevs);
        void WritePlotfile(int iteration, amrex::Real time);
        void WriteCheckpoint();

        virtual void PostProcessBaseGrids(amrex::BoxArray& box_array) const override;


        // Accessible to HydroEXA top class
        amrex::Vector<amrex::Real> t_new;
        amrex::Vector<amrex::Real> t_old;
        amrex::Vector<amrex::Real> dt;
        amrex::Vector<int> istep;
        amrex::Vector<int> nsubsteps;
        
    private:
        // Internal helper functions for level management
        void FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
        void FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
    
        void AverageDown (amrex::Vector<amrex::MultiFab>& arr);
        void AverageDownTo(int crse_lev, amrex::Vector<amrex::MultiFab>& arr);
        
        void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);
        void GetTerrainData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);
        
        void TerrainMapStaticToDynamic(int lev, amrex::MultiFab& dynTerrain);
        void FluidMapStaticToDynamic(int lev);


        // Manages all I/O operations (checkpointing, plotfiles, restarts)
        std::shared_ptr<IOHandler> IO;
        HDF5SpatialMetadata metadata;

        int ncomp_U;
        int ngrow_U;

        int ncomp_Terrain;
        int ngrow_Terrain;


        // --- Dynamic Simulation State ---
        // These are the core state variables that evolve in time and are stored at each AMR level. They are resized and redefined as levels are created or destroyed.
        amrex::Vector<amrex::MultiFab> U_new;
        amrex::Vector<amrex::MultiFab> U_old;
        amrex::Vector<amrex::MultiFab> DynamicTerrain;
        
        mutable std::vector<int> pure_fluid_boxes;
        mutable std::vector<int> boundary_boxes;
        
        // Boundary conditions
        amrex::BCRec phys_bc;                     // Physical BC types (0-5: Interior/Inflow/Outflow/Symmetry/SlipWall/NoSlipWall)
        amrex::BCRec terrain_bc;                  // Physical BC types (0-5: Interior/Inflow/Outflow/Symmetry/SlipWall/NoSlipWall)
        amrex::Vector<amrex::BCRec> U_bcs;        // Per-component mathematical BCRecs
        amrex::Vector<amrex::BCRec> Terrain_bcs;  // Per-component mathematical BCRecs

        // --- Static Database State ---
        // Ground Truth Layer: Split across MPI ranks uniformly at ith Level resolution.
        // These three members define the permanent, unchanging database layout.
        amrex::BoxArray             static_ba;
        amrex::DistributionMapping  static_dm;
        amrex::Geometry             static_geom;   // <--- NEW: Dedicated geometry for the static layer
        amrex::MultiFab             StaticTerrain;
        int static_terrain_lev;
        
        std::unique_ptr<amrex::MultiFab> StaticFluid;

        // Flux registers for coarse-fine synchronization
        amrex::Vector<std::unique_ptr<amrex::FluxRegister>> flux_reg;

        std::string hdf5_path;
        std::string restart_chkfile;

        RuntimeParameters runtime_p;
        IOParameters      io_p;
        AMRParameters     amr_p;
        PhysicsParameters physics_p;

        SolverVariant solver; 
};

#endif