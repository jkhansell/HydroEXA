// include/core/AmrMeshState.H
#ifndef AMRMESHSTATE_H_
#define AMRMESHSTATE_H_
#include <memory>
#include <vector>

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>

#include <io/IOHandler.H>
#include <utils/Params.H>


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
        void Regrid(int lbase, amrex::Real time);

            
        // Destructor
        virtual ~AmrMeshState() = default;

        // Standard 5 AmrCore overrides
        virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        virtual void ClearLevel(int lev) override;
        virtual void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

        // Public Accessors 
        amrex::MultiFab& get_U_new(int lev)   { return U_new[lev]; }
        amrex::MultiFab& get_terrain(int lev) { return DynamicTerrain[lev]; }
        amrex::MultiFab& get_static_terrain() { return StaticTerrain; }

        const amrex::MultiFab& get_U_new(int lev)   const { return U_new[lev]; }
        const amrex::MultiFab& get_terrain(int lev) const { return DynamicTerrain[lev]; }
        const amrex::MultiFab& get_static_terrain() const { return StaticTerrain; }

        // PascalCase accessors for the Checkpointer
        amrex::MultiFab& GetUNew(int lev) { return U_new[lev]; }
        const amrex::MultiFab& GetUNew(int lev) const { return U_new[lev]; }
        amrex::MultiFab& GetUOld(int lev) { return U_old[lev]; }
        const amrex::MultiFab& GetUOld(int lev) const { return U_old[lev]; }
        amrex::MultiFab& GetTerrain(int lev) { return DynamicTerrain[lev]; }
        const amrex::MultiFab& GetTerrain(int lev) const { return DynamicTerrain[lev]; }

        amrex::Vector<amrex::Real>& GetTNew() { return t_new; }
        const amrex::Vector<amrex::Real>& GetTNew() const { return t_new; }
        amrex::Vector<amrex::Real>& GetTOld() { return t_old; }
        const amrex::Vector<amrex::Real>& GetTOld() const { return t_old; }
        amrex::Vector<amrex::Real>& GetDt() { return dt; }
        const amrex::Vector<amrex::Real>& GetDt() const { return dt; }

        int get_ncomp_U() const { return ncomp_U; }
        int get_ncomp_Terrain() const { return ncomp_Terrain; }

        bool is_flux_reg_allocated(int lev) const { return flux_reg[lev] != nullptr; }
        amrex::FluxRegister* get_flux_reg(int lev) { return flux_reg[lev].get(); }

        amrex::Vector<amrex::Real>& get_t_new() {return t_new;}
        amrex::Vector<amrex::Real>& get_t_old() {return t_old;}
        amrex::Vector<amrex::Real>& get_dt() {return dt;}

        const amrex::Vector<amrex::Real>& get_t_new() const {return t_new;}
        const amrex::Vector<amrex::Real>& get_t_old() const {return t_old;}
        const amrex::Vector<amrex::Real>& get_dt() const {return dt;}

        int NumLevels() const {return finest_level + 1;}
        int FinestLevel() const {return finest_level;}
        void SetFinestLevel(int lev) { finest_level = lev; }

        void BuildLevel(int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
            SetBoxArray(lev, ba);
            SetDistributionMap(lev, dm);
            U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
            U_old[lev].define(ba, dm, ncomp_U, ngrow_U);
            DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);
        }

        void BuildFluxRegister(int lev) {
            if (lev > 0 && do_reflux) {
                flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
                    boxArray(lev), DistributionMap(lev), refRatio(lev - 1), lev, ncomp_U
                );
            }
        }
        // Utils
        void ResizeLevels(int nlevs);

    protected: 
        virtual void PostProcessBaseGrids(amrex::BoxArray& box_array) const override;

    private:
        // Internal helper functions for level management
        void FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
        void FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
        void FillCoarsePatchTerrain (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
    
        void AverageDown (amrex::Vector<amrex::MultiFab>& arr);
        void AverageDownTo(int crse_lev, amrex::Vector<amrex::MultiFab>& arr);
        
        void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);
        void GetTerrainData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);
        
        void TerrainMapStaticToDynamic(int lev);
        void FluidMapStaticToDynamic(int lev);


        // Manages all I/O operations (checkpointing, plotfiles, restarts)
        std::shared_ptr<IOHandler> IO;
        HDF5SpatialMetadata metadata;

        int ncomp_U;
        int ngrow_U;

        int ncomp_Terrain;
        int ngrow_Terrain;

        // Refluxing and subcycling controls
        bool do_reflux{true};              // Whether to perform refluxing for flux correction at

        // --- Dynamic Simulation State ---
        // These are the core state variables that evolve in time and are stored at each AMR level. They are resized and redefined as levels are created or destroyed.
        amrex::Vector<amrex::MultiFab> U_new;
        amrex::Vector<amrex::MultiFab> U_old;
        amrex::Vector<amrex::MultiFab> DynamicTerrain;
        
        mutable std::vector<int> pure_fluid_boxes;
        mutable std::vector<int> boundary_boxes;

        amrex::Vector<amrex::Real> t_new;
        amrex::Vector<amrex::Real> t_old;
        amrex::Vector<amrex::Real> dt;
        
        // Boundary conditions
        amrex::Vector<amrex::BCRec> U_bcs;        // 1-component
        amrex::Vector<amrex::BCRec> Terrain_bcs;  // 1-component

        // --- Static Database State ---
        // Ground Truth Layer: Split across MPI ranks uniformly at ith Level resolution.
        // These three members define the permanent, unchanging database layout.
        amrex::BoxArray             static_ba;
        amrex::DistributionMapping  static_dm;
        amrex::Geometry             static_geom;   // <--- NEW: Dedicated geometry for the static layer
        amrex::MultiFab             StaticTerrain;
        
        std::unique_ptr<amrex::MultiFab> StaticFluid;
        
        int static_terrain_lev;


        // Flux registers for coarse-fine synchronization
        amrex::Vector<std::unique_ptr<amrex::FluxRegister>> flux_reg;

        std::string hdf5_path;
        std::string restart_chkfile;

        RuntimeParameters runtime_p;
        IOParameters      io_p;
        AMRParameters     amr_p;
        PhysicsParameters physics_p;
};

#endif