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

/**
 * \file AmrMeshState.H
 * \brief Declaration of the AMR state machine and the state it owns.
 *
 * One class, four implementation files: construction and the terrain
 * database, the AmrCore overrides, the time loop, and output.
 */

/**
 * \brief The AMR hierarchy, the state on it, the terrain under it and the solver.
 *
 * Everything that has to exist once per run and per level lives here: the
 * hierarchy itself (as an ``amrex::AmrCore`` subclass), the conserved state at
 * every level, the terrain at every level, the boundary records, the flux
 * registers, the solver, and the I/O handler.  What it does *not* contain is
 * any discretization: the fluxes, the timestep and the tagging criterion are
 * the solver's, reached through ``SolverVariant``.
 *
 * The class is split across four translation units by concern --
 * ``_Init`` (construction, terrain database, solver), ``_AMR`` (the AmrCore
 * overrides, tagging, pruning, filling), ``_Physics`` (timestep, subcycling,
 * level advance) and ``_Utils`` (resizing, output, diagnostics) -- so a change
 * to regridding does not require reading the time loop.
 *
 * **Two-phase initialization.**  The constructor only records what it was given
 * and hands the mesh description to ``AmrCore``; ``Initialize()`` then reads
 * the terrain, allocates the levels and constructs the solver.  That split
 * exists because AMReX calls the virtual overrides during mesh construction,
 * and a half-built object whose solver does not exist yet would be called with
 * one.  Nothing may be advanced before ``Initialize()`` returns.
 *
 * **Terrain as a database.**  ``StaticTerrain`` is the ground truth: one
 * MultiFab at ``static_terrain_lev`` resolution, loaded once from HDF5 and never
 * modified.  Every AMR level's ``DynamicTerrain`` is a *mapping* of it, restricted
 * with a NODATA-aware average or prolonged with conservative interpolation, and
 * ``StaticFluid`` is the matching initial-condition field.  This is why a level
 * can be created at any depth without a parent level to copy from, and why the
 * bed a fine level sees is the bed the DEM actually contains rather than an
 * interpolation of a coarser reading.
 *
 * **Two state buffers.**  ``U_new`` and ``U_old`` are swapped *by the solver*
 * (see ``SolverContext``); after ``compute_fluxes`` the new time level is in
 * ``U_new``.  ``Eta`` is a cache of ``h + z``, refreshed after every mutation of
 * either buffer, because the free surface is what crosses a coarse-fine
 * boundary and depth is not.
 */
class AmrMeshState : public amrex::AmrCore {
    public:

        /**
         * \brief Default-construct an empty state: no mesh, no levels, no solver.
         *
         * The hierarchy it inherits is empty, so nothing here can be advanced;
         * it exists so the class can be a member of another object that is
         * constructed before the run parameters are known.
         */
        AmrMeshState() : amrex::AmrCore() {}

        /**
         * \brief Record the run configuration and describe the mesh to AmrCore.
         *
         * Does not touch terrain, does not allocate levels, does not build the
         * solver: call ``Initialize()`` for that.
         *
         * The parameter structs are taken **by value** and stored, so the
         * caller's copies are not the runtime source of truth after this point;
         * editing them later has no effect.
         *
         * \param io_handler     Shared I/O handler, already configured.
         * \param terrain_ref_lev Resolution level the static DEM is defined at.
         * \param meta           Spatial metadata of the input HDF5 file.
         * \param input          Input file path, for terrain and restart data.
         * \param runtime_params Clock and stop time.
         * \param io_params      Output cadence and formats.
         * \param amr_params     Regridding, refluxing and subcycling switches.
         * \param physics_params CFL, model name, tagging thresholds, BC codes.
         * \param rb             Physical domain.
         * \param max_level_in   Deepest level.
         * \param n_cell_in      Cells at level 0.
         * \param coord          Coordinate system.
         * \param ref_ratios     Refinement ratio of each level.
         */
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

        /**
         * \brief Second phase: build the terrain database, levels and solver.
         *
         * Must be called exactly once, after the constructor, before anything
         * is advanced.  Order matters inside: terrain first, because the fluid
         * reconstruction on every level needs that level's own bed.
         */
        void Initialize();

        /*void Initialize(const int ncomp_U = 4, const int ncomp_Terrain = 1) {
            this->ncomp_U = ncomp_U;
            this->ncomp_Terrain = ncomp_Terrain;
        }*/

        /// Load the static terrain and initial fluid database from the input file.
        void InitializeTerrainFluid();
        /// Select the solver named by ``phys.model``; aborts on an unknown name.
        void InitializeSolver();

        /**
         * \brief Declared for an older regridding path; never defined.
         *
         * Regridding now goes through ``AmrCore::regrid`` (called from
         * ``TimeStepWithSubcycling``), which invokes the overrides below.
         */
        void Regrid(int lbase, amrex::Real time);

            
        // Destructor
        /// Releases the hierarchy through AmrCore; no manual level teardown needed.
        virtual ~AmrMeshState() = default;

        // Standard 5 AmrCore overrides
        /// Create a level from the static database: terrain first, then fluid.
        virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        /// Create a level by filling from its parent, then filling its ghosts.
        virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        /// Redefine an existing level after a regrid, preserving its state.
        virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;
        /// Release the level's state and flux register.
        virtual void ClearLevel(int lev) override;
        /// Tag refinement cells; delegates the criterion to the solver.
        virtual void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

        /// Components in the conserved state (h, hu, hv).
        int get_ncomp_U() const { return ncomp_U; }
        /// Components in the terrain (bed elevation, roughness).
        int get_ncomp_Terrain() const { return ncomp_Terrain; }

        /// Whether level \p lev has a flux register (it does not below the finest).
        bool is_flux_reg_allocated(int lev) const { return flux_reg[lev] != nullptr; }
        /// Flux register covering the \p lev-1 / \p lev interface.
        amrex::FluxRegister* get_flux_reg(int lev) { return flux_reg[lev].get(); }

        /// Build the view a solver is handed; see ``SolverContext``.
        SolverContext GetSolverContext();
        /// Gather the fields a checkpoint writer must save.
        CheckpointerContext GetCheckpointerContext();

        // Time stepping
        /// Ask the solver for the per-level timesteps and store them in \c dt.
        void ComputeDt();
        /// Recursive subcycled advance: advance \p lev, then its children, then reflux.
        void TimeStepWithSubcycling(int lev, amrex::Real time, int iteration);
        /// Advance one level once: fill ghosts, compute fluxes, average down.
        void AdvanceLevel(int lev, amrex::Real time, amrex::Real dt);

        // Utils
        /// Resize every per-level container to \p nlevs.
        void ResizeLevels(int nlevs);
        /// Write a plotfile at this iteration and time.
        void WritePlotfile(int iteration, amrex::Real time);
        /// Write a checkpoint through the I/O handler.
        void WriteCheckpoint();

        // Mass diagnostics / conservation tracking
        /// Compare current totals with the t=0 baseline and report the drift.
        void ComputeMassDiagnostics(amrex::Real time, int iteration);
        /// Record mass, momentum and kinetic energy at t=0 as the baseline.
        void StoreInitialMassMomentum();

        /// Drop level-0 blocks that contain no valid terrain at all.
        virtual void PostProcessBaseGrids(amrex::BoxArray& box_array) const override;

        /**
         * \brief Fill a state MultiFab from this level and the coarser one.
         *
         * Fills momenta and ghosts; depth is rebuilt from eta afterwards
         * because h must never be transported across a resolution change.  See
         * ``amrex::FillPatchFrom``.
         */
        void FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
        /// Fill terrain; its BCs are always transmissive.
        void FillPatchTerrain (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
        /// Fill the free surface, the quantity that crosses coarse-fine boundaries.
        void FillPatchEta   (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp);
        /// Refresh the eta cache for \p lev from one of the two state buffers.
        void ComputeEta(int lev, bool from_U_old = false);
        // R2 helper: derive η = h + z from an arbitrary source U into dst,
        // over grown tiles (src must carry valid ghosts) + FillBoundary dst.
        /// Derive eta from an arbitrary state into \p dst, including ghosts.
        void DeriveEtaFrom(int lev, const amrex::MultiFab& Usrc, amrex::MultiFab& dst);

        // Accessible to HydroEXA top class
        amrex::Vector<amrex::Real> t_new;   ///< Time of \c U_new, per level.
        amrex::Vector<amrex::Real> t_old;   ///< Time of \c U_old, per level.
        amrex::Vector<amrex::Real> dt;      ///< Timestep, per level.
        amrex::Vector<int> istep;           ///< Coarse-level step counter, per level.
        amrex::Vector<int> nsubsteps;       ///< Subcycles per advance: the ratio, or 1 in lockstep.
        
    private:

        
        /// Fill a level's ghosts from its parent only; no eta/momentum split.
        void FillCoarsePatchTerrain (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs);
        /// Fill the free surface from the parent, then rebuild depth.
        void FillCoarsePatchEta (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs);
        /**
         * \brief Fill a component range from the parent level.
         *
         * The coarse source is read from \p scomp and written to \p dcomp, so
         * one routine covers both "all components" and "momenta only" (depth
         * must never cross a resolution change; see ``FillPatchEta``).
         */
        void FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs,
                              int scomp = 0, int dcomp = 0, int ncomp = -1);

        /// Restrict every level onto the one below, finest to coarsest.
        void AverageDown (amrex::Vector<amrex::MultiFab>& arr);
        /// Restrict levels above \p crse_lev onto it.
        void AverageDownTo(int crse_lev, amrex::Vector<amrex::MultiFab>& arr);
        
        /// Gather time-interpolated state sources for a fill.
        void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);
        /// Gather terrain sources; terrain is static, so only one time is returned.
        void GetTerrainData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);
        /// Gather eta sources, deriving the scratch buffers the two-time blend needs.
        void GetEtaData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);
        
        /// Map the static terrain database onto \p dynTerrain at this level's resolution.
        void TerrainMapStaticToDynamic(int lev, amrex::MultiFab& dynTerrain);
        /// Map the static initial fluid state onto this level, rebuilding depth from eta.
        void FluidMapStaticToDynamic(int lev);


        // Manages all I/O operations (checkpointing, plotfiles, restarts)
        std::shared_ptr<IOHandler> IO;      ///< All reads and writes go through this.
        HDF5SpatialMetadata metadata;       ///< Geometry of the input file, for restart.

        int ncomp_U;        ///< State components (3: h, hu, hv).
        int ngrow_U;        ///< State ghost width, from the AMR parameters.

        int ncomp_Terrain;  ///< Terrain components (2: bed, roughness).
        int ngrow_Terrain;  ///< Terrain ghost width.


        // --- Dynamic Simulation State ---
        // These are the core state variables that evolve in time and are stored at each AMR level. They are resized and redefined as levels are created or destroyed.
        amrex::Vector<amrex::MultiFab> U_new;   ///< State at t^{n+1}; the solver swaps this with \c U_old.
        amrex::Vector<amrex::MultiFab> U_old;   ///< State at t^n after the swap.
        amrex::Vector<amrex::MultiFab> Eta;       // cached η = h + z, refreshed after every U mutation (feeds ctx.Eta → Roe)
        // η workspaces for GetEtaData (Rule R2: η is derived on demand from
        // U_old/U_new, never stored as state). The blend branch of GetEtaData
        // needs BOTH simultaneously, hence two buffers. Overwritten wholesale
        // on every call — no invariant to maintain, cannot go stale.
        amrex::Vector<amrex::MultiFab> EtaScratchOld;   ///< Eta derived from \c U_old, for time interpolation.
        amrex::Vector<amrex::MultiFab> EtaScratchNew;   ///< Eta derived from \c U_new, for time interpolation.
        amrex::Vector<amrex::MultiFab> DynamicTerrain;  ///< Per-level terrain, mapped from the static database.
        
        /// Level-0 box indices with no NODATA anywhere; filled during pruning.
        mutable std::vector<int> pure_fluid_boxes;
        /// Level-0 box indices that touch NODATA; filled during pruning.
        mutable std::vector<int> boundary_boxes;
        
        // Boundary conditions
        amrex::Vector<amrex::BCRec> U_bcs;        // Per-component mathematical BCRecs
        amrex::Vector<amrex::BCRec> Eta_bcs;      // Per-component mathematical BCRecs
        amrex::Vector<amrex::BCRec> Terrain_bcs;  // Per-component mathematical BCRecs

        // In AmrMeshState (member variables):
        amrex::Vector<HydroPhysBC> U_physbc;
        amrex::Vector<HydroPhysBC> Eta_physbc;
        amrex::Vector<HydroPhysBC> Terrain_physbc;

        // --- Static Database State ---
        // Ground Truth Layer: Split across MPI ranks uniformly at ith Level resolution.
        // These three members define the permanent, unchanging database layout.
        amrex::BoxArray             static_ba;      ///< Layout of the static database.
        amrex::DistributionMapping  static_dm;      ///< Rank assignment of the static database.
        amrex::Geometry             static_geom;   // <--- NEW: Dedicated geometry for the static layer
        amrex::MultiFab             StaticTerrain;  ///< Ground-truth terrain; never modified after load.
        int static_terrain_lev;                     ///< Resolution level the static database is defined at.
        
        std::unique_ptr<amrex::MultiFab> StaticFluid;  ///< Initial fluid state at the static resolution.

        // Flux registers for coarse-fine synchronization
        amrex::Vector<std::unique_ptr<amrex::FluxRegister>> flux_reg;  ///< One per interface; empty below the finest.

        /**
         * \brief Per level: has the terrain ghost ring been filled since the
         * level was created or regridded?
         *
         * Terrain is static while the Exner coupling is unwired, so the ghost
         * cells only change when the level's boxes change.  FillPatchTerrain
         * returns immediately when this is set, which removes a full coarse-fine
         * interpolation from every step.  Clear it wherever DynamicTerrain is
         * redefined.
         */
        amrex::Vector<int> terrain_ghosts_valid;

        std::string hdf5_path;          ///< Resolved path of the input HDF5 file.
        std::string restart_chkfile;    ///< Checkpoint to resume from; empty means cold start.

        RuntimeParameters runtime_p;    ///< Stored copy of the run parameters.
        IOParameters      io_p;         ///< Stored copy of the I/O parameters.
        AMRParameters     amr_p;        ///< Stored copy of the AMR parameters.
        PhysicsParameters physics_p;    ///< Stored copy of the physics parameters.

        SolverVariant solver;           ///< The selected model; throws if used before Initialize.

        // Conservation error tracking (stored at t=0, compared in ComputeMassDiagnostics)
        amrex::Real initial_mass{0.0};      ///< Total water mass at t=0.
        amrex::Real initial_mom_x{0.0};     ///< Total x-momentum at t=0.
        amrex::Real initial_mom_y{0.0};     ///< Total y-momentum at t=0.
        amrex::Real initial_ke{0.0};        ///< Total kinetic energy at t=0.
        bool initial_mass_stored{false};    ///< Whether the baseline has been recorded.
};

#endif
