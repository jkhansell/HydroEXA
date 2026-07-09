#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>


#include <state/AmrMeshState.H>
#include <boundaries/EmptyFill.H>


AmrMeshState::AmrMeshState(std::shared_ptr<IOHandler> io_handler,
                           int terrain_static_terrain_lev, 
                           HDF5SpatialMetadata& meta, 
                           std::string input, 
                           /*AmrCore params*/
                           const amrex::RealBox& rb, 
                           int max_level_in,
                           const amrex::Vector<int>& n_cell_in, 
                           int coord,
                           const amrex::Vector<amrex::IntVect>& ref_ratios)
    : amrex::AmrCore(&rb, max_level_in, n_cell_in, coord, ref_ratios),
      IO(io_handler),
      static_terrain_lev(terrain_static_terrain_lev), 
      metadata(meta), 
      hdf5_path(input),
      ncomp_U(3),
      ngrow_U(1),
      ncomp_Terrain(2),
      ngrow_Terrain(0)
{
    // At this point, the base grids are structured natively by AMReX!
    // We can explicitly tune performance boundaries from the safety of the constructor body.
    this->SetMaxGridSize(64);
    this->SetBlockingFactor(8);

    // 2. Resize containers dynamically
    U_bcs.resize(ncomp_U);
    Terrain_bcs.resize(ncomp_Terrain);

    // 3. Populate fluid boundary conditions record tracking
    for (int comp = 0; comp < ncomp_U; ++comp) 
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            if (Geom(0).isPeriodic()[idim]) 
            {
                // Periodic boundaries are tracked internally by AMReX routines
                U_bcs[comp].setLo(idim, amrex::BCType::int_dir);
                U_bcs[comp].setHi(idim, amrex::BCType::int_dir);
            } 
            else 
            {
                // Direct non-periodic walls and open outflows to our HydroEXAFillExtDir GPU functor
                U_bcs[comp].setLo(idim, amrex::BCType::ext_dir);
                U_bcs[comp].setHi(idim, amrex::BCType::ext_dir);
            }
        }
    }

    // 4. Populate terrain boundary conditions records (pure internal/extrapolated)
    for (int comp = 0; comp < ncomp_Terrain; ++comp) 
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            // Terrain geometry is fixed; we pass int_dir so that the interpolator fills ghost cells safely
            Terrain_bcs[comp].setLo(idim, amrex::BCType::foextrap);
            Terrain_bcs[comp].setHi(idim, amrex::BCType::foextrap);
        }
    }

}


void AmrMeshState::Initialize() {
    if (restart_chkfile == "") {
        // start simulation from the beginning
        const amrex::Real time = 0.0;

        InitializeStaticTerrain();
        InitFromScratch(time); // Calls PostProcessBaseGrids to prune the mesh using NODATA
        PostInit(); // Post Init Routine to fill lev < static_terrain_lev
        
        amrex::Print() << "[STUB] Skipping checkpoint write.\n";
    }
    else {
        amrex::Print() << "[STUB] Skipping checkpoint read.\n";
    }

    amrex::Print() << "[STUB] Skipping plotfile write.\n";
}

void AmrMeshState::PostProcessBaseGrids(amrex::BoxArray& box_array) const
{
    amrex::Print() << "[DEBUG PRUNE] Intercepting Level 0 layout to prune inactive NODATA zones on the GPU.\n";
    
    int scale_factor = 1 << static_terrain_lev; 
    amrex::IntVect ref_ratio_vect(AMREX_D_DECL(scale_factor, scale_factor, 1));

    amrex::BoxList active_list;
    const amrex::Real nodata_val = -9999.0;

    // Loop through every candidate Level 0 box AMReX wants to generate
    for (int b = 0; b < box_array.size(); ++b)
    {
        const amrex::Box& coarse_box = box_array[b];

        // Project the coarse index coordinates up to the fine file-resolution bounds
        amrex::Box fine_inspection_window = amrex::refine(coarse_box, ref_ratio_vect);

        // Define a safe local tracker element
        int local_box_contains_valid_data = 0;

        // Iterate over local chunks of the persistent StaticTerrain MultiFab
        for (amrex::MFIter mfi(StaticTerrain); mfi.isValid(); ++mfi)
        {
            const amrex::Box& valid_patch = mfi.validbox();
            
            // STRICT GRID ACCESSIBILITY SAFE GUARD: Only look at intersections
            // that are completely guaranteed to belong to this rank's local memory block allocation!
            amrex::Box safe_intersection = fine_inspection_window & valid_patch;

            if (!safe_intersection.isEmpty())
            {
                // Retrieve the device view
                auto const& terrain_arr = StaticTerrain.const_array(mfi);

                amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
                amrex::ReduceData<int> reduce_data(reduce_op);
                using ReduceTuple = decltype(reduce_data)::Type;

                // Fire the reduction loop strictly bound to the safe_intersection box coordinates
                reduce_op.eval(safe_intersection, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
                {
                    amrex::Real val = terrain_arr(i, j, k, 0);
                    
                    if (!isnan(val) && amrex::Math::abs(val - nodata_val) > 1e-1) {
                        return 1;
                    }
                    return 0;
                });

                ReduceTuple local_tuple = reduce_data.value(); // Safely blocks and downloads value
                if (amrex::get<0>(local_tuple) > 0) {
                    local_box_contains_valid_data = 1;
                }
            }
        }

        // Coordinate Cross-Rank Parallel Synchronization across the entire communicator
        int global_keep = local_box_contains_valid_data;
        amrex::ParallelDescriptor::ReduceIntMax(global_keep);

        if (global_keep > 0) {
            active_list.push_back(coarse_box); // Keep the box
        } else {
            amrex::Print() << "             [PRUNE] Dropped entirely inactive mountain/ocean box: " << coarse_box << "\n";
        }
    }

    // Update AMReX's original grid layout arrays with our pruned structure list
    amrex::Print() << "[DEBUG PRUNE DONE] Original box count: " << box_array.size() 
                   << " | Pruned active box count: " << active_list.size() << "\n";

    // ... [Previous Pruning logic inside PostProcessBaseGrids] ...

    if (active_list.isEmpty()) {
        amrex::Print() << "\n=========================================================================\n"
                       << "!!! CRITICAL WARNING: Every single grid box was pruned away!           !!!\n"
                       << "=========================================================================\n\n";
    } else {
        // 1. Safe to overwrite the in-place BoxArray reference parameter
        box_array = amrex::BoxArray(std::move(active_list));

        // 2. REBUILD Parent AmrCore/AmrMesh DM for Level 0 directly!
        // This synchronizes the base class dmap vector array with our pruned topology.
        // We look at level 0 because this hook specifically processes the base grids.
        int lev = 0; 
        
        // Use const_cast because PostProcessBaseGrids is marked as a 'const' method,
        // but we explicitly need to update our own core structural mapping parameters.
        auto* non_const_this = const_cast<AmrMeshState*>(this);
        
        // Re-generate a balanced layout map and save it into the parent's dmap collection
        non_const_this->SetDistributionMap(lev, amrex::DistributionMapping(box_array));
        
        amrex::Print() << "[DEBUG PRUNE] Successfully reset parent AmrCore Level 0 DistributionMapping.\n";
    }
}

void AmrMeshState::PostInit() {
    amrex::Print() << "[DEBUG INIT] Executing unified PostInit down-averaging sweeps.\n";

    // Cascade downward from our known high-resolution file anchor down to Level 0
    for (int lev = static_terrain_lev - 1; lev >= 0; --lev) {
        amrex::Print() << "             Averaging Level " << lev+1 << " -> Level " << lev << "\n";
        
        // Down-sample Fluid State variables (Enforces absolute conservation of mass and momentum)
        amrex::average_down(U_new[lev+1], U_new[lev],
                            geom[lev+1], geom[lev],
                            0, U_new[lev].nComp(), refRatio(lev));

        // Down-sample Topography features (Enforces exact volumetric grid matching)
        amrex::average_down(DynamicTerrain[lev+1], DynamicTerrain[lev],
                            geom[lev+1], geom[lev],
                            0, DynamicTerrain[lev].nComp(), refRatio(lev));
    }
    amrex::Print() << "[DEBUG INIT] PostInit complete. Full mesh tree is synchronized.\n";
}

void AmrMeshState::InitializeStaticTerrain() {

    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(metadata.global_nx - 1, metadata.global_ny - 1, 0));
    amrex::Box domain(dom_lo, dom_hi);

    // If your AMR Geometry was initialized blindly, you should ensure Geom(2) 
    // matches this exact bounding box to prevent ParallelCopy misalignments.
    // e.g., SetGeometryBase(global_domain, physical_real_box);

    // 4. Distribute the newly discovered global domain evenly
    static_ba.define(domain);

    // 1024x1024 chunks for lean MPI routing (IDK if this would work for interpolation we'll need to 
    // test it, but it should be fine for the initial conditions 
    // since we read them in at the finest level and then average down)
    static_ba.maxSize(128); 

    static_dm.define(static_ba);
    StaticTerrain.define(static_ba, static_dm, ncomp_Terrain, 0); // No ghost cells for static terrain

    // 5. Finally, execute the parallel hyperslab readers we wrote earlier.
    // Every rank now confidently knows its coordinates are within the true file bounds.
    for (amrex::MFIter mfi(StaticTerrain); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& z_arr = StaticTerrain.array(mfi);
        
        IO->ReadHDF5Hyperslab(z_arr, bx, hdf5_path, "bathymetry"); 
    }

    amrex::Print() << "[DEBUG INIT] Standalone StaticTerrain MultiFab successfully loaded.\n";
}

void AmrMeshState::ResizeLevels(int nlevs) {
    U_new.resize(nlevs);
    U_old.resize(nlevs);
    DynamicTerrain.resize(nlevs);

    t_new.resize(nlevs);
    t_old.resize(nlevs);
    dt.resize(nlevs);

    flux_reg.resize(nlevs);
}

void AmrMeshState::MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                           const amrex::BoxArray& ba, 
                                           const amrex::DistributionMapping& dm) 
{
    amrex::Print() << "[DEBUG MakeNewLevelFromScratch] Entered Function.\n";

    // CRITICAL GUARD: If pruning dropped all boxes, ba.size() is 0.
    // Standard level initializations cannot allocate over empty layouts.
    if (ba.empty() || ba.size() == 0) {
        amrex::Print() << "[WARNING LEVEL " << lev 
                       << "] No active grid patches remain after post-process pruning! Skipping allocation.\n";
        return; 
    }

    // Allocate current AMR tier MultiFabs
    U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
    DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);

    amrex::Print() << "[DEBUG MakeNewLevelFromScratch] Defined MultiFabs\n";

    // -------------------------------------------------------------------------
    // STEP 2: Terrain Mapping Operations
    // -------------------------------------------------------------------------
    if (lev == static_terrain_lev) {
        // ParallelCopy brings data over from our standalone StaticTerrain MultiFab
        DynamicTerrain[lev].ParallelCopy(StaticTerrain, 0, 0, ncomp_Terrain);
    } 
    else if (lev < static_terrain_lev) {
        DynamicTerrain[lev].setVal(0.0); // Handled by a downstream average_down pass later
    } 
    else {
        // Interpolate down from immediate parent
        FillCoarsePatch(lev, time, DynamicTerrain[lev], Terrain_bcs, 0, ncomp_Terrain);
    }

    amrex::Print() << "[DEBUG MakeNewLevelFromScratch] After FillCoarsePatch\n";


    // -------------------------------------------------------------------------
    // STEP 3: Data-Driven Fluid Mapping (h, hu, hv)
    // -------------------------------------------------------------------------
    if (lev == static_terrain_lev) {
        // Every process reads its designated subdomain tile directly from the HDF5 file
        for (amrex::MFIter mfi(U_new[lev]); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            auto const& u_arr = U_new[lev].array(mfi);
            
            // Read dataset "fluid_h" from the file directly into Component 0 (h)
            IO->ReadHDF5Hyperslab(u_arr, bx, hdf5_path, "fluid", 0); 
            
            // Read momentum components into Component 1 (hu) and Component 2 (hv)
            IO->ReadHDF5Hyperslab(u_arr, bx, hdf5_path, "fluid", 1); 
            IO->ReadHDF5Hyperslab(u_arr, bx, hdf5_path, "fluid", 2); 
        }
    } 
    else if (lev < static_terrain_lev) {
        // Coarser tiers are initialized to zero. They will be filled via average_down
        // once the reference data level finishes loading.
        U_new[lev].setVal(0.0);
    } 
    else {
        // Sub-grid fine tiers: Since AMReX initializes from level 0 upwards,
        // the parent level (lev-1) is guaranteed to be fully configured.
        FillCoarsePatch(lev, time, U_new[lev], U_bcs, 0, ncomp_U);
    }

    amrex::Print() << "[DEBUG INIT] Level " << lev << " Fluid & Terrain States Initialized.\n";
}


void AmrMeshState::MakeNewLevelFromCoarse(int lev, amrex::Real time, 
                                          const amrex::BoxArray& ba, 
                                          const amrex::DistributionMapping& dm) 
{
    // 1. DYNAMIC SIZE VERIFICATION: Ensure our state vectors can hold this level index

    // 2. Allocate the brand new level structures cleanly
    U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
    U_old[lev].define(ba, dm, ncomp_U, ngrow_U);
    DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);

    // Track time-stamps for standard evolution sequences
    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // 3. Dynamic Flux Register Hook
    if (lev > 0 && do_reflux) {
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba, dm, refRatio(lev - 1), lev, ncomp_U
        );
    } else {
        flux_reg[lev] = nullptr;
    }

    // 4. DYNAMIC TERRAIN ROUTING FOR NEWLY SPAWNED GRIDS

    if (lev == static_terrain_lev) {
        // Safe distributed cross-rank layout copy from our persistent baseline ground truth
        DynamicTerrain[lev].ParallelCopy(StaticTerrain, 0, 0, ncomp_Terrain);
    } 
    else if (lev < static_terrain_lev) {
        // Coarser tiers are calculated via down-averaging sweeps
        DynamicTerrain[lev].setVal(0.0); 
    } 
    else {
        // Sub-grid fine tiers: Project topography down from parent level (lev-1) 
        // to maintain conservation rules across newly refinement patches
        FillCoarsePatch(lev, time, DynamicTerrain[lev], Terrain_bcs, 0, ncomp_Terrain);
    }

    // 5. Populate the newly created fluid states with conservative interpolation
    // from the underlying coarser level mesh state data
    FillCoarsePatch(lev, time, U_new[lev], U_bcs, 0, ncomp_U);
}

void AmrMeshState::Regrid(int lbase, amrex::Real time) {
    amrex::Print() << "[DEBUG REGRID] Intercepting regrid pass at lbase = " << lbase << "\n";

    // 1. FORWARD TO THE NATIVE AMREX CORE BACKEND
    // This executes the exact block of code you provided, updating all box layers
    regrid(lbase, time);

    // 2. RUN OUR SYNCHRONIZATION POST-PROCESS SWEEP
    // Now that finest_level is updated, we safely anchor our down-averaging routines
    PostCoarsePatchSync();
    
}

void AmrMeshState::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
    
    MultiFab new_U(ba, dm, ncomp_U, ngrow_U);
    MultiFab old_U(ba, dm, ncomp_U, ngrow_U);
    MultiFab new_DT(ba, dm, ncomp_Terrain, ngrow_Terrain);
    
    FillPatch(lev, time, new_U, U_bcs, 0, ncomp_U);
    
    std::swap(new_U, U_new[lev]);
    std::swap(old_U, U_old[lev]);

    t_new[lev] = time; 
    t_old[lev] = time - 1.e200;

    if (lev > 0 && do_reflux) {
        flux_reg[lev].reset(new FluxRegister(ba, dm, refRatio(lev-1), lev, ncomp_U));
    }
}

void AmrMeshState::ClearLevel(int lev) {
    U_new[lev].clear();
    U_old[lev].clear();
    DynamicTerrain[lev].clear();
}

void AmrMeshState::ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) {

}

void AmrMeshState::GetData(int lev, Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime) {
    data.clear();
    datatime.clear();

    const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
        data.push_back(&U_new[lev]);
        datatime.push_back(t_new[lev]);
    }
    else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
        data.push_back(&U_old[lev]);
        datatime.push_back(t_old[lev]);
    }
    else
    {
        data.push_back(&U_old[lev]);
        data.push_back(&U_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }    
}

void AmrMeshState::FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    if (lev == 0) {
        Vector<MultiFab*> smf;
        Vector<Real> stime;
        GetData(0, time, smf, stime);

        if(Gpu::inLaunchRegion()) {
            GpuBndryFuncFab<EmptyFill> gpu_bndry_func(EmptyFill{});
            PhysBCFunct<GpuBndryFuncFab<EmptyFill> > physbc(geom[lev],bcs,gpu_bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                        geom[lev], physbc, 0);
        } else {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev],bcs,bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                        geom[lev], physbc, 0);
        }
    } else {
        Vector<MultiFab*> cmf, fmf;
        Vector<Real> ctime, ftime;
        GetData(lev-1, time, cmf, ctime);
        GetData(lev  , time, fmf, ftime);

        Interpolater* mapper = &cell_cons_interp;

        if(Gpu::inLaunchRegion()) {
            GpuBndryFuncFab<EmptyFill> gpu_bndry_func(EmptyFill{});
            PhysBCFunct<GpuBndryFuncFab<EmptyFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
            PhysBCFunct<GpuBndryFuncFab<EmptyFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

            amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                      0, icomp, ncomp, geom[lev-1], geom[lev],
                                      cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                      mapper, bcs, 0);
        } else {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
            PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

            amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                      0, icomp, ncomp, geom[lev-1], geom[lev],
                                      cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                      mapper, bcs, 0);
        }
    }
}

void AmrMeshState::FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf,  amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    BL_ASSERT(lev > 0);

    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetData(lev-1, time, cmf, ctime);
    Interpolater* mapper = &cell_cons_interp;

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    if(Gpu::inLaunchRegion())
    {
        GpuBndryFuncFab<EmptyFill> gpu_bndry_func(EmptyFill{});
        PhysBCFunct<GpuBndryFuncFab<EmptyFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
        PhysBCFunct<GpuBndryFuncFab<EmptyFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
    else
    {
        CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
        PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
        PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }

}

void AmrMeshState::AverageDown (amrex::Vector<amrex::MultiFab>& arr) {
    for (int lev = finest_level-1; lev >= 0; --lev) {
        amrex::average_down(arr[lev+1], arr[lev],
                            geom[lev+1], geom[lev],
                            0, arr[lev].nComp(), refRatio(lev));
    }
}

void AmrMeshState::AverageDownTo(int crse_lev, amrex::Vector<amrex::MultiFab>&  arr) {
    amrex::average_down(arr[crse_lev+1], arr[crse_lev],
                        geom[crse_lev+1], geom[crse_lev],
                        0, arr[crse_lev].nComp(), refRatio(crse_lev));
}


void AmrMeshState::PostCoarsePatchSync()
{
    amrex::Print() << "[DEBUG REGRID] Executing macro-level down-averaging synchronization...\n";

    // A. Synchronize the dynamic calculation terrain layout down to Level 0
    // If runtime grids shifted, coarse backgrounds must match the fine averages
    for (int lev = static_terrain_lev - 1; lev >= 0; --lev)
    {
        if (lev + 1 <= finestLevel() && LevelDefined(lev) && LevelDefined(lev + 1))
        {
            amrex::average_down(
                DynamicTerrain[lev+1], DynamicTerrain[lev],
                geom[lev+1], geom[lev],
                0, ncomp_Terrain, refRatio(lev));
        }
    }

    // B. Synchronize the active shallow-water fluid matrices (U_new)
    // Ensures perfect conservation of mass (h) and momentum (hu, hv) across full domain tree
    int current_finest = finestLevel();
    for (int lev = current_finest - 1; lev >= 0; --lev)
    {
        
        if (LevelDefined(lev) && LevelDefined(lev + 1))
        {
            amrex::average_down(
                U_new[lev+1], U_new[lev],
                geom[lev+1], geom[lev],
                0, ncomp_U, refRatio(lev));
        }
    }

    amrex::Print() << "[DEBUG REGRID] Regrid synchronization phase complete.\n";
}
