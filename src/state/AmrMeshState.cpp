

#include <state/AmrMeshState.H>
#include <boundaries/EmptyFill.H>


AmrMeshState::AmrMeshState(std::shared_ptr<IOHandler> io_handler,
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
                           const amrex::Vector<amrex::IntVect>& ref_ratios)
    : amrex::AmrCore(&rb, max_level_in, n_cell_in, coord, ref_ratios),
      IO(io_handler),
      static_terrain_lev(terrain_ref_lev), 
      metadata(meta), 
      hdf5_path(input),
      ncomp_U(3),
      ngrow_U(1),
      ncomp_Terrain(1),
      ngrow_Terrain(1), 
      runtime_p(runtime_params), 
      io_p(io_params), 
      amr_p(amr_params), 
      physics_p(physics_params) 
{
    // 2. Resize containers dynamically
    ResizeLevels(max_level + 1); // Resizes all variables
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


void AmrMeshState::ResizeLevels(int nlevs) {
    U_new.resize(nlevs);
    U_old.resize(nlevs);
    DynamicTerrain.resize(nlevs);

    t_new.resize(nlevs);
    t_old.resize(nlevs);
    dt.resize(nlevs);

    flux_reg.resize(nlevs);
}


void AmrMeshState::Initialize() {
    amrex::Real time = 0.0;
    int initial_step = 0;

    if (restart_chkfile == "") {
        // start simulation from the beginning

        amrex::Print() << "FINEST LEVEL: " << finest_level << " MAX_LEVEL: "<< max_level <<  "\n";
        
        InitializeSolver();
        InitializeTerrainFluid();
        InitFromScratch(time); // Calls PostProcessBaseGrids to prune the mesh using NODATA
        //PostInit(); // Post Init Routine to fill lev < static_terrain_lev
        amrex::Print() << "FINEST LEVEL: " << finest_level << " MAX_LEVEL: "<< max_level <<  "\n";

        // deallocating initial static fluid MultiFab pointer since we         
        // already allocated the initial DynamicFluid Multifab and now refinement or coarsening 
        // is tackled by physics
        StaticFluid.reset();
   
        amrex::Print() << "[STUB] Skipping checkpoint write.\n";
    }
    else {
        amrex::Print() << "[STUB] Skipping checkpoint read.\n";
    }

    // --- DIAGNOSTICS START ---
    amrex::Print() << "\n[DIAGNOSTIC] --- Level 0 Status ---\n";
    amrex::Print() << "[DIAGNOSTIC] U_new size (number of levels): " << U_new.size() << "\n";
    
    if (U_new.size() > 0) {
        amrex::Print() << "[DIAGNOSTIC] U_new[0] empty(): " << (U_new[0].empty() ? "YES" : "NO") << "\n";
        amrex::Print() << "[DIAGNOSTIC] U_new[0] BoxArray size (number of boxes): " << U_new[0].boxArray().size() << "\n";
        
        // Let's check DynamicTerrain too just in case
        amrex::Print() << "[DIAGNOSTIC] DynamicTerrain[0] empty(): " << (U_new[0].empty() ? "YES" : "NO") << "\n";
        amrex::Print() << "[DIAGNOSTIC] DynamicTerrain[0] BoxArray size (number of boxes): " << U_new[0].boxArray().size() << "\n";
    }
    // --- DIAGNOSTICS END ---

    IO->WritePlotfile(
        U_new, DynamicTerrain, 
        initial_step, time, 
        Geom(), refRatio(), 
        finest_level
    );
}

void AmrMeshState::CheckNaNsAndValidInMultiFab(const amrex::MultiFab& mf, int& global_has_valid, int& global_has_nan, int comp)
{
    BL_PROFILE("CheckNaNsAndValidInMultiFab");

    int local_has_valid = 0;
    int local_has_nan   = 0;

    // Simultaneously check for ANY valid cells AND ANY NaN cells
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_op;
    amrex::ReduceData<int, int> reduce_data(reduce_op);
    using ReduceTuple = decltype(reduce_data)::Type;

    // Iterate through available valid tiles
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& fab_arr = mf.const_array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {
            amrex::Real val = fab_arr(i, j, k, comp);
            int is_nan   = amrex::isnan(val) ? 1 : 0;
            int is_valid = !is_nan ? 1 : 0;
            
            return {is_valid, is_nan};
        });
    }

    // Accumulate the findings locally for this rank
    ReduceTuple local_tuple = reduce_data.value();
    local_has_valid = amrex::get<0>(local_tuple);
    local_has_nan   = amrex::get<1>(local_tuple);

    // Sync state flags globally across all MPI ranks
    global_has_valid = local_has_valid;
    global_has_nan   = local_has_nan;
    amrex::ParallelDescriptor::ReduceIntMax(global_has_valid);
    amrex::ParallelDescriptor::ReduceIntMax(global_has_nan);
}

void AmrMeshState::InitializeTerrainFluid() {

    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(metadata.global_nx - 1, metadata.global_ny - 1, 0));
    amrex::Box domain(dom_lo, dom_hi);

    // 1. Define the dedicated Geometry for the Static Terrain.
    // We borrow the physical bounding box (RealBox), coordinate system, 
    // and periodicity from the base AMR level (Level 0) so they perfectly align in physical space.
    amrex::Vector<amrex::Real> prob_lo = { metadata.prob_lo_x, metadata.prob_lo_y, 0.0 };
    amrex::Vector<amrex::Real> prob_hi = { metadata.prob_hi_x, metadata.prob_hi_y, 0.0 }; 
    amrex::RealBox rb(prob_lo.data(), prob_hi.data());
    
    //amrex::RealBox rb = Geom(0).ProbDomain();
    
    int coord = Geom(0).Coord();
    amrex::Array<int, AMREX_SPACEDIM> is_per = Geom(0).isPeriodic();
    
    static_geom.define(domain, rb, coord, is_per);
    amrex::Print() << "[DEBUG GEOM] Static Grid dx: " << static_geom.CellSize(0) 
               << " | File Metadata dx: " << metadata.dx << "\n";

    // 2. Distribute the newly discovered global domain evenly
    static_ba.define(domain);

    // 1024x1024 chunks for lean MPI routing (128 is a good max size for testing)
    static_ba.maxSize(512); 

    static_dm.define(static_ba);
    
    // 3. Allocate the MultiFab
    StaticTerrain.define(static_ba, static_dm, ncomp_Terrain, 2); // No ghost cells for static terrain
    
    // we make this a dynamic pointer so that we can release after initialization
    StaticFluid = std::make_unique<amrex::MultiFab>(static_ba, static_dm, ncomp_U, 2); // No ghost cells for static terrain

    StaticTerrain.setVal(-9999.0);
    StaticFluid->setVal(-9999.0);


    // 4. Execute the parallel hyperslab readers
    for (amrex::MFIter mfi(StaticTerrain); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& z_arr = StaticTerrain.array(mfi);
        auto const& u_arr = StaticFluid->array(mfi);
        
        IO->ReadHDF5Hyperslab(z_arr, bx, "bathymetry", 0); 
        IO->ReadHDF5HyperslabComponents(u_arr, bx, "fluid", 0, ncomp_U);
    }

    StaticTerrain.FillBoundary(static_geom.periodicity());
    StaticFluid->FillBoundary(static_geom.periodicity());

    // -------------------------------------------------------------
    // Topography Diagnostic: Check component 0 (Zb)
    // -------------------------------------------------------------
    int terrain_has_valid = 0;
    int terrain_has_nan   = 0;
    CheckNaNsAndValidInMultiFab(StaticTerrain, terrain_has_valid, terrain_has_nan, 0);

    amrex::Print() << "\n============================================\n"
                << "[DIAGNOSTIC] StaticTerrain Status Check:\n"
                << "  -> Has Valid Cells: " << (terrain_has_valid ? "YES" : "NO") << "\n"
                << "  -> Has NaN Cells  : " << (terrain_has_nan   ? "YES" : "NO") << "\n";


    // -------------------------------------------------------------
    // Fluid State Diagnostic: Loop through comps 0, 1, 2 (h, hu, hv)
    // -------------------------------------------------------------
    int fluid_has_valid = 0;
    int fluid_has_nan   = 0;

    for (int comp = 0; comp < 3; ++comp) {
        int local_comp_valid = 0;
        int local_comp_nan   = 0;
        CheckNaNsAndValidInMultiFab(*StaticFluid, local_comp_valid, local_comp_nan, comp);
        
        fluid_has_valid = std::max(fluid_has_valid, local_comp_valid);
        fluid_has_nan   = std::max(fluid_has_nan, local_comp_nan);
    }

    amrex::Print() << "[DIAGNOSTIC] StaticFluid Status Check (3 Comps):\n"
                << "  -> Has Valid Fluid Cells: " << (fluid_has_valid ? "YES" : "NO") << "\n"
                << "  -> Has NaN Fluid Cells  : " << (fluid_has_nan   ? "YES" : "NO") << "\n"
                << "============================================\n\n";

    amrex::Print() << "[DEBUG INIT] Standalone StaticTerrain MultiFab successfully loaded.\n";
}

void AmrMeshState::InitializeSolver()
{
    if (physics_p.model == "Roe") {
        solver = Roe{};
    } else {  // fill with else ifs later
        amrex::Abort("Unknown solver type: " + physics_p.model);
    }
}

void AmrMeshState::TerrainMapStaticToDynamic(int lev)
{
    amrex::MultiFab& amr_mf = DynamicTerrain[lev];
    
    amr_mf.setVal(-9999.0);

    const amrex::Geometry& amr_geom = geom[lev];
    
    // 2. Static is Finer than AMR: Restrict (Average Down)
    if (static_terrain_lev > lev) 
    {
        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(amr_geom.CellSize(idim) / static_geom.CellSize(idim)));
            amrex::Print() << "[DEBUG TerrainMapStaticToDynamic] Ratio: " << ratio[idim] << " Dim: " << idim << "\n";
            
        }
    
        // Use the Geometry-aware 7-argument overload that natively bridges mismatched BoxArrays
        amrex::average_down(StaticTerrain, amr_mf, 
                            static_geom, amr_geom, 
                            0, ncomp_Terrain, ratio);
    } 
    
    // 3. Static is Coarser than AMR: Prolongate (Interpolate from the custom standalone layer)
    else 
    {
        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(static_geom.CellSize(idim) / amr_geom.CellSize(idim)));
        }

        amrex::Real dummy_time = 0.0;

        amrex::Interpolater* mapper = &amrex::pc_interp;

        if (amrex::Gpu::inLaunchRegion())
        {
            // Set up hardware-agnostic boundary functions using your EmptyFill approach
            amrex::GpuBndryFuncFab<EmptyFill> gpu_bndry_func(EmptyFill{});
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<EmptyFill>> cphysbc(static_geom, Terrain_bcs, gpu_bndry_func);
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<EmptyFill>> fphysbc(amr_geom,    Terrain_bcs, gpu_bndry_func);

            // 15-argument standard compiler-compliant call
            amrex::InterpFromCoarseLevel(amr_mf, dummy_time, StaticTerrain, 0, 0, ncomp_Terrain,
                                         static_geom, amr_geom,
                                         cphysbc, 0, fphysbc, 0, ratio,
                                         mapper, Terrain_bcs, 0);
        }
        else
        {

            // Host CPU equivalent mapping path
            amrex::CpuBndryFuncFab bndry_func(nullptr);
            amrex::PhysBCFunct<amrex::CpuBndryFuncFab> cphysbc(static_geom, Terrain_bcs, bndry_func);
            amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fphysbc(amr_geom,    Terrain_bcs, bndry_func);

            // 15-argument standard compiler-compliant call
            amrex::InterpFromCoarseLevel(amr_mf, dummy_time, StaticTerrain, 0, 0, ncomp_Terrain,
                                         static_geom, amr_geom,
                                         cphysbc, 0, fphysbc, 0, ratio,
                                         mapper, Terrain_bcs, 0);
        }
    }

    amr_mf.FillBoundary(geom[lev].periodicity());
}

void AmrMeshState::FluidMapStaticToDynamic(int lev)
{
    amrex::MultiFab& amr_mf = U_new[lev];
    
    amr_mf.setVal(-9999.0);

    const amrex::Geometry& amr_geom = geom[lev];
    
    // 2. Static is Finer than AMR: Restrict (Average Down)
    if (static_terrain_lev > lev) 
    {
        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(amr_geom.CellSize(idim) / static_geom.CellSize(idim)));
        }
        
        // Use the Geometry-aware 7-argument overload that natively bridges mismatched BoxArrays
        amrex::average_down(*StaticFluid, amr_mf, 
                            static_geom, amr_geom, 
                            0, ncomp_U, ratio);
    } 
    
    // 3. Static is Coarser than AMR: Prolongate (Interpolate from the custom standalone layer)
    else 
    {
        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(static_geom.CellSize(idim) / amr_geom.CellSize(idim)));
        }

        amrex::Real dummy_time = 0.0;

        amrex::Interpolater* mapper = &amrex::cell_cons_interp;

        if (amrex::Gpu::inLaunchRegion())
        {
            // Set up hardware-agnostic boundary functions using your EmptyFill approach
            amrex::GpuBndryFuncFab<EmptyFill> gpu_bndry_func(EmptyFill{});
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<EmptyFill>> cphysbc(static_geom, U_bcs, gpu_bndry_func);
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<EmptyFill>> fphysbc(amr_geom,    U_bcs, gpu_bndry_func);

            // 15-argument standard compiler-compliant call
            amrex::InterpFromCoarseLevel(amr_mf, dummy_time, *StaticFluid, 0, 0, ncomp_U,
                                         static_geom, amr_geom,
                                         cphysbc, 0, fphysbc, 0, ratio,
                                         mapper, U_bcs, 0);
        }
        else
        {

            // Host CPU equivalent mapping path
            amrex::CpuBndryFuncFab bndry_func(nullptr);
            amrex::PhysBCFunct<amrex::CpuBndryFuncFab> cphysbc(static_geom, U_bcs, bndry_func);
            amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fphysbc(amr_geom,    U_bcs, bndry_func);

            // 15-argument standard compiler-compliant call
            amrex::InterpFromCoarseLevel(amr_mf, dummy_time, *StaticFluid, 0, 0, ncomp_U,
                                         static_geom, amr_geom,
                                         cphysbc, 0, fphysbc, 0, ratio,
                                         mapper, U_bcs, 0);
        }
    }

    amr_mf.FillBoundary(geom[lev].periodicity());
}


void AmrMeshState::MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                           const amrex::BoxArray& ba, 
                                           const amrex::DistributionMapping& dm) 
{
    // 1. Always allocate the MultiFabs for this level
    U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
    U_old[lev].define(ba, dm, ncomp_U, ngrow_U);
    DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    TerrainMapStaticToDynamic(lev);
    FluidMapStaticToDynamic(lev);

    amrex::Print() << "[DEBUG] Level " << lev << " fully initialized without CoarsePatch dependency.\n";
}


void AmrMeshState::PostProcessBaseGrids(amrex::BoxArray& box_array) const
{
    const int num_boxes = box_array.size();
    if (num_boxes == 0) return;

    // Vectors to hold local box validity across ranks (num_boxes is global)
    std::vector<int> local_has_valid(num_boxes, 0);
    std::vector<int> local_has_nan(num_boxes, 0);

    // Compute refinement factor for StaticTerrain's index space
    const int scale = 1 << static_terrain_lev;
    const amrex::IntVect ref_ratio(AMREX_D_DECL(scale, scale, scale));

    // Outer loop over local StaticTerrain patches (Only scans locally owned GPU data)
    for (amrex::MFIter mfi(StaticTerrain); mfi.isValid(); ++mfi)
    {
        const amrex::Box& patch_box = mfi.validbox();
        auto const& terrain_arr = StaticTerrain.const_array(mfi);

        for (int b = 0; b < num_boxes; ++b)
        {
            // Upscale the coarse Level 0 box to match StaticTerrain's index resolution
            amrex::Box scaled_box = amrex::refine(box_array[b], ref_ratio);
            amrex::Box intersection = scaled_box & patch_box;

            if (intersection.ok())
            {
                amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_op;
                amrex::ReduceData<int, int> reduce_data(reduce_op);
                using ReduceTuple = decltype(reduce_data)::Type;

                reduce_op.eval(intersection, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
                {
                    amrex::Real val = terrain_arr(i, j, k, 0);
                    
                    // Safe GPU Floating-point and NoData (-9999) check
                    bool is_nodata = (val <= static_cast<amrex::Real>(-9998.0));
                    
                    return {!is_nodata ? 1 : 0, is_nodata ? 1 : 0};
                });

                ReduceTuple local_tuple = reduce_data.value(); // Handles stream sync automatically
                local_has_valid[b] = std::max(local_has_valid[b], amrex::get<0>(local_tuple));
                local_has_nan[b]   = std::max(local_has_nan[b],   amrex::get<1>(local_tuple));
            }
        }
    }

    // Single global MPI reduction across all ranks
    amrex::ParallelDescriptor::ReduceIntMax(local_has_valid.data(), num_boxes);
    amrex::ParallelDescriptor::ReduceIntMax(local_has_nan.data(), num_boxes);

    // Rebuild active BoxList and update class index arrays
    amrex::BoxList active_list;
    pure_fluid_boxes.clear();
    boundary_boxes.clear();

    for (int b = 0; b < num_boxes; ++b) 
    {
        if (local_has_valid[b] > 0) 
        {
            active_list.push_back(box_array[b]);
            int new_idx = static_cast<int>(active_list.size()) - 1;

            if (local_has_nan[b] > 0) {
                boundary_boxes.push_back(new_idx);
            } else {
                pure_fluid_boxes.push_back(new_idx);
            }
        }
    }

    amrex::Print() << "[DEBUG PRUNE] Successfully pruned Level 0 layout: Retained " 
                   << active_list.size() << " / " << num_boxes << " boxes.\n";

    // Safety check to preserve domain integrity if all boxes evaluate empty
    if (active_list.isEmpty()) {
        amrex::Print() << "[WARNING PRUNE] All boxes were pruned! Keeping full original domain layout.\n";
        for (int b = 0; b < num_boxes; ++b) {
            pure_fluid_boxes.push_back(b);
        }
    } else {
        box_array = amrex::BoxArray(std::move(active_list));
    }
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
        
    // We use our terrain static representation
    TerrainMapStaticToDynamic(lev);

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

    FillCoarsePatch(lev, time, U_new[lev], U_bcs, 0, ncomp_U);
}

void AmrMeshState::Regrid(int lbase, amrex::Real time) {
    amrex::Print() << "[DEBUG REGRID] Intercepting regrid pass at lbase = " << lbase << "\n";
    // 1. FORWARD TO THE NATIVE AMREX CORE BACKEND
    // This executes the exact block of code you provided, updating all box layers
    regrid(lbase, time);
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

void AmrMeshState::ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{   

    if (lev >= max_level) return;

    if (time == 0.0) {
        // Use our high-res VRAM StaticTerrain to decide where to refine
        const amrex::MultiFab& terrain = DynamicTerrain[lev];
        const amrex::Real slope_threshold = physics_p.z_grad_thresh[lev];
        const amrex::Real* dx = Geom(lev).CellSize();

        for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            const auto tag = tags.array(mfi);
            const auto z = terrain.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                // Check slope to tag for refinement
                
                amrex::Real z_east  = z(i+1, j, k, 0);
                amrex::Real z_west  = z(i-1, j, k, 0);
                amrex::Real z_north = z(i, j+1, k, 0);
                amrex::Real z_south = z(i, j-1, k, 0);
                
                bool edge = (z_east == -9999) || (z_west == -9999) || (z_north == -9999) || (z_south == -9999);

                if (edge) { return; }
                
                amrex::Real dzdx = (z(i+1,j,k,0) - z(i-1,j,k,0)) / (2.0 * dx[0]);
                amrex::Real dzdy = (z(i,j+1,k,0) - z(i,j-1,k,0)) / (2.0 * dx[1]);
                
                // Tag if slope is high
                if (std::sqrt(dzdx*dzdx + dzdy*dzdy) > slope_threshold) {
                    tag(i, j, k) = amrex::TagBox::SET;
                }
            });
        }
    }
}

void AmrMeshState::GetTerrainData(int lev, Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime) {
    data.clear();
    datatime.clear();
    data.push_back(&DynamicTerrain[lev]);
    datatime.push_back(t_new[lev]);
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


void AmrMeshState::FillCoarsePatchTerrain (int lev, amrex::Real time, amrex::MultiFab& mf,  amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    BL_ASSERT(lev > 0);

    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetTerrainData(lev-1, time, cmf, ctime);
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


