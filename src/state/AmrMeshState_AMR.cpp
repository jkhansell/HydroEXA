#include <state/AmrMeshState.H>
#include <utils/MaskedAverageDown.H>
#include <boundaries/EmptyFill.H>
#include <utils/Logging.H>


void AmrMeshState::TerrainMapStaticToDynamic(int lev, amrex::MultiFab& dynTerrain)
{

    // Ensure source coarse terrain has valid ghost halos prior to interpolation
    StaticTerrain.FillBoundary(static_geom.periodicity());

    if (StaticTerrain.contains_nan()) {
        LOG(INFO, "StaticTerrain contains NaNs BEFORE average_down!\n");
    }

    
    dynTerrain.setVal(-9999.0);

    const amrex::Geometry& amr_geom = geom[lev];
    
    // 2. Static is Finer than AMR: Restrict (Average Down)
    if (static_terrain_lev > lev)
    {
        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(amr_geom.CellSize(idim) / static_geom.CellSize(idim)));
        }

        // Use the Geometry-aware 7-argument overload that natively bridges mismatched BoxArrays
        amrex::masked_average_down(StaticTerrain, dynTerrain, static_geom, amr_geom, 0, ncomp_Terrain, ratio, -9999.0);
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
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<EmptyFill>> cphysbc(static_geom, Terrain_bcs, gpu_bndry_func);
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<EmptyFill>> fphysbc(amr_geom,    Terrain_bcs, gpu_bndry_func);

            // 15-argument standard compiler-compliant call
            amrex::InterpFromCoarseLevel(dynTerrain, dummy_time, StaticTerrain, 0, 0, ncomp_Terrain,
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
            amrex::InterpFromCoarseLevel(dynTerrain, dummy_time, StaticTerrain, 0, 0, ncomp_Terrain,
                                         static_geom, amr_geom,
                                         cphysbc, 0, fphysbc, 0, ratio,
                                         mapper, Terrain_bcs, 0);
        }
    }

    dynTerrain.FillBoundary(geom[lev].periodicity());
}

void AmrMeshState::FluidMapStaticToDynamic(int lev)
{
    amrex::MultiFab& amr_mf = U_new[lev];
    
    amr_mf.setVal(0.0);

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

/* ---------------------- AMReX AmrCore Base Functions  ----------------------*/


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

    TerrainMapStaticToDynamic(lev, DynamicTerrain[lev]);
    FluidMapStaticToDynamic(lev);

    DynamicTerrain[lev].FillBoundary(Geom(lev).periodicity());
    U_new[lev].FillBoundary(Geom(lev).periodicity());
    U_old[lev].FillBoundary(Geom(lev).periodicity());

    if (lev > 0 && amr_p.do_reflux) {
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba, dm, refRatio(lev - 1), lev, ncomp_U
        );
    } else {
        flux_reg[lev] = nullptr;
    }

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

    LOG(DIAG, "Successfully pruned Level 0 layout: Retained "
                   + std::to_string(active_list.size()) + " / " + std::to_string(num_boxes) + " boxes.");

    // Safety check to preserve domain integrity if all boxes evaluate empty
    if (active_list.isEmpty()) {
        LOG_WARN("All boxes were pruned! Keeping full original domain layout.");
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
    TerrainMapStaticToDynamic(lev, DynamicTerrain[lev]);

    // Track time-stamps for standard evolution sequences
    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // 3. Dynamic Flux Register Hook
    if (lev > 0 && amr_p.do_reflux) {
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba, dm, refRatio(lev - 1), lev, ncomp_U
        );
    } else {
        flux_reg[lev] = nullptr;
    }

    FillCoarsePatch(lev, time, U_new[lev], U_bcs, 0, ncomp_U);
}


void AmrMeshState::Regrid(int lbase, amrex::Real time) {
    LOG(DIAG, "Intercepting regrid pass at lbase = " + std::to_string(lbase));
    // 1. FORWARD TO THE NATIVE AMREX CORE BACKEND
    // This executes the exact block of code you provided, updating all box layers
    regrid(lbase, time);
}

void AmrMeshState::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
    
    MultiFab new_U(ba, dm, ncomp_U, ngrow_U);
    MultiFab old_U(ba, dm, ncomp_U, ngrow_U);
    MultiFab new_DT(ba, dm, ncomp_Terrain, ngrow_Terrain);
    
    FillPatch(lev, time, new_U, U_bcs, 0, ncomp_U);
    TerrainMapStaticToDynamic(lev, new_DT);
    

    std::swap(new_U, U_new[lev]);
    std::swap(old_U, U_old[lev]);
    std::swap(new_DT, DynamicTerrain[lev]);

    t_new[lev] = time; 
    t_old[lev] = time - 1.e200;

    if (lev > 0 && amr_p.do_reflux) {
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba, dm, refRatio(lev - 1), lev, ncomp_U
        );
    } else {
        flux_reg[lev] = nullptr;
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
    solver.tag_cells(tags, U_new[lev], DynamicTerrain[lev], geom[lev], lev, time, physics_p);
}

/* ---------------------- AMReX AmrCore Base Functions  ----------------------*/


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
