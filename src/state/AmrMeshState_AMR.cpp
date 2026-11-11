#include <state/AmrMeshState.H>
#include <utils/MaskedAverageDown.H>
#include <utils/FillPatchUtils.H>
#include <boundaries/BCFill.H>
#include <utils/Logging.H>
#include <AMReX_PhysBCFunct.H>


void AmrMeshState::TerrainMapStaticToDynamic(int lev, amrex::MultiFab& dynTerrain)
{

    // Ensure StaticTerrain ghost cells are filled across periodic boundaries
    StaticTerrain.FillBoundary(static_geom.periodicity());

    if (StaticTerrain.contains_nan()) {
        LOG(INFO, "StaticTerrain contains NaNs BEFORE average_down!\n");
    }

    dynTerrain.setVal(-9999.0);
    const amrex::Geometry& amr_geom = Geom()[lev];

    // 2. Static is Finer than AMR: Restrict (Average Down)
    if (static_terrain_lev > lev)
    {

        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(amr_geom.CellSize(idim) / static_geom.CellSize(idim)));
        }

        amrex::masked_average_down(StaticTerrain, dynTerrain, static_geom, amr_geom, 0, ncomp_Terrain, ratio, -9999.0);
    } 
    // 3. Static is Coarser than AMR: Prolongate (Interpolate)
    else 
    {

        // Refinement ratio: how many fine cells (AMR) fit in one coarse cell (Static)
        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(static_geom.CellSize(idim) / amr_geom.CellSize(idim)));
        }

        amrex::Real dummy_time = 0.0;


        amrex::GpuBndryFuncFab<ExternalBCFill> bndry_func(ExternalBCFill{});
        using BndryPhysBC = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExternalBCFill>>;

        // Separate physical BC objects for Coarse and Fine Geometries
        BndryPhysBC cphysbc(static_geom, Terrain_bcs, bndry_func);
        BndryPhysBC fphysbc(amr_geom,    Terrain_bcs, bndry_func);

        amrex::InterpFromCoarseLevel(
            dynTerrain,             // Destination (Fine)
            dummy_time, 
            StaticTerrain,          // Source (Coarse)
            0, 0, ncomp_Terrain,
            static_geom, amr_geom,   // Coarse Geom, Fine Geom
            cphysbc, 0,             // Coarse BC handler & comp
            fphysbc, 0,             // Fine BC handler & comp
            ratio,
            &amrex::pc_interp, 
            Terrain_bcs, 0
        );
        
    }

    dynTerrain.FillBoundary(Geom()[lev].periodicity());
}

void AmrMeshState::FluidMapStaticToDynamic(int lev)
{
    amrex::MultiFab& amr_mf = U_new[lev];
    
    amr_mf.setVal(0.0);

    const amrex::Geometry& amr_geom = Geom()[lev];

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

        {
            amrex::GpuBndryFuncFab<ExternalBCFill> bndry_func(ExternalBCFill{});
            using BndryPhysBC = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<ExternalBCFill>>;
            BndryPhysBC physbc(static_geom, U_bcs, bndry_func);
            amrex::InterpFromCoarseLevel(amr_mf, dummy_time, *StaticFluid, 0, 0, ncomp_U,
                                         static_geom, amr_geom,
                                         physbc, 0, physbc, 0, ratio,
                                         &amrex::cell_cons_interp, U_bcs, 0);
        }
    }

    amr_mf.FillBoundary(Geom()[lev].periodicity());
}

/* ---------------------- AMReX AmrCore Base Functions  ----------------------*/

void AmrMeshState::ComputeEta(int lev, bool from_U_old)
{
    for (amrex::MFIter mfi(Eta[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        // compute_fluxes_Impl swaps U_new/U_old by reference in ctx.
        // After swap + update: U_new holds the updated state (t^{n+1}),
        // U_old holds the old state (t^n).
        // from_U_old=true  → read U_old (NOT used currently, reserved)
        // from_U_old=false → read U_new (updated state, correct after compute_fluxes)
        auto const& u = from_U_old ? U_old[lev].const_array(mfi)
                                   : U_new[lev].const_array(mfi);
        auto const& z   = DynamicTerrain[lev].const_array(mfi);
        auto const& eta = Eta[lev].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            eta(i,j,k,0) = u(i,j,k,0) + z(i,j,k,0);
        });
    }
}

void AmrMeshState::MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                           const amrex::BoxArray& ba, 
                                           const amrex::DistributionMapping& dm) 
{

    // 1. Always allocate the MultiFabs for this level
    U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
    U_old[lev].define(ba, dm, ncomp_U, ngrow_U);
    Eta[lev].define(ba, dm, 1, ngrow_U);
    Eta_old[lev].define(ba, dm, 1, ngrow_U);
    DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    TerrainMapStaticToDynamic(lev, DynamicTerrain[lev]);
    FluidMapStaticToDynamic(lev);
    ComputeEta(lev, false);

    // Recompute h = max(0, eta - z) from the live free surface.
    // FluidMapStaticToDynamic and TerrainMapStaticToDynamic use different
    // interpolation methods (cell_cons_interp vs pc_interp), so h+z may not
    // equal the interpolated eta.  Reconstructing h guarantees eta = h+z
    // exactly, which is required for the C-property.
    for (amrex::MFIter mfi(U_new[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& u   = U_new[lev].array(mfi);
        auto const& eta = Eta[lev].array(mfi);
        auto const& z   = DynamicTerrain[lev].array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            u(i,j,k,0) = std::max(amrex::Real(0.0), eta(i,j,k,0) - z(i,j,k,0));
        });
    }

    U_new[lev].FillBoundary(Geom(lev).periodicity());
    U_old[lev].FillBoundary(Geom(lev).periodicity());
    Eta[lev].FillBoundary(Geom(lev).periodicity());
    Eta_old[lev].FillBoundary(Geom(lev).periodicity());
    // At t=0 there is no previous step; Eta_old = Eta initially.
    amrex::MultiFab::Copy(Eta_old[lev], Eta[lev], 0, 0, 1, 0);
    DynamicTerrain[lev].FillBoundary(Geom(lev).periodicity());

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

    BL_ASSERT(lev > 0);

    // Track time-stamps for standard evolution sequences
    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // 2. Allocate the brand new level structures cleanly
    U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
    U_old[lev].define(ba, dm, ncomp_U, ngrow_U);
    Eta[lev].define(ba, dm, 1, ngrow_U);
    Eta_old[lev].define(ba, dm, 1, ngrow_U);
    DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);
        
    // We use our terrain static representation
    TerrainMapStaticToDynamic(lev, DynamicTerrain[lev]);

    FillCoarsePatchTerrain(lev, time, DynamicTerrain[lev], Terrain_bcs);

    // A newly created level must start from the coarse state.  This call was
    // missing, which left U_new uninitialised until the first FillPatch.
    // Eta is then derived from that state instead of being interpolated on its
    // own, so h and eta are consistent by construction.
    FillCoarsePatch(lev, time, U_new[lev], U_bcs);
    ComputeEta(lev, false);

    // Recompute h = max(0, eta - z) from the live free surface.
    // FillCoarsePatch and FillCoarsePatchTerrain use different interpolation
    // methods, so h+z may not equal the interpolated eta.  Reconstructing h
    // guarantees eta = h+z exactly, which is required for the C-property.
    for (amrex::MFIter mfi(U_new[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& u   = U_new[lev].array(mfi);
        auto const& eta = Eta[lev].array(mfi);
        auto const& z   = DynamicTerrain[lev].array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            u(i,j,k,0) = std::max(amrex::Real(0.0), eta(i,j,k,0) - z(i,j,k,0));
        });
    }

    U_new[lev].FillBoundary(Geom(lev).periodicity());
    U_old[lev].FillBoundary(Geom(lev).periodicity());
    Eta[lev].FillBoundary(Geom(lev).periodicity());
    Eta_old[lev].FillBoundary(Geom(lev).periodicity());
    // At creation there is no previous step; Eta_old = Eta initially.
    amrex::MultiFab::Copy(Eta_old[lev], Eta[lev], 0, 0, 1, 0);

    // 3. Dynamic Flux Register Hook
    if (lev > 0 && amr_p.do_reflux) {
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba, dm, refRatio(lev - 1), lev, ncomp_U
        );
    } else {
        flux_reg[lev] = nullptr;
    }

}

void AmrMeshState::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {

    amrex::MultiFab new_U(ba, dm, ncomp_U, ngrow_U);
    amrex::MultiFab old_U(ba, dm, ncomp_U, ngrow_U);
    amrex::MultiFab new_Eta(ba, dm, 1, ngrow_U);
    amrex::MultiFab new_Eta_old(ba, dm, 1, ngrow_U);
    amrex::MultiFab new_DT(ba, dm, ncomp_Terrain, ngrow_Terrain);
    
    FillPatch(lev, time, new_U, U_bcs, 0, ncomp_U);
    TerrainMapStaticToDynamic(lev, new_DT);
        
    std::swap(new_U, U_new[lev]);
    std::swap(old_U, U_old[lev]);
    std::swap(new_Eta, Eta[lev]);
    std::swap(new_Eta_old, Eta_old[lev]);
    std::swap(new_DT, DynamicTerrain[lev]);

    ComputeEta(lev, false);

    // Recompute h = max(0, eta - z) from the live free surface.
    for (amrex::MFIter mfi(U_new[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& u   = U_new[lev].array(mfi);
        auto const& eta = Eta[lev].array(mfi);
        auto const& z   = DynamicTerrain[lev].array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            u(i,j,k,0) = std::max(amrex::Real(0.0), eta(i,j,k,0) - z(i,j,k,0));
        });
    }

    t_new[lev] = time; 
    t_old[lev] = time - 1.e200;

    // No previous step at remake; Eta_old = Eta initially.
    amrex::MultiFab::Copy(Eta_old[lev], Eta[lev], 0, 0, 1, 0);

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
    Eta[lev].clear();
    Eta_old[lev].clear();
    DynamicTerrain[lev].clear();
}

void AmrMeshState::ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{   
    if (lev >= max_level) return;
    solver.tag_cells(tags, U_new[lev], DynamicTerrain[lev], Geom()[lev], lev, time, physics_p);
}

/* ---------------------- AMReX AmrCore Base Functions  ----------------------*/


void AmrMeshState::FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    if (lev == 0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        GetData(0, time, smf, stime);

        amrex::FillPatchFrom(mf, time,
                             {}, stime, smf, stime,
                             0, icomp, ncomp,
                             Geom()[0], Geom()[0],
                             bcs, amrex::IntVect::TheZeroVector(),
                             &amrex::cell_cons_interp);
    } else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        GetData(lev-1, time, cmf, ctime);
        GetData(lev  , time, fmf, ftime);

        amrex::FillPatchFrom(mf, time,
                             cmf, ctime, fmf, ftime,
                             0, icomp, ncomp,
                             Geom()[lev-1], Geom()[lev],
                             bcs, refRatio(lev-1));
    }
}

void AmrMeshState::FillPatchTerrain (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    if (lev == 0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        GetTerrainData(0, time, smf, stime);

        amrex::FillPatchFrom(mf, time,
                             {}, stime, smf, stime,
                             0, icomp, ncomp,
                             Geom()[0], Geom()[0],
                             bcs, amrex::IntVect::TheZeroVector(),
                             &amrex::pc_interp);
    } else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        GetTerrainData(lev-1, time, cmf, ctime);
        GetTerrainData(lev  , time, fmf, ftime);

        amrex::FillPatchFrom(mf, time,
                             cmf, ctime, fmf, ftime,
                             0, icomp, ncomp,
                             Geom()[lev-1], Geom()[lev],
                             bcs, refRatio(lev-1),
                             &amrex::pc_interp);
    }
}

void AmrMeshState::FillPatchEta (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    if (lev == 0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        GetEtaData(0, time, smf, stime);

        amrex::FillPatchFrom(mf, time,
                             {}, stime, smf, stime,
                             0, icomp, ncomp,
                             Geom()[0], Geom()[0],
                             bcs, amrex::IntVect::TheZeroVector(),
                             &amrex::cell_cons_interp);
    } else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        GetEtaData(lev-1, time, cmf, ctime);
        GetEtaData(lev  , time, fmf, ftime);

        amrex::FillPatchFrom(mf, time,
                             cmf, ctime, fmf, ftime,
                             0, icomp, ncomp,
                             Geom()[lev-1], Geom()[lev],
                             bcs, refRatio(lev-1),
                             &amrex::cell_cons_interp);
    }
}

void AmrMeshState::FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs) {
    BL_ASSERT(lev > 0);

    amrex::Vector<amrex::MultiFab*> cmf;
    amrex::Vector<amrex::Real> ctime;
    GetData(lev-1, time, cmf, ctime);

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    amrex::FillCoarsePatchFrom(mf, time, *cmf[0], 0, 0, ncomp_U,
                               Geom()[lev-1], Geom()[lev],
                               bcs, refRatio(lev-1));
}


void AmrMeshState::FillCoarsePatchTerrain (
        int lev, amrex::Real time, amrex::MultiFab& mf,  amrex::Vector<amrex::BCRec> bcs
) {
    BL_ASSERT(lev > 0);

    amrex::Vector<amrex::MultiFab*> cmf;
    amrex::Vector<amrex::Real> ctime;
    GetTerrainData(lev-1, time, cmf, ctime);

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatchTerrain: how did this happen?");
    }

    amrex::FillCoarsePatchFrom(mf, time, *cmf[0], 0, 0, ncomp_Terrain,
                               Geom()[lev-1], Geom()[lev],
                               bcs, refRatio(lev-1), &amrex::pc_interp);
}

void AmrMeshState::FillCoarsePatchEta (
    int lev, amrex::Real time, amrex::MultiFab& mf,  amrex::Vector<amrex::BCRec> bcs
) {
    BL_ASSERT(lev > 0);

    amrex::Vector<amrex::MultiFab*> cmf;
    amrex::Vector<amrex::Real> ctime;
    GetEtaData(lev-1, time, cmf, ctime);

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatchEta: how did this happen?");
    }

    amrex::FillCoarsePatchFrom(mf, time, *cmf[0], 0, 0, 1,
                               Geom()[lev-1], Geom()[lev],
                               bcs, refRatio(lev-1), &amrex::cell_cons_interp);
}

void AmrMeshState::FillCoarsePatchMomentum (
    int lev, amrex::Real time, amrex::MultiFab& mf,  amrex::Vector<amrex::BCRec> bcs
) {
    BL_ASSERT(lev > 0);

    amrex::Vector<amrex::MultiFab*> cmf;
    amrex::Vector<amrex::Real> ctime;
    GetData(lev-1, time, cmf, ctime);

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatchMomentum: how did this happen?");
    }

    amrex::FillCoarsePatchFrom(mf, time, *cmf[0], 1, 1, 2,
                               Geom()[lev-1], Geom()[lev],
                               bcs, refRatio(lev-1));
}



void AmrMeshState::AverageDown (amrex::Vector<amrex::MultiFab>& arr) {
    for (int lev = finestLevel()-1; lev >= 0; --lev) {
        amrex::average_down(arr[lev+1], arr[lev],
                            Geom()[lev+1], Geom()[lev],
                            0, arr[lev].nComp(), refRatio(lev));
    }
}

void AmrMeshState::AverageDownTo(int crse_lev, amrex::Vector<amrex::MultiFab>&  arr) {
    amrex::average_down(arr[crse_lev+1], arr[crse_lev],
                        Geom()[crse_lev+1], Geom()[crse_lev],
                        0, arr[crse_lev].nComp(), refRatio(crse_lev));
}
