#include <state/AmrMeshState.H>
#include <utils/MaskedAverageDown.H>
#include <utils/FillPatchUtils.H>
#include <boundaries/BCFill.H>
#include <utils/Logging.H>


void AmrMeshState::TerrainMapStaticToDynamic(int lev, amrex::MultiFab& dynTerrain)
{
    BL_PROFILE("AmrMeshState::TerrainMapStaticToDynamic");

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

        // One-off mapping, run when the level is created: the fillers are
        // locals because the coarse side is the static database, which has no
        // AMR level (and therefore no cached HydroPhysBC) of its own.
        HydroPhysBC cphysbc(static_geom, Terrain_bcs);
        HydroPhysBC fphysbc(amr_geom,    Terrain_bcs);

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

namespace {
// Extended __device__ lambdas cannot appear inside private member functions
// (nvcc restriction), so the FluidMap kernels live here as free functions.

// eta_s = h + z on valid cells; eta = z at NODATA so the single primitive
// h = max(0, eta - z) yields a dry cell with no branch (Rule R4).
void fill_eta_s (const amrex::MultiFab& fluid, const amrex::MultiFab& terr,
                 amrex::MultiFab& eta_s)
{
    for (amrex::MFIter mfi(eta_s, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& fl = fluid.const_array(mfi);
        auto const& tr = terr.const_array(mfi);
        auto const& es = eta_s.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::Real z = tr(i,j,k,0);
            es(i,j,k,0) = (tr(i,j,k,1) == -9999.0) ? z : fl(i,j,k,0) + z;
        });
    }
}

// h = max(0, eta - z) on valid cells (Rule R3: the one rebuild primitive).
void reconstruct_h_from_eta (amrex::MultiFab& U, const amrex::MultiFab& eta,
                             const amrex::MultiFab& terr)
{
    for (amrex::MFIter mfi(U, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& u  = U.array(mfi);
        auto const& ef = eta.const_array(mfi);
        auto const& z  = terr.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            u(i,j,k,0) = std::max(amrex::Real(0.0), ef(i,j,k,0) - z(i,j,k,0));
        });
    }
}
} // anonymous namespace

void AmrMeshState::FluidMapStaticToDynamic(int lev)
{
    BL_PROFILE("AmrMeshState::FluidMapStaticToDynamic");

    amrex::MultiFab& amr_mf = U_new[lev];

    amr_mf.setVal(0.0);

    const amrex::Geometry& amr_geom = Geom()[lev];

    // Rule R1 (plans/IdealAMR.md): h is NEVER transported across a
    // resolution change.  Build eta_s = h + z at the static layer, move the
    // free surface (constant-preserving under any interpolator), move the
    // momenta as momenta, then reconstruct h on THIS level's own z via the
    // single primitive h = max(0, eta - z).
    amrex::MultiFab eta_s(static_ba, static_dm, 1, 2);
    fill_eta_s(*StaticFluid, StaticTerrain, eta_s);
    eta_s.FillBoundary(static_geom.periodicity());

    // 2. Static is Finer than AMR: Restrict (Average Down)
    if (static_terrain_lev > lev)
    {
        amrex::IntVect ratio;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            ratio[idim] = static_cast<int>(std::round(amr_geom.CellSize(idim) / static_geom.CellSize(idim)));
        }

        // Momenta: conservative averaging, comps 1..ncomp_U-1
        amrex::average_down(*StaticFluid, amr_mf,
                            static_geom, amr_geom,
                            1, ncomp_U - 1, ratio);

        // eta: conservative averaging into a scratch, then reconstruct h
        amrex::MultiFab eta_f(amr_mf.boxArray(), amr_mf.DistributionMap(), 1, 0);
        amrex::average_down(eta_s, eta_f, static_geom, amr_geom, 0, 1, ratio);

        const amrex::Real eta_f_min = eta_f.min(0);
        const amrex::Real eta_f_max = eta_f.max(0);
        LOG_STREAM(DIAG) << "[FluidMap eta] lev=" << lev << " restrict: eta_f min/max = "
                         << eta_f_min << " / " << eta_f_max;

        reconstruct_h_from_eta(amr_mf, eta_f, DynamicTerrain[lev]);
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
            // Momenta: prolongate comps 1..ncomp_U-1 of the static fluid
            HydroPhysBC cphysbc(static_geom, U_bcs);
            HydroPhysBC fphysbc(amr_geom,    U_bcs);
            amrex::InterpFromCoarseLevel(amr_mf, dummy_time, *StaticFluid, 1, 1, ncomp_U - 1,
                                         static_geom, amr_geom,
                                         cphysbc, 1, fphysbc, 1, ratio,
                                         &amrex::cell_cons_interp, U_bcs, 1);

            // eta: prolongate the free surface into a scratch
            amrex::MultiFab eta_f(amr_mf.boxArray(), amr_mf.DistributionMap(), 1, ngrow_U);
            HydroPhysBC cphysbc_eta(static_geom, Eta_bcs);
            HydroPhysBC fphysbc_eta(amr_geom,    Eta_bcs);
            amrex::InterpFromCoarseLevel(eta_f, dummy_time, eta_s, 0, 0, 1,
                                         static_geom, amr_geom,
                                         cphysbc_eta, 0, fphysbc_eta, 0, ratio,
                                         &amrex::cell_cons_interp, Eta_bcs, 0);

            const amrex::Real eta_f_min = eta_f.min(0);
            const amrex::Real eta_f_max = eta_f.max(0);
            LOG_STREAM(DIAG) << "[FluidMap eta] lev=" << lev << " prolongate: eta_f min/max = "
                             << eta_f_min << " / " << eta_f_max;

            // Reconstruct h on this level's own bathymetry (single primitive)
            reconstruct_h_from_eta(amr_mf, eta_f, DynamicTerrain[lev]);
        }
    }

    amr_mf.FillBoundary(Geom()[lev].periodicity());
}

/* ---------------------- AMReX AmrCore Base Functions  ----------------------*/

void AmrMeshState::ComputeEta(int lev, bool from_U_old)
{
    BL_PROFILE("AmrMeshState::ComputeEta");
    DeriveEtaFrom(lev, from_U_old ? U_old[lev] : U_new[lev], Eta[lev]);
}

void AmrMeshState::DeriveEtaFrom(int lev, const amrex::MultiFab& Usrc, amrex::MultiFab& dst)
{
    BL_PROFILE("AmrMeshState::DeriveEtaFrom");
    for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // compute_fluxes_Impl swaps U_new/U_old by reference in ctx.
        // After swap + update: U_new holds the updated state (t^{n+1}),
        // U_old holds the old state (t^n).
        // Growntilebox: the fill machinery (FillPatchTwoLevels /
        // FillCoarsePatch) reads coarse-source ghosts via ParallelCopy, so
        // η must exist on the ghost ring too — Usrc carries valid ghosts.
        const amrex::Box& bx = mfi.growntilebox();
        auto const& u   = Usrc.const_array(mfi);
        auto const& z   = DynamicTerrain[lev].const_array(mfi);
        auto const& eta = dst.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            eta(i,j,k,0) = u(i,j,k,0) + z(i,j,k,0);
        });
    }

    dst.FillBoundary(Geom(lev).periodicity());
}

void AmrMeshState::MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                           const amrex::BoxArray& ba, 
                                           const amrex::DistributionMapping& dm) 
{
    BL_PROFILE("AmrMeshState::MakeNewLevelFromScratch");

    // 1. Always allocate the MultiFabs for this level
    U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
    U_old[lev].define(ba, dm, ncomp_U, ngrow_U);
    Eta[lev].define(ba, dm, 1, ngrow_U);
    EtaScratchOld[lev].define(ba, dm, 1, ngrow_U);
    EtaScratchNew[lev].define(ba, dm, 1, ngrow_U);
    DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);
    terrain_ghosts_valid[lev] = 0;   // new boxes: the ghost ring must be refilled

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    TerrainMapStaticToDynamic(lev, DynamicTerrain[lev]);
    FluidMapStaticToDynamic(lev);
    ComputeEta(lev, false);

    U_physbc[lev].define(geom[lev], U_bcs);
    Terrain_physbc[lev].define(geom[lev], Terrain_bcs);
    Eta_physbc[lev].define(geom[lev], Eta_bcs);

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

    DynamicTerrain[lev].FillBoundary(Geom(lev).periodicity());

    if (amr_p.do_reflux)
    {
        if (lev > 0) {
            flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
                ba, dm, refRatio(lev - 1), lev, ncomp_U
            );
        } else {
            flux_reg[lev] = nullptr;
        }

        // Solvers that accumulate side-selected face fluxes (Roe today) own
        // per-level buffers that must be redefined whenever the boxes change.
        // with_flux_buffers skips solvers that have none.
        solver.with_flux_buffers([&](auto& s) {
            // ResizeLevels runs in the constructor, before the solver exists, so
            // the buffers may still be unsized here.  Grow them on demand so the
            // allocation never depends on initialisation order.
            if (static_cast<int>(s.flux_fine.size()) < lev + 1) {
                s.flux_fine.resize(lev + 1);
                s.flux_crse.resize(lev + 1);
            }

            // This level acts as fine when lev > 0, and as coarse while a
            // finer level exists (on regrid, lev+1 may be created/destroyed).
            const bool has_crse = (lev > 0);
            const bool has_fine = (lev < maxLevel());

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                amrex::BoxArray face_ba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(dir));

                s.flux_fine[lev][dir].clear();
                if (has_crse) {
                    s.flux_fine[lev][dir].define(face_ba, dm, ncomp_U, 0);
                }

                s.flux_crse[lev][dir].clear();
                if (has_fine) {
                    s.flux_crse[lev][dir].define(face_ba, dm, ncomp_U, 0);
                }
            }
        });
    }
}

void AmrMeshState::PostProcessBaseGrids(amrex::BoxArray& box_array) const
{
    BL_PROFILE("AmrMeshState::PostProcessBaseGrids");

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
    BL_PROFILE("AmrMeshState::MakeNewLevelFromCoarse");

    BL_ASSERT(lev > 0);

    // Track time-stamps for standard evolution sequences
    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // 2. Allocate the brand new level structures cleanly
    U_new[lev].define(ba, dm, ncomp_U, ngrow_U);
    U_old[lev].define(ba, dm, ncomp_U, ngrow_U);
    Eta[lev].define(ba, dm, 1, ngrow_U);
    EtaScratchOld[lev].define(ba, dm, 1, ngrow_U);
    EtaScratchNew[lev].define(ba, dm, 1, ngrow_U);
    DynamicTerrain[lev].define(ba, dm, ncomp_Terrain, ngrow_Terrain);
    terrain_ghosts_valid[lev] = 0;   // new boxes: the ghost ring must be refilled

    U_physbc[lev].define(geom[lev], U_bcs);
    Terrain_physbc[lev].define(geom[lev], Terrain_bcs);
    Eta_physbc[lev].define(geom[lev], Eta_bcs);

    // We use our terrain static representation
    TerrainMapStaticToDynamic(lev, DynamicTerrain[lev]);
    FillCoarsePatchTerrain(lev, time, DynamicTerrain[lev], Terrain_bcs);

    // A newly created level must start from the coarse state.  Rule R1: the
    // free surface eta is what crosses the resolution change (constant-
    // preserving), momenta cross as momenta, and h is reconstructed on THIS
    // level's own z.  (The previous version FillCoarsePatch'd U — i.e. it
    // interpolated h — which taints eta by the h-interpolation error.)
    {
        amrex::MultiFab eta_new(ba, dm, 1, ngrow_U);
        FillCoarsePatchEta(lev, time, eta_new, Eta_bcs);
        // Momenta only: depth is rebuilt from eta on this level's own z below.
        FillCoarsePatch(lev, time, U_new[lev], U_bcs, 1, 1, ncomp_U - 1);

        const amrex::Real eta_new_min = eta_new.min(0);
        const amrex::Real eta_new_max = eta_new.max(0);
        LOG_STREAM(DIAG) << "[MakeNewLevelFromCoarse eta] lev=" << lev
                         << " eta min/max = " << eta_new_min << " / " << eta_new_max;

        for (amrex::MFIter mfi(U_new[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            auto const& u   = U_new[lev].array(mfi);
            auto const& ef  = eta_new.const_array(mfi);
            auto const& z   = DynamicTerrain[lev].const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                u(i,j,k,0) = std::max(amrex::Real(0.0), ef(i,j,k,0) - z(i,j,k,0));
            });
        }
    }
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

    // 3. Dynamic Flux Register Hook
    if (amr_p.do_reflux)
    {
        if (lev > 0) {
            flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
                ba, dm, refRatio(lev - 1), lev, ncomp_U
            );
        } else {
            flux_reg[lev] = nullptr;
        }

        // Solvers that accumulate side-selected face fluxes (Roe today) own
        // per-level buffers that must be redefined whenever the boxes change.
        // with_flux_buffers skips solvers that have none.
        solver.with_flux_buffers([&](auto& s) {
            // ResizeLevels runs in the constructor, before the solver exists, so
            // the buffers may still be unsized here.  Grow them on demand so the
            // allocation never depends on initialisation order.
            if (static_cast<int>(s.flux_fine.size()) < lev + 1) {
                s.flux_fine.resize(lev + 1);
                s.flux_crse.resize(lev + 1);
            }

            // This level acts as fine when lev > 0, and as coarse while a
            // finer level exists (on regrid, lev+1 may be created/destroyed).
            const bool has_crse = (lev > 0);
            const bool has_fine = (lev < maxLevel());

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                amrex::BoxArray face_ba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(dir));

                s.flux_fine[lev][dir].clear();
                if (has_crse) {
                    s.flux_fine[lev][dir].define(face_ba, dm, ncomp_U, 0);
                }

                s.flux_crse[lev][dir].clear();
                if (has_fine) {
                    s.flux_crse[lev][dir].define(face_ba, dm, ncomp_U, 0);
                }
            }
        });
    }
}

void AmrMeshState::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) {
    BL_PROFILE("AmrMeshState::RemakeLevel");

    amrex::MultiFab new_U(ba, dm, ncomp_U, ngrow_U);
    amrex::MultiFab old_U(ba, dm, ncomp_U, ngrow_U);
    amrex::MultiFab new_Eta(ba, dm, 1, ngrow_U);
    amrex::MultiFab new_EtaScratchOld(ba, dm, 1, ngrow_U);
    amrex::MultiFab new_EtaScratchNew(ba, dm, 1, ngrow_U);
    amrex::MultiFab new_DT(ba, dm, ncomp_Terrain, ngrow_Terrain);
    
    U_physbc[lev].define(geom[lev], U_bcs);
    Terrain_physbc[lev].define(geom[lev], Terrain_bcs);
    Eta_physbc[lev].define(geom[lev], Eta_bcs);

    TerrainMapStaticToDynamic(lev, new_DT);

    // Rule R1: transport eta (+ momenta) across the remake, reconstruct h
    // on the new level's own z.  (Previously U was FillPatch'd, i.e. h
    // was interpolated across the resolution change.)
    {
        amrex::MultiFab eta_new(ba, dm, 1, ngrow_U);
        FillPatchEta(lev, time, eta_new, Eta_bcs, 0, 1);
        FillPatch(lev, time, new_U, U_bcs, 1, ncomp_U - 1);   // momenta only

        const amrex::Real eta_new_min = eta_new.min(0);
        const amrex::Real eta_new_max = eta_new.max(0);
        
        for (amrex::MFIter mfi(new_U, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            auto const& u   = new_U.array(mfi);
            auto const& ef  = eta_new.const_array(mfi);
            auto const& z   = new_DT.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                u(i,j,k,0) = std::max(amrex::Real(0.0), ef(i,j,k,0) - z(i,j,k,0));
            });
        }
    }
        
    std::swap(new_U, U_new[lev]);
    std::swap(old_U, U_old[lev]);
    std::swap(new_Eta, Eta[lev]);
    std::swap(new_EtaScratchOld, EtaScratchOld[lev]);
    std::swap(new_EtaScratchNew, EtaScratchNew[lev]);
    std::swap(new_DT, DynamicTerrain[lev]);
    terrain_ghosts_valid[lev] = 0;   // new boxes: the ghost ring must be refilled

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

    if (amr_p.do_reflux)
    {
        if (lev > 0) {
            flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
                ba, dm, refRatio(lev - 1), lev, ncomp_U
            );
        } else {
            flux_reg[lev] = nullptr;
        }

        // Solvers that accumulate side-selected face fluxes (Roe today) own
        // per-level buffers that must be redefined whenever the boxes change.
        // with_flux_buffers skips solvers that have none.
        solver.with_flux_buffers([&](auto& s) {
            // ResizeLevels runs in the constructor, before the solver exists, so
            // the buffers may still be unsized here.  Grow them on demand so the
            // allocation never depends on initialisation order.
            if (static_cast<int>(s.flux_fine.size()) < lev + 1) {
                s.flux_fine.resize(lev + 1);
                s.flux_crse.resize(lev + 1);
            }

            // This level acts as fine when lev > 0, and as coarse while a
            // finer level exists (on regrid, lev+1 may be created/destroyed).
            const bool has_crse = (lev > 0);
            const bool has_fine = (lev < maxLevel());

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                amrex::BoxArray face_ba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(dir));

                s.flux_fine[lev][dir].clear();
                if (has_crse) {
                    s.flux_fine[lev][dir].define(face_ba, dm, ncomp_U, 0);
                }

                s.flux_crse[lev][dir].clear();
                if (has_fine) {
                    s.flux_crse[lev][dir].define(face_ba, dm, ncomp_U, 0);
                }
            }
        });
    }
}

void AmrMeshState::ClearLevel(int lev) {
    U_new[lev].clear();
    U_old[lev].clear();
    Eta[lev].clear();
    EtaScratchOld[lev].clear();
    EtaScratchNew[lev].clear();
    DynamicTerrain[lev].clear();

    // Release the cached boundary fillers with the level.  They are rebuilt by
    // whichever MakeNewLevel* / RemakeLevel recreates the level; a default
    // HydroPhysBC is inert (see its operator()).
    U_physbc[lev]       = HydroPhysBC{};
    Eta_physbc[lev]     = HydroPhysBC{};
    Terrain_physbc[lev] = HydroPhysBC{};
}

void AmrMeshState::ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{   
    BL_PROFILE("AmrMeshState::ErrorEst");

    if (lev >= max_level) return;
    solver.tag_cells(tags, U_new[lev], DynamicTerrain[lev], Geom()[lev], lev, time, physics_p);
}

/* ---------------------- AMReX AmrCore Base Functions  ----------------------*/


void AmrMeshState::FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    BL_PROFILE("AmrMeshState::FillPatch");

    if (lev == 0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        GetData(0, time, smf, stime);

        amrex::FillPatchFrom(mf, time,
                             {}, stime, smf, stime,
                             icomp, icomp, ncomp,
                             Geom()[0], Geom()[0],
                             U_physbc[0], U_physbc[0],
                             bcs, 
                             amrex::IntVect::TheZeroVector(),
                             &amrex::cell_cons_interp);
    } else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        GetData(lev-1, time, cmf, ctime);
        GetData(lev  , time, fmf, ftime);

        amrex::FillPatchFrom(mf, time,
                             cmf, ctime, fmf, ftime,
                             icomp, icomp, ncomp,
                             Geom()[lev-1], Geom()[lev],
                             U_physbc[lev-1], U_physbc[lev],
                             bcs, refRatio(lev-1));
    }
}

void AmrMeshState::FillPatchTerrain (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    BL_PROFILE("AmrMeshState::FillPatchTerrain");

    // Static terrain: the ghost ring is unchanged since the level was created
    // or regridded, so there is nothing to refill.  The flag is cleared where
    // DynamicTerrain is redefined.
    if (lev < static_cast<int>(terrain_ghosts_valid.size()) && terrain_ghosts_valid[lev]) {
        return;
    }

    if (lev == 0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        GetTerrainData(0, time, smf, stime);

        amrex::FillPatchFrom(mf, time,
                             {}, stime, smf, stime,
                             icomp, icomp, ncomp,
                             Geom()[0], Geom()[0],
                             Terrain_physbc[0], Terrain_physbc[0],
                             bcs, amrex::IntVect::TheZeroVector(),
                             &amrex::pc_interp);
    } else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        GetTerrainData(lev-1, time, cmf, ctime);
        GetTerrainData(lev  , time, fmf, ftime);

        amrex::FillPatchFrom(mf, time,
                             cmf, ctime, fmf, ftime,
                             icomp, icomp, ncomp,
                             Geom()[lev-1], Geom()[lev],
                             Terrain_physbc[lev-1], Terrain_physbc[lev],
                             bcs, refRatio(lev-1),
                             &amrex::pc_interp);
    }

    terrain_ghosts_valid[lev] = 1;
}

void AmrMeshState::FillPatchEta (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs, int icomp, int ncomp) {
    BL_PROFILE("AmrMeshState::FillPatchEta");

    if (lev == 0) {
        amrex::Vector<amrex::MultiFab*> smf;
        amrex::Vector<amrex::Real> stime;
        GetEtaData(0, time, smf, stime);

        amrex::FillPatchFrom(mf, time,
                             {}, stime, smf, stime,
                             icomp, icomp, ncomp,
                             Geom()[0], Geom()[0],
                             Eta_physbc[0], Eta_physbc[0],
                             bcs, amrex::IntVect::TheZeroVector(),
                             &amrex::cell_cons_interp);
    } else {
        amrex::Vector<amrex::MultiFab*> cmf, fmf;
        amrex::Vector<amrex::Real> ctime, ftime;
        GetEtaData(lev-1, time, cmf, ctime);
        GetEtaData(lev  , time, fmf, ftime);

        amrex::FillPatchFrom(mf, time,
                             cmf, ctime, fmf, ftime,
                             icomp, icomp, ncomp,
                             Geom()[lev-1], Geom()[lev],
                             Eta_physbc[lev-1], Eta_physbc[lev],
                             bcs, refRatio(lev-1),
                             &amrex::cell_cons_interp);
    }
}

void AmrMeshState::FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, amrex::Vector<amrex::BCRec> bcs,
                                    int scomp, int dcomp, int ncomp) {
    BL_ASSERT(lev > 0);

    if (ncomp < 0) { ncomp = ncomp_U; }

    amrex::Vector<amrex::MultiFab*> cmf;
    amrex::Vector<amrex::Real> ctime;
    GetData(lev-1, time, cmf, ctime);

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    amrex::FillCoarsePatchFrom(mf, time, *cmf[0], scomp, dcomp, ncomp,
                               Geom()[lev-1], Geom()[lev],
                               U_physbc[lev-1], U_physbc[lev],
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
                               Terrain_physbc[lev-1], Terrain_physbc[lev],
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
                               Eta_physbc[lev-1], Eta_physbc[lev],
                               bcs, refRatio(lev-1), &amrex::cell_cons_interp);
}

void AmrMeshState::AverageDown (amrex::Vector<amrex::MultiFab>& arr) {
    BL_PROFILE("AmrMeshState::AverageDown");

    for (int lev = finestLevel()-1; lev >= 0; --lev) {
        amrex::average_down(arr[lev+1], arr[lev],
                            Geom()[lev+1], Geom()[lev],
                            0, arr[lev].nComp(), refRatio(lev));
    }
}

void AmrMeshState::AverageDownTo(int crse_lev, amrex::Vector<amrex::MultiFab>&  arr) {
    BL_PROFILE("AmrMeshState::AverageDownTo");

    amrex::average_down(arr[crse_lev+1], arr[crse_lev],
                        Geom()[crse_lev+1], Geom()[crse_lev],
                        0, arr[crse_lev].nComp(), refRatio(crse_lev));
}