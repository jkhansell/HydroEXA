#include <hydroexa/HydroEXA.H>
#include <hydroexa/RoeSolver.H>

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunchFunctsC.H>
#include <AMReX_iMultiFab.H>

using namespace amrex;

Real
HydroEXA::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("HydroEXA::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    // Since Terrain_Type is static, copy the valid old data to the new data after swap
    MultiFab::Copy(get_new_data(Terrain_Type), get_old_data(Terrain_Type), 0, 0, NUM_TERRAIN, 0);

    MultiFab& StateNew = get_new_data(State_Type);
    MultiFab& TerrainNew = get_new_data(Terrain_Type);
    
    // Obtain grid metrics from the level's underlying Geometry object
    const auto& geom = Geom();
    const Real dx = geom.CellSize(0);
    const Real dy = geom.CellSize(1);

    FluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        HydroEXA& fine_level = getLevel(level+1);
        fr_as_crse = fine_level.flux_reg.get();
    }

    FluxRegister* fr_as_fine = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = flux_reg.get();
    }

    if (fr_as_crse) {
        fr_as_crse->setVal(Real(0.0));
    }

    MultiFab StateBorder(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());
    FillPatch(*this, StateBorder, NUM_GROW, time, State_Type, 0, NUM_STATE);

    MultiFab TerrainBorder(grids,dmap,NUM_TERRAIN,NUM_GROW,MFInfo(),Factory());
    FillPatch(*this, TerrainBorder, NUM_GROW, time, Terrain_Type, 0, NUM_TERRAIN);

    // ------------------------------------------------------------------------
    // 1. ALLOCATE DUAL FLUXES
    // ------------------------------------------------------------------------
    MultiFab fx_minus, fx_plus;
    MultiFab fy_minus, fy_plus;

    {
        // X-direction face allocations (nodal in 0)
        BoxArray ba_x = grids;
        ba_x.surroundingNodes(0);
        fx_minus.define(ba_x, dmap, NUM_STATE, 1);
        fx_plus.define(ba_x,  dmap, NUM_STATE, 1);

        // Y-direction face allocations (nodal in 1)
        BoxArray ba_y = grids;
        ba_y.surroundingNodes(1);
        fy_minus.define(ba_y, dmap, NUM_STATE, 1);
        fy_plus.define(ba_y,  dmap, NUM_STATE, 1);
    }

    // ------------------------------------------------------------------------
    // 2. COMPUTE DIRECTIONAL ADVANCED RIEMANN DISCONTINUITIES
    // ------------------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(StateNew, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx_x = mfi.nodaltilebox(0); 
            Box bx_y = mfi.nodaltilebox(1); 

            Array4<Real const> const& statein = StateBorder.array(mfi);
            Array4<Real const> const& terrain = TerrainBorder.array(mfi);

            Array4<Real> const& fx_m = fx_minus.array(mfi);
            Array4<Real> const& fx_p = fx_plus.array(mfi);
            Array4<Real> const& fy_m = fy_minus.array(mfi);
            Array4<Real> const& fy_p = fy_plus.array(mfi);

            // X-fluxes pass (dir = 0)
            compute_amrex_effective_fluxes(bx_x, statein, terrain, fx_m, fx_p, dt, dx, 0);
            
            // Y-fluxes pass (dir = 1)
            compute_amrex_effective_fluxes(bx_y, statein, terrain, fy_m, fy_p, dt, dy, 1);
        }
    }

    // Add this right before Section 3:
    fx_minus.FillBoundary(geom.periodicity());
    fx_plus.FillBoundary(geom.periodicity());
    fy_minus.FillBoundary(geom.periodicity());
    fy_plus.FillBoundary(geom.periodicity());

    // ------------------------------------------------------------------------
    // 3. EXECUTE LOCAL CONSERVATIVE CELL UPDATE STEP
    // ------------------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(StateNew, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            Array4<Real const> const& fx_m = fx_minus.array(mfi);
            Array4<Real const> const& fx_p = fx_plus.array(mfi);
            Array4<Real const> const& fy_m = fy_minus.array(mfi);
            Array4<Real const> const& fy_p = fy_plus.array(mfi);
            
            Array4<Real const> const& state_old = StateBorder.array(mfi);
            Array4<Real> const&       state_new = StateNew.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
            {
                for (int n = 0; n < NUM_STATE; ++n) {
                    state_new(i,j,k,n) = state_old(i,j,k,n)
                        - (dt / dx) * (fx_m(i+1, j, k, n) - fx_p(i, j, k, n))
                        - (dt / dy) * (fy_m(i, j+1, k, n) - fy_p(i, j, k, n));
                }
            });
        }
    }

    // ------------------------------------------------------------------------
    // 4. MAP DUAL FLUXES INTO AMREX SINGLE-VALUED FLUX REGISTERS VIA MASK
    // ------------------------------------------------------------------------
    if (do_reflux) 
    {
        MultiFab fluxes_crse[AMREX_SPACEDIM];
        MultiFab fluxes_fine[AMREX_SPACEDIM];

        if (fr_as_crse) {
            BoxArray ba_x = grids;
            ba_x.surroundingNodes(0);
            fluxes_crse[0].define(ba_x, dmap, NUM_STATE, 0);
            fluxes_crse[0].setVal(0.0);

            BoxArray ba_y = grids;
            ba_y.surroundingNodes(1);
            fluxes_crse[1].define(ba_y, dmap, NUM_STATE, 0);
            fluxes_crse[1].setVal(0.0);
        }

        if (fr_as_fine) {
            BoxArray ba_x = grids;
            ba_x.surroundingNodes(0);
            fluxes_fine[0].define(ba_x, dmap, NUM_STATE, 0);
            fluxes_fine[0].setVal(0.0);

            BoxArray ba_y = grids;
            ba_y.surroundingNodes(1);
            fluxes_fine[1].define(ba_y, dmap, NUM_STATE, 0);
            fluxes_fine[1].setVal(0.0);
        }

        if (fr_as_crse)
        {
            // 1 = Valid internal cell on this AMR level, 0 = Covered by coarse cell / boundary
            iMultiFab fine_mask = makeFineMask(StateNew, getLevel(level+1).grids, parent->refRatio(level), 0, 1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            {
                for (MFIter mfi(StateNew, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box bx_x = mfi.nodaltilebox(0);
                    Box bx_y = mfi.nodaltilebox(1);

                    auto const& mask = fine_mask.array(mfi);
                    
                    auto const& fx_m   = fx_minus.array(mfi);
                    auto const& fx_p   = fx_plus.array(mfi);
                    auto const& reg_x  = fluxes_crse[0].array(mfi);

                    auto const& fy_m   = fy_minus.array(mfi);
                    auto const& fy_p   = fy_plus.array(mfi);
                    auto const& reg_y  = fluxes_crse[1].array(mfi);
      
                    amrex::ParallelFor(bx_x,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int left_is_fine  = mask(i-1, j, k);
                        int right_is_fine = mask(i,   j, k);

                        for (int n = 0; n < NUM_STATE; ++n) {
                            if (left_is_fine == 1 && right_is_fine == 0) {
                                reg_x(i, j, k, n) = fx_p(i, j, k, n); // Coarse cell is right, receives fx_p
                            } else if (left_is_fine == 0 && right_is_fine == 1) {
                                reg_x(i, j, k, n) = fx_m(i, j, k, n); // Coarse cell is left, leaves fx_m
                            } else {
                                reg_x(i, j, k, n) = 0.0;
                            }
                        }
                    });

                    amrex::ParallelFor(bx_y,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int bottom_is_fine = mask(i, j-1, k);
                        int top_is_fine    = mask(i, j,   k);

                        for (int n = 0; n < NUM_STATE; ++n) {
                            if (bottom_is_fine == 1 && top_is_fine == 0) {
                                reg_y(i, j, k, n) = fy_p(i, j, k, n); // Coarse cell is top, receives fy_p
                            } else if (bottom_is_fine == 0 && top_is_fine == 1) {
                                reg_y(i, j, k, n) = fy_m(i, j, k, n); // Coarse cell is bottom, leaves fy_m
                            } else {
                                reg_y(i, j, k, n) = 0.0;
                            }
                        }
                    });
                }
            }
        }

        if (fr_as_fine)
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            {
                for (MFIter mfi(StateNew, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box bx_x = mfi.nodaltilebox(0);
                    Box bx_y = mfi.nodaltilebox(1);
                    Box bx_grid = mfi.validbox();

                    auto const& fx_m   = fx_minus.array(mfi);
                    auto const& fx_p   = fx_plus.array(mfi);
                    auto const& reg_x  = fluxes_fine[0].array(mfi);

                    auto const& fy_m   = fy_minus.array(mfi);
                    auto const& fy_p   = fy_plus.array(mfi);
                    auto const& reg_y  = fluxes_fine[1].array(mfi);
      
                    amrex::ParallelFor(bx_x,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int is_low_boundary = (i == bx_grid.smallEnd(0));
                        int is_high_boundary = (i == bx_grid.bigEnd(0) + 1);

                        for (int n = 0; n < NUM_STATE; ++n) {
                            if (is_low_boundary) {
                                reg_x(i, j, k, n) = fx_p(i, j, k, n); // Western boundary: fine cell is right (i), uses fx_p
                            } else if (is_high_boundary) {
                                reg_x(i, j, k, n) = fx_m(i, j, k, n); // Eastern boundary: fine cell is left (i-1), uses fx_m
                            } else {
                                reg_x(i, j, k, n) = 0.0;
                            }
                        }
                    });

                    amrex::ParallelFor(bx_y,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int is_low_boundary = (j == bx_grid.smallEnd(1));
                        int is_high_boundary = (j == bx_grid.bigEnd(1) + 1);

                        for (int n = 0; n < NUM_STATE; ++n) {
                            if (is_low_boundary) {
                                reg_y(i, j, k, n) = fy_p(i, j, k, n); // Southern boundary: fine cell is top (j), uses fy_p
                            } else if (is_high_boundary) {
                                reg_y(i, j, k, n) = fy_m(i, j, k, n); // Northern boundary: fine cell is bottom (j-1), uses fy_m
                            } else {
                                reg_y(i, j, k, n) = 0.0;
                            }
                        }
                    });
                }
            }
        }

        // ------------------------------------------------------------------------
        // 5. SYNCHRONIZE WITH AMREX FLUX REGISTERS
        // ------------------------------------------------------------------------
        if (fr_as_crse) {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                // In 2D, the face area (length) is the spacing of the OTHER dimension
                const Real dA = (idim == 0) ? dy : dx;
                const Real scale = -dt * dA;
                
                // Note: Classic FluxRegister uses FluxRegister::ADD or FluxRegister::COPY
                fr_as_crse->CrseInit(fluxes_crse[idim], idim, 0, 0, NUM_STATE, scale, FluxRegister::ADD);
            }
        }

        if (fr_as_fine) {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                const Real dA = (idim == 0) ? dy : dx;
                const Real scale = dt * dA;
                
                fr_as_fine->FineAdd(fluxes_fine[idim], idim, 0, 0, NUM_STATE, scale);
            }
        }
    }

    return dt;
}

// =========================================================================
// DIRECTIONAL FLUX WRAPPER (CHOOSE COMPONENT: 0 = X-Faces, 1 = Y-Faces)
// =========================================================================
void HydroEXA::compute_amrex_effective_fluxes(
    const Box& face_box,
    Array4<Real const> const& U, 
    Array4<Real const> const& terrain, 
    Array4<Real> const& D_minus_mf, 
    Array4<Real> const& D_plus_mf, 
    const Real dt, const Real dx, const int dir)
{
    const Real nx = (dir == 0) ? 1.0 : 0.0;
    const Real ny = (dir == 1) ? 1.0 : 0.0;

    const int offset_x = (dir == 0) ? 1 : 0;
    const int offset_y = (dir == 1) ? 1 : 0;

    amrex::ParallelFor(face_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        int li = i - offset_x; int lj = j - offset_y;
        int ri = i;            int rj = j;

        Real hi  = U(li, lj, k, H);
        Real hui = U(li, lj, k, HU);
        Real hvi = U(li, lj, k, HV);
        Real zi  = terrain(li, lj, k, 0);
        Real ni  = terrain(li, lj, k, 1);

        Real hj  = U(ri, rj, k, H);
        Real huj = U(ri, rj, k, HU);
        Real hvj = U(ri, rj, k, HV);
        Real zj  = terrain(ri, rj, k, 0);
        Real nj  = terrain(ri, rj, k, 1);

        Real D_minus[3];
        Real D_plus[3];

        roeSolver(hi, hui, hvi, zi, ni,
                  hj, huj, hvj, zj, nj,
                  D_minus, D_plus, dt, dx, nx, ny);

        for (int comp = 0; comp < 3; ++comp) {
            D_minus_mf(i, j, k, comp) = D_minus[comp];
            D_plus_mf(i, j, k, comp)  = D_plus[comp];
        }
    });
}