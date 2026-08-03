#include <solvers/Roe.H>
#include <solvers/kernels/Roe.H>

void
Roe::compute_amrex_effective_fluxes(
    const amrex::Box& face_box,
    amrex::Array4<amrex::Real const> const& U,
    amrex::Array4<amrex::Real const> const& terrain,
    amrex::Array4<amrex::Real> const& D_minus_mf,
    amrex::Array4<amrex::Real> const& D_plus_mf,
    const amrex::Real dt, const amrex::Real dx, const int dir)
{
    const amrex::Real nx = (dir == 0) ? 1.0 : 0.0;
    const amrex::Real ny = (dir == 1) ? 1.0 : 0.0;

    const int offset_x = (dir == 0) ? 1 : 0;
    const int offset_y = (dir == 1) ? 1 : 0;

    amrex::ParallelFor(face_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        int ri = i;            int rj = j;
        int li = i - offset_x; int lj = j - offset_y;

        amrex::Real hi  = U(li, lj, k, 0);
        amrex::Real hui = U(li, lj, k, 1);
        amrex::Real hvi = U(li, lj, k, 2);
        amrex::Real zi  = terrain(li, lj, k, 0);
        amrex::Real ni  = terrain(li, lj, k, 1);

        amrex::Real hj  = U(ri, rj, k, 0);
        amrex::Real huj = U(ri, rj, k, 1);
        amrex::Real hvj = U(ri, rj, k, 2);
        amrex::Real zj  = terrain(ri, rj, k, 0);
        amrex::Real nj  = terrain(ri, rj, k, 1);

        amrex::Real D_minus[3], D_plus[3];
        roeSolver(hi, hui, hvi, zi, ni,
                  hj, huj, hvj, zj, nj,
                  D_minus, D_plus, dt, dx, nx, ny);

        for (int c = 0; c < 3; ++c) {
            D_minus_mf(i, j, k, c) = D_minus[c];
            D_plus_mf(i, j, k, c) = D_plus[c];
        }
    });
}

amrex::Real
Roe::compute_dt_Impl(const amrex::Vector<amrex::MultiFab>& U,
                     const amrex::Vector<amrex::MultiFab>& terrain,
                     const amrex::Vector<amrex::Geometry>& geom,
                     int lev, amrex::Real cfl) {
    BL_PROFILE("Roe::compute_dt_Impl");

    amrex::Print() << "ComputeDT: "
    << "Level " << lev
    << " h=["
    << U[lev].min(0) << ", "
    << U[lev].max(0) << "] "
    << "hu=["
    << U[lev].min(1) << ", "
    << U[lev].max(1) << "] "
    << "hv=["
    << U[lev].min(2) << ", "
    << U[lev].max(2) << "]\n";

    // 1. Use the incoming function parameters instead of hidden mesh_state dependencies
    const auto dx = geom[lev].CellSizeArray();
    const amrex::MultiFab& S = U[lev];
    const amrex::MultiFab& Terrain = terrain[lev];

    amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);

    using ReduceTuple = decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {
        for (amrex::MFIter mfi(S, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            // Rename to avoid shadowing the argument parameter vector 'U'
            const auto& arr = S.const_array(mfi);
            const auto& z_arr   = Terrain.array(mfi);
            
            const amrex::Real dxx = dx[0];
            const amrex::Real dyy = dx[1];
            
            reduce_op.eval(bx, reduce_data, 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    
                    if (std::abs(z_arr(i,j,k,0) - (-9999.0)) < Tolerances::TOL10()) {
                        return 1.e30;
                    }

                    amrex::Real h = arr(i,j,k,0);
                    amrex::Real dt_cell = 1.e6; 
                    
                    if (h > 1.e-12) { 
                        amrex::Real u = arr(i,j,k,1) / h;
                        amrex::Real v = arr(i,j,k,2) / h;
                        // Fixed: Correct namespace resolution for GRAV
                        amrex::Real c = std::sqrt(PhysConstants::GRAV() * h); 

                        amrex::Real dt_x = dxx / (std::abs(u) + c);
                        amrex::Real dt_y = dyy / (std::abs(v) + c);

                        dt_cell = amrex::min(dt_x, dt_y);
                    }
                    return dt_cell;
                }
            );
        }
    }

    // Extract the LOCAL minimum from the GPU to the CPU
    ReduceTuple local_dt = reduce_data.value();
    amrex::Real dt_est = amrex::get<0>(local_dt);

    // Apply CFL multiplier to the global safe estimate
    dt_est *= cfl;

    return dt_est;
}

void Roe::tag_cells_Impl(amrex::TagBoxArray& tags,
                         const amrex::MultiFab& U,
                         const amrex::MultiFab& terrain,
                         const amrex::Geometry& geom,
                         int lev,
                         amrex::Real time,
                         const PhysicsParameters& phys_params)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real z_thresh = phys_params.z_grad_thresh[lev];
    const amrex::Real h_thresh = phys_params.h_grad_thresh[lev];
    const amrex::Real u_thresh = phys_params.umag_grad_thresh[lev];

    for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto tag_arr = tags.array(mfi);
        const auto u_arr   = U.array(mfi);
        const auto z_arr   = terrain.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Check NODATA bounds (-9999.0)
            if (z_arr(i+1,j,k,0) == -9999.0 || z_arr(i-1,j,k,0) == -9999.0 ||
                z_arr(i,j+1,k,0) == -9999.0 || z_arr(i,j-1,k,0) == -9999.0) {
                return;
            }

            bool tag_cell = false;

            // 1. Initial Step (t = 0): Tag based on Bathymetry Slope
            if (time == 0.0) {
                amrex::Real dzdx = (z_arr(i+1,j,k,0) - z_arr(i-1,j,k,0)) / (2.0 * dx[0]);
                amrex::Real dzdy = (z_arr(i,j+1,k,0) - z_arr(i,j-1,k,0)) / (2.0 * dx[1]);
                if (std::sqrt(dzdx*dzdx + dzdy*dzdy) > z_thresh) {
                    tag_cell = true;
                }
            } 
            // 2. Dynamic Steps (t > 0): Tag based on Fluid Depth (h) & Velocity Magnitude (|u|) Gradients
            else {
                // Component 0: h, Component 1: hu, Component 2: hv
                amrex::Real h_c = u_arr(i,j,k,0);
                
                // Depth gradients
                amrex::Real dhdx = (u_arr(i+1,j,k,0) - u_arr(i-1,j,k,0)) / (2.0 * dx[0]);
                amrex::Real dhdy = (u_arr(i,j+1,k,0) - u_arr(i,j-1,k,0)) / (2.0 * dx[1]);
                if (std::sqrt(dhdx*dhdx + dhdy*dhdy) > h_thresh) {
                    tag_cell = true;
                }

                // Velocity magnitude gradients (using small epsilon for dry bed safety)
                if (!tag_cell) {
                    constexpr amrex::Real eps = 1.0e-6;
                    
                    auto get_u_mag = [=](int ii, int jj, int kk) {
                        amrex::Real h  = u_arr(ii,jj,kk,0);
                        amrex::Real hu = u_arr(ii,jj,kk,1);
                        amrex::Real hv = u_arr(ii,jj,kk,2);
                        if (h < eps) return 0.0;
                        amrex::Real u = hu / h;
                        amrex::Real v = hv / h;
                        return std::sqrt(u*u + v*v);
                    };

                    amrex::Real u_mag_E = get_u_mag(i+1, j, k);
                    amrex::Real u_mag_W = get_u_mag(i-1, j, k);
                    amrex::Real u_mag_N = get_u_mag(i, j+1, k);
                    amrex::Real u_mag_S = get_u_mag(i, j-1, k);

                    amrex::Real dudx = (u_mag_E - u_mag_W) / (2.0 * dx[0]);
                    amrex::Real dudy = (u_mag_N - u_mag_S) / (2.0 * dx[1]);

                    if (std::sqrt(dudx*dudx + dudy*dudy) > u_thresh) {
                        tag_cell = true;
                    }
                }
            }

            if (tag_cell) {
                tag_arr(i, j, k) = amrex::TagBox::SET;
            }
        });
    }
}

void
Roe::compute_fluxes_Impl(SolverContext ctx, int lev, amrex::Real dt, amrex::Real time) {
    
    std::swap(ctx.U_new[lev], ctx.U_old[lev]);

    amrex::MultiFab& U_n = ctx.U_new[lev];
    amrex::MultiFab& U_o = ctx.U_old[lev];
    amrex::MultiFab& Terrain  = ctx.Terrain[lev];

    AMREX_ASSERT_WITH_MESSAGE(U_n.nComp() >= 3,
        "Roe solver requires at least 3 state components (H, HU, HV) but U_n.nComp() = "
        + std::to_string(U_n.nComp()));


    const auto& amr_geom = ctx.geom[lev];
    const amrex::Real dx = amr_geom.CellSize(0);
    const amrex::Real dy = amr_geom.CellSize(1);

    // ------------------------------------------------------------------------
    // 1. FLUX REGISTER POINTERS (AmrCore-style)
    //    flux_reg[lev]     = register lev receives fluxes FROM (lev-1 -> lev)
    //    flux_reg[lev + 1] = register lev contributes fluxes TO   (lev -> lev+1)
    // ------------------------------------------------------------------------
    amrex::FluxRegister* fr_as_crse = nullptr;
    if (ctx.do_reflux && lev < ctx.finest_level) {
        fr_as_crse = ctx.FluxRegisters[lev+1].get();
    }

    amrex::FluxRegister* fr_as_fine = nullptr;
    if (ctx.do_reflux && lev > 0) {
        fr_as_fine = ctx.FluxRegisters[lev].get();
    }

    // ------------------------------------------------------------------------
    // 2. FILL PATCH (ghost cells)
    // ------------------------------------------------------------------------
    // amrex::MultiFab U_border(ctx.grids[lev], ctx.dmap[lev], U_n.nComp(), U_n.nGrow());
    // ctx.FillPatch(lev, time, U_border, ctx.UBCs, 0, U_n.nComp());

    // amrex::MultiFab Terrain_border(ctx.grids[lev], ctx.dmap[lev], Terrain.nComp(), Terrain.nGrow());
    // ctx.FillPatch(lev, time, Terrain_border, ctx.TerrainBCs, 0, Terrain.nComp());

    // ------------------------------------------------------------------------
    // 3. ALLOCATE DUAL FLUXES (nodal in flux direction)
    // ------------------------------------------------------------------------
    amrex::MultiFab fx_minus, fx_plus;
    amrex::MultiFab fy_minus, fy_plus;

    {
        amrex::BoxArray ba_x = ctx.grids[lev];
        ba_x.surroundingNodes(0);
        fx_minus.define(ba_x, ctx.dmap[lev], U_n.nComp(), 1);
        fx_plus.define(ba_x, ctx.dmap[lev], U_n.nComp(), 1);

        amrex::BoxArray ba_y = ctx.grids[lev];
        ba_y.surroundingNodes(1);
        fy_minus.define(ba_y, ctx.dmap[lev], U_n.nComp(), 1);
        fy_plus.define(ba_y, ctx.dmap[lev], U_n.nComp(), 1);

        fx_minus.setVal(0.0);
        fy_minus.setVal(0.0);
        fx_plus.setVal(0.0);
        fy_plus.setVal(0.0);

        // Fill boundary for fluxes (needed for coarse-fine interface detection)
        fx_minus.FillBoundary(amr_geom.periodicity());
        fx_plus.FillBoundary(amr_geom.periodicity());
        fy_minus.FillBoundary(amr_geom.periodicity());
        fy_plus.FillBoundary(amr_geom.periodicity());
    }

    // ------------------------------------------------------------------------
    // 4. COMPUTE DIRECTIONAL ADVANCED RIEMANN FLUXES
    // ------------------------------------------------------------------------
    // Capture for device lambda capture (avoid capturing `this`)
    const amrex::Real dx_local = dx;
    const amrex::Real dy_local = dy;
    const amrex::Real dt_local = dt;


    U_o.FillBoundary(amr_geom.periodicity());
    Terrain.FillBoundary(amr_geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (amrex::MFIter mfi(U_o, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box bx_x = mfi.nodaltilebox(0);
            amrex::Box bx_y = mfi.nodaltilebox(1);

            auto const& statein = U_o.array(mfi);
            auto const& z_arr = Terrain.array(mfi);

            compute_amrex_effective_fluxes(
                bx_x, statein, z_arr,
                fx_minus.array(mfi), fx_plus.array(mfi),
                dt_local, dx_local, 0
            );

            compute_amrex_effective_fluxes(
                bx_y, statein, z_arr,
                fy_minus.array(mfi), fy_plus.array(mfi),
                dt_local, dy_local, 1
            );
        }
    }

    // Fill boundary for fluxes (needed for coarse-fine interface detection)
    fx_minus.FillBoundary(amr_geom.periodicity());
    fx_plus.FillBoundary(amr_geom.periodicity());
    fy_minus.FillBoundary(amr_geom.periodicity());
    fy_plus.FillBoundary(amr_geom.periodicity());

    // ------------------------------------------------------------------------
    // 5. LOCAL CONSERVATIVE CELL UPDATE
    // ------------------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (amrex::MFIter mfi(U_o, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const int ncomp = U_o.nComp();

            const amrex::Box& bx = mfi.validbox();
            auto const& fx_m = fx_minus.array(mfi);
            auto const& fx_p = fx_plus.array(mfi);
            auto const& fy_m = fy_minus.array(mfi);
            auto const& fy_p = fy_plus.array(mfi);
            auto const& u_o  = U_o.array(mfi);
            auto const& u_n  = U_n.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                for (int n = 0; n < ncomp; ++n) {
                    u_n(i, j, k, n) = u_o(i, j, k, n)
                        - (dt_local / dx) * (fx_m(i + 1, j, k, n) - fx_p(i, j, k, n))
                        - (dt_local / dy) * (fy_m(i, j + 1, k, n) - fy_p(i, j, k, n));
                }
            });
        }
    }

    amrex::Print()
        << "Level " << lev
        << " h=["
        << U_n.min(0) << ", "
        << U_n.max(0) << "] "
        << "hu=["
        << U_n.min(1) << ", "
        << U_n.max(1) << "] "
        << "hv=["
        << U_n.min(2) << ", "
        << U_n.max(2) << "]\n";

    // ------------------------------------------------------------------------
    // 6. MAP FLUXES INTO AMREX FLUX REGISTERS (coarse-fine interfaces)
    // ------------------------------------------------------------------------
    if (ctx.do_reflux)
    {
        amrex::MultiFab fluxes_crse[AMREX_SPACEDIM];
        amrex::MultiFab fluxes_fine[AMREX_SPACEDIM];

        if (fr_as_crse)
        {
            amrex::BoxArray ba_x = ctx.grids[lev];
            ba_x.surroundingNodes(0);
            fluxes_crse[0].define(ba_x, ctx.dmap[lev], U_n.nComp(), 0);
            fluxes_crse[0].setVal(0.0);
            fluxes_crse[0].FillBoundary(amr_geom.periodicity());


            amrex::BoxArray ba_y = ctx.grids[lev];
            ba_y.surroundingNodes(1);
            fluxes_crse[1].define(ba_y, ctx.dmap[lev], U_n.nComp(), 0);
            fluxes_crse[1].setVal(0.0);
            fluxes_crse[1].FillBoundary(amr_geom.periodicity());
        
        }

        if (fr_as_fine)
        {
            amrex::BoxArray ba_x = ctx.grids[lev];
            ba_x.surroundingNodes(0);
            fluxes_fine[0].define(ba_x, ctx.dmap[lev], U_n.nComp(), 0);
            fluxes_fine[0].setVal(0.0);
            fluxes_fine[0].FillBoundary(amr_geom.periodicity());

            amrex::BoxArray ba_y = ctx.grids[lev];
            ba_y.surroundingNodes(1);
            fluxes_fine[1].define(ba_y, ctx.dmap[lev], U_n.nComp(), 0);
            fluxes_fine[1].setVal(0.0);
            fluxes_fine[1].FillBoundary(amr_geom.periodicity());

        }

        // --- 6a. Coarse-fine fluxes: this level contributes to coarser level ---
        if (fr_as_crse)
        {
            // fine_mask: 1 = valid internal cell on this level, 0 = covered by coarse
            amrex::iMultiFab fine_mask = ctx.makeFineMask(U_n, ctx.grids[lev + 1], ctx.RefRatio(lev), 0, 1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            {
                for (amrex::MFIter mfi(U_n, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const int ncomp = U_n.nComp();

                    amrex::Box bx_x = mfi.nodaltilebox(0);
                    amrex::Box bx_y = mfi.nodaltilebox(1);

                    auto const& mask = fine_mask.array(mfi);
                    auto const& fx_m = fx_minus.array(mfi);
                    auto const& fx_p = fx_plus.array(mfi);
                    auto const& reg_x = fluxes_crse[0].array(mfi);
                    auto const& fy_m = fy_minus.array(mfi);
                    auto const& fy_p = fy_plus.array(mfi);
                    auto const& reg_y = fluxes_crse[1].array(mfi);

                    amrex::ParallelFor(bx_x,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        int left_is_fine  = mask(i - 1, j, k);
                        int right_is_fine = mask(i, j, k);

                        for (int n = 0; n < ncomp; ++n) {
                            if (left_is_fine == 1 && right_is_fine == 0) {
                                reg_x(i, j, k, n) = fx_p(i, j, k, n);
                            } else if (left_is_fine == 0 && right_is_fine == 1) {
                                reg_x(i, j, k, n) = fx_m(i, j, k, n);
                            } else {
                                reg_x(i, j, k, n) = 0.0;
                            }
                        }
                    });

                    amrex::ParallelFor(bx_y,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        int bottom_is_fine = mask(i, j - 1, k);
                        int top_is_fine    = mask(i, j, k);

                        for (int n = 0; n < ncomp; ++n) {
                            if (bottom_is_fine == 1 && top_is_fine == 0) {
                                reg_y(i, j, k, n) = fy_p(i, j, k, n);
                            } else if (bottom_is_fine == 0 && top_is_fine == 1) {
                                reg_y(i, j, k, n) = fy_m(i, j, k, n);
                            } else {
                                reg_y(i, j, k, n) = 0.0;
                            }
                        }
                    });
                }
            }
        }

        // --- 6b. Coarse-fine fluxes: this level contributes to finer level ---
        if (fr_as_fine)
        {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            {
                for (amrex::MFIter mfi(U_n, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const int ncomp = U_n.nComp();

                    amrex::Box bx_x = mfi.nodaltilebox(0);
                    amrex::Box bx_y = mfi.nodaltilebox(1);
                    amrex::Box bx_grid = mfi.validbox();

                    auto const& fx_m = fx_minus.array(mfi);
                    auto const& fx_p = fx_plus.array(mfi);
                    auto const& reg_x = fluxes_fine[0].array(mfi);

                    auto const& fy_m = fy_minus.array(mfi);
                    auto const& fy_p = fy_plus.array(mfi);
                    auto const& reg_y = fluxes_fine[1].array(mfi);

                    amrex::ParallelFor(bx_x,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int is_low_boundary = (i == bx_grid.smallEnd(0));
                        int is_high_boundary = (i == bx_grid.bigEnd(0) + 1);

                        for (int n = 0; n < ncomp; ++n) {
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

                        for (int n = 0; n < ncomp; ++n) {
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
        // 7. SYNCHRONIZE WITH FLUX REGISTERS
        // ------------------------------------------------------------------------
        if (fr_as_crse)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const amrex::Real dA = (idim == 0) ? dy : dx;
                const amrex::Real scale = -dt*dA;
                fr_as_crse->CrseInit(fluxes_crse[idim], idim, 0, 0, U_n.nComp(), scale,
                                     amrex::FluxRegister::ADD);
            }
        }

        if (fr_as_fine)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const amrex::Real dA = (idim == 0) ? dy : dx;
                const amrex::Real scale = dt*dA;
                fr_as_fine->FineAdd(fluxes_fine[idim], idim, 0, 0, U_n.nComp(), scale);
            }
        }
    }
}

