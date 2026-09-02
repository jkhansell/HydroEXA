#include <solvers/Roe.H>
#include <solvers/kernels/Roe.H>
#include <boundaries/BCFill.H>

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
        int li = i - offset_x; int lj = j - offset_y;
        int ri = i;            int rj = j;

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

        amrex::Real D_minus[3] = {}; 
        amrex::Real D_plus[3] = {};
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
    amrex::MultiFab& Terrain = ctx.Terrain[lev];

    AMREX_ASSERT_WITH_MESSAGE(U_n.nComp() >= 3,
        "Roe solver requires at least 3 state components (H, HU, HV) but U_n.nComp() = "
        + std::to_string(U_n.nComp()));

    const auto& amr_geom = ctx.geom[lev];
    const amrex::Real dx = amr_geom.CellSize(0);
    const amrex::Real dy = amr_geom.CellSize(1);

    // ------------------------------------------------------------------------
    // 1. FLUX REGISTER POINTERS
    // ------------------------------------------------------------------------
    amrex::FluxRegister* fr_as_crse = nullptr;
    if (ctx.do_reflux && lev < ctx.finest_level) {
        fr_as_crse = ctx.FluxRegisters[lev+1].get();
    }

    amrex::FluxRegister* fr_as_fine = nullptr;
    if (ctx.do_reflux && lev > 0) {
        fr_as_fine = ctx.FluxRegisters[lev].get();
    }

    if (fr_as_crse) {
        fr_as_crse->setVal(0.0);
    }

    // ------------------------------------------------------------------------
    // 2. ALLOCATE DUAL EFFECTIVE FLUXES (FL and FR)
    // ------------------------------------------------------------------------
    amrex::MultiFab D_minus_mf[AMREX_SPACEDIM];
    amrex::MultiFab D_plus_mf[AMREX_SPACEDIM];

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        amrex::BoxArray ba = ctx.grids[lev];
        ba.surroundingNodes(dir);
        D_minus_mf[dir].define(ba, ctx.dmap[lev], U_n.nComp(), 1);
        D_plus_mf[dir].define(ba, ctx.dmap[lev], U_n.nComp(), 1);

        D_minus_mf[dir].setVal(0.0);
        D_plus_mf[dir].setVal(0.0);
    }

    // ------------------------------------------------------------------------
    // 3. COMPUTE DIRECTIONAL DUAL EFFECTIVE FLUXES
    // ------------------------------------------------------------------------
    const amrex::Real dx_local = dx;
    const amrex::Real dy_local = dy;
    const amrex::Real dt_local = dt;

    // fill coarse/fine boundaries
    // ctx.FillPatch(lev, time, U_o, ctx.UBCs, 0, U_o.nComp());
    // ctx.FillPatch(lev, time, Terrain, ctx.TerrainBCs, 0, Terrain.nComp());
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

            // Output FL and FR directly from roeSolver
            compute_amrex_effective_fluxes(
                bx_x, statein, z_arr,
                D_minus_mf[0].array(mfi), D_plus_mf[0].array(mfi),
                dt_local, dx_local, 0
            );

            compute_amrex_effective_fluxes(
                bx_y, statein, z_arr,
                D_minus_mf[1].array(mfi), D_plus_mf[1].array(mfi),
                dt_local, dy_local, 1
            );
        }
    }

    // After compute_amrex_effective_fluxes loops finish:
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) { 
        D_minus_mf[dir].FillBoundary(amr_geom.periodicity());
        D_plus_mf[dir].FillBoundary(amr_geom.periodicity());
    }

    // ------------------------------------------------------------------------
    // 4. LOCAL CONSERVATIVE CELL UPDATE (Exact Fluctuation Differencing)
    // ------------------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (amrex::MFIter mfi(U_o, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const int ncomp = U_o.nComp();
            const amrex::Box& bx = mfi.tilebox();

            auto const& Dminus_x = D_minus_mf[0].array(mfi);
            auto const& Dplus_x = D_plus_mf[0].array(mfi);

            auto const& Dminus_y = D_minus_mf[1].array(mfi);
            auto const& Dplus_y = D_plus_mf[1].array(mfi);

            auto const& u_o = U_o.array(mfi);
            auto const& u_n = U_n.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int n = 0; n < ncomp; ++n)
                {
                    u_n(i,j,k,n) = u_o(i,j,k,n)
                                 - (dt_local/dx_local) * (Dminus_x(i+1,j,k,n) - Dplus_x(i,j,k,n))
                                 - (dt_local/dy_local) * (Dminus_y(i,j+1,k,n) - Dplus_y(i,j,k,n));
                }
            });
        }
    }

    amrex::Print()
        << "Level " << lev
        << " h=["  << U_n.min(0) << ", " << U_n.max(0) << "] "
        << "hu=[" << U_n.min(1) << ", " << U_n.max(1) << "] "
        << "hv=[" << U_n.min(2) << ", " << U_n.max(2) << "]\n";

    // ------------------------------------------------------------------------
    // 5. PREPARE AND SYNCHRONIZE WITH FLUX REGISTERS
    // ------------------------------------------------------------------------
    if (fr_as_crse || fr_as_fine)
    {
        amrex::MultiFab flux_fine[AMREX_SPACEDIM];
        amrex::MultiFab flux_crse[AMREX_SPACEDIM];

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            amrex::BoxArray ba = ctx.grids[lev];
            ba.surroundingNodes(dir);

            if (fr_as_fine)
            {
                flux_fine[dir].define(ba, ctx.dmap[lev], U_n.nComp(), 0);
                flux_fine[dir].setVal(0.0);
            }

            if (fr_as_crse)
            {
                flux_crse[dir].define(ba, ctx.dmap[lev], U_n.nComp(), 0);
                flux_crse[dir].setVal(0.0);
            }
        }

        // --------------------------------------------------------------------
        // Coarse coverage mask
        //
        // 0 = Coarse cell
        // 1 = Covered by finer level
        // --------------------------------------------------------------------
        std::unique_ptr<amrex::iMultiFab> coarse_mask;

        if (fr_as_crse)
        {
            coarse_mask = std::make_unique<amrex::iMultiFab>(makeFineMask(U_n,ctx.grids[lev+1],ctx.RefRatio(lev),0,1));
        }

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
        {
            for (amrex::MFIter mfi(U_o, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const int ncomp = U_o.nComp();
                const amrex::Box& valid_bx = mfi.validbox();

                auto const& mask = fr_as_crse ? coarse_mask->const_array(mfi) : amrex::Array4<int const>();

                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    amrex::Box nbx = mfi.nodaltilebox(dir);

                    auto const& Dm = D_minus_mf[dir].array(mfi);   // D-
                    auto const& Dp = D_plus_mf[dir].array(mfi);   // D+

                    auto const& ff = fr_as_fine ? flux_fine[dir].array(mfi) : amrex::Array4<amrex::Real>();
                    auto const& fc = fr_as_crse ? flux_crse[dir].array(mfi) : amrex::Array4<amrex::Real>();

                    const int i_low  = valid_bx.smallEnd(dir);
                    const int i_high = valid_bx.bigEnd(dir) + 1;

                    amrex::ParallelFor(nbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int face_idx = (dir == 0) ? i : (dir == 1) ? j : k;

                        int left_is_fine  = 0;
                        int right_is_fine = 0;

                        if (fr_as_crse)
                        {
                            if (dir == 0)
                            {
                                left_is_fine  = mask(i-1,j,k);
                                right_is_fine = mask(i  ,j,k);
                            }
                            else
                            {
                                left_is_fine  = mask(i,j-1,k);
                                right_is_fine = mask(i,j  ,k);
                            }
                        }

                        for (int n = 0; n < ncomp; ++n)
                        {
                            //--------------------------------------------------
                            // COARSE REGISTER
                            //--------------------------------------------------
                            if (fr_as_crse)
                            {
                                if (left_is_fine == 0 && right_is_fine == 1)
                                {
                                    // Coarse cell is LEFT
                                    fc(i,j,k,n) = Dm(i,j,k,n);
                                }
                                else if (left_is_fine == 1 && right_is_fine == 0)
                                {
                                    // Coarse cell is RIGHT
                                    fc(i,j,k,n) = Dp(i,j,k,n);
                                }
                            }

                            //--------------------------------------------------
                            // FINE REGISTER
                            //--------------------------------------------------
                            if (fr_as_fine)
                            {
                                if (face_idx == i_low)
                                {
                                    // Fine cell is RIGHT
                                    ff(i,j,k,n) = Dp(i,j,k,n);
                                }
                                else if (face_idx == i_high)
                                {
                                    // Fine cell is LEFT
                                    ff(i,j,k,n) = Dm(i,j,k,n);
                                }
                            }
                        }
                    });
                }
            }
        }

        // Send mapped boundary fluxes to AMReX FluxRegisters
        if (fr_as_crse)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const amrex::Real dA = (idim == 0) ? dy : dx;
                const amrex::Real scale = -dt*dA;
                fr_as_crse->CrseInit(flux_crse[idim], idim, 0, 0, U_n.nComp(), scale,
                                     amrex::FluxRegister::ADD);
            }
        }

        if (fr_as_fine)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const amrex::Real dA = (idim == 0) ? dy : dx;
                const amrex::Real scale = dt*dA;
                fr_as_fine->FineAdd(flux_fine[idim], idim, 0, 0, U_n.nComp(), scale);
            }
        }
    }
}
