#include <solvers/HydrostaticRoe.H>
#include <solvers/kernels/HydrostaticRoe.H>
#include <boundaries/BCFill.H>
#include <utils/Constants.H>

amrex::Real
HydrostaticRoe::compute_dt_Impl(const amrex::Vector<amrex::MultiFab>& U,
                     const amrex::Vector<amrex::MultiFab>& terrain,
                     const amrex::Vector<amrex::Geometry>& geom,
                     int lev, amrex::Real cfl) {
    BL_PROFILE("HydrostaticRoe::compute_dt_Impl");
    
    // 1. Use the incoming function parameters instead of hidden mesh_state dependencies
    const auto dx = geom[lev].CellSizeArray();
    const amrex::MultiFab& S = U[lev];
    const amrex::MultiFab& Terrain = terrain[lev];

    amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);

    using ReduceTuple = decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if(amrex::Gpu::notInLaunchRegion())
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

void HydrostaticRoe::tag_cells_Impl(amrex::TagBoxArray& tags,
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

            if (tag_cell) {
                tag_arr(i, j, k) = amrex::TagBox::SET;
            }
        });
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real vanLeerLimiter(const amrex::Real r)
{
    return (r + abs(r)) / (amrex::Real(1) + abs(r));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real vanLeerSlope(
    const amrex::Real qim1,
    const amrex::Real qi,
    const amrex::Real qip1)
{
    const amrex::Real dm = qi - qim1;
    const amrex::Real dp = qip1 - qi;

    if (std::abs(dp) <= Tolerances::TOL12()) {
        return 0.0;
    }

    const amrex::Real r = dm / dp;

    return vanLeerLimiter(r) * dp;
}

void
HydrostaticRoe::compute_amrex_effective_fluxes(
    const amrex::Box& face_box,
    amrex::Array4<amrex::Real const> const& U,
    amrex::Array4<amrex::Real const> const& terrain,
    amrex::Array4<amrex::Real> const& F,
    const amrex::Real dt,
    const amrex::Real dx,
    const int dir)
{
    const int ox = (dir == 0) ? 1 : 0;
    const int oy = (dir == 1) ? 1 : 0;

    const amrex::Real nx = (dir == 0) ? 1.0 : 0.0;
    const amrex::Real ny = (dir == 1) ? 1.0 : 0.0;

    amrex::ParallelFor(
        face_box,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            const int li = i - ox;
            const int lj = j - oy;

            const int ri = i;
            const int rj = j;

            const int lim = li - ox;
            const int ljm = lj - oy;

            const int rip = ri + ox;
            const int rjp = rj + oy;

            // ---------------------------------------------------------
            // Face bathymetry
            // ---------------------------------------------------------

            const amrex::Real zL = terrain(li, lj, k, 0);
            const amrex::Real zR = terrain(ri, rj, k, 0);

            const amrex::Real zf = std::max(zL, zR);

            // ---------------------------------------------------------
            // Reconstruct eta
            // ---------------------------------------------------------

            amrex::Real etaLf;
            amrex::Real etaRf;

            {
                const amrex::Real etaLm = U(lim, ljm, k, 0) + terrain(lim, ljm, k, 0);
                const amrex::Real etaL = U(li, lj, k, 0) + zL;
                const amrex::Real etaR = U(ri, rj, k, 0) + zR;
                const amrex::Real etaRp = U(rip, rjp, k, 0) + terrain(rip, rjp, k, 0);

                const amrex::Real slopeL = vanLeerSlope(etaLm, etaL, etaR);
                const amrex::Real slopeR = vanLeerSlope(etaL, etaR, etaRp);

                etaLf = etaL + 0.5 * slopeL;
                etaRf = etaR - 0.5 * slopeR;
            }

            // ---------------------------------------------------------
            // Reconstruct u
            // ---------------------------------------------------------

            amrex::Real uLf;
            amrex::Real uRf;

            {
                const amrex::Real hLm = U(lim, ljm, k, 0);
                const amrex::Real hL  = U(li, lj, k, 0);
                const amrex::Real hR  = U(ri, rj, k, 0);
                const amrex::Real hRp = U(rip, rjp, k, 0);

                const amrex::Real uLm =
                    (hLm > Tolerances::TOL12())
                        ? U(lim, ljm, k, 1) / hLm
                        : 0.0;

                const amrex::Real uL =
                    (hL > Tolerances::TOL12())
                        ? U(li, lj, k, 1) / hL
                        : 0.0;

                const amrex::Real uR =
                    (hR > Tolerances::TOL12())
                        ? U(ri, rj, k, 1) / hR
                        : 0.0;

                const amrex::Real uRp =
                    (hRp > Tolerances::TOL12())
                        ? U(rip, rjp, k, 1) / hRp
                        : 0.0;

                uLf = uL + 0.5 * vanLeerSlope(uLm, uL, uR);
                uRf = uR - 0.5 * vanLeerSlope(uL, uR, uRp);
            }

            // ---------------------------------------------------------
            // Reconstruct v
            // ---------------------------------------------------------

            amrex::Real vLf;
            amrex::Real vRf;

            {
                const amrex::Real hLm = U(lim, ljm, k, 0);
                const amrex::Real hL  = U(li, lj, k, 0);
                const amrex::Real hR  = U(ri, rj, k, 0);
                const amrex::Real hRp = U(rip, rjp, k, 0);

                const amrex::Real vLm =
                    (hLm > Tolerances::TOL12())
                        ? U(lim, ljm, k, 2) / hLm
                        : 0.0;

                const amrex::Real vL =
                    (hL > Tolerances::TOL12())
                        ? U(li, lj, k, 2) / hL
                        : 0.0;

                const amrex::Real vR =
                    (hR > Tolerances::TOL12())
                        ? U(ri, rj, k, 2) / hR
                        : 0.0;

                const amrex::Real vRp =
                    (hRp > Tolerances::TOL12())
                        ? U(rip, rjp, k, 2) / hRp
                        : 0.0;

                vLf = vL + 0.5 * vanLeerSlope(vLm, vL, vR);
                vRf = vR - 0.5 * vanLeerSlope(vL, vR, vRp);
            }

            // ---------------------------------------------------------
            // Hydrostatic reconstruction
            // ---------------------------------------------------------

            const amrex::Real hLf = std::max(Tolerances::TOLDRY(), etaLf - zf);
            const amrex::Real hRf = std::max(Tolerances::TOLDRY(), etaRf - zf);

            const amrex::Real huLf = hLf * uLf;
            const amrex::Real hvLf = hLf * vLf;

            const amrex::Real huRf = hRf * uRf;
            const amrex::Real hvRf = hRf * vRf;

            // ---------------------------------------------------------
            // Homogeneous Roe
            // ---------------------------------------------------------

            amrex::Real flux[3] = {};

            roeSolver(
                hLf, huLf, hvLf,
                hRf, huRf, hvRf,
                flux,
                dt, dx, nx, ny);

            F(i, j, k, 0) = flux[0];
            F(i, j, k, 1) = flux[1];
            F(i, j, k, 2) = flux[2];
        });
}

void
HydrostaticRoe::compute_fluxes_Impl(SolverContext ctx, int lev, amrex::Real dt, amrex::Real time) {

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
    amrex::MultiFab F[AMREX_SPACEDIM];
    amrex::MultiFab U_ext, Terrain_ext;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        amrex::BoxArray ba = ctx.grids[lev];
        ba.surroundingNodes(dir);
        F[dir].define(ba, ctx.dmap[lev], U_n.nComp(), 0);
        F[dir].setVal(0.0);
    }

    // MUSCL reconstruction and flux calculation need 2 ghosts
    U_ext.define(ctx.grids[lev], ctx.dmap[lev], U_n.nComp(), 2); 
    Terrain_ext.define(ctx.grids[lev], ctx.dmap[lev], Terrain.nComp(), 2);

    // ----------------------------------- -------------------------------------
    // 3. COMPUTE DIRECTIONAL DUAL EFFECTIVE FLUXES
    // ------------------------------------------------------------------------
    const amrex::Real dx_local = dx;
    const amrex::Real dy_local = dy;
    const amrex::Real dt_local = dt;

    // fill coarse/fine boundaries
    ctx.FillPatch(lev, time, U_ext, ctx.UBCs, 0, U_o.nComp());
    ctx.FillPatchTerrain(lev, time, Terrain_ext, ctx.TerrainBCs, 0, Terrain.nComp());

#ifdef AMREX_USE_OMP    
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        for (amrex::MFIter mfi(U_o, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const& statein = U_ext.array(mfi);
            auto const& z_arr = Terrain_ext.array(mfi);

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                
                amrex::Box face_box = mfi.nodaltilebox(dir);
                // Output FL and FR directly from roeSolver

                amrex::Real ds = (dir == 0) ? dx_local : dy_local; 
                compute_amrex_effective_fluxes(
                    face_box, statein, z_arr,
                    F[dir].array(mfi),
                    dt_local, ds, dir
                );
            }
        }
    }


    // ------------------------------------------------------------------------
    // 4. LOCAL CONSERVATIVE CELL UPDATE (Exact Fluctuation Differencing)
    // ------------------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        for (amrex::MFIter mfi(U_o, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const int ncomp = U_o.nComp();
            const amrex::Box& bx = mfi.tilebox();

            auto const& F_x = F[0].array(mfi);
            auto const& F_y = F[1].array(mfi);

            auto const& u_o = U_o.array(mfi);
            auto const& u_n = U_n.array(mfi);
            auto const& terrain   = Terrain.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int comp = 0; comp < ncomp; ++comp)
                {
                    u_n(i,j,k,comp) = u_o(i,j,k,comp)
                        - (dt_local / dx_local)
                            * (F_x(i+1,j,k,comp) - F_x(i,j,k,comp))
                        - (dt_local / dy_local)
                            * (F_y(i,j+1,k,comp) - F_y(i,j,k,comp));
                }

                const amrex::Real h = u_n(i,j,k,0);

                if (h >= Tolerances::TOLDRY())
                {
                    const amrex::Real hu = u_n(i,j,k,1);
                    const amrex::Real hv = u_n(i,j,k,2);
                    const amrex::Real nsq = terrain(i,j,k,1) * terrain(i,j,k,1);

                    const amrex::Real inv_h = 1.0 / h;
                    const amrex::Real u = hu * inv_h;
                    const amrex::Real v = hv * inv_h;
                    const amrex::Real modM = std::sqrt(u*u + v*v);

                    if (nsq > Tolerances::TOL12() && modM > Tolerances::TOL12())
                    {
                        const amrex::Real h43 = h * std::cbrt(h);

                        const amrex::Real tt =
                            dt_local * PhysConstants::GRAV() * nsq * modM / h43;

                        const amrex::Real ff =
                            std::sqrt(1.0 + 4.0 * tt);

                        const amrex::Real factor =
                            2.0 / (1.0 + ff);

                        u_n(i,j,k,1) = hu * factor;
                        u_n(i,j,k,2) = hv * factor;
                    }
                }

                
            });
        }
    }

    
    // ------------------------------------------------------------------------
    // 5. PREPARE AND SYNCHRONIZE WITH FLUX REGISTERS
    // ------------------------------------------------------------------------
    if (ctx.do_reflux) {
        // Send mapped boundary fluxes to AMReX FluxRegisters
        if (fr_as_crse)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const amrex::Real dA = (idim == 0) ? dy : dx;
                const amrex::Real scale = -dt*dA;
                fr_as_crse->CrseInit(F[idim], idim, 0, 0, U_n.nComp(), scale,
                                    amrex::FluxRegister::COPY);
            }
        }

        if (fr_as_fine)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const amrex::Real dA = (idim == 0) ? dy : dx;
                const amrex::Real scale = dt*dA;
                fr_as_fine->FineAdd(F[idim], idim, 0, 0, U_n.nComp(), scale);
            }
        }
    }
}
