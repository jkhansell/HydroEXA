
void
Roe::compute_fluxes_Impl(AmrMeshState& mesh_state, int lev){
    BL_PROFILE("Roe::compute_fluxes_Impl");
    


}

Real
Roe::compute_dt_Impl(AmrMeshState& mesh_state, int lev) {
    BL_PROFILE("Roe::compute_dt_Impl");
    
    const auto dx = mesh_state.geom[lev].CellSizeArray();
    const MultiFab& S = mesh_state.U_new[lev];

    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            const auto& U = S.const_array(mfi);

            const Real dxx = dx[0];
            const Real dyy = dx[1];
            
            reduce_op.eval(bx, reduce_data, 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    Real h = U(i,j,k,0);
                    Real dt_cell = 1.e6; 

                    if (h > 1.e-12) { 
                        Real u = U(i,j,k,1) / h;
                        Real v = U(i,j,k,2) / h;
                        amrex::Real c = std::sqrt(GRAV * h); 

                        Real dt_x = dxx / (std::abs(u) + c);
                        Real dt_y = dyy / (std::abs(v) + c);

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

    // Apply CFL multiplier to the local estimate
    dt_est *= cfl;

    return dt_est;
}


