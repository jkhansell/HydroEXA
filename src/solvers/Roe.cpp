#include <solvers/Roe.H>

void
Roe::compute_fluxes_Impl(amrex::Vector<amrex::MultiFab>& U,
                         amrex::Vector<amrex::MultiFab>& terrain,
                         const amrex::Vector<amrex::Geometry>& geom,
                         int lev){
    BL_PROFILE("Roe::compute_fluxes_Impl");
    
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

            const amrex::Real dxx = dx[0];
            const amrex::Real dyy = dx[1];
            
            reduce_op.eval(bx, reduce_data, 
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    amrex::Real h = arr(i,j,k,0);
                    amrex::Real dt_cell = 1.e6; 

                    if (h > 1.e-12) { 
                        amrex::Real u = arr(i,j,k,1) / h;
                        amrex::Real v = arr(i,j,k,2) / h;
                        // Fixed: Correct namespace resolution for GRAV
                        amrex::Real c = std::sqrt(PhysConstants::GRAV * h); 

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

    // 2. Fixed: Synchronize time steps across all parallel MPI ranks to prevent deadlocks/drift
    amrex::ParallelDescriptor::ReduceRealMin(dt_est);

    // Apply CFL multiplier to the global safe estimate
    dt_est *= cfl;

    return dt_est;
}