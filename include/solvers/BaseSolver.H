// include/solvers/BaseSolver.H
#ifndef BASE_SOLVER_H_
#define BASE_SOLVER_H_

#include <AMReX_Box.H>
#include <AMReX_Array4.H>

template <typename Derived>
struct BaseSolver {
    void compute_fluxes(AmrMeshState& mesh_state, int lev) {
        static_cast<Derived*>(this)->compute_fluxes_Impl(mesh_state, lev);
    }

    Real compute_dt_level(AmrMeshState& mesh_state, int lev) {
        return static_cast<Derived*>(this)->compute_dt_Impl(mesh_state, lev);
    }

    void compute_dt(AmrMeshState& mesh_state) {
        Vector<Real> dt_tmp(mesh_state.finest_level + 1);
        for (int lev = 0; lev <= mesh_state.finest_level; ++lev) {
            dt_tmp[lev] = compute_dt_level(mesh_state, lev);
        }

        // 1. Parallel reduction to get the true minimums across all MPI ranks
        ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

        // 2. Start with the dt calculated for the finest level
        int max_lev = mesh_state.finest_level;
        mesh_state.dt[max_lev] = dt_tmp[max_lev];

        // 3. Work backwards from the finest level to the coarsest level
        //    dt_coarse = dt_fine * ref_ratio
        for (int lev = max_lev - 1; lev >= 0; --lev) {
            // refRatio(lev)[0] is the refinement ratio between 'lev' and 'lev+1'
            mesh_state.dt[lev] = mesh_state.dt[lev + 1] * mesh_state.refRatio(lev)[0];
            
            // Safety check: Ensure the scaled dt doesn't accidentally 
            // violate the original physics constraints of this coarser level
            mesh_state.dt[lev] = std::min(mesh_state.dt[lev], dt_tmp[lev]);
        }
    }
    
    int get_ncomp_Impl(){
        return static_cast<Derived*>(this)->get_ncomp_Impl();
    }

    int get_ncomp_terrain_Impl(){
        return static_cast<Derived*>(this)->get_ncomp_terrain_Impl();
    }
};

#endif

