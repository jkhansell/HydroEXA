#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

#include "hydroexa/HydroLevelBld.hpp"

int main(int argc, char* argv[]) {
    // Initialize AMReX (handles MPI and Kokkos/GPU init internally)
    amrex::Initialize(argc, argv);
    
    {

        HydroLevelBld hydro_bld;
        amrex::Amr HydroEXA(&hydro_bld);
        
        amrex::Real stop_time = 1.0;
        int max_step = -1;
        
        amrex::ParmParse pp;
        pp.query("t_final", stop_time);
        pp.query("max_steps", max_step);
        
        HydroEXA.init(0., stop_time);
        
        amrex::Print() << "=======================================\n";
        amrex::Print() << "          HydroEXA Initialized         \n";
        amrex::Print() << "=======================================\n";

        while ( HydroEXA.okToContinue() 
             && (HydroEXA.levelSteps(0) < max_step || max_step < 0) 
             && (HydroEXA.cumTime() < stop_time || stop_time < 0.0) )
        {
            HydroEXA.coarseTimeStep(stop_time);
        }
    }
    
    amrex::Print() << "Finalizing HydroEXA...\n";
    amrex::Finalize();
    return 0;
}
