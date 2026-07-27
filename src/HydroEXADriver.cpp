#include <string>
#include <HydroEXA.H>
#include <utils/Logging.H>

using namespace amrex;

int main(int argc, char* argv[]){
    amrex::Initialize(argc,argv);

    {  // start of scope where Fortran librarys are visible

         // timer for profiling
        BL_PROFILE("main()");

        // wallclock time
        const auto strt_total = amrex::second();

        // constructor - reads in parameters from inputs file
        //             - sizes multilevel arrays and data structures

        HydroEXA hydroexa;      // Param read here ONLY C++ can't call virtual functions in the constructor, so we defer solver construction to Initialize()

        hydroexa.Initialize();  // sets up the mesh state, including geometry, initial conditions, and constructs the solver variant based on the input parameters

        //hydroexa.Compute();     // main time-stepping loop, which calls the solver's compute_fluxes and compute_dt methods at each iteration

        // hydroexa.Finalize();    // any necessary cleanup, output final diagnostics, etc.

        // wallclock time
        auto end_total = amrex::second() - strt_total;

        if (hydroexa.Verbose()) {
            // print wallclock time
            ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
            LOG(INFO, "\nTotal Time: " + std::to_string(end_total) + '\n');
        }
    }  // end of scope where Fortran librarys are visible

    amrex::Finalize();

    return 0;

}