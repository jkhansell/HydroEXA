// NativePlotFile.cpp
// Free-function plotfile writer using AMReX's native plotfile format.
// Extracted from IOHandler::WritePlotfile to decouple native I/O
// from the IOHandler class itself.

#include <io/NativePlotFile.H>

// STD
#include <string>
#include <vector>

// AMReX
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>

// Local
#include <utils/Logging.H>

void WriteNativePlotfile(
    const std::string& prefix,
    int iteration,
    amrex::Real time,
    const amrex::Vector<amrex::MultiFab>& U,
    const amrex::Vector<amrex::MultiFab>& Terrain,
    const amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<amrex::IntVect>& ref_ratio,
    int finest_level)
{
    // 1. Safely calculate the TRUE number of active, fully allocated levels.
    int num_active_levels = 0;
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (U[lev].boxArray().empty() || Terrain[lev].boxArray().empty())
        {
            break;
        }
        num_active_levels++;
    }

    if (num_active_levels == 0)
    {
        LOG_WARN("No valid levels found. Skipping plotfile.");
        return;
    }

    std::string plotfilename = amrex::Concatenate(prefix, iteration, 5);

    // 2. Set variable string descriptors safely
    int ncomp_U = U[0].nComp();
    int ncomp_Terrain = Terrain[0].nComp();
    int total_comps = ncomp_U + ncomp_Terrain;

    amrex::Vector<std::string> varnames;
    varnames.push_back("h_fluid");
    varnames.push_back("hu_momentum");
    varnames.push_back("hv_momentum");
    varnames.push_back("z_bathymetry");

    // SAFETY CATCH: If you ever change ncomps (e.g., adding roughness),
    // this prevents AMReX from crashing due to an out-of-bounds string read.
    while (varnames.size() < static_cast<size_t>(total_comps))
    {
        varnames.push_back("extra_comp_" + std::to_string(varnames.size()));
    }

    // 3. Set up pointers and temporary multi-component Fabs safely
    amrex::Vector<const amrex::MultiFab*> output_mf(num_active_levels);
    amrex::Vector<amrex::MultiFab> temp_mf(num_active_levels);

    for (int lev = 0; lev < num_active_levels; ++lev)
    {
        // Allocate local temporary space tracking identical layout geometry
        temp_mf[lev].define(U[lev].boxArray(), U[lev].DistributionMap(), total_comps, 0);

        // Copy fluid variables (first ncomp_U components)
        amrex::MultiFab::Copy(temp_mf[lev], U[lev], 0, 0, ncomp_U, 0);
        // Copy terrain/bathymetry (next ncomp_Terrain components)
        amrex::MultiFab::Copy(temp_mf[lev], Terrain[lev], 0, ncomp_U, ncomp_Terrain, 0);

        // Guarantee a valid memory address is provided
        output_mf[lev] = &temp_mf[lev];
    }

    // 4. Track local integer state steps
    amrex::Vector<int> istep(num_active_levels, iteration);

    // 5. Fire parallel output dump sequence
    amrex::WriteMultiLevelPlotfile(
        plotfilename,
        num_active_levels,
        output_mf,
        varnames,
        geom,
        time,
        istep,
        ref_ratio);
}
