#include <AMReX_ParmParse.H>
#include <HydroEXA.H>

HydroEXA::HydroEXA() {
    // Constructor can be used to set up any necessary data structures or 
    // read in configuration parameters if needed. For now, we'll keep it simple.
    BL_PROFILE("HydroEXA::Constructor");
    {
        amrex::ParmParse pp("io");
        pp.query("chk_freq", io_params.chk_freq);
        pp.query("chk_type", io_params.chk_type);    // HDF5, Native
        pp.query("plot_freq", io_params.plot_freq);
        pp.query("async_io", io_params.async_io);    // bool for whether to do asynchronous I/O
        
        pp.query("input_file", io_params.input_file);  // HDF5 file containing initial conditions (e.g., DEM)
        pp.query("dataset_name", io_params.dataset_name);  // HDF5 file containing initial conditions (e.g., DEM)
        pp.query("plot_file", io_params.plot_file);
        pp.query("chk_file", io_params.chk_file);
        
        // AMReX native checkpoint file
        pp.query("restart", io_params.restart_chkfile);
    }

    {
        // Traditionally, these have prefix, amr.
        amrex::ParmParse pp("amr");
        
        pp.query("regrid_int", amr_params.regrid_int);           // frequency of regridding
        pp.query("do_reflux", amr_params.do_reflux);             // whether to perform flux correction
        pp.query("do_subcycle", amr_params.do_subcycle);           // we place this here since it affects the time stepping and regridding logic
        pp.query("terrain_ref_lev", amr_params.terrain_ref_lev); // AMR level at which to read the initial conditions (default 0)
        pp.query("max_level", amr_params.max_amr_level);           // Number of levels is max_level + 1
        amr_params.max_amr_level = amr_params.max_amr_level + 1;
    }

    {
        // read in an array of thresholds for the gradients of umag and h, which are the tagging threshold
        amrex::ParmParse pp("HydroEXA");

        pp.query("model", physics_params.model);   // 0 SWE, 1 SWE-Exner 
        pp.query("cfl", physics_params.cfl);

        int n = pp.countval("h_grad_thresh");
        if (n > 0) {
            pp.getarr("h_grad_thresh", physics_params.h_grad_thresh, 0, n);
        }

        n = pp.countval("umag_grad_thresh");
        if (n > 0) {
            pp.getarr("umag_grad_thresh", physics_params.umag_grad_thresh, 0, n);
        }
    }
}

void
HydroEXA::Initialize() {
    // Initialize the mesh state, including geometry, initial conditions, etc.
    BL_PROFILE("HydroEXA::Initialize");

    IO = std::make_shared<IOHandler>();
    HDF5SpatialMetadata metadata;
    IO->ReadHDF5Metadata(io_params.input_file, io_params.dataset_name, metadata);
    
    // 1. Determine the raw resolution reduction factor down to Level 0
    int refinement_scale = 1 << amr_params.terrain_ref_lev;
    int base_nx_raw = metadata.global_nx / refinement_scale;
    int base_ny_raw = metadata.global_ny / refinement_scale;

    // 2. Fetch or match your gridding blocking factor (e.g., 8)
    int bf = 8; 
    amrex::ParmParse pp_amr("amr");
    pp_amr.query("blocking_factor", bf); // Checks if user overrode it via inputs

    // Round up the base cell configuration to an exact multiple of the blocking factor
    int base_nx = ((base_nx_raw + bf - 1) / bf) * bf;
    int base_ny = ((base_ny_raw + bf - 1) / bf) * bf;
    int base_nz = 1;

    // 3. Compute Level-0 grid spacing metrics to find the updated physical boundary
    amrex::Real level0_dx = metadata.dx * static_cast<amrex::Real>(refinement_scale);
    amrex::Real level0_dy = metadata.dy * static_cast<amrex::Real>(refinement_scale);

    // Dynamic right-edge expansion to account for grid padding without stretching cells
    amrex::Real padded_prob_hi_x = metadata.prob_lo_x + (static_cast<amrex::Real>(base_nx) * level0_dx);
    amrex::Real padded_prob_hi_y = metadata.prob_lo_y + (static_cast<amrex::Real>(base_ny) * level0_dy);

    // Set Up Continuous Physical Coordinate Coordinates using padded high-bounds
    amrex::RealBox real_domain(
        { AMREX_D_DECL(metadata.prob_lo_x, metadata.prob_lo_y, 0.0) }, 
        { AMREX_D_DECL(padded_prob_hi_x, padded_prob_hi_y, 1.0) }
    );

    amrex::Vector<int> n_cell = { AMREX_D_DECL(base_nx, base_ny, base_nz) };
    
    int coord_sys = 0; // Cartesian
    
    // Enforce programmatic factor-of-two behavior agnostically
    amrex::Vector<amrex::IntVect> ref_ratios(amr_params.max_amr_level);

    for (int lev = 0; lev < amr_params.max_amr_level; ++lev) {
        ref_ratios[lev] = amrex::IntVect(
            AMREX_D_DECL(2, 2, 1)
        );
    }

    // Allocate the dynamic pointer state cleanly with updated base dimensions
    MeshState = std::make_unique<AmrMeshState>(
        IO, amr_params.terrain_ref_lev, metadata, io_params.input_file, 
        /*AmrCore params*/
        real_domain, amr_params.max_amr_level, n_cell, coord_sys, ref_ratios
    );

    MeshState->Initialize();
}

void
HydroEXA::Compute() {
    BL_PROFILE("HydroEXA::Compute");
    

}

void 
HydroEXA::Finalize() {
    BL_PROFILE("HydroEXA::Finalize");
    // Perform any necessary cleanup, final I/O, etc.
}

