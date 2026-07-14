#include <AMReX_ParmParse.H>
#include <HydroEXA.H>

HydroEXA::HydroEXA() {
    ReadParameters();
}

void HydroEXA::ReadParameters() {
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

        n = pp.countval("z_grad_thresh");
        if (n > 0) {
            pp.getarr("z_grad_thresh", physics_params.z_grad_thresh, 0, n);
        }
    }
}

void HydroEXA::Initialize() {
    BL_PROFILE("HydroEXA::Initialize");

    IO = std::make_shared<IOHandler>(io_params.input_file);
    HDF5SpatialMetadata metadata;

    IO->ReadHDF5Metadata(io_params.dataset_name, metadata);
    
    int refinement_scale = 1 << amr_params.terrain_ref_lev;
    int base_nx_raw = metadata.global_nx / refinement_scale;
    int base_ny_raw = metadata.global_ny / refinement_scale;

    int bf = 8; 
    amrex::ParmParse pp_amr("amr");
    pp_amr.query("blocking_factor", bf);

    int base_nx = ((base_nx_raw + bf - 1) / bf) * bf;
    int base_ny = ((base_ny_raw + bf - 1) / bf) * bf;
    int base_nz = 0;

    amrex::Real level0_dx = (metadata.prob_hi_x - metadata.prob_lo_x) / base_nx;
    amrex::Real level0_dy = (metadata.prob_hi_y - metadata.prob_lo_y) / base_ny;


    // amrex::Real padded_prob_hi_x = metadata.prob_lo_x + (static_cast<amrex::Real>(base_nx) * level0_dx);
    // amrex::Real padded_prob_hi_y = metadata.prob_lo_y + (static_cast<amrex::Real>(base_ny) * level0_dy);

    // =======================================================================
    // GEOMETRY VERIFICATION DIAGNOSTICS
    // =======================================================================
    amrex::Print() << "\n============================================================\n"
                   << "[GEOM DIAGNOSTIC] Verifying HDF5 to AMReX Structural Mapping\n"
                   << "============================================================\n"
                   << "  -> Raw HDF5 Resolution      : " << metadata.global_nx << " x " << metadata.global_ny << "\n"
                   << "  -> Reference Terrain Level  : " << amr_params.terrain_ref_lev << " (Refinement Scale: " << refinement_scale << "x)\n"
                   << "  -> Target Level 0 Raw Size  : " << base_nx_raw << " x " << base_ny_raw << "\n"
                   << "  -> Grid Blocking Factor     : " << bf << "\n"
                   << "  -> Padded Level 0 Cell Count: [" << base_nx << ", " << base_ny << ", " << base_nz << "]\n"
                   << "  -> Calculated Cell Spacing  : dx=" << level0_dx << " m, dy=" << level0_dy << " m\n"
                   << "  -> Physical Domain Low Bound: [" << metadata.prob_lo_x << ", " << metadata.prob_lo_y << ", 0.0]\n"
                   << "  -> Physical Domain High Bound: [" << metadata.prob_hi_x << ", " << metadata.prob_hi_y << ", 1.0]\n"
                   << "============================================================\n\n";

    // =======================================================================
    // CRITICAL FIX: OVERRIDE AMREX GEOMETRY PARAMETERS FORCEFULLY
    // =======================================================================
    
    amrex::Vector<amrex::Real> prob_lo = { metadata.prob_lo_x, metadata.prob_lo_y, 0.0 };
    amrex::Vector<amrex::Real> prob_hi = { metadata.prob_hi_x, metadata.prob_hi_y, 0.0 };
    amrex::Vector<int> n_cell_arr = { base_nx, base_ny, base_nz };
    amrex::RealBox real_domain(prob_lo.data(), prob_hi.data());
    // =======================================================================

    amrex::Vector<amrex::IntVect> ref_ratios(amr_params.max_amr_level);
    for (int lev = 0; lev < amr_params.max_amr_level; ++lev) {
        ref_ratios[lev] = amrex::IntVect(AMREX_D_DECL(2, 2, 0));
    }

    int coord_sys = 0; // Cartesian

    MeshState = std::make_unique<AmrMeshState>(
        IO, amr_params.terrain_ref_lev, metadata, io_params.input_file, 
        runtime_params, io_params, amr_params, physics_params,
        real_domain, amr_params.max_amr_level, n_cell_arr, coord_sys, ref_ratios
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

