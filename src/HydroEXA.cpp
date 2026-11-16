#include <AMReX_ParmParse.H>
#include <HydroEXA.H>
#include <utils/Logging.H>

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
        amrex::ParmParse pp("HydroEXA");
        pp.query("model", physics_params.model);
        pp.query("cfl", physics_params.cfl);
        pp.query("max_time", runtime_params.max_time);
        pp.query("mass_diag_freq", runtime_params.mass_diag_freq);

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

        // Boundary conditions: physical BC type integers (0-5)
        n = pp.countval("lo_bc");
        if (n > 0) {
            amrex::Vector<int> lo_bc_tmp(n);
            pp.getarr("lo_bc", lo_bc_tmp, 0, n);
            for (int i = 0; i < AMREX_SPACEDIM && i < n; ++i) {
                physics_params.bc_params.lo_bc[i] = lo_bc_tmp[i];
            }
        }
        n = pp.countval("hi_bc");
        if (n > 0) {
            amrex::Vector<int> hi_bc_tmp(n);
            pp.getarr("hi_bc", hi_bc_tmp, 0, n);
            for (int i = 0; i < AMREX_SPACEDIM && i < n; ++i) {
                physics_params.bc_params.hi_bc[i] = hi_bc_tmp[i];
            }
        }
    }
}

void HydroEXA::Initialize()
{
    BL_PROFILE("HydroEXA::Initialize");

    IO = std::make_shared<IOHandler>(io_params.input_file);
    IO->SetPlotfilePrefix(io_params.plot_file);
    IO->SetPlotfileWriterType(io_params.plot_type);

    HDF5SpatialMetadata metadata;
    IO->ReadHDF5Metadata("terrain", metadata);

    // =======================================================================
    // GEOMETRY CONTRACT
    //
    // The HDF5 physical domain is authoritative.
    //
    // HydroEXA requires:
    //   1. Positive dimensions
    //   2. dx == dy
    //   3. HDF5 dimensions compatible with terrain_ref_lev
    //   4. Level-0 dimensions compatible with blocking_factor
    //   5. The physical domain must not be padded or modified
    // =======================================================================

    amrex::ParmParse pp_amr("amr");
    int bf = 8;
    pp_amr.query("blocking_factor", bf);

    const int terrain_ref_lev = amr_params.terrain_ref_lev;
    const int refinement_scale = 1 << terrain_ref_lev;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        bf > 0, 
        "amr.blocking_factor must be greater than zero."
    );
    
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        terrain_ref_lev >= 0, 
        "amr.terrain_ref_lev must be non-negative."
    );

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        metadata.global_nx > 0 && metadata.global_ny > 0, 
        "HDF5 terrain dimensions must be positive."
    );

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        metadata.dx > 0.0 && metadata.dy > 0.0,
        "HDF5 terrain cell spacing must be positive."
    );

    // -----------------------------------------------------------------------
    // 1. HydroEXA requires square cells.
    //
    // The tolerance only accounts for floating-point representation. It must
    // not be used to accept materially different resolutions.
    // -----------------------------------------------------------------------

    const amrex::Real geometry_tolerance = 1.e-12 * std::max(amrex::Real(1.0), std::max(std::abs(metadata.dx), std::abs(metadata.dy)));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::abs(metadata.dx - metadata.dy) <= geometry_tolerance, 
        "HydroEXA requires square HDF5 cells: dx must equal dy."
    );

    const amrex::Real hdf5_cell_size = 0.5 * (metadata.dx + metadata.dy);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        metadata.global_nx % refinement_scale == 0, 
        "HDF5 global_nx must be divisible by 2^amr.terrain_ref_lev."
    );

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        metadata.global_ny % refinement_scale == 0, 
        "HDF5 global_ny must be divisible by 2^amr.terrain_ref_lev."
    );

    const int base_nx = metadata.global_nx / refinement_scale;
    const int base_ny = metadata.global_ny / refinement_scale;
    const int base_nz = 0;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        base_nx > 0 && base_ny > 0, 
        "Derived level-0 grid dimensions must be positive."
    );
    
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        base_nx % bf == 0, 
        "Level-0 nx is incompatible with amr.blocking_factor. HydroEXA will not pad the physical domain."
    );
    
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        base_ny % bf == 0, 
        "Level-0 ny is incompatible with amr.blocking_factor. HydroEXA will not pad the physical domain."
    );

    const amrex::Real level0_cell_size = hdf5_cell_size * static_cast<amrex::Real>(refinement_scale);
    const amrex::Real level0_dx = level0_cell_size;
    const amrex::Real level0_dy = level0_cell_size;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::abs(level0_dx - level0_dy) <= geometry_tolerance, 
        "HydroEXA requires square level-0 cells: dx must equal dy."
    );

    const amrex::Real domain_length_x = metadata.prob_hi_x - metadata.prob_lo_x;
    const amrex::Real domain_length_y = metadata.prob_hi_y - metadata.prob_lo_y;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        domain_length_x > 0.0 && domain_length_y > 0.0, 
        "HDF5 physical domain must have positive extents."
    );

    const amrex::Real extent_tolerance = 1.e-10 * std::max(amrex::Real(1.0), std::max(std::abs(domain_length_x), std::abs(domain_length_y)));

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::abs(static_cast<amrex::Real>(base_nx) * level0_dx - domain_length_x) <= extent_tolerance,
        "HDF5 x extent is inconsistent with global_nx, terrain_ref_lev, and dx."
    );
    
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::abs(static_cast<amrex::Real>(base_ny) * level0_dy - domain_length_y) <= extent_tolerance,
        "HDF5 y extent is inconsistent with global_ny, terrain_ref_lev, and dy."
    );

    LOG(INFO, "\n============================================================\n"
              "[GEOM DIAGNOSTIC] HydroEXA Geometry Verification\n"
              "============================================================\n"
              "  -> HDF5 Resolution          : " + std::to_string(metadata.global_nx) + " x " + std::to_string(metadata.global_ny) + "\n"
              "  -> HDF5 Cell Size           : " + std::to_string(hdf5_cell_size) + " m\n"
              "  -> HDF5 dx / dy             : " + std::to_string(metadata.dx) + " / " + std::to_string(metadata.dy) + " m\n"
              "  -> Terrain Reference Level  : " + std::to_string(terrain_ref_lev) + "\n"
              "  -> Refinement Scale         : " + std::to_string(refinement_scale) + "x\n"
              "  -> Level 0 Resolution       : " + std::to_string(base_nx) + " x " + std::to_string(base_ny) + "\n"
              "  -> Blocking Factor          : " + std::to_string(bf) + "\n"
              "  -> Level 0 Cell Size        : " + std::to_string(level0_cell_size) + " m\n"
              "  -> Physical Domain Low      : [" + std::to_string(metadata.prob_lo_x) + ", " + std::to_string(metadata.prob_lo_y) + "]\n"
              "  -> Physical Domain High     : [" + std::to_string(metadata.prob_hi_x) + ", " + std::to_string(metadata.prob_hi_y) + "]\n"
              "  -> Physical Extent          : " + std::to_string(domain_length_x) + " x " + std::to_string(domain_length_y) + " m\n"
              "  -> Geometry Status          : VALID\n"
              "============================================================\n\n");

    amrex::Vector<amrex::Real> prob_lo = {metadata.prob_lo_x, metadata.prob_lo_y, 0.0};
    amrex::Vector<amrex::Real> prob_hi = {metadata.prob_hi_x, metadata.prob_hi_y, 0.0};
    amrex::Vector<int> n_cell_arr = {base_nx, base_ny, base_nz};
    amrex::RealBox real_domain(prob_lo.data(), prob_hi.data());

    amrex::Vector<amrex::IntVect> ref_ratios(amr_params.max_amr_level);
    for (int lev = 0; lev < amr_params.max_amr_level; ++lev) ref_ratios[lev] = amrex::IntVect(AMREX_D_DECL(2, 2, 0));

    const int coord_sys = 0;

    MeshState = std::make_unique<AmrMeshState>(
        IO, amr_params.terrain_ref_lev, metadata, io_params.input_file,
        runtime_params, io_params, amr_params, physics_params,
        real_domain, amr_params.max_amr_level, n_cell_arr, coord_sys, ref_ratios);

    MeshState->Initialize();
}

void
HydroEXA::Compute() {
    BL_PROFILE("HydroEXA::Compute");

    amrex::Real plot_time_next = (io_params.plot_freq > 0) ? io_params.plot_freq : -1.0;
    amrex::Real chk_time_next  = (io_params.chk_freq  > 0) ? io_params.chk_freq  : -1.0;
    amrex::Real mass_time_next = (runtime_params.mass_diag_freq > 0) ? runtime_params.mass_diag_freq : -1.0;
    int plot_iter = 0;
    int chk_iter  = 0;

    MeshState->WritePlotfile(plot_iter, runtime_params.etime);
    plot_iter++;

    while (runtime_params.etime < runtime_params.max_time)
    {
        const double wall_start = amrex::ParallelDescriptor::second();

        // CFL
        MeshState->ComputeDt();

        // Advance
        MeshState->TimeStepWithSubcycling(0, runtime_params.etime, runtime_params.iteration);

        runtime_params.etime = MeshState->t_new[0];
        runtime_params.iteration++;

        // Mass & momentum conservation diagnostics
        if (runtime_params.mass_diag_freq > 0 &&
            runtime_params.etime >= mass_time_next)
        {
            MeshState->ComputeMassDiagnostics(runtime_params.etime, runtime_params.iteration);
            mass_time_next += runtime_params.mass_diag_freq;
        }

        const double wall_end = amrex::ParallelDescriptor::second();

        if (Verbose() && amrex::ParallelDescriptor::IOProcessor())
        {
            LOG(INFO,
                "[Step "
                + std::to_string(runtime_params.iteration)
                + "]"
                + " t=" + std::to_string(runtime_params.etime)
                + "  dt=" + std::to_string(MeshState->dt[0])
                + "  wall=" + std::to_string(wall_end - wall_start) + " s\n");
        }

        // Plotfiles
        if (io_params.plot_freq > 0 &&
            runtime_params.etime >= plot_time_next)
        {
            MeshState->WritePlotfile(plot_iter, runtime_params.etime);
            plot_time_next += io_params.plot_freq;
            plot_iter++;
        }

        // Checkpoints
        if (io_params.chk_freq > 0 &&
            runtime_params.etime >= chk_time_next)
        {
            MeshState->WriteCheckpoint();
            chk_time_next += io_params.chk_freq;
            chk_iter++;
        }
    }

    if (Verbose()) {
        LOG(INFO, "\nSimulation complete: t = " + std::to_string(runtime_params.etime)
               + " (" + std::to_string(runtime_params.iteration) + " coarse steps)\n");
    }
}

void 
HydroEXA::Finalize() {
    BL_PROFILE("HydroEXA::Finalize");
    // Perform any necessary cleanup, final I/O, etc.
}

