#ifndef PARAMS_H
#define PARAMS_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

// ============================================================================
// Parameter Groups
// ============================================================================

struct RuntimeParameters
{
    amrex::Real etime{0.0};       // Current physical simulation time
    amrex::Real max_time{1.0};    // Simulation stop time (physical time)
    int iteration{0};             // Iteration counter
};

struct IOParameters 
{
    amrex::Real io_freq{1.0};                 // Frequency of I/O operations (in physical time)
    amrex::Real chk_freq{-1.0};               // Checkpoint frequency (in physical time)
    amrex::Real plot_freq{-1.0};              // Plotfile frequency (in physical time)
    bool async_io{false};                     // Whether to use asynchronous I/O
    std::string chk_type{"native"};           // Checkpoint file format
    std::string plot_file{"plt"};             // Plotfile prefix
    std::string chk_file{"chk"};              // Checkpoint file prefix
    std::string input_file{"input.h5"};       // HDF5 file containing initial conditions (e.g., DEM)
    std::string restart_chkfile{"restart"};   // Checkpoint file to restart from (empty means no restart)
};

struct AMRParameters 
{
    int regrid_int{2};            // Regridding frequency (in iterations)
    int do_reflux{1};             // Perform refluxing for flux correction at coarse-fine interfaces
    int do_subcycle{1};           // Subcycle in time (if false, all levels advance with the same dt)
    int terrain_ref_lev{0};       // Level at which to read in terrain
    int max_amr_level{0};         // Maximum level for the AMR grid (if -1, it is set to the number of levels)
};

struct PhysicsParameters 
{
    amrex::Real cfl{0.5};                         // CFL number for time step calculation
    std::string model{"Roe"};                     // Model selection: 0 for SWE, 1 for SWE-Exner
    amrex::Vector<amrex::Real> h_grad_thresh;     // Thresholds for tagging based on height gradients
    amrex::Vector<amrex::Real> umag_grad_thresh;  // Threshold for tagging based on velocity magnitude gradients
    amrex::Vector<amrex::Real> z_grad_thresh;     // Threshold for tagging based on velocity magnitude gradients
};


#endif