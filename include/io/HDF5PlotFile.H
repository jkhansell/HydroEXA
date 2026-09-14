#ifndef HDF5_PLOTFILE_H
#define HDF5_PLOTFILE_H

#include <AMReX_REAL.H>
#include <string>

// Forward declarations
namespace amrex {
    class MultiFab;
    struct Geometry;
}

// Write a multi-level plotfile using HDF5 format with MPI-IO.
// Takes raw AMReX types — no abstraction layer.
//
// Parameters:
//   prefix        : Plotfile filename prefix (e.g. "h5plt", "output")
//   iteration     : Current iteration number (used in filename concatenation)
//   time          : Physical time
//   U             : Vector of MultiFab per level (fluid variables)
//   Terrain       : Vector of MultiFab per level (terrain/bathymetry)
//   geom          : Vector of Geometry per level
//   ref_ratio     : Vector of ref ratio per level
//   finest_level  : Finest active level index
//   compression   : HDF5 compression filter name (e.g. "gzip", "shuffle,gzip", or "" for none)
void WriteHDF5Plotfile(
    const std::string& prefix,
    int iteration,
    amrex::Real time,
    const amrex::Vector<amrex::MultiFab>& U,
    const amrex::Vector<amrex::MultiFab>& Terrain,
    const amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<amrex::Real>& ref_ratio,
    int finest_level,
    const std::string& compression = "");

#endif // HDF5_PLOTFILE_H
