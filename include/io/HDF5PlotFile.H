#ifndef HDF5_PLOTFILE_H
#define HDF5_PLOTFILE_H

#include <string>

#include <AMReX_PlotFileUtil.H>

// Write a multi-level plotfile using AMReX's HDF5 plotfile writer.
// Requires AMReX built with USE_HDF5=TRUE and parallel HDF5.
//
// Parameters:
//   prefix        : Plotfile filename prefix (e.g. "h5plt", "output")
//   iteration     : Current iteration number (used in filename concatenation)
//   time          : Physical time
//   U             : Vector of MultiFab per level (fluid variables)
//   Terrain       : Vector of MultiFab per level (terrain/bathymetry)
//   geom          : Vector of Geometry per level
//   ref_ratio     : Vector of ref ratio (IntVect) per level
//   finest_level  : Finest active level index
//   compression   : HDF5 compression filter string (e.g. "gzip", "" for none)
void WriteHDF5Plotfile(
    const std::string& prefix,
    int iteration,
    amrex::Real time,
    const amrex::Vector<amrex::MultiFab>& U,
    const amrex::Vector<amrex::MultiFab>& Terrain,
    const amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<amrex::IntVect>& ref_ratio,
    int finest_level,
    const std::string& compression = "");

#endif // HDF5_PLOTFILE_H
