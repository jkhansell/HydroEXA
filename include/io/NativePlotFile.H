#ifndef NATIVE_PLOTFILE_H
#define NATIVE_PLOTFILE_H

#include <string>

#include <AMReX_PlotFileUtil.H>

// Write a multi-level plotfile using AMReX's native format.
// Takes raw AMReX types — no abstraction layer.
//
// Parameters:
//   prefix        : Plotfile filename prefix (e.g. "plt", "output")
//   iteration     : Current iteration number (used in filename concatenation)
//   time          : Physical time
//   U             : Vector of MultiFab per level (fluid variables)
//   Terrain       : Vector of MultiFab per level (terrain/bathymetry)
//   geom          : Vector of Geometry per level
//   ref_ratio     : Vector of ref ratio per level
//   finest_level  : Finest active level index
void WriteNativePlotfile(
    const std::string& prefix,
    int iteration,
    amrex::Real time,
    const amrex::Vector<amrex::MultiFab>& U,
    const amrex::Vector<amrex::MultiFab>& Terrain,
    const amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<amrex::IntVect>& ref_ratio,
    int finest_level);

#endif // NATIVE_PLOTFILE_H
