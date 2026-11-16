#ifndef HDF5_PLOTFILE_H
#define HDF5_PLOTFILE_H

#include <string>

#include <AMReX_PlotFileUtil.H>

/**
 * \file io/HDF5PlotFile.H
 * \brief Plotfile writer using AMReX's HDF5 format.
 *
 * Same output as ``NativePlotFile.H`` -- one directory per call, state and
 * terrain concatenated into one multi-component field -- in a single-file,
 * optionally compressed format instead of a VisMF tree.
 *
 * The trade is portability for size: this path only exists when AMReX itself
 * was built with HDF5 and parallel HDF5, so a CPU-only build without those
 * libraries can still run the solver, it just cannot use ``io.plot_type =
 * hdf5``.  Readers then need AMReX's HDF5 plotfile support rather than the
 * plain VisMF reader.
 */

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
/**
 * \brief Write \p prefix\ + iteration as an HDF5 plotfile.
 *
 * The level count actually written is not \p finest_level: the loop stops at
 * the first level whose state or terrain has an empty BoxArray.  A regrid can
 * leave a level allocated but empty, and trusting \p finest_level would then
 * write a plotfile with a hole in it.
 *
 * Component names are fixed: ``h_fluid``, ``hu_momentum``, ``hv_momentum``,
 * ``z_bathymetry``, and then ``extra_comp_N`` for anything beyond -- which
 * today is the terrain's roughness/mask component.  The padding loop exists so
 * a component added to the state cannot overrun the name list.
 *
 * \param prefix       Directory-name prefix.
 * \param iteration    Step counter, concatenated with five digits.
 * \param time         Physical time, for the Header.
 * \param U            Per-level state.
 * \param Terrain      Per-level terrain; appended after the state components.
 * \param geom         Per-level geometry.
 * \param ref_ratio    Per-level refinement ratios.
 * \param finest_level Deepest level that may be written.
 * \param compression  HDF5 filter, e.g. ``gzip``; empty means uncompressed.
 */
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
