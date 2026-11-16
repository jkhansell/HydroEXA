#ifndef NATIVE_PLOTFILE_H
#define NATIVE_PLOTFILE_H

#include <string>

#include <AMReX_PlotFileUtil.H>

/**
 * \file io/NativePlotFile.H
 * \brief Plotfile writer using AMReX's native (VisMF) format.
 *
 * The default writer, and the one that always exists: it needs nothing beyond
 * AMReX itself, so every build can produce output.  A plotfile is a directory
 * named by prefix and step, holding a text ``Header`` and one VisMF directory
 * per level.
 *
 * The state and the terrain are concatenated into a single multi-component
 * field rather than written as two files, so a plotfile carries both sides of
 * the ``h``/``z`` pair and an analysis script needs only one path.  The
 * component names are ``h_fluid``, ``hu_momentum``, ``hv_momentum``,
 * ``z_bathymetry``, then ``extra_comp_N`` for the terrain's remaining
 * components.
 *
 * This header deliberately exposes free functions rather than a class: the
 * writer holds no state, and the caller already has every container it needs.
 */

/**
 * \brief Write \p prefix\ + iteration as a native plotfile.
 *
 * The number of levels written is determined by scanning for the first level
 * with an empty state or terrain BoxArray, not by trusting \p finest_level: a
 * regrid can leave a level allocated and empty, and writing it would produce a
 * plotfile with a gap that readers cannot reconstruct.
 *
 * \param prefix       Directory-name prefix.
 * \param iteration    Step counter, concatenated with five digits.
 * \param time         Physical time, for the Header.
 * \param U            Per-level state.
 * \param Terrain      Per-level terrain; appended after the state components.
 * \param geom         Per-level geometry.
 * \param ref_ratio    Per-level refinement ratios.
 * \param finest_level Deepest level that may be written.
 */
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
