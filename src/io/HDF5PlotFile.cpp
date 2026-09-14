// HDF5PlotFile.cpp
// Free-function plotfile writer using HDF5 format with MPI-IO.
// Mirrors the structure of AMReX's native plotfile but stores data
// in HDF5 with a compatible directory layout for yt/VisIt reading.
//
// HDF5 plotfile structure (group-per-level + group-per-variable):
//   plotfilename/
//     level<N>/          (one group per active level)
//       cellCentered/
//         <varname>/     (one group per variable)
//           data/        (dataset: [ny,nx] per cell, chunked)
//           boxArray/    (dataset: tile bounds)
//           distribution/ (dataset: rank per tile)
//         meta/          (geometry metadata)
//       nodeCentered/    (optional, for node-centered data)
//     time/              (dataset: simulation time)
//     iteration/         (dataset: iteration number)
//     refRatio/          (dataset: ref ratios)
//     geometry/          (per-level geometry: dx, problo, prohi, is_periodic)

#include "HDF5PlotFile.H"

// STD
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

// HDF5 (parallel)
#include <hdf5.h>
#include <h5p.h>

// AMReX
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

// Local
#include <utils/Logging.H>

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Create a parallel HDF5 file with MPI-IO
hid_t open_parallel_file(const std::string& filename, hid_t fcpl_id) {
    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, fcpl_id, H5P_DEFAULT);
    if (file < 0) {
        amrex::Error("HDF5PlotFile: Failed to create plotfile '" + filename + "'");
    }
    return file;
}

// Write a simple 1D dataset
void write_dataset_1d(hid_t group, const char* name, const hsize_t* dims,
                      const void* data, hsize_t maxdims) {
    hid_t dataspace = H5Screate_simple(1, dims, maxdims);
    hid_t dataset = H5Dcreate2(group, name, H5T_NATIVE_INT64, dataspace,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

// Write a 2D dataset (ny, nx)
void write_dataset_2d(hid_t group, const char* name, const hsize_t* dims,
                      const void* data, const hsize_t* maxdims,
                      const hsize_t* chunks, hid_t dcpl_id) {
    hid_t dataspace = H5Screate_simple(2, dims, maxdims);
    hid_t dspace_stored = H5Screate_simple(2, maxdims, maxdims);

    hid_t dataset = H5Dcreate2(group, name, H5T_NATIVE_REAL, dataspace,
                               H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

    // Write the actual data (may be smaller than max dims for the last tile)
    H5Dwrite(dataset, H5T_NATIVE_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(dspace_stored);
}

// Write geometry attributes for a level
void write_level_geometry(hid_t level_group, const amrex::Geometry& geom,
                          int lev, const amrex::Vector<amrex::Real>& ref_ratio,
                          const amrex::Vector<amrex::Geometry>& all_geom) {
    // dx
    amrex::Vector<amrex::Real> dx(AMREX_SPACEDIM);
    geom.CellSize(dx.data());
    write_dataset_1d(level_group, "dx", &AMREX_SPACEDIM, dx.data(), AMREX_SPACEDIM);

    // problo
    amrex::Real problo[AMREX_SPACEDIM];
    geom.ProbLo(problo);
    write_dataset_1d(level_group, "prob_lo", &AMREX_SPACEDIM, problo, AMREX_SPACEDIM);

    // prohi
    amrex::Real prohi[AMREX_SPACEDIM];
    geom.ProbHi(prohi);
    write_dataset_1d(level_group, "prob_hi", &AMREX_SPACEDIM, prohi, AMREX_SPACEDIM);

    // is_periodic
    amrex::Vector<int> periodic(AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        periodic[i] = geom.IsPeriodic(i) ? 1 : 0;
    }
    write_dataset_1d(level_group, "is_periodic", &AMREX_SPACEDIM, periodic.data(), AMREX_SPACEDIM);

    // refRatio (for this level relative to level 0)
    if (lev > 0) {
        hsize_t refDims = 1;
        write_dataset_1d(level_group, "refRatio", &refDims,
                         &ref_ratio[lev], &refDims);
    }
}

} // anonymous namespace

// ============================================================================
// Main implementation
// ============================================================================

void WriteHDF5Plotfile(
    const std::string& prefix,
    int iteration,
    amrex::Real time,
    const amrex::Vector<amrex::MultiFab>& U,
    const amrex::Vector<amrex::MultiFab>& Terrain,
    const amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<amrex::Real>& ref_ratio,
    int finest_level,
    const std::string& compression)
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
        LOG_WARN("No valid levels found. Skipping HDF5 plotfile.");
        return;
    }

    const int ncomp_U = U[0].nComp();
    const int ncomp_Terrain = Terrain[0].nComp();
    const int total_comps = ncomp_U + ncomp_Terrain;

    // Variable names (same as native)
    amrex::Vector<std::string> varnames;
    varnames.push_back("h_fluid");
    varnames.push_back("hu_momentum");
    varnames.push_back("hv_momentum");
    varnames.push_back("z_bathymetry");
    while (static_cast<int>(varnames.size()) < total_comps)
    {
        varnames.push_back("extra_comp_" + std::to_string(varnames.size()));
    }

    // Build plotfile directory name
    std::string plotdirname = amrex::Concatenate(prefix, iteration, 5);

    // 2. Setup HDF5 parallel file properties
    hid_t fcpl_id = H5Pcreate(H5P_FILE_CREATE);
    H5Pset_driver(fcpl_id, H5P_FILE_DRIVER_MPIIO,
                  static_cast<void*>(amrex::ParallelDescriptor::Communicator().ptr()));

    // Apply compression if requested
    if (!compression.empty())
    {
        // Parse compression string (e.g. "shuffle,gzip" or "gzip")
        bool apply_shuffle = (compression.find("shuffle") != std::string::npos);
        bool apply_gzip = (compression.find("gzip") != std::string::npos);

        if (apply_gzip)
        {
            H5Pset_layout(fcpl_id, H5D_COMPRESS);
            int gzip_level = 4; // default moderate compression
            if (apply_shuffle)
            {
                H5Pset_shuffle(fcpl_id);
            }
            H5Pset_deflate(fcpl_id, gzip_level);
        }
        else
        {
            // Unknown compression — fall back to no compression
        }
    }

    hid_t file = open_parallel_file(plotdirname, fcpl_id);

    // 3. Write top-level metadata
    {
        hid_t root_group = H5Gopen(file, "/", H5P_DEFAULT);

        // Time
        hsize_t timeDims = 1;
        write_dataset_1d(root_group, "time", &timeDims, &time, &timeDims);

        // Iteration
        hsize_t iterDims = 1;
        write_dataset_1d(root_group, "iteration", &iterDims, &iteration, &iterDims);

        // Ref ratios (per level, starting from lev 1)
        hsize_t nRefDims = std::max(0, finest_level);
        if (nRefDims > 0)
        {
            amrex::Vector<hsize_t> refVals(nRefDims);
            for (int i = 0; i < finest_level; ++i)
            {
                refVals[i] = static_cast<hsize_t>(ref_ratio[i + 1]);
            }
            write_dataset_1d(root_group, "refRatio", &nRefDims, refVals.data(), &nRefDims);
        }

        H5Gclose(root_group);
    }

    // 4. Create compression property list for data datasets
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if (!compression.empty())
    {
        bool apply_shuffle = (compression.find("shuffle") != std::string::npos);
        bool apply_gzip = (compression.find("gzip") != std::string::npos);

        if (apply_gzip)
        {
            H5Pset_layout(dcpl_id, H5D_COMPRESS);
            if (apply_shuffle)
            {
                H5Pset_shuffle(dcpl_id);
            }
            H5Pset_deflate(dcpl_id, 4);
        }
    }

    // 5. Write per-level data
    for (int lev = 0; lev < num_active_levels; ++lev)
    {
        std::string levGroupName = "level" + std::to_string(lev);
        hid_t lev_group = H5Gcreate2(file, levGroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // --- cellCentered group ---
        hid_t cc_group = H5Gcreate2(lev_group, "cellCentered", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Write each variable
        for (int comp = 0; comp < total_comps; ++comp)
        {
            std::string varGroupName = varnames[comp];
            hid_t var_group = H5Gcreate2(cc_group, varGroupName.c_str(),
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            const amrex::MultiFab& source_mf = (comp < ncomp_U) ? U[lev] : Terrain[lev];
            int src_comp = (comp < ncomp_U) ? comp : comp - ncomp_U;

            // Write data tile by tile (HDF5 doesn't support non-contiguous MultiFab memory directly)
            const amrex::BoxArray& ba = source_mf.boxArray();
            const amrex::DistributionMapping& dm = source_mf.DistributionMapping();

            hsize_t numTiles = static_cast<hsize_t>(ba.size());

            // Store tile bounds
            amrex::Vector<hsize_t> boxData(numTiles * 2 * AMREX_SPACEDIM);
            for (hsize_t t = 0; t < numTiles; ++t)
            {
                amrex::Box bx = ba[t];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    boxData[t * 2 * AMREX_SPACEDIM + d * 2] = static_cast<hsize_t>(bx.smallEnd(d));
                    boxData[t * 2 * AMREX_SPACEDIM + d * 2 + 1] = static_cast<hsize_t>(bx.bigEnd(d));
                }
            }
            {
                hsize_t boxDims[1] = {static_cast<hsize_t>(boxData.size())};
                hsize_t boxMaxDims[1] = {static_cast<hsize_t>(boxData.size())};
                write_dataset_2d(var_group, "boxArray", boxDims, boxData.data(), boxMaxDims, boxDims, dcpl_id);
            }

            // Store distribution mapping (which rank owns each tile)
            amrex::Vector<hsize_t> distData(numTiles);
            for (hsize_t t = 0; t < numTiles; ++t)
            {
                distData[t] = static_cast<hsize_t>(dm[t]);
            }
            {
                hsize_t distDims[1] = {numTiles};
                hsize_t distMaxDims[1] = {numTiles};
                write_dataset_2d(var_group, "distribution", distDims, distData.data(), distMaxDims, distDims, dcpl_id);
            }

            // Write actual data — tile by tile, concatenated into a 2D array
            // For AMR, each tile is a 2D slice. We store all tiles as [numTiles, maxCells].
            // But the standard plotfile HDF5 convention stores per-tile data separately
            // or as a flat array with box bounds.
            //
            // Convention: store as a flat dataset "data" with shape [total_cells],
            // and use boxArray to reconstruct.

            // First pass: compute total cells and gather tile data
            hsize_t totalCells = 0;
            amrex::Vector<amrex::Real> tileData;
            amrex::Vector<hsize_t> tileCellCounts(numTiles);
            amrex::Vector<amrex::Real> tempBuffer;

            for (hsize_t t = 0; t < numTiles; ++t)
            {
                amrex::Box bx = ba[t];
                hsize_t nCells = static_cast<hsize_t>(bx.numPts());
                tileCellCounts[t] = nCells;
                totalCells += nCells;

                // Extract tile data
                tempBuffer.resize(nCells);
                amrex::Box validBx = bx & source_mf.boxArray()[t];
                amrex::Array4<const amrex::Real> arr = source_mf.const_array();
                for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
                {
                    amrex::IntVect iv(i, j, k);
                    if (validBx.contains(iv))
                    {
                        tempBuffer[static_cast<size_t>((iv - bx.smallEnd()) * AMREX_SPACEDIM +
                                                       (AMREX_SPACEDIM - 1))] = arr(iv, src_comp);
                    }
                    else
                    {
                        tempBuffer[static_cast<size_t>((iv - bx.smallEnd()) * AMREX_SPACEDIM +
                                                       (AMREX_SPACEDIM - 1))] = 0.0;
                    }
                }

                // Simpler: iterate the box in index order
                tileData.resize(totalCells);
                hsize_t offset = totalCells - nCells;
                int idx = 0;
                for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
                {
                    amrex::IntVect iv(i, j, k);
                    if (validBx.contains(iv))
                    {
                        tileData[offset + idx] = arr(iv, src_comp);
                    }
                    else
                    {
                        tileData[offset + idx] = 0.0;
                    }
                    idx++;
                }
            }

            // Write data as a flat dataset
            {
                hsize_t dataDims[1] = {totalCells};
                hsize_t dataMaxDims[1] = {totalCells};
                write_dataset_2d(var_group, "data", dataDims, tileData.data(), dataMaxDims, dataDims, dcpl_id);
            }

            H5Gclose(var_group);
        }

        H5Gclose(cc_group);

        // Write geometry for this level
        write_level_geometry(lev_group, geom[lev], lev, ref_ratio, geom);

        H5Gclose(lev_group);
    }

    // 6. Cleanup
    H5Pclose(dcpl_id);
    H5Pclose(fcpl_id);
    H5Fclose(file);

    LOG_INFO("Wrote HDF5 plotfile: " << plotdirname);
}
