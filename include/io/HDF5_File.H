#ifndef HDF5_FILE_H
#define HDF5_FILE_H

#include <string>
#include <unordered_map>

#include <hdf5.h>
#include <mpi.h>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

/**
 * \file io/HDF5_File.H
 * \brief Read-only, parallel HDF5 access to the input file.
 *
 * The input file holds terrain and the initial fields; the solver never writes
 * to it.  That is why this reader has no write path, no dataset creation and no
 * chunking policy -- it opens the file, caches the dataset handles, and hands
 * out the descriptors a hyperslab read needs.
 *
 * Two conventions are fixed here and assumed everywhere above:
 *
 * - **Precision is the build's.**  The HDF5 native type is chosen from
 *   ``sizeof(amrex::Real)``, so a DOUBLE build reads the data as doubles.  A
 *   file written as float64 read by a SINGLE build is not converted, it is
 *   reinterpreted; the inputs file and the build must agree.
 * - **The last two dimensions are spatial.**  A 3-D dataset is
 *   ``(component, ny, nx)`` and a 2-D dataset is ``(ny, nx)``, so a
 *   single-component field may carry a leading singleton.  Index order is
 *   row-major in ``y`` and ``x``, matching AMReX's ``(i, j)``.
 */

/**
 * \brief Cached handle and extents for one dataset in the input file.
 *
 * Opened lazily by ``HDF5Reader::Dataset`` and closed by the reader's
 * destructor, so a dataset is opened at most once per run.
 */
struct DatasetInfo
{
    hid_t dataset{-1};  ///< HDF5 dataset handle; -1 until opened.
    int nx{0};          ///< Extent in x (the last dimension).
    int ny{0};          ///< Extent in y (the second-to-last dimension).
    int ndims{0};       ///< Rank of the dataset as stored, 2 or 3.
};

/**
 * \brief Owns the open file and the dataset cache for the run.
 *
 * MPIO with the AMReX communicator: the file is opened collectively, so every
 * rank must construct the reader, and the destructor's ``H5Fclose`` is a
 * collective operation as well.  Transfers use ``H5FD_MPIO_INDEPENDENT``, which
 * lets each rank read its own hyperslab without a collective call -- the
 * decomposition is decided by the caller (``IOHandler``), not by HDF5.
 */
class HDF5Reader
{
public:

    /**
     * \brief Open \p filename for reading on every rank.
     *
     * Aborts if the file cannot be opened; there is no meaningful degraded mode
     * for a missing terrain file.
     *
     * \param filename Path to the input HDF5 file.
     */
    explicit HDF5Reader(const std::string& filename)
    {
        hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist, amrex::ParallelDescriptor::Communicator(), MPI_INFO_NULL);

        m_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist);

        H5Pclose(plist);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_file >= 0, "Failed to open HDF5 file.");

        m_xfer = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(m_xfer, H5FD_MPIO_INDEPENDENT);

        m_datatype = (sizeof(amrex::Real) == 8)
            ? H5T_NATIVE_DOUBLE
            : H5T_NATIVE_FLOAT;
    }

    /// Close every cached dataset, then the transfer list, then the file.
    ~HDF5Reader()
    {
        for (auto& ds : m_datasets)
            H5Dclose(ds.second.dataset);

        if (m_xfer >= 0)
            H5Pclose(m_xfer);

        if (m_file >= 0)
            H5Fclose(m_file);
    }

    /**
     * \brief Look up (and on first use, open) a dataset by name.
     *
     * The returned reference stays valid for the reader's lifetime: the map is
     * only ever grown, and rehashing does not move the elements.
     *
     * \param name Dataset path inside the file.
     * \return Extents and handle; the handle is -1 only for a failed open,
     *         which aborts rather than returning.
     */
    DatasetInfo& Dataset(const std::string& name)
    {
        auto it = m_datasets.find(name);
        if (it != m_datasets.end())
            return it->second;

        DatasetInfo info;
        info.dataset = H5Dopen2(m_file, name.c_str(), H5P_DEFAULT);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(info.dataset >= 0, "Failed to open dataset.");

        hid_t filespace = H5Dget_space(info.dataset);

        info.ndims = H5Sget_simple_extent_ndims(filespace);

        hsize_t dims[3] = {0,0,0};
        H5Sget_simple_extent_dims(filespace, dims, nullptr);

        // 3-D means a leading component dimension; only the trailing two are
        // spatial, which is what lets a single-component field be stored with a
        // singleton first axis and read by the same code.
        if (info.ndims == 3)
        {
            info.ny = static_cast<int>(dims[1]);
            info.nx = static_cast<int>(dims[2]);
        }
        else
        {
            info.ny = static_cast<int>(dims[0]);
            info.nx = static_cast<int>(dims[1]);
        }

        H5Sclose(filespace);

        auto result = m_datasets.emplace(name, std::move(info));
        return result.first->second;
    }

    /// Native HDF5 type to transfer as; fixed by the build's precision.
    hid_t Datatype() const
    {
        return m_datatype;
    }

    /// Transfer property list for a hyperslab read (independent, as set above).
    hid_t TransferProperty() const
    {
        return m_xfer;
    }

private:

    hid_t m_file{-1};       ///< Open file handle.
    hid_t m_xfer{-1};       ///< Dataset transfer properties.
    hid_t m_datatype{-1};   ///< Native type matching ``amrex::Real``.

    std::unordered_map<std::string, DatasetInfo> m_datasets;  ///< Name to handle, opened once.
};

#endif
