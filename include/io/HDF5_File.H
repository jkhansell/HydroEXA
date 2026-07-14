#ifndef HDF5_FILE_H
#define HDF5_FILE_H

#include <string>
#include <unordered_map>

#include <hdf5.h>
#include <mpi.h>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

struct DatasetInfo
{
    hid_t dataset{-1};
    int nx{0};
    int ny{0};
    int ndims{0};
};

class HDF5Reader
{
public:

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

    ~HDF5Reader()
    {
        for (auto& ds : m_datasets)
            H5Dclose(ds.second.dataset);

        if (m_xfer >= 0)
            H5Pclose(m_xfer);

        if (m_file >= 0)
            H5Fclose(m_file);
    }

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

    hid_t Datatype() const
    {
        return m_datatype;
    }

    hid_t TransferProperty() const
    {
        return m_xfer;
    }

private:

    hid_t m_file{-1};
    hid_t m_xfer{-1};
    hid_t m_datatype{-1};

    std::unordered_map<std::string, DatasetInfo> m_datasets;
};

#endif