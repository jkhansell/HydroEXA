
// STD includes
#include <string>

// HDF5
#include <hdf5.h>

// AMReX includes
#include <AMReX_ParallelDescriptor.H>

// local includes 
#include <io/IOHandler.H>

void IOHandler::ReadHDF5Hyperslab(amrex::Array4<amrex::Real> const& arr, 
                                  const amrex::Box& bx, 
                                  const std::string& hdf5_filename,
                                  const std::string& dataset_name,
                                  int dst_comp) 
{
    // i_lo, j_lo, nx, ny represent what AMReX allocated for this tile
    const int i_lo = bx.smallEnd(0);
    const int j_lo = bx.smallEnd(1);
    const int nx   = bx.length(0);
    const int ny   = bx.length(1);

    // 1. Query HDF5 to find the exact matrix bounds inside the file
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, amrex::ParallelDescriptor::Communicator(), MPI_INFO_NULL);
    hid_t file_id = H5Fopen(hdf5_filename.c_str(), H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    hid_t dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    hid_t filespace_id = H5Dget_space(dataset_id);
    int file_ndims = H5Sget_simple_extent_ndims(filespace_id);
    hsize_t file_dims[3] = {0, 0, 0};
    H5Sget_simple_extent_dims(filespace_id, file_dims, NULL);

    int h5_ny = (file_ndims == 3) ? static_cast<int>(file_dims[1]) : static_cast<int>(file_dims[0]);
    int h5_nx = (file_ndims == 3) ? static_cast<int>(file_dims[2]) : static_cast<int>(file_dims[1]);

    // 2. Define a bounding box representing the valid HDF5 file content bounds
    amrex::Box file_box(amrex::IntVect(0,0), amrex::IntVect(h5_nx - 1, h5_ny - 1));

    // 3. Compute the geometric intersection using AMReX's built-in & operator
    amrex::Box valid_intersect_box = bx & file_box;

    // Create a host buffer that matches the exact shape AMReX allocated (including padding)
    const amrex::Real nodata_val = -9999.0;
    amrex::Gpu::PinnedVector<amrex::Real> host_buffer(nx * ny, nodata_val);

    if (valid_intersect_box.ok()) // True if this rank owns data inside the file bounds
    {
        // Dimensions of the valid chunk we are about to read
        int read_nx = valid_intersect_box.length(0);
        int read_ny = valid_intersect_box.length(1);

        hsize_t offset[3] = {0, 0, 0};
        hsize_t count[3]  = {1, 1, 1};
        hsize_t mem_dims[3] = {1, 1, 1};

        if (file_ndims == 3) {
            offset[0] = static_cast<hsize_t>(dst_comp);
            offset[1] = static_cast<hsize_t>(valid_intersect_box.smallEnd(1));
            offset[2] = static_cast<hsize_t>(valid_intersect_box.smallEnd(0));
            count[1]  = static_cast<hsize_t>(read_ny);
            count[2]  = static_cast<hsize_t>(read_nx);
            mem_dims[1] = count[1]; mem_dims[2] = count[2];
        } else {
            offset[0] = static_cast<hsize_t>(valid_intersect_box.smallEnd(1));
            offset[1] = static_cast<hsize_t>(valid_intersect_box.smallEnd(0));
            count[0]  = static_cast<hsize_t>(read_ny);
            count[1]  = static_cast<hsize_t>(read_nx);
            mem_dims[0] = count[0]; mem_dims[1] = count[1];
        }

        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
        hid_t memspace_id = H5Screate_simple(file_ndims, mem_dims, NULL);

        // Read the continuous unpadded segment from HDF5 into a temporary buffer
        amrex::Vector<amrex::Real> temp_buf(read_nx * read_ny);
        hid_t datatype = (sizeof(amrex::Real) == 8) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
        hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);

        H5Dread(dataset_id, datatype, memspace_id, filespace_id, xfer_plist, temp_buf.data());

        H5Pclose(xfer_plist);
        H5Sclose(memspace_id);

        // Map the valid unpadded data points into our main host buffer layout
        for (int j = valid_intersect_box.smallEnd(1); j <= valid_intersect_box.bigEnd(1); ++j) {
            for (int i = valid_intersect_box.smallEnd(0); i <= valid_intersect_box.bigEnd(0); ++i) {
                int temp_idx = (j - valid_intersect_box.smallEnd(1)) * read_nx + (i - valid_intersect_box.smallEnd(0));
                int host_idx = (j - j_lo) * nx + (i - i_lo);
                host_buffer[host_idx] = temp_buf[temp_idx];
            }
        }
    }
    else 
    {
        // MPI Collective requirement: Ranks completely inside the dead zone 
        // must still participate in the collective H5Dread call.
        H5Sselect_none(filespace_id);
        hid_t memspace_id = H5Screate(H5S_NULL);
        hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        
        amrex::Real dummy;
        hid_t datatype = (sizeof(amrex::Real) == 8) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
        H5Dread(dataset_id, datatype, memspace_id, filespace_id, xfer_plist, &dummy);
        
        H5Pclose(xfer_plist);
        H5Sclose(memspace_id);
    }

    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    // =====================================================================
    // STEP B: GPU Mapping and Dynamic Padding Assignment
    // =====================================================================
    amrex::Real const* raw_host_ptr = host_buffer.data();

    // Launch the kernel across the entire allocated box 'bx' (including padding)
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // 1. Is this cell outside the true physical domain of our HDF5 file?
        if (i >= h5_nx || j >= h5_ny || i < 0 || j < 0) {
            // Yes: Set it to a safe padding value (dry land or zero fluid height)
            arr(i, j, k, dst_comp) = 0.0;
        } 
        else {
            // No: This cell is inside the true domain. Fetch it from the buffer.
            int flat_idx = (j - j_lo) * nx + (i - i_lo);
            amrex::Real val = raw_host_ptr[flat_idx];

            if (amrex::Math::abs(val - nodata_val) < 1e-1) {
                arr(i, j, k, dst_comp) = 0.0; // Handled nodata mask cells from python
            } else {
                arr(i, j, k, dst_comp) = val; // Valid interior simulation data
            }
        }
    });

    amrex::Gpu::streamSynchronize();
}



void IOHandler::ReadHDF5Metadata(const std::string& hdf5_path, 
                                 const std::string& dataset_name, 
                                 HDF5SpatialMetadata& meta)
{
    // Continuous length-8 broadcast payload
    // Layout: [nx, ny, dx, dy, prob_lo_x, prob_lo_y, prob_hi_x, prob_hi_y]
    double global_payload[8] = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    // ---------------------------------------------------------
    // STEP 1: IO Processor (Rank 0) parses file parameters
    // ---------------------------------------------------------
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        hid_t file_id = H5Fopen(hdf5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id >= 0, "Failed to open HDF5 file.");

        hid_t dset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dset_id >= 0, "Failed to open dataset.");

        // Extract matrix dims
        hid_t space_id = H5Dget_space(dset_id);
        int ndims = H5Sget_simple_extent_ndims(space_id);
        // ACCEPT EITHER 2D OR 3D DATASETS
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ndims == 2 || ndims == 3, 
            "Dataset is neither a 2D matrix nor a 3D multi-component stack!");
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space_id, dims, NULL);
        
        // HDF5 Row-Major [Y][X] counts
        double nx_val = static_cast<double>(dims[1]);
        double ny_val = static_cast<double>(dims[0]);

        double dx = 1.0, dy = 1.0;
        double x_ll = 0.0, y_ll = 0.0;

        // Parse attributes
        if (H5Aexists(dset_id, "dx") > 0) {
            hid_t attr = H5Aopen(dset_id, "dx", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &dx);
            H5Aclose(attr);
        }
        if (H5Aexists(dset_id, "dy") > 0) {
            hid_t attr = H5Aopen(dset_id, "dy", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &dy);
            H5Aclose(attr);
        } else {
            dy = dx;
        }

        if (H5Aexists(dset_id, "x_ll") > 0) {
            hid_t attr = H5Aopen(dset_id, "x_ll", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &x_ll);
            H5Aclose(attr);
        }
        if (H5Aexists(dset_id, "y_ll") > 0) {
            hid_t attr = H5Aopen(dset_id, "y_ll", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &y_ll);
            H5Aclose(attr);
        }

        // Compute physical domain bounds
        double prob_hi_x = x_ll + (nx_val * dx);
        double prob_hi_y = y_ll + (ny_val * dy);

        // Pack the entire structural layout sequentially
        global_payload[0] = nx_val;
        global_payload[1] = ny_val;
        global_payload[2] = dx;
        global_payload[3] = dy;
        global_payload[4] = x_ll;
        global_payload[5] = y_ll;
        global_payload[6] = prob_hi_x;
        global_payload[7] = prob_hi_y;

        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
    }

    // ---------------------------------------------------------
    // STEP 2: Single Unified Collective Broadcast
    // ---------------------------------------------------------
    int io_rank = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(global_payload, 8, io_rank);

    // Unpack variables back into the structural configuration object
    meta.global_nx = static_cast<int>(global_payload[0]);
    meta.global_ny = static_cast<int>(global_payload[1]);
    meta.dx        = global_payload[2];
    meta.dy        = global_payload[3];
    meta.prob_lo_x = global_payload[4];
    meta.prob_lo_y = global_payload[5];
    meta.prob_hi_x = global_payload[6];
    meta.prob_hi_y = global_payload[7];
}