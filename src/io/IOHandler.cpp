
// STD includes
#include <string>

// HDF5
#include <hdf5.h>

// AMReX includes
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>

// local includes
#include <io/IOHandler.H>

void IOHandler::ReadHDF5Hyperslab(
    amrex::Array4<amrex::Real> const &arr,
    const amrex::Box &bx,
    const std::string &dataset_name,
    int dst_comp)
{
    const amrex::Real nodata_val = -9999;
    const int i_lo = bx.smallEnd(0);
    const int j_lo = bx.smallEnd(1);
    const int nx = bx.length(0);
    const int ny = bx.length(1);

    DatasetInfo &ds = hdf5_reader->Dataset(dataset_name);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ds.ndims == 2, "ReadHDF5Hyperslab() expects a 2D dataset.");

    const int h5_nx = ds.nx;
    const int h5_ny = ds.ny;

    amrex::Box file_box(amrex::IntVect(0, 0), amrex::IntVect(h5_nx - 1, h5_ny - 1));
    amrex::Box valid_box = bx & file_box;

    // Use value-initialization (0.0) to avoid any random junk leaking through 
    amrex::Gpu::PinnedVector<amrex::Real> host_buffer(nx * ny);

    if (valid_box.ok())
    {
        hid_t dataset_id = ds.dataset;
        hid_t filespace_id = H5Dget_space(dataset_id);

        const int read_nx = valid_box.length(0);
        const int read_ny = valid_box.length(1);

        hsize_t file_offset[2] = {
            static_cast<hsize_t>(valid_box.smallEnd(1)),
            static_cast<hsize_t>(valid_box.smallEnd(0))};

        hsize_t count[2] = {
            static_cast<hsize_t>(read_ny),
            static_cast<hsize_t>(read_nx)};

        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, file_offset, nullptr, count, nullptr);

        hsize_t mem_dims[2] = {static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)};
        hid_t memspace = H5Screate_simple(2, mem_dims, nullptr);

        hsize_t mem_offset[2] = {
            static_cast<hsize_t>(valid_box.smallEnd(1) - j_lo),
            static_cast<hsize_t>(valid_box.smallEnd(0) - i_lo)};

        H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, nullptr, count, nullptr);

        // FIX: Explicitly create a clean independent data transfer property list 
        // to bypass any conflicting global collective configurations.
        hid_t ind_dxpl = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(ind_dxpl, H5FD_MPIO_INDEPENDENT);

        H5Dread(
            dataset_id,
            hdf5_reader->Datatype(),
            memspace,
            filespace_id,
            ind_dxpl, // Use our verified independent transfer handle
            host_buffer.data());

        H5Pclose(ind_dxpl);
        H5Sclose(memspace);
        H5Sclose(filespace_id);
    }

    // Allocate Device Vector
    amrex::Gpu::DeviceVector<amrex::Real> device_buffer(host_buffer.size());
    
    // Copy asynchronous to device
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host_buffer.begin(), host_buffer.end(), device_buffer.begin());
    
    // FIX: Force the GPU stream to block and wait for the host data payload 
    // to complete transfer before continuing CPU processing.
    amrex::Gpu::streamSynchronize(); 

    const amrex::Real *device_ptr = device_buffer.data();

    const bool is_valid_ok = valid_box.ok();
    const int v_ilo = is_valid_ok ? valid_box.smallEnd(0) : int(1e9);
    const int v_ihi = is_valid_ok ? valid_box.bigEnd(0)  : int(-1e9);
    const int v_jlo = is_valid_ok ? valid_box.smallEnd(1) : int(1e9);
    const int v_jhi = is_valid_ok ? valid_box.bigEnd(1)  : int(-1e9);

    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            if (i >= v_ilo && i <= v_ihi && j >= v_jlo && j <= v_jhi)
            {
                int idx = (j - j_lo) * nx + (i - i_lo);
                arr(i, j, k, dst_comp) = device_ptr[idx];
            }
            else
            {
                arr(i, j, k, dst_comp) = nodata_val; 
            }
        });

    // Finalize stream before exit
    amrex::Gpu::streamSynchronize();
}


void IOHandler::ReadHDF5HyperslabComponents(
    amrex::Array4<amrex::Real> const &arr,
    const amrex::Box &bx,
    const std::string &dataset_name,
    int first_comp,
    int ncomp)
{

    const amrex::Real nodata_val = -9999;

    const int i_lo = bx.smallEnd(0);
    const int j_lo = bx.smallEnd(1);
    const int nx = bx.length(0);
    const int ny = bx.length(1);

    DatasetInfo &ds = hdf5_reader->Dataset(dataset_name);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ds.ndims == 3, "ReadHDF5HyperslabComponents() expects a 3D dataset.");

    const int h5_nx = ds.nx;
    const int h5_ny = ds.ny;

    hid_t dataset_id = ds.dataset;
    hid_t filespace_id = H5Dget_space(dataset_id);

    amrex::Box file_box(amrex::IntVect(0, 0), amrex::IntVect(h5_nx - 1, h5_ny - 1));
    amrex::Box valid_box = bx & file_box;

    // but avoiding the initialization loop saves a massive amount of CPU time.
    amrex::Gpu::PinnedVector<amrex::Real> host_buffer(nx * ny * ncomp);

    if (valid_box.ok())
    {
        const int read_nx = valid_box.length(0);
        const int read_ny = valid_box.length(1);

        hsize_t file_offset[3] = {
            static_cast<hsize_t>(first_comp),
            static_cast<hsize_t>(valid_box.smallEnd(1)),
            static_cast<hsize_t>(valid_box.smallEnd(0))};

        hsize_t count[3] = {
            static_cast<hsize_t>(ncomp),
            static_cast<hsize_t>(read_ny),
            static_cast<hsize_t>(read_nx)};

        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, file_offset, nullptr, count, nullptr);

        // Define the memory space layout to map directly into host_buffer
        hsize_t mem_dims[3] = {
            static_cast<hsize_t>(ncomp),
            static_cast<hsize_t>(ny),
            static_cast<hsize_t>(nx)};
            
        hid_t memspace = H5Screate_simple(3, mem_dims, nullptr);

        hsize_t mem_offset[3] = {
            0,
            static_cast<hsize_t>(valid_box.smallEnd(1) - j_lo),
            static_cast<hsize_t>(valid_box.smallEnd(0) - i_lo)};

        // Read directly into host_buffer at the correct offset
        H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, nullptr, count, nullptr);

        H5Dread(
            dataset_id,
            hdf5_reader->Datatype(),
            memspace,
            filespace_id,
            hdf5_reader->TransferProperty(),
            host_buffer.data());

        H5Sclose(memspace);
    }

    H5Sclose(filespace_id);

    // Transfer the data from pinned memory to device memory explicitly 
    // to avoid extreme PCIe zero-copy bottlenecks during kernel execution.
    amrex::Gpu::DeviceVector<amrex::Real> device_buffer(host_buffer.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host_buffer.begin(), host_buffer.end(), device_buffer.begin());
    amrex::Gpu::streamSynchronize(); 
    
    const amrex::Real *device_ptr = device_buffer.data();

    amrex::ParallelFor(
        bx,
        ncomp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
        {
            if (i < 0 || j < 0 || i >= h5_nx || j >= h5_ny)
            {
                arr(i, j, k, first_comp + n) = nodata_val;
            }
            else
            {
                int idx = n * nx * ny + (j - j_lo) * nx + (i - i_lo);
                arr(i, j, k, first_comp + n) = device_ptr[idx];
            }
        });

    amrex::Gpu::streamSynchronize();
}

void IOHandler::ReadHDF5Metadata(const std::string &dataset_name,
                                 HDF5SpatialMetadata &meta)
{
    // Layout:
    // [nx, ny, dx, dy, prob_lo_x, prob_lo_y, prob_hi_x, prob_hi_y]
    double global_payload[8] = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(hdf5_reader, "HDF5Reader has not been initialized.");

        auto ds = hdf5_reader->Dataset(dataset_name);
        
        hid_t dset_id = ds.dataset;
        hid_t space_id = H5Dget_space(dset_id);

        double dx = 1.0;
        double x_ll = 0.0;
        double dy = 1.0;
        double y_ll = 0.0;

        if (H5Aexists(dset_id, "dx") > 0)
        {
            hid_t attr = H5Aopen(dset_id, "dx", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &dx);
            H5Aclose(attr);
        }

        if (H5Aexists(dset_id, "dy") > 0)
        {
            hid_t attr = H5Aopen(dset_id, "dy", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &dy);
            H5Aclose(attr);
        }
        else
        {
            dy = dx;
        }

        if (H5Aexists(dset_id, "x_ll") > 0)
        {
            hid_t attr = H5Aopen(dset_id, "x_ll", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &x_ll);
            H5Aclose(attr);
        }

        if (H5Aexists(dset_id, "y_ll") > 0)
        {
            hid_t attr = H5Aopen(dset_id, "y_ll", H5P_DEFAULT);
            H5Aread(attr, H5T_NATIVE_DOUBLE, &y_ll);
            H5Aclose(attr);
        }

        global_payload[0] = ds.nx;
        global_payload[1] = ds.ny;
        global_payload[2] = dx;
        global_payload[3] = dy;
        global_payload[4] = x_ll;
        global_payload[5] = y_ll;
        global_payload[6] = x_ll + ds.nx * dx;
        global_payload[7] = y_ll + ds.ny * dy;

        // This must still be closed; H5Dget_space() creates a new object.
        H5Sclose(space_id);
    }

    amrex::ParallelDescriptor::Bcast(global_payload, 8,
                                     amrex::ParallelDescriptor::IOProcessorNumber());

    meta.global_nx = static_cast<int>(global_payload[0]);
    meta.global_ny = static_cast<int>(global_payload[1]);
    meta.dx = global_payload[2];
    meta.dy = global_payload[3];
    meta.prob_lo_x = global_payload[4];
    meta.prob_lo_y = global_payload[5];
    meta.prob_hi_x = global_payload[6];
    meta.prob_hi_y = global_payload[7];
}

void IOHandler::CloseHDF5()
{
    hdf5_reader.reset(); // closes the file immediately
}

void IOHandler::WritePlotfile(
    const amrex::Vector<amrex::MultiFab> &U,
    const amrex::Vector<amrex::MultiFab> &Terrain,
    int iteration,
    double time,
    const amrex::Vector<amrex::Geometry> &geom,
    const amrex::Vector<amrex::IntVect> &ref_ratio,
    int finest_level)
{
    // 1. Safely calculate the TRUE number of active, fully allocated levels.
    // This entirely prevents the "nullptr" gap segfault.
    int num_active_levels = 0;
    for (int lev = 0; lev <= finest_level; ++lev) {
        // If a level's BoxArray hasn't been defined yet, we break out.
        // We do NOT use 'continue', because AMReX requires contiguous level arrays.
        if (U[lev].boxArray().empty() || Terrain[lev].boxArray().empty()) {
            break; 
        }
        num_active_levels++;
    }

    if (num_active_levels == 0) {
        amrex::Print() << "[IOHandler] Warning: No valid levels found. Skipping plotfile.\n";
        return;
    }

    std::string plotfilename = amrex::Concatenate("plt", iteration, 5);
    amrex::Print() << "[IOHandler] Initializing plotfile bundle output: " << plotfilename 
                   << " with " << num_active_levels << " active levels.\n";

    // 2. Set variable string descriptors safely
    int ncomp_U = U[0].nComp();
    int ncomp_Terrain = Terrain[0].nComp();
    int total_comps = ncomp_U + ncomp_Terrain;

    amrex::Vector<std::string> varnames;
    varnames.push_back("h_fluid");
    varnames.push_back("hu_momentum");
    varnames.push_back("hv_momentum");
    varnames.push_back("z_bathymetry");
    
    // SAFETY CATCH: If you ever change ncomps (e.g., adding roughness), 
    // this prevents AMReX from crashing due to an out-of-bounds string read.
    while (varnames.size() < total_comps) {
        varnames.push_back("extra_comp_" + std::to_string(varnames.size()));
    }

    // 3. Set up pointers and temporary multi-component Fabs safely
    amrex::Vector<const amrex::MultiFab *> output_mf(num_active_levels);
    amrex::Vector<amrex::MultiFab> temp_mf(num_active_levels);

    for (int lev = 0; lev < num_active_levels; ++lev)
    {
        // Allocate local temporary space tracking identical layout geometry
        temp_mf[lev].define(U[lev].boxArray(), U[lev].DistributionMap(), total_comps, 0);

        // Map components sequentially across memory block offsets
        amrex::MultiFab::Copy(temp_mf[lev], U[lev],       0, 0,       ncomp_U,       0);
        amrex::MultiFab::Copy(temp_mf[lev], Terrain[lev], 0, ncomp_U, ncomp_Terrain, 0);

        // Guarantee a valid memory address is provided
        output_mf[lev] = &temp_mf[lev];
    }

    // 4. Track local integer state steps matching AMReX criteria signatures
    amrex::Vector<int> istep(num_active_levels, iteration);

    // 5. Fire parallel output dump sequence
    // Notice we pass the full, untouched `geom` and `ref_ratio` arrays. 
    // AMReX handles `max_level` sized tracking arrays natively without issue.
    amrex::WriteMultiLevelPlotfile(
        plotfilename,
        num_active_levels,
        output_mf,
        varnames,
        geom,           
        time,
        istep,
        ref_ratio);     

    amrex::Print() << "[IOHandler] Plotfile write finalized cleanly on disk.\n";
}