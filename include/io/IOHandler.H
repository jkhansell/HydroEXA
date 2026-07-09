#ifndef IOHANDLER_H_
#define IOHANDLER_H_

#include <memory>
#include <string>


#include <AMReX_MultiFabUtil.H>

#include <io/Checkpointer.H>
#include <io/PlotFileWriter.H>

// Embedded metadata container for geometric properties
struct HDF5SpatialMetadata {
    int global_nx = 0;
    int global_ny = 0;
    double dx = 1.0;
    double dy = 1.0;
    double prob_lo_x = 0.0;
    double prob_lo_y = 0.0;
    double prob_hi_x = 1.0;
    double prob_hi_y = 1.0;
};

class IOHandler
{
public:
    IOHandler() = default;
    ~IOHandler() = default;

    void GetHDF5Extents(const std::string& hdf5_path, const std::string& dataset_name, int& global_nx, int& global_ny);

    void ReadHDF5Hyperslab(amrex::Array4<amrex::Real> const& z_arr, 
                           const amrex::Box& bx, 
                           const std::string& hdf5_filename,
                           const std::string& dataset_name = "bathymetry",
                           int dst_comp = 0);


    void ReadHDF5Metadata(const std::string& hdf5_path, const std::string& dataset_name, HDF5SpatialMetadata& meta);

    void SetCheckpointer(std::unique_ptr<Checkpointer> checkpointer);

    void SetPlotfileWriter(std::unique_ptr<PlotfileWriter> writer);

    void WriteCheckpoint(
        const amrex::Vector<amrex::MultiFab> U, 
        const amrex::Vector<amrex::MultiFab> Terrain, 
        int iteration, 
        double time
    );

    void ReadCheckpoint(
        amrex::Vector<amrex::MultiFab> &U, 
        amrex::Vector<amrex::MultiFab> &Terrain,
        double &time, 
        const std::string& checkpoint_file
    );

    void WritePlotfile(
        const amrex::Vector<amrex::MultiFab> U, 
        const amrex::Vector<amrex::MultiFab> Terrain, 
        int iteration, 
        double time
    );

private:
    std::unique_ptr<Checkpointer> m_checkpointer;
    std::unique_ptr<PlotfileWriter> m_plotfile_writer;
};

#endif