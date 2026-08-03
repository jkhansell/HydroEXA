#ifndef IOHANDLER_H_
#define IOHANDLER_H_

#include <memory>
#include <string>

#include <AMReX_MultiFabUtil.H>

#include <io/Checkpointer.H>
#include <io/CheckpointerContext.H>
#include <io/PlotFileWriter.H>
#include <io/HDF5_File.H>

// Embedded metadata container for geometric properties
struct HDF5SpatialMetadata
{
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
    explicit IOHandler(const std::string &hdf5_path)
        : hdf5_reader(std::make_unique<HDF5Reader>(hdf5_path))
    {
    }

    ~IOHandler() = default;

    void GetHDF5Extents(const std::string &dataset_name, int &global_nx, int &global_ny);

    void ReadHDF5Hyperslab(amrex::Array4<amrex::Real> const &arr,
                           const amrex::Box &bx,
                           const std::string &dataset_name,
                           int dst_comp);

    void ReadHDF5HyperslabComponents(amrex::Array4<amrex::Real> const &arr,
                                     const amrex::Box &bx,
                                     const std::string &dataset_name,
                                     int first_comp,
                                     int ncomp,
                                     const amrex::Real nodata_val
                                    );

    void ReadHDF5Metadata(const std::string &dataset_name,
                          HDF5SpatialMetadata &meta);

    void CloseHDF5();

    void SetPlotfileWriter(std::unique_ptr<PlotfileWriter> writer);

    void WriteCheckpoint(const CheckpointerContext& ctx, int iteration);

    void ReadCheckpoint(
        const CheckpointerContext& ctx,
        int& iteration,
        amrex::Real& time,
        int do_reflux,
        const std::string& restart_chkfile);

    // In your IOHandler class public section:
    void WritePlotfile(
        const amrex::Vector<amrex::MultiFab> &U,
        const amrex::Vector<amrex::MultiFab> &Terrain,
        int iteration,
        double time,
        const amrex::Vector<amrex::Geometry> &geom,
        const amrex::Vector<amrex::IntVect> &ref_ratio,
        int finest_level);

protected:
    std::unique_ptr<HDF5Reader> hdf5_reader;

private:
    Checkpointer checkpointer;
    std::unique_ptr<PlotfileWriter> plotfile_writer;
    std::string chk_file{"chk"};

public:
    void SetCheckpointFile(const std::string& chk) { chk_file = chk; }
};

#endif