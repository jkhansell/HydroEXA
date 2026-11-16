#ifndef IOHANDLER_H_
#define IOHANDLER_H_

#include <memory>
#include <string>

#include <AMReX_MultiFabUtil.H>

#include <io/Checkpointer.H>
#include <io/CheckpointerContext.H>
#include <io/HDF5_File.H>

/**
 * \file io/IOHandler.H
 * \brief The single entry point for reading the input and writing output.
 *
 * Three jobs that share one thing -- the file prefixes and formats from the
 * inputs file -- and nothing else:
 *
 * - **read** terrain and initial fields out of the HDF5 input, one hyperslab
 *   per grid box;
 * - **checkpoint** through ``Checkpointer`` (native VisMF, currently the only
 *   format);
 * - **plot** through one of two free functions, native or HDF5, chosen by
 *   ``io.plot_type``.
 *
 * The abstraction stops at the file type.  The plotfile writers take raw AMReX
 * containers, so a caller does not have to know which format is in use, and the
 * readers take an ``Array4`` and a ``Box``, so the AMR code does not have to
 * know HDF5 exists.
 *
 * Splitting this from ``AmrMeshState`` is what keeps the state class free of
 * file formats: it asks for data at a box, and gets data at a box.
 */

/**
 * \brief Geometry of the input file, read once and carried by value.
 *
 * The static terrain layer has to be defined before any AMR level exists, so
 * its geometry cannot come from ``Geom(0)`` -- that would be circular.  It is
 * read from the dataset's attributes instead (``dx``, ``dy`` defaulting to
 * ``dx``, and ``x_ll``/``y_ll`` defaulting to zero), which makes the file the
 * authority on its own extent.
 */
struct HDF5SpatialMetadata
{
    int global_nx = 0;          ///< Cells in x, from the dataset extent.
    int global_ny = 0;          ///< Cells in y.
    amrex::Real dx = 1.0;       ///< Cell size in x, from the ``dx`` attribute.
    amrex::Real dy = 1.0;       ///< Cell size in y, from ``dy`` or ``dx``.
    amrex::Real prob_lo_x = 0.0;    ///< Lower corner in x.
    amrex::Real prob_lo_y = 0.0;    ///< Lower corner in y.
    amrex::Real prob_hi_x = 1.0;    ///< Upper corner in x, computed from the above.
    amrex::Real prob_hi_y = 1.0;    ///< Upper corner in y, computed from the above.
};

/**
 * \brief Owns the input reader and the checkpointer, and dispatches plotfiles.
 *
 * One instance per run, held by ``HydroEXA`` and passed to ``AmrMeshState`` as a
 * shared pointer, so the terrain reader and the output prefixes are shared
 * rather than duplicated.
 */
class IOHandler
{
public:
    /**
     * \brief Open the input file for parallel reading.
     * \param hdf5_path Path of the HDF5 input; also passed to the reader.
     */
    explicit IOHandler(const std::string &hdf5_path)
        : hdf5_reader(std::make_unique<HDF5Reader>(hdf5_path))
    {
    }

    /// Releases the reader, which closes the file collectively.
    ~IOHandler() = default;

    /**
     * \brief Global extents of a dataset, ignoring any component dimension.
     *
     * \param dataset_name Dataset path in the input file.
     * \param global_nx    [out] Cells in x.
     * \param global_ny    [out] Cells in y.
     */
    void GetHDF5Extents(const std::string &dataset_name, int &global_nx, int &global_ny);

    /**
     * \brief Read one component of a box into \p arr at component \p dst_comp.
     *
     * A single-component convenience wrapper around the multi-component reader.
     *
     * \param arr          Destination array.
     * \param bx           Box to read (clipped to the file's extent).
     * \param dataset_name Dataset path.
     * \param dst_comp     Destination component.
     */
    void ReadHDF5Hyperslab(amrex::Array4<amrex::Real> const &arr,
                           const amrex::Box &bx,
                           const std::string &dataset_name,
                           int dst_comp);

    /**
     * \brief Read \p ncomp components of a box, filling uncovered cells with NODATA.
     *
     * The requested box is intersected with the dataset's extent before the
     * read, and everything outside that intersection is set to \p nodata_val.
     * That is the mechanism by which a domain larger than the DEM gets an
     * invalid border rather than uninitialised memory -- and therefore why the
     * solver sees a sentinel it can recognise instead of a random bed.
     *
     * \param arr          Destination array.
     * \param bx           Box to read, in AMR index space.
     * \param dataset_name Dataset path; must be 3-D.
     * \param first_comp   First source component in the dataset.
     * \param ncomp        Number of components to read.
     * \param nodata_val   Sentinel written where the file has no data.
     */
    void ReadHDF5HyperslabComponents(amrex::Array4<amrex::Real> const &arr,
                                     const amrex::Box &bx,
                                     const std::string &dataset_name,
                                     int first_comp,
                                     int ncomp,
                                     const amrex::Real nodata_val
                                    );

    /**
     * \brief Read extents and geometry attributes, then broadcast them.
     *
     * \param dataset_name Dataset to inspect.
     * \param meta         [out] Extents, cell sizes and corners.
     */
    void ReadHDF5Metadata(const std::string &dataset_name,
                          HDF5SpatialMetadata &meta);

    /// Release the reader, closing the input file; safe to call once per run.
    void CloseHDF5();

    /**
     * \brief Write a checkpoint.
     * \param ctx       State, hierarchy and clock to save.
     * \param iteration Step counter for the Header.
     */
    void WriteCheckpoint(const CheckpointerContext& ctx, int iteration);

    /**
     * \brief Read a checkpoint.
     *
     * Implemented, but not called: restart is not wired into
     * ``AmrMeshState::Initialize`` yet, so setting ``io.restart_chkfile`` has no
     * effect on a run.
     *
     * \param ctx              [in,out] State and hierarchy to fill.
     * \param iteration        [out] Step counter from the file.
     * \param time             [out] Physical time from the file.
     * \param do_reflux        Whether the run refluxes.
     * \param restart_chkfile  Prefix of the checkpoint to read.
     */
    void ReadCheckpoint(
        const CheckpointerContext& ctx,
        int& iteration,
        amrex::Real& time,
        int do_reflux,
        const std::string& restart_chkfile);

    // In your IOHandler class public section:
    /**
     * \brief Write a plotfile in the format selected by ``io.plot_type``.
     *
     * Both writers concatenate the state and the terrain into one
     * multi-component field rather than writing two files, so a plotfile is
     * everything needed to restart an analysis.
     *
     * \param U            Per-level state.
     * \param Terrain      Per-level terrain.
     * \param iteration    Step counter, concatenated into the directory name.
     * \param time         Physical time, for the Header.
     * \param geom         Per-level geometry.
     * \param ref_ratio    Per-level refinement ratios.
     * \param finest_level Deepest level to write.
     */
    void WritePlotfile(
        const amrex::Vector<amrex::MultiFab> &U,
        const amrex::Vector<amrex::MultiFab> &Terrain,
        int iteration,
        amrex::Real time,
        const amrex::Vector<amrex::Geometry> &geom,
        const amrex::Vector<amrex::IntVect> &ref_ratio,
        int finest_level);

protected:
    std::unique_ptr<HDF5Reader> hdf5_reader;    ///< Input file reader; null after CloseHDF5.

private:
    Checkpointer checkpointer;                  ///< Checkpoint reader/writer.
    std::string chk_file{"chk"};                ///< Checkpoint prefix; see setter note.
    std::string plotfile_prefix{"plt"};         ///< Plotfile prefix.
    std::string plotfile_writer_type{"native"}; // "native" or "hdf5"

public:
    /// Set the checkpoint prefix.  Note: the driver does not call this today,
    /// so checkpoints always land under ``chk`` and ``io.chk_file`` is ignored.
    void SetCheckpointFile(const std::string& chk) { chk_file = chk; }

    /// Set the plotfile prefix, from ``io.plot_file``.
    void SetPlotfilePrefix(const std::string& prefix) { plotfile_prefix = prefix; }

    /// Select the plotfile writer: ``native`` (default) or ``hdf5``.
    void SetPlotfileWriterType(const std::string& type) { plotfile_writer_type = type; }
};

#endif
