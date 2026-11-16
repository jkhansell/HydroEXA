#ifndef CHECKPOINTER_H
#define CHECKPOINTER_H

#include <AMReX_REAL.H>
#include <string>

#include <io/CheckpointerContext.H>

/**
 * \file Checkpointer.H
 * \brief Read and write a restart checkpoint: a Header plus one VisMF per level.
 *
 * The format is AMReX's, not HydroEXA's: a text ``Header`` holding the level
 * count, the clock and the per-level box layout, followed by
 * ``Level_<n>/U_new``, ``Level_<n>/U_old`` and ``Level_<n>/Terrain`` as VisMF
 * directories.  Reusing VisMF means a checkpoint is readable by AMReX tooling
 * and needs no format version of its own; the cost is that it is a directory
 * tree rather than a single file.
 *
 * Only the native format is implemented.  ``io.chk_type`` is parsed but not
 * consulted: an HDF5 checkpoint would need a writer that does not exist yet.
 */

/**
 * \brief Serialises and parses checkpoints; stateless itself.
 *
 * The Header is written and read by rank 0 and broadcast, so every rank sees
 * the same level layout; the field data goes through VisMF, which handles the
 * per-rank files.
 */
class Checkpointer
{
public:
    /**
     * \brief Write one checkpoint.
     *
     * \param ctx       State, hierarchy and clock to save.
     * \param iteration Coarse-level step counter, for the Header.
     * \param time      Physical time, for the Header.
     * \param chk_file  Output prefix (``chk`` unless the driver set another).
     */
    void Write(
        const CheckpointerContext& ctx,
        int iteration,
        amrex::Real time,
        const std::string& chk_file);

    /**
     * \brief Read a checkpoint's Header and fields.
     *
     * Fills the clock and the per-level box arrays first, then reads the
     * MultiFabs.  The caller must have defined the levels at the recorded
     * layout before the field data can land in them -- which is the step the
     * state class has not implemented yet, so nothing calls this today.
     *
     * \param ctx              [in,out] State and hierarchy to fill.
     * \param iteration        [out] Step counter from the file.
     * \param time             [out] Physical time from the file.
     * \param do_reflux        Whether the run refluxes; a register cannot be
     *                         rebuilt from a file that did not store one.
     * \param restart_chkfile  Prefix of the checkpoint to read.
     */
    void Read(
        const CheckpointerContext& ctx,
        int& iteration,
        amrex::Real& time,
        int do_reflux,
        const std::string& restart_chkfile);
};

#endif
