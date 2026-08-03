#include <io/Checkpointer.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <utils/Logging.H>
#include <fstream>
#include <sstream>

void Checkpointer::Write(
    const CheckpointerContext& ctx,
    int iteration,
    amrex::Real time,
    const std::string& chk_file)
{
    const std::string checkpoint_name = amrex::Concatenate(chk_file, iteration);
    const int nlevels = ctx.finest_level + 1;

    LOG(INFO, "Writing checkpoint " + checkpoint_name);

    amrex::PreBuildDirectorHierarchy(checkpoint_name, "Level_", nlevels, true);

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::string header_name = checkpoint_name + "/Header";

        amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);

        std::ofstream header_file;
        header_file.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        header_file.open(header_name.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

        if (!header_file.good())
        {
            amrex::FileOpenFailed(header_name);
        }

        header_file.precision(17);

        header_file << "Checkpoint file for HydroEXA\n";
        header_file << ctx.finest_level << "\n";
        header_file << iteration << "\n";
        header_file << time << "\n";

        for (auto v : ctx.dt)       { header_file << v << " "; }  header_file << "\n";
        for (auto v : ctx.t_new)    { header_file << v << " "; }  header_file << "\n";
        for (auto v : ctx.t_old)    { header_file << v << " "; }  header_file << "\n";

        for (int lev = 0; lev <= ctx.finest_level; ++lev)
        {
            ctx.grids[lev].writeOn(header_file);
            header_file << '\n';
        }
    }

    for (int lev = 0; lev <= ctx.finest_level; ++lev)
    {
        amrex::VisMF::Write(ctx.U_new[lev],
            amrex::MultiFabFileFullPrefix(lev, checkpoint_name, "Level_", "U_new"));
        amrex::VisMF::Write(ctx.U_old[lev],
            amrex::MultiFabFileFullPrefix(lev, checkpoint_name, "Level_", "U_old"));
        amrex::VisMF::Write(ctx.Terrain[lev],
            amrex::MultiFabFileFullPrefix(lev, checkpoint_name, "Level_", "Terrain"));
    }
}

namespace {
    void GotoNextLine (std::istream& is)
    {
        constexpr std::streamsize bl_ignore_max { 100000 };
        is.ignore(bl_ignore_max, '\n');
    }
}

void Checkpointer::Read(
    const CheckpointerContext& ctx,
    int& iteration,
    amrex::Real& time,
    int do_reflux,
    const std::string& restart_chkfile)
{
    LOG(INFO, "Restart from checkpoint " + restart_chkfile);

    std::string file_path(restart_chkfile + "/Header");

    amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::GetIOBufferSize());
    amrex::Vector<char> file_char_ptr;
    amrex::ParallelDescriptor::ReadAndBcastFile(file_path, file_char_ptr);
    std::string file_str(file_char_ptr.dataPtr());
    std::istringstream is(file_str, std::istringstream::in);

    std::string line, word;
    std::getline(is, line); // title

    int finest_level;
    is >> finest_level;
    GotoNextLine(is);

    // iteration
    std::getline(is, line);
    { std::istringstream lis(line); lis >> iteration; }

    // time
    std::getline(is, line);
    { std::istringstream lis(line); lis >> time; }

    // dt
    std::getline(is, line);
    {
        std::istringstream lis(line);
        for (int i = 0; lis >> word; ++i) { ctx.dt[i] = std::stod(word); }
    }

    // t_new
    std::getline(is, line);
    {
        std::istringstream lis(line);
        for (int i = 0; lis >> word; ++i) { ctx.t_new[i] = std::stod(word); }
    }

    // t_old
    std::getline(is, line);
    {
        std::istringstream lis(line);
        for (int i = 0; lis >> word; ++i) { ctx.t_old[i] = std::stod(word); }
    }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        amrex::BoxArray ba;
        ba.readFrom(is);
        GotoNextLine(is);

        amrex::DistributionMapping dm(ba, amrex::ParallelDescriptor::NProcs());

        // BuildLevel is called by the caller (AmrMeshState) after Read returns.
        // Here we only store the box arrays for later reconstruction.
        ctx.grids[lev] = ba;
    }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        amrex::VisMF::Read(ctx.U_new[lev],
            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "U_new"));
        amrex::VisMF::Read(ctx.U_old[lev],
            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "U_old"));
        amrex::VisMF::Read(ctx.Terrain[lev],
            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "Terrain"));
    }
}