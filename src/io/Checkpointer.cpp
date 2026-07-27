#include <io/Checkpointer.H>
#include <state/AmrMeshState.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <utils/Logging.H>
#include <fstream>
#include <sstream>

void Checkpointer::Write(
    const AmrMeshState& state, 
    int iteration, 
    amrex::Real time, 
    const std::string& chk_file)
{
    const std::string checkpoint_name = amrex::Concatenate(chk_file, iteration);

    LOG(INFO, "Writing checkpoint " + checkpoint_name);

    const int finest_level = state.FinestLevel();
    const int nlevels = finest_level + 1;

    amrex::PreBuildDirectorHierarchy(
        checkpoint_name,
        "Level_",
        nlevels,
        true);

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::string header_name = checkpoint_name + "/Header";

        amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);

        std::ofstream header_file;

        header_file.rdbuf()->pubsetbuf(
            io_buffer.dataPtr(),
            io_buffer.size());

        header_file.open(
            header_name.c_str(),
            std::ofstream::out |
            std::ofstream::trunc |
            std::ofstream::binary);

        if (!header_file.good())
        {
            amrex::FileOpenFailed(header_name);
        }

        header_file.precision(17);

        header_file << "Checkpoint file for HydroEXA\n";

        header_file << finest_level << "\n";

        header_file << iteration << "\n";

        header_file << time << "\n";

        const auto& dt = state.GetDt();
        for (auto v : dt)
        {
            header_file << v << " ";
        }
        header_file << "\n";

        const auto& t_new = state.GetTNew();
        for (auto v : t_new)
        {
            header_file << v << " ";
        }
        header_file << "\n";

        const auto& t_old = state.GetTOld();
        for (auto v : t_old)
        {
            header_file << v << " ";
        }
        header_file << "\n";

        for (int lev = 0; lev <= finest_level; ++lev)
        {
            state.boxArray(lev).writeOn(header_file);
            header_file << '\n';
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        amrex::VisMF::Write(state.GetUNew(lev),
            amrex::MultiFabFileFullPrefix(lev, checkpoint_name, "Level_", "U_new"));

        amrex::VisMF::Write(state.GetUOld(lev),
            amrex::MultiFabFileFullPrefix(lev, checkpoint_name, "Level_", "U_old"));

        amrex::VisMF::Write(state.GetTerrain(lev),
            amrex::MultiFabFileFullPrefix(lev, checkpoint_name, "Level_", "Terrain"));
    }
}

namespace {
    // utility to skip to next line in Header
    void GotoNextLine (std::istream& is)
    {
        constexpr std::streamsize bl_ignore_max { 100000 };
        is.ignore(bl_ignore_max, '\n');
    }
}

void Checkpointer::Read(
    AmrMeshState& state, 
    int& iteration, 
    amrex::Real& time, 
    int do_reflux, 
    const std::string& restart_chkfile) {

    LOG(INFO, "Restart from checkpoint " + restart_chkfile);

    // Header
    std::string File(restart_chkfile + "/Header");

    amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::GetIOBufferSize());

    amrex::Vector<char> fileCharPtr;
    amrex::ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    // read in title line
    std::getline(is, line);

    int finest_level;
    // read in finest_level
    is >> finest_level;

    state.SetFinestLevel(finest_level);
    state.ResizeLevels(finest_level + 1);
    
    GotoNextLine(is);

    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0; 
        auto& dt = state.GetDt();
        while (lis >> word) {
            dt[i++] = std::stod(word);
        }
    }

    std::getline(is, line);
    { // Read an array of t_new
        std::istringstream lis(line);
        int i = 0; 
        auto& t_new = state.GetTNew();
        while (lis >> word) {
            t_new[i++] = std::stod(word);
        }
    }

    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        auto& t_old = state.GetTOld();
        while (lis >> word) {
            t_old[i++] = std::stod(word);
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        amrex::BoxArray ba;

        ba.readFrom(is);
        GotoNextLine(is);

        amrex::DistributionMapping dm(ba, amrex::ParallelDescriptor::NProcs());

        state.BuildLevel(lev, ba, dm);

        if (lev > 0 && do_reflux) {
            state.BuildFluxRegister(lev);
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        amrex::VisMF::Read(state.GetUNew(lev),
            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "U_new"));

        amrex::VisMF::Read(state.GetUOld(lev),
            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "U_old"));

        amrex::VisMF::Read(state.GetTerrain(lev),
            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "Terrain"));
    }
}