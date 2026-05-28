
#include <hydroexa/HydroEXA.H>

using namespace amrex;

void
HydroEXA::restart (Amr& papa, std::istream& is, bool bReadSpecial)
{
    AmrLevel::restart(papa,is,bReadSpecial);

    if (do_reflux && level > 0) {
        flux_reg.reset(new FluxRegister(grids,dmap,crse_ratio,level,NUM_STATE));
    }

    buildMetrics();
}

void
HydroEXA::checkPoint (const std::string& dir, std::ostream& os, VisMF::How how, bool dump_old)
{
    AmrLevel::checkPoint(dir, os, how, dump_old);
}

void
HydroEXA::writePlotFile (const std::string& dir, std::ostream& os, VisMF::How how)
{
    BL_PROFILE("HydroEXA::writePlotFile()");
    AmrLevel::writePlotFile(dir, os, how);
}
