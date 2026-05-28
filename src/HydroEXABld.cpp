
#include <AMReX_LevelBld.H>
#include <hydroexa/HydroEXA.H>

using namespace amrex;

class HydroEXABld
    :
    public LevelBld
{
    virtual void variableSetUp () override;
    virtual void variableCleanUp () override;
    virtual AmrLevel *operator() () override;
    virtual AmrLevel *operator() (Amr&            papa,
                                  int             lev,
                                  const Geometry& level_geom,
                                  const BoxArray& ba,
                                  const DistributionMapping& dm,
                                  Real            time) override;
};

HydroEXABld HydroEXA_bld;

LevelBld*
getLevelBld ()
{
    return &HydroEXA_bld;
}

void
HydroEXABld::variableSetUp ()
{
    HydroEXA::variableSetUp();
}

void
HydroEXABld::variableCleanUp ()
{
    HydroEXA::variableCleanUp();
}

AmrLevel*
HydroEXABld::operator() ()
{
    return new HydroEXA;
}

AmrLevel*
HydroEXABld::operator() (Amr&            papa,
                    int             lev,
                    const Geometry& level_geom,
                    const BoxArray& ba,
                    const DistributionMapping& dm,
                    Real            time)
{
    return new HydroEXA(papa, lev, level_geom, ba, dm, time);
}
