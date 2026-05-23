#ifndef _HYDROEXA_HYDROLEVELBLD_H_
#define _HYDROEXA_HYDROLEVELBLD_H_

#include <AMReX_LevelBld.H>
#include "hydroexa/HydroLevel.H"

class HydroLevelBld : public amrex::LevelBld {
public:
    HydroLevelBld() = default;
    virtual ~HydroLevelBld() override = default;

    virtual void variableSetUp() override {
        HydroLevel::variableSetUp();
    }
    virtual void variableCleanUp() override {
        HydroLevel::variableCleanUp();
    }
    virtual amrex::AmrLevel* operator() () override {
        return new HydroLevel();
    }
    virtual amrex::AmrLevel* operator() (amrex::Amr& papa, int lev, 
                                         const amrex::Geometry& level_geom,
                                         const amrex::BoxArray& ba, 
                                         const amrex::DistributionMapping& dm,
                                         amrex::Real time) override {
        return new HydroLevel(papa, lev, level_geom, ba, dm, time);
    }
};

#endif
