// include/solvers/SolverContext.H
#ifndef SOLVER_CONTEXT_H
#define SOLVER_CONTEXT_H


#include <memory>
#include <functional>

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>

struct SolverContext
{
    //==========================================================================
    // Simulation state
    //==========================================================================

    amrex::Vector<amrex::MultiFab>& U_new;
    amrex::Vector<amrex::MultiFab>& U_old;
    amrex::Vector<amrex::MultiFab>& Terrain;

    //==========================================================================
    // Mesh hierarchy
    //==========================================================================

    // These are required for allocating temporary MultiFabs
    const amrex::Vector<amrex::Geometry>& geom;
    const amrex::Vector<amrex::BoxArray>& grids;
    const amrex::Vector<amrex::DistributionMapping>& dmap;
    std::function<amrex::IntVect(int)> RefRatio;

    //==========================================================================
    // Boundary conditions
    //==========================================================================

    const amrex::Vector<amrex::BCRec>& UBCs;
    const amrex::Vector<amrex::BCRec>& TerrainBCs;

    //==========================================================================
    // Refluxing
    //==========================================================================

    amrex::Vector<std::unique_ptr<amrex::FluxRegister>>& FluxRegisters;

    bool do_reflux;
    int finest_level;

    //==========================================================================
    // Mesh services
    //==========================================================================

    std::function<void(
        int,
        amrex::Real,
        amrex::MultiFab&,
        const amrex::Vector<amrex::BCRec>&,
        int,
        int)> FillPatch;

    std::function<amrex::iMultiFab(
        const amrex::MultiFab&,
        const amrex::BoxArray&,
        const amrex::IntVect&,
        int,
        int)> makeFineMask;

};

#endif