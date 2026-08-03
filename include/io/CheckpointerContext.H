#ifndef CHECKPOINTER_CONTEXT_H
#define CHECKPOINTER_CONTEXT_H

#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Vector.H>

struct CheckpointerContext
{
    // Evolving fields — non-const so Checkpointer::Read can write into them
    amrex::Vector<amrex::MultiFab>& U_new;
    amrex::Vector<amrex::MultiFab>& U_old;
    amrex::Vector<amrex::MultiFab>& Terrain;

    // AMR hierarchy — grids is non-const because Read populates it from file
    amrex::Vector<amrex::BoxArray>&     grids;
    const amrex::Vector<amrex::DistributionMapping>& dmap;

    // Time state
    amrex::Vector<amrex::Real>& dt;
    amrex::Vector<amrex::Real>& t_new;
    amrex::Vector<amrex::Real>& t_old;

    // Metadata
    int finest_level;
    int iteration;
    const amrex::Vector<int>& istep;
};

#endif