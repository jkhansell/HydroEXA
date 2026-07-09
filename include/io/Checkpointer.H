#ifndef CHECKPOINTER_H
#define CHECKPOINTER_H

#include <AMReX_REAL.H>

// 1. Forward declaration 
class AmrMeshState;

class Checkpointer
{
public:
    void Write(
        const AmrMeshState &state,
        int iteration,
        amrex::Real time,
        const std::string &chkfile);

    void Read(
        AmrMeshState& state, 
        int& iteration, 
        amrex::Real& time, 
        int do_reflux, 
        const std::string& restart_chkfile);
};

#endif