#ifndef CHECKPOINTER_H
#define CHECKPOINTER_H

#include <AMReX_REAL.H>
#include <string>

#include <io/CheckpointerContext.H>

class Checkpointer
{
public:
    void Write(
        const CheckpointerContext& ctx,
        int iteration,
        amrex::Real time,
        const std::string& chk_file);

    void Read(
        const CheckpointerContext& ctx,
        int& iteration,
        amrex::Real& time,
        int do_reflux,
        const std::string& restart_chkfile);
};

#endif