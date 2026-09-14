#ifndef BCFILL_H
#define BCFILL_H

#include <AMReX_BCRec.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include <utils/Params.H>


enum class ExternalBC : int
{
    Reflective, 
    Transmissive
};

inline int ScalarBC(int bc)
{
    return static_cast<ExternalBC>(bc) == ExternalBC::Reflective ? amrex::BCType::reflect_even : amrex::BCType::foextrap;
}

inline int NormalBC(int bc)
{
    return static_cast<ExternalBC>(bc) == ExternalBC::Reflective ? amrex::BCType::reflect_odd : amrex::BCType::foextrap;
}

inline int TangentialBC(int bc)
{
    return static_cast<ExternalBC>(bc) == ExternalBC::Reflective ? amrex::BCType::reflect_even : amrex::BCType::foextrap;
}

inline void SetScalarBC(amrex::BCRec& bc, BCParams& bc_params)
{
    bc.setLo(0, ScalarBC(bc_params.lo_bc[0]));
    bc.setHi(0, ScalarBC(bc_params.hi_bc[0]));
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, ScalarBC(bc_params.lo_bc[1]));
    bc.setHi(1, ScalarBC(bc_params.hi_bc[1]));
#endif
}

inline void SetXVelocityBC(amrex::BCRec& bc, BCParams& bc_params)
{
    bc.setLo(0, NormalBC(bc_params.lo_bc[0]));
    bc.setHi(0, NormalBC(bc_params.hi_bc[0]));
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, TangentialBC(bc_params.lo_bc[1]));
    bc.setHi(1, TangentialBC(bc_params.hi_bc[1]));
#endif
}

inline void SetYVelocityBC(amrex::BCRec& bc, BCParams& bc_params)
{
    bc.setLo(0, TangentialBC(bc_params.lo_bc[0]));
    bc.setHi(0, TangentialBC(bc_params.hi_bc[0]));
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1, NormalBC(bc_params.lo_bc[1]));
    bc.setHi(1, NormalBC(bc_params.hi_bc[1]));
#endif
}

struct ExternalBCFill
{
    AMREX_GPU_DEVICE
    void operator()(const amrex::IntVect& iv,
                    amrex::Array4<amrex::Real> const& a,
                    int dcomp, int ncomp,
                    amrex::GeometryData const& geom,
                    amrex::Real,
                    const amrex::BCRec* bcr,
                    int bcomp, int) const
    {

    }
};


#endif