#include <AMReX_MultiFabUtil.H>
#include <AMReX_Random.H>
#include <numbers>
#include <sstream>
#include <iostream>

#include <utils/MaskedAverageDown.H>
#include <AMReX_BLProfiler.H>

namespace amrex {

void masked_average_down (const amrex::MultiFab& S_fine, amrex::MultiFab& S_crse,
                          const amrex::Geometry& fgeom,  const amrex::Geometry& cgeom,
                          int scomp, int ncomp, const amrex::IntVect& ratio,
                          amrex::Real nodata_val)
{
    amrex::ignore_unused(cgeom);
    BL_PROFILE("amrex::masked_average_down");

    if (S_fine.is_nodal() || S_crse.is_nodal()) {
        amrex::Error("Can't use masked_average_down for nodal MultiFab!");
    }

    AMREX_ASSERT(S_crse.nComp() >= scomp + ncomp &&
                 S_fine.is_cell_centered() && S_crse.is_cell_centered());

    //
    // Coarsen() the fine stuff on processors owning the fine data.
    //
    const amrex::BoxArray& fine_BA = S_fine.boxArray();
    const amrex::DistributionMapping& fine_dm = S_fine.DistributionMap();
    amrex::BoxArray crse_S_fine_BA = fine_BA;
    crse_S_fine_BA.coarsen(ratio);

    amrex::MultiFab crse_S_fine(crse_S_fine_BA, fine_dm, ncomp, S_crse.nGrow(),
                                 amrex::MFInfo(), amrex::FArrayBoxFactory());

    amrex::MultiFab fvolume;
    fgeom.GetVolume(fvolume, fine_BA, fine_dm, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    
    for (amrex::MFIter mfi(crse_S_fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4<Real> const& crsearr = crse_S_fine.array(mfi);
        amrex::Array4<Real const> const& finearr = S_fine.const_array(mfi);
        amrex::Array4<Real const> const& finevolarr = fvolume.const_array(mfi);


        amrex::ParallelFor(bx, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        
        {
            masked_avgdown_with_vol(i,j,k,n,crsearr,finearr,finevolarr,
                                    0,scomp,ratio,nodata_val);
        });
    }

    S_crse.ParallelCopy(crse_S_fine, 0, scomp, ncomp);
}

} // namespace amrex
