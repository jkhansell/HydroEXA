#ifndef MASKEDAVERAGEDOWN_H
#define MASKEDAVERAGEDOWN_H

#include <AMReX_Gpu.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <cmath>

namespace amrex {

    // GPU device kernel for volume-weighted down-averaging with NODATA/NaN masking.
    // A coarse cell becomes nodata when fewer than 50% of its fine cells are valid.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void masked_avgdown_with_vol (int i, int j, int k, int n,
                                  Array4<Real> const& crse,
                                  Array4<Real const> const& fine,
                                  Array4<Real const> const& fv,
                                  int ccomp, int fcomp,
                                  IntVect const& ratio,
                                  Real nodata_val) noexcept
    {
        const int facx = ratio[0];
        const int facy = ratio[1];
        const int facz = (AMREX_SPACEDIM == 3) ? ratio[2] : 1;

        const int ii = i * facx;
        const int jj = j * facy;
        const int kk = (AMREX_SPACEDIM == 3) ? k * facz : k;

        Real cd = 0.0, cv = 0.0;
        int nvalid = 0;
        int total  = facx * facy * facz;

        for (int kref = 0; kref < facz; ++kref) {
            for (int jref = 0; jref < facy; ++jref) {
                //#pragma unroll
                for (int iref = 0; iref < facx; ++iref) {
                    Real val = fine(ii+iref, jj+jref, kk+kref, n+fcomp);
                    if (!std::isnan(val) && std::abs(val - nodata_val) > 1e-3) {
                        ++nvalid;
                        cv += fv(ii+iref, jj+jref, kk+kref, 0);
                        cd   += val * fv(ii+iref, jj+jref, kk+kref, 0);
                    }
                }
            }
        }

        // Require >= 50% valid fine cells; otherwise mark nodata
        if (nvalid * 2 >= total && cv > 0.0) {
            crse(i, j, k, n + ccomp) = cd / cv;
        } else {
            crse(i, j, k, n + ccomp) = nodata_val;
        }
    }

    void masked_average_down (const amrex::MultiFab& S_fine, amrex::MultiFab& S_crse,
                              const amrex::Geometry& fgeom,  const amrex::Geometry& cgeom,
                              int scomp, int ncomp, const amrex::IntVect& ratio,
                              amrex::Real nodata_val = -9999.0);
}

#endif
