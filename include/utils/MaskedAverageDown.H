#ifndef MASKEDAVERAGEDOWN_H
#define MASKEDAVERAGEDOWN_H

#include <AMReX_Gpu.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <cmath>

/**
 * \file MaskedAverageDown.H
 * \brief Volume-weighted fine-to-coarse restriction that keeps NODATA out.
 *
 * AMReX's ``average_down`` treats every fine cell as data.  Terrain does not
 * work that way here: a DEM is a rectangle of valid cells surrounded by
 * -9999.0 sentinels, and an average that includes sentinels produces a coarse
 * bed elevation near -3000 m, which then propagates through the solver as a
 * cliff at the edge of the domain.
 *
 * So the restriction is a *masked* one: invalid fine cells are dropped from
 * both numerator and denominator, and a coarse cell with too few valid
 * children is itself marked invalid.  Invalid therefore stays invalid, and
 * never becomes a wrong number.
 */

namespace amrex {

    /**
     * \brief Restrict one coarse cell from its fine children, ignoring NODATA.
     *
     * The kernel implements the cell-local half of the operation; the
     * whole-MultiFab pass is ``masked_average_down``.
     *
     * Volume weighting is what makes the average well defined when children do
     * not share a volume: on a non-Cartesian geometry (``RZ``, where ``fv``
     * grows with radius) an unweighted mean of the children is not the mean
     * over the coarse cell.  HydroEXA runs Cartesian, where ``fv = dx*dy`` is
     * the same for every child, so the weight cancels and the result is the
     * mean of the valid children — but the weight is carried rather than
     * assumed, so the operation does not silently become wrong elsewhere.
     *
     * A fine cell counts as valid when its value is neither NaN nor within
     * ``1e-3`` of \p nodata_val.  The tolerance is absolute and deliberately
     * loose: the sentinel travels through HDF5 as a float, and a value that
     * arrives as -9998.99997 is still NODATA, not a bed 10 km below sea level.
     *
     * The 50 % rule is the modelling decision in this function.  Averaging the
     * valid minority would let a handful of wet fine cells pull a mostly
     * invalid coarse cell down to a plausible-looking bed elevation.  Requiring
     * at least half the children means the coarse mask is at worst half a
     * coarse cell more pessimistic than the fine one, and an invalid region
     * never becomes valid terrain by restriction.
     *
     * \param i,j,k      Coarse cell index (``k`` is ignored in 2-D).
     * \param n          Component loop index added to \p ccomp and \p fcomp.
     * \param crse       Destination coarse array.
     * \param fine       Source fine array.
     * \param fv         Fine-cell volumes, one component.
     * \param ccomp      First destination component in \p crse.
     * \param fcomp      First source component in \p fine.
     * \param ratio      Refinement ratio in each direction.
     * \param nodata_val Sentinel marking an invalid cell.
     */
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

    /**
     * \brief Volume-weighted, NODATA-aware fine-to-coarse restriction.
     *
     * The whole-MultiFab driver: it coarsens the fine BoxArray, obtains cell
     * volumes from \p fgeom, runs ``masked_avgdown_with_vol`` over every cell
     * and component, and copies the result into \p S_crse.  Cell-centred only
     * — a nodal MultiFab has no volume to weight by and aborts.
     *
     * \param S_fine     Source fine MultiFab (cell-centred).
     * \param S_crse     Destination coarse MultiFab (cell-centred); components
     *                   ``scomp .. scomp+ncomp-1`` are overwritten.
     * \param fgeom      Fine-level geometry; supplies the cell volumes.
     * \param cgeom      Coarse-level geometry; currently unused.
     * \param scomp      First component to write in \p S_crse.
     * \param ncomp      Number of components to restrict.
     * \param ratio      Refinement ratio, fine cells per coarse cell.
     * \param nodata_val Sentinel marking an invalid cell, -9999.0 by default.
     */
    void masked_average_down (const amrex::MultiFab& S_fine, amrex::MultiFab& S_crse,
                              const amrex::Geometry& fgeom,  const amrex::Geometry& cgeom,
                              int scomp, int ncomp, const amrex::IntVect& ratio,
                              amrex::Real nodata_val = -9999.0);
}

#endif
