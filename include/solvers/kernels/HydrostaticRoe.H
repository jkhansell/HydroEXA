#ifndef KERNELS_HYDROSTATIC_ROE_H
#define KERNELS_HYDROSTATIC_ROE_H

#include <AMReX_REAL.H>
#include <utils/Constants.H>

/**
 * \file kernels/HydrostaticRoe.H
 * \brief Device kernel for the homogeneous Roe flux on reconstructed face states.
 *
 * The same Riemann solver as ``kernels/Roe.H`` with everything removed that the
 * well-balanced solver does elsewhere: no bed slope, no friction, no sentinel
 * test, and no momentum rotation other than the face normal.  What arrives here
 * is already a pair of face states produced by the MUSCL reconstruction and the
 * hydrostatic depth cap, and what leaves is one physical flux.
 *
 * That division is deliberate rather than an accident of layering.  The
 * well-balanced property comes from the reconstruction and the pressure
 * correction, which are applied by the caller *around* the Riemann problem.  If
 * this kernel also added a bed-source term, a lake at rest would be corrected
 * twice and the lake would start to move -- the exact failure the hydrostatic
 * solver exists to avoid.
 *
 * The function is named ``roeSolver``, the same as the classic solver's, with a
 * different parameter list.  Only one of the two headers is ever included by a
 * translation unit (each solver's ``.cpp`` includes its own), so this is a
 * contained overload rather than a collision.  Do not include both.
 */

/**
 * \brief Roe flux through one face, from two reconstructed states.
 *
 * \param hi,hui,hvi  Left face state: depth and momentum, already capped at the
 *                    face bed and already reconstructed.
 * \param hj,huj,hvj  Right face state.
 * \param F           [out] Physical flux through the face, three components.
 * \param dt          Timestep; carried for signature symmetry with the classic
 *                    kernel, which needs it for its positivity guard.  This one
 *                    has no source term to limit, so it is unused.
 * \param dx          Face-normal cell size; likewise unused here.
 * \param nx,ny       Face-normal unit vector.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void roeSolver(
    amrex::Real hi, amrex::Real hui, amrex::Real hvi,
    amrex::Real hj, amrex::Real huj, amrex::Real hvj,
    amrex::Real F[3],       // Array to return F
    const amrex::Real dt,
    const amrex::Real dx,
    const amrex::Real nx,
    const amrex::Real ny
) {

    // --- dry handling ---
    // A face with no water on either side must produce no flux, and the
    // reconstruction upstream of here can legitimately hand over a zero depth.
    // TOLDRY is smaller than the dry/wet reporting threshold, so a cell has to
    // fall further to be dry in the flux than to be reported dry.
    hi = (hi >= Tolerances::TOLDRY()) ? hi : 0.0;
    hj = (hj >= Tolerances::TOLDRY()) ? hj : 0.0;

    amrex::Real sqrt_i = (hi > 0.0) ? std::sqrt(hi) : 0.0;
    amrex::Real sqrt_j = (hj > 0.0) ? std::sqrt(hj) : 0.0;

    // --- primitive variables ---
    amrex::Real ui = (hi > 0.0) ? hui / hi : 0.0;
    amrex::Real vi = (hi > 0.0) ? hvi / hi : 0.0;

    amrex::Real uj = (hj > 0.0) ? huj / hj : 0.0;
    amrex::Real vj = (hj > 0.0) ? hvj / hj : 0.0;

    // --- rotate ---
    amrex::Real uhati = ui * nx + vi * ny;
    amrex::Real vhati = -ui * ny + vi * nx;

    amrex::Real uhatj = uj * nx + vj * ny;
    amrex::Real vhatj = -uj * ny + vj * nx;

    // --- Roe averages ---
    // Arithmetic-mean depth, floored at TOLDRY so that c = sqrt(g h) stays
    // positive even for a face between two dry cells, and square-root-weighted
    // velocities as the Roe linearisation requires.
    amrex::Real htilde = (0.5 * (hi + hj) > Tolerances::TOLDRY()) ? 0.5 * (hi + hj) : Tolerances::TOLDRY();
    amrex::Real ctilde = std::sqrt(PhysConstants::GRAV() * htilde);

    amrex::Real denom_sqrt = sqrt_i + sqrt_j + Tolerances::TOL12();

    amrex::Real utilde = (uhati * sqrt_i + uhatj * sqrt_j) / denom_sqrt;
    amrex::Real vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / denom_sqrt;

    amrex::Real lambda0 = utilde - ctilde;
    amrex::Real lambda1 = utilde;
    amrex::Real lambda2 = utilde + ctilde;

    // --- entropy fix Harten Hyman---
    // As in the classic kernel: only where an acoustic characteristic is sonic,
    // because that is where the plain Roe flux would resolve a rarefaction as an
    // expansion shock.
    {
        amrex::Real ei = uhati - std::sqrt(PhysConstants::GRAV() * hi);
        amrex::Real ej = uhatj - std::sqrt(PhysConstants::GRAV() * hj);

        if (ei < 0.0 && ej > 0.0) {
            lambda0  = ei * (ej - lambda0) / (ej - ei);
        }

        ei = uhati + std::sqrt(PhysConstants::GRAV() * hi);
        ej = uhatj + std::sqrt(PhysConstants::GRAV() * hj);

        if (ei < 0.0 && ej > 0.0) {
            lambda2  = ej * (lambda2 - ei) / (ej - ei);
        }
    }

    amrex::Real denom = 2.0 * ctilde + Tolerances::TOL12();

    // --- differences ---
    amrex::Real dh  = hj - hi;
    amrex::Real dhu = hj * uhatj - hi * uhati;
    amrex::Real dhv = hj * vhatj - hi * vhati;

    // --- wave strengths ---
    amrex::Real alpha0 = (lambda2 * dh - dhu) / denom;
    amrex::Real alpha1 = (-vtilde * dh + dhv) / ctilde;
    amrex::Real alpha2 = (-lambda0 * dh + dhu) / denom;

    // Initialize physical center-flux component
    F[0] = 0.5 * (hi * uhati + hj * uhatj);
    F[1] = 0.5 * (hi * uhati * uhati + 0.5 * PhysConstants::GRAV() * hi * hi +
                  hj * uhatj * uhatj + 0.5 * PhysConstants::GRAV() * hj * hj);
    F[2] = 0.5 * (hi * uhati * vhati + hj * uhatj * vhatj);

    // --- Compute Wave Contributions ---
    // The upwind part: subtract 0.5|lambda_k| alpha_k along each right
    // eigenvector.  No source distribution here -- that is the caller's
    // pressure correction, applied after this returns.
    for (int k = 0; k < 3; k++) {
        amrex::Real lambda_k = (k == 0) ? lambda0 : ((k == 1) ? lambda1 : lambda2);
        amrex::Real alpha_k  = (k == 0) ? alpha0  : ((k == 1) ? alpha1  : alpha2);

        amrex::Real e0 = (k == 1) ? 0.0     : 1.0;
        amrex::Real e1 = (k == 0) ? lambda0 : (k == 1 ? 0.0     : lambda2);
        amrex::Real e2 = (k == 0) ? vtilde  : (k == 1 ? ctilde  : vtilde);

        amrex::Real flux = 0.5 * std::abs(lambda_k) * alpha_k;
        
        F[0] -= flux * e0; 
        F[1] -= flux * e1; 
        F[2] -= flux * e2;
    }

    // --- Rotate Back Both Effective Fluxes into Global Coordinates ---
    {
        amrex::Real tmp1 = F[1];
        amrex::Real tmp2 = F[2];
        F[1] = tmp1 * nx - tmp2 * ny;
        F[2] = tmp1 * ny + tmp2 * nx;
    }
}

#endif
