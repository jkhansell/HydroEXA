#ifndef KERNELS_ROE_H
#define KERNELS_ROE_H

#include <AMReX_REAL.H>
#include <utils/Constants.H>

/**
 * \file kernels/Roe.H
 * \brief Device kernel for the classic Roe flux, with the bed and friction
 *        terms split across the waves.
 *
 * One call per face.  The kernel returns three things rather than one -- the
 * physical flux and the two halves of an upwinded source -- because the
 * classic solver applies the bed slope and friction *inside* the Riemann
 * problem instead of as a cell update afterwards.  Splitting the source across
 * the waves is what allows each half to be delivered to the cell the wave is
 * leaving, so the result is an exact fluctuation-differencing update rather
 * than a central source plus a correction.
 *
 * Everything happens in face-normal coordinates: the velocities are rotated
 * onto the face, the Riemann problem is solved in 1-D, and the momentum
 * components are rotated back.  At the current call sites ``(nx, ny)`` is an
 * axis unit vector, so the rotation is trivial, but keeping it general is what
 * lets the kernel ignore which direction it was called for.
 *
 * The companion ``kernels/HydrostaticRoe.H`` declares a function with the same
 * name and a different signature.  That collision is deliberate and contained:
 * each solver's ``.cpp`` includes exactly one of the two, so the overloads
 * never meet in a single translation unit.  Do not include both.
 */

/**
 * \brief Roe flux and upwinded source halves for one face.
 *
 * \param hi,hui,hvi   Left-cell state: depth and the two momentum components.
 * \param zi           Left-cell bed elevation.
 * \param ni           Left-cell Manning roughness coefficient.
 * \param hj,huj,hvj   Right-cell state.
 * \param zj           Right-cell bed elevation.
 * \param nj           Right-cell Manning roughness coefficient.
 * \param F            [out] Physical flux through the face, three components.
 * \param S_minus      [out] Source share carried by right-going waves
 *                     (``lambda > 0``); the right cell sees ``F - S_minus``.
 * \param S_plus       [out] Source share carried by left-going waves
 *                     (``lambda < 0``); the left cell sees ``F + S_plus``.
 * \param dt           Timestep; only the positivity guard below uses it.
 * \param dx           Face-normal cell size, for the friction length scale.
 * \param nx,ny        Face-normal unit vector.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void roeSolver(
    amrex::Real hi, amrex::Real hui, amrex::Real hvi, amrex::Real zi, amrex::Real ni,
    amrex::Real hj, amrex::Real huj, amrex::Real hvj, amrex::Real zj, amrex::Real nj,
    amrex::Real F[3],       // Array to return F - S_minus (for right cell j)
    amrex::Real S_minus[3],       // Array to return F - S_minus (for right cell j)
    amrex::Real S_plus[3],        // Array to return F + S_plus  (for left cell i)
    const amrex::Real dt,
    const amrex::Real dx,
    const amrex::Real nx,
    const amrex::Real ny
) {

    // --- dry handling ---
    // Below TOLDRY the cell holds no water, so depth is zeroed *before* any
    // division.  TOLDRY is smaller than the dry/wet reporting threshold on
    // purpose: a cell must fall further to be treated as dry inside the flux
    // than it must to be reported dry, which stops the shoreline chattering.
    hi = (hi >= Tolerances::TOLDRY()) ? hi : 0.0;
    hj = (hj >= Tolerances::TOLDRY()) ? hj : 0.0;

    amrex::Real sqrt_i = (hi > 0.0) ? std::sqrt(hi) : 0.0;
    amrex::Real sqrt_j = (hj > 0.0) ? std::sqrt(hj) : 0.0;

    // --- primitive variables ---
    // Guarded rather than divided blindly: on a dry cell the momentum is
    // already zero, and 0/0 would poison the Roe averages with a NaN that
    // propagates to every face the cell touches.
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
    // Depth is the arithmetic mean, floored at TOLDRY, while the velocities
    // use the square-root weighting the Roe linearisation requires.  The floor
    // is not cosmetic: it keeps the wave speed c = sqrt(g h) strictly positive,
    // so a face between two dry cells still has finite eigenvalues.
    amrex::Real htilde = (0.5 * (hi + hj) > Tolerances::TOLDRY()) ? 0.5 * (hi + hj) : Tolerances::TOLDRY();
    amrex::Real ctilde = std::sqrt(PhysConstants::GRAV() * htilde);

    amrex::Real denom_sqrt = sqrt_i + sqrt_j + Tolerances::TOL12();

    amrex::Real utilde = (uhati * sqrt_i + uhatj * sqrt_j) / denom_sqrt;
    amrex::Real vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / denom_sqrt;

    amrex::Real lambda0 = utilde - ctilde;
    amrex::Real lambda1 = utilde;
    amrex::Real lambda2 = utilde + ctilde;

    // --- entropy fix Harten Hyman---
    // Applied only where an acoustic wave straddles zero, i.e. where the
    // characteristic is sonic.  Without it a transonic rarefaction is resolved
    // as an expansion shock: the scheme converges to a weak solution that the
    // physics does not admit.
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
    // Project the state jump onto the three right eigenvectors: the two
    // acoustic waves and the contact.  denom and ctilde carry a tolerance
    // because both can be zero on a dry face.
    amrex::Real alpha0 = (lambda2 * dh - dhu) / denom;
    amrex::Real alpha1 = (-vtilde * dh + dhv) / ctilde;
    amrex::Real alpha2 = (-lambda0 * dh + dhu) / denom;

    // --- source terms ---
    // Two discretizations of the bed-slope thrust compete here.  thrust_a is
    // the centred -g h dz; thrust_b is the depth-limited upwind form, which
    // measures the thrust over the wetted part of the step rather than the
    // whole column.  The choice below is the one that does not over-deepen a
    // cell sitting on a wetting front, where the centred form would invent
    // momentum from a bed the water has not reached yet.
    amrex::Real dz = zj - zi;
    amrex::Real di = hi + zi;
    amrex::Real dj = hj + zj;
    amrex::Real dd = dj - di;

    amrex::Real thrust, tau;

    {
        amrex::Real thrust_a = -PhysConstants::GRAV() * htilde * dz;

        amrex::Real dztilde;
        if ((dz >= 0.0) && (di < zj)) {
            dztilde = hi;
        } else if ((dz < 0.0) && (dj < zi)) {
            dztilde = hj;
        } else {
            dztilde = dz;
        }

        amrex::Real hr = (dz >= 0.0) ? hi : hj;

        amrex::Real thrust_b = -PhysConstants::GRAV() * (hr - 0.5 * std::abs(dztilde)) * dztilde;

        thrust = ((dz * dd >= 0.0) && (utilde * dz > 0.0))
                        ? std::max(thrust_a, thrust_b)
                        : thrust_b;

        // Manning friction, per unit length across the face.  h^(4/3) is
        // written cbrt(h)*h; the max() floor keeps a nearly dry cell from
        // producing an unbounded drag coefficient, which would otherwise
        // freeze the timestep.
        amrex::Real ntilde = 0.5 * (ni + nj);

        amrex::Real sf = (ntilde * ntilde *
                    std::sqrt(utilde * utilde + vtilde * vtilde) * utilde) /
                    std::max(Tolerances::TOLDRY(), htilde * std::cbrt(htilde));

        tau = PhysConstants::GRAV() * htilde * sf * dx;
    }

    // --- split the source across the waves ---
    // Only the two acoustic waves carry it, with equal and opposite shares
    // (beta2 = -beta0), so the source changes momentum without adding mass.
    // The contact wave (beta1) carries nothing: friction and bed slope act on
    // the momentum equations, not on continuity.
    amrex::Real beta0 = (thrust - tau) / denom;
    amrex::Real beta1 = 0.0;
    amrex::Real beta2 = -beta0;

    {
        // compute stars (short lifetime)
        // Positivity guard.  Each acoustic wave aims to leave a star depth
        // behind it; if that depth would be negative, and the wave arrives
        // within the local timestep, the source share is limited to the value
        // that just empties the cell instead.  Without it a drying front
        // produces a negative depth that the update has to clamp away, and the
        // mass lost in that clamp is not accounted for anywhere.
        const amrex::Real lambda0_safe = (std::abs(lambda0) > Tolerances::TOL12()) ? lambda0 : (lambda0 >= 0.0 ? Tolerances::TOL12() : -Tolerances::TOL12());
        const amrex::Real lambda2_safe = (std::abs(lambda2) > Tolerances::TOL12()) ? lambda2 : (lambda2 >= 0.0 ? Tolerances::TOL12() : -Tolerances::TOL12());

        const amrex::Real h_i1star = hi + alpha0 - (beta0 / lambda0_safe);
        const amrex::Real h_j3star = hj - alpha2 + (beta2 / lambda2_safe);

        // compute local dt
        amrex::Real maxlam = std::abs(lambda0);
        maxlam = std::max(maxlam, std::abs(lambda1));
        maxlam = std::max(maxlam, std::abs(lambda2));
        const amrex::Real loc_dt = dx / (maxlam + Tolerances::TOL12());

        const amrex::Real denom1 = hi - h_i1star;
        const amrex::Real denom1_safe = (std::abs(denom1) > Tolerances::TOL12()) ? denom1 : (denom1 >= 0.0 ? Tolerances::TOL12() : -Tolerances::TOL12());
        const amrex::Real denom3 = hj - h_j3star;
        const amrex::Real denom3_safe = (std::abs(denom3) > Tolerances::TOL12()) ? denom3 : (denom3 >= 0.0 ? Tolerances::TOL12() : -Tolerances::TOL12());

        const amrex::Real dt1star = (dx * 0.5 / lambda0_safe) * (hi / denom1_safe);
        const amrex::Real dt3star = (dx * 0.5 / lambda2_safe) * (hj / denom3_safe);

        if ((h_i1star < -Tolerances::TOLDRY()) && (h_j3star > Tolerances::TOLDRY()) && (dt1star < loc_dt)) {
            const amrex::Real beta1min = -(hi + alpha0) * std::abs(lambda0);
            const amrex::Real beta3min = -(hj - alpha2) * lambda2;

            if (-beta1min >= beta3min) {
                beta0 = beta1min;
            }
            beta2 = -beta0;
        }

        if ((h_i1star > Tolerances::TOLDRY()) && (h_j3star < -Tolerances::TOLDRY()) && (dt3star < loc_dt)) {
            const amrex::Real beta1min = -(hi + alpha0) * std::abs(lambda0);
            const amrex::Real beta3min = -(hj - alpha2) * lambda2;

            if (-beta3min >= beta1min) {
                beta2 = beta3min;
            }
            beta0 = -beta2;
        }
    }

    for (int k = 0; k < 3; k++) {
        S_minus[k]  = 0.0; 
        S_plus[k]   = 0.0; 
    } 
    
    // Initialize physical center-flux component
    F[0] = 0.5 * (hi * uhati + hj * uhatj);
    F[1] = 0.5 * (hi * uhati * uhati + 0.5 * PhysConstants::GRAV() * hi * hi +
                  hj * uhatj * uhatj + 0.5 * PhysConstants::GRAV() * hj * hj);
    F[2] = 0.5 * (hi * uhati * vhati + hj * uhatj * vhatj);

    // --- Compute Wave Contributions ---
    // Each wave removes 0.5|lambda_k| alpha_k from the central flux: that is
    // the upwind part of the Roe flux.  The same loop distributes the source
    // share of the wave to whichever side the wave is leaving, which is why
    // the two happen together rather than in separate passes.
    for (int k = 0; k < 3; k++) {
        amrex::Real lambda_k = (k == 0) ? lambda0 : ((k == 1) ? lambda1 : lambda2);
        amrex::Real alpha_k  = (k == 0) ? alpha0  : ((k == 1) ? alpha1  : alpha2);
        amrex::Real beta_k   = (k == 0) ? beta0   : ((k == 1) ? beta1   : beta2);

        amrex::Real e0 = (k == 1) ? 0.0     : 1.0;
        amrex::Real e1 = (k == 0) ? lambda0 : (k == 1 ? 0.0     : lambda2);
        amrex::Real e2 = (k == 0) ? vtilde  : (k == 1 ? ctilde  : vtilde);

        amrex::Real flux = 0.5 * std::abs(lambda_k) * alpha_k;
        
        F[0] -= flux * e0; 
        F[1] -= flux * e1; 
        F[2] -= flux * e2;

        // Symmetric Upwind Distribution of Source Terms
        if (lambda_k > 0.0) {
            S_minus[0] += beta_k * e0;
            S_minus[1] += beta_k * e1;
            S_minus[2] += beta_k * e2;
        } else if (lambda_k < 0.0) { 
            S_plus[0] += beta_k * e0;
            S_plus[1] += beta_k * e1;
            S_plus[2] += beta_k * e2;
        }
    }

    // --- Rotate Back Both Effective Fluxes into Global Coordinates ---
    {
        amrex::Real tmp1 = F[1];
        amrex::Real tmp2 = F[2];
        F[1] = tmp1 * nx - tmp2 * ny;
        F[2] = tmp1 * ny + tmp2 * nx;

        tmp1 = S_minus[1];
        tmp2 = S_minus[2];
        S_minus[1] = tmp1 * nx - tmp2 * ny;
        S_minus[2] = tmp1 * ny + tmp2 * nx;

        tmp1 = S_plus[1];
        tmp2 = S_plus[2];
        S_plus[1] = tmp1 * nx - tmp2 * ny;
        S_plus[2] = tmp1 * ny + tmp2 * nx;
    }
}

#endif
