#ifndef HYDROEXA_ROESOLVER_H
#define HYDROEXA_ROESOLVER_H

#include <AMReX_REAL.H>
#include <AMReX_Math.H>
#include <cmath>

#include <hydroexa/Constants.H>

using namespace amrex;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void roeSolver(
    Real hi, Real hui, Real hvi, Real zi, Real ni,
    Real hj, Real huj, Real hvj, Real zj, Real nj,
    Real D_minus[3],       // Array to return F - S_minus (for left cell i)
    Real D_plus[3],        // Array to return F + S_plus  (for right cell j)
    const Real dt,
    const Real dx,
    const Real nx,
    const Real ny
) {

    // --- dry handling ---
    hi = (hi >= TOLDRY) ? hi : 0.0;
    hj = (hj >= TOLDRY) ? hj : 0.0;

    Real sqrt_i = (hi > 0.0) ? std::sqrt(hi) : 0.0;
    Real sqrt_j = (hj > 0.0) ? std::sqrt(hj) : 0.0;

    // --- primitive variables ---
    Real ui = (hi > 0.0) ? hui / hi : 0.0;
    Real vi = (hi > 0.0) ? hvi / hi : 0.0;

    Real uj = (hj > 0.0) ? huj / hj : 0.0;
    Real vj = (hj > 0.0) ? hvj / hj : 0.0;

    // --- rotate ---
    Real uhati = ui * nx + vi * ny;
    Real vhati = -ui * ny + vi * nx;

    Real uhatj = uj * nx + vj * ny;
    Real vhatj = -uj * ny + vj * nx;

    // --- Roe averages ---
    Real htilde = (0.5 * (hi + hj) > TOLDRY) ? 0.5 * (hi + hj) : TOLDRY;
    Real ctilde = std::sqrt(GRAV * htilde);

    Real denom_sqrt = sqrt_i + sqrt_j + TOL12;

    Real utilde = (uhati * sqrt_i + uhatj * sqrt_j) / denom_sqrt;
    Real vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / denom_sqrt;

    Real lambda0 = utilde - ctilde;
    Real lambda1 = utilde;
    Real lambda2 = utilde + ctilde;
	
    Real lambdaE0 = 0.0;
    Real lambdaE2 = 0.0;

    // --- entropy fix ---
    {
        Real ei = uhati - std::sqrt(GRAV * hi);
        Real ej = uhatj - std::sqrt(GRAV * hj);

        if (ei < 0.0 && ej > 0.0) {
            lambdaE0 = lambda0 - ei * (ej - lambda0) / (ej - ei);
            lambda0  = ei * (ej - lambda0) / (ej - ei);
        }

        ei = uhati + std::sqrt(GRAV * hi);
        ej = uhatj + std::sqrt(GRAV * hj);

        if (ei < 0.0 && ej > 0.0) {
            lambdaE2 = lambda2 - ej * (lambda2 - ei) / (ej - ei);
            lambda2  = ej * (lambda2 - ei) / (ej - ei);
        }
    }

    Real denom = 2.0 * ctilde + TOL12;

    // --- differences ---
    Real dh  = hj - hi;
    Real dhu = hj * uhatj - hi * uhati;
    Real dhv = hj * vhatj - hi * vhati;

    Real alpha0 = (lambda2 * dh - dhu) / denom;
    Real alpha1 = (-vtilde * dh + dhv) / ctilde;
    Real alpha2 = (-lambda0 * dh + dhu) / denom;

    // --- source terms ---
    Real dz = zj - zi;
    Real di = hi + zi;
    Real dj = hj + zj;
    Real dd = dj - di;

    Real thrust, tau;
	
    {
        Real thrust_a = -GRAV * htilde * dz;

        Real dztilde;
        if ((dz >= 0.0) && (di < zj)) {
            dztilde = hi;
        } else if ((dz < 0.0) && (dj < zi)) {
            dztilde = hj;
        } else {
            dztilde = dz;
        }

        Real hr = (dz >= 0.0) ? hi : hj;

        Real thrust_b = -GRAV * (hr - 0.5 * std::abs(dztilde)) * dztilde;

        thrust = ((dz * dd >= 0.0) && (utilde * dz > 0.0))
                        ? std::max(thrust_a, thrust_b)
                        : thrust_b;

        Real ntilde = 0.5 * (ni + nj);

        Real sf = (ntilde * ntilde *
                    std::sqrt(utilde * utilde + vtilde * vtilde) *
                    utilde) /
                    std::max(TOLDRY, htilde * std::cbrt(htilde));

        tau = GRAV * htilde * sf * dx;
    }
	
    Real beta0 = (thrust - tau) / denom;
    Real beta1 = 0.0;
    Real beta2 = -beta0;

    {
        // compute stars (short lifetime)
        const Real lambda0_safe = (std::abs(lambda0) > TOL12) ? lambda0 : (lambda0 >= 0.0 ? TOL12 : -TOL12);
        const Real lambda2_safe = (std::abs(lambda2) > TOL12) ? lambda2 : (lambda2 >= 0.0 ? TOL12 : -TOL12);

        const Real h_i1star = hi + alpha0 - (beta0 / lambda0_safe);
        const Real h_j3star = hj - alpha2 + (beta2 / lambda2_safe);

        // compute local dt
        Real maxlam = std::abs(lambda0);
        maxlam = std::max(maxlam, std::abs(lambda1));
        maxlam = std::max(maxlam, std::abs(lambda2));
        const Real loc_dt = dx / (maxlam + TOL12);

        const Real denom1 = hi - h_i1star;
        const Real denom1_safe = (std::abs(denom1) > TOL12) ? denom1 : (denom1 >= 0.0 ? TOL12 : -TOL12);
        const Real denom3 = hj - h_j3star;
        const Real denom3_safe = (std::abs(denom3) > TOL12) ? denom3 : (denom3 >= 0.0 ? TOL12 : -TOL12);

        const double dt1star = (dx * 0.5 / lambda0_safe) * (hi / denom1_safe);
        const double dt3star = (dx * 0.5 / lambda2_safe) * (hj / denom3_safe);

        if ((h_i1star < -TOLDRY) && (h_j3star > TOLDRY) && (dt1star < loc_dt)) {
            const double beta1min = -(hi + alpha0) * std::abs(lambda0);
            const double beta3min = -(hj - alpha2) * lambda2;

            if (-beta1min >= beta3min) {
                beta0 = beta1min;
            }
            beta2 = -beta0;
        }

        if ((h_i1star > TOLDRY) && (h_j3star < -TOLDRY) && (dt3star < loc_dt)) {
            const double beta1min = -(hi + alpha0) * std::abs(lambda0);
            const double beta3min = -(hj - alpha2) * lambda2;

            if (-beta3min >= beta1min) {
                beta2 = beta3min;
            }
            beta0 = -beta2;
        }
    }

    // --- Allocate temporary structures for rotated updates ---
    Real F[3];
    Real S_minus[3] = {0.0, 0.0, 0.0};
    Real S_plus[3]  = {0.0, 0.0, 0.0};

    // Initialize physical center-flux component
    F[0] = 0.5 * (hi * uhati + hj * uhatj);
    F[1] = 0.5 * (hi * uhati * uhati + 0.5 * GRAV * hi * hi +
                  hj * uhatj * uhatj + 0.5 * GRAV * hj * hj);
    F[2] = 0.5 * (hi * uhati * vhati + hj * uhatj * vhatj);

    // --- Compute Wave Contributions ---
    for (int k = 0; k < 3; k++) {
        Real lambda_k = (k == 0) ? lambda0 : ((k == 1) ? lambda1 : lambda2);
        Real alpha_k  = (k == 0) ? alpha0  : ((k == 1) ? alpha1  : alpha2);
        Real beta_k   = (k == 0) ? beta0   : ((k == 1) ? beta1   : beta2);

        Real e0 = (k == 1) ? 0.0    : 1.0;
        Real e1 = (k == 0) ? lambda0 : (k == 1 ? 0.0     : lambda2);
        Real e2 = (k == 0) ? vtilde  : (k == 1 ? ctilde  : vtilde);

        Real flux = 0.5 * std::abs(lambda_k) * alpha_k;
		
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

    // --- Assemble D_minus and D_plus ---
    D_minus[0] = F[0] - S_minus[0];
    D_minus[1] = F[1] - S_minus[1];
    D_minus[2] = F[2] - S_minus[2];

    D_plus[0]  = F[0] + S_plus[0];
    D_plus[1]  = F[1] + S_plus[1];
    D_plus[2]  = F[2] + S_plus[2];

    // --- Rotate Back Both Effective Fluxes into Global Coordinates ---
    {
        Real tmp1 = D_minus[1];
        Real tmp2 = D_minus[2];
        D_minus[1] = tmp1 * nx - tmp2 * ny;
        D_minus[2] = tmp1 * ny + tmp2 * nx;

        tmp1 = D_plus[1];
        tmp2 = D_plus[2];
        D_plus[1] = tmp1 * nx - tmp2 * ny;
        D_plus[2] = tmp1 * ny + tmp2 * nx;
    }
}


#endif