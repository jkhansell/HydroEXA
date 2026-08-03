#ifndef KERNELS_ROE_H
#define KERNELS_ROE_H

#include <AMReX_REAL.H>
#include <utils/Constants.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void roeSolver(
    amrex::Real hi, amrex::Real hui, amrex::Real hvi, amrex::Real zi, amrex::Real ni,
    amrex::Real hj, amrex::Real huj, amrex::Real hvj, amrex::Real zj, amrex::Real nj,
    amrex::Real D_minus[3],       // Array to return F - S_minus (for left cell i)
    amrex::Real D_plus[3],        // Array to return F + S_plus  (for right cell j)
    const amrex::Real dt,
    const amrex::Real dx,
    const amrex::Real nx,
    const amrex::Real ny
) {

    // --- dry handling ---
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
    amrex::Real htilde = (0.5 * (hi + hj) > Tolerances::TOLDRY()) ? 0.5 * (hi + hj) : Tolerances::TOLDRY();
    amrex::Real ctilde = std::sqrt(PhysConstants::GRAV() * htilde);

    amrex::Real denom_sqrt = sqrt_i + sqrt_j + Tolerances::TOL12();

    amrex::Real utilde = (uhati * sqrt_i + uhatj * sqrt_j) / denom_sqrt;
    amrex::Real vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / denom_sqrt;

    amrex::Real lambda0 = utilde - ctilde;
    amrex::Real lambda1 = utilde;
    amrex::Real lambda2 = utilde + ctilde;
	
    amrex::Real lambdaE0 = 0.0;
    amrex::Real lambdaE2 = 0.0;

    // --- entropy fix ---
    {
        amrex::Real ei = uhati - std::sqrt(PhysConstants::GRAV() * hi);
        amrex::Real ej = uhatj - std::sqrt(PhysConstants::GRAV() * hj);

        if (ei < 0.0 && ej > 0.0) {
            lambdaE0 = lambda0 - ei * (ej - lambda0) / (ej - ei);
            lambda0  = ei * (ej - lambda0) / (ej - ei);
        }

        ei = uhati + std::sqrt(PhysConstants::GRAV() * hi);
        ej = uhatj + std::sqrt(PhysConstants::GRAV() * hj);

        if (ei < 0.0 && ej > 0.0) {
            lambdaE2 = lambda2 - ej * (lambda2 - ei) / (ej - ei);
            lambda2  = ej * (lambda2 - ei) / (ej - ei);
        }
    }

    amrex::Real denom = 2.0 * ctilde + Tolerances::TOL12();

    // --- differences ---
    amrex::Real dh  = hj - hi;
    amrex::Real dhu = hj * uhatj - hi * uhati;
    amrex::Real dhv = hj * vhatj - hi * vhati;

    amrex::Real alpha0 = (lambda2 * dh - dhu) / denom;
    amrex::Real alpha1 = (-vtilde * dh + dhv) / ctilde;
    amrex::Real alpha2 = (-lambda0 * dh + dhu) / denom;

    // --- source terms ---
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

        amrex::Real ntilde = 0.5 * (ni + nj);

        amrex::Real sf = (ntilde * ntilde *
                    std::sqrt(utilde * utilde + vtilde * vtilde) *
                    utilde) /
                    std::max(Tolerances::TOLDRY(), htilde * std::cbrt(htilde));

        tau = PhysConstants::GRAV() * htilde * sf * dx;
    }
	
    amrex::Real beta0 = (thrust - tau) / denom;
    amrex::Real beta1 = 0.0;
    amrex::Real beta2 = -beta0;

    {
        // compute stars (short lifetime)
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

        const double dt1star = (dx * 0.5 / lambda0_safe) * (hi / denom1_safe);
        const double dt3star = (dx * 0.5 / lambda2_safe) * (hj / denom3_safe);

        if ((h_i1star < -Tolerances::TOLDRY()) && (h_j3star > Tolerances::TOLDRY()) && (dt1star < loc_dt)) {
            const double beta1min = -(hi + alpha0) * std::abs(lambda0);
            const double beta3min = -(hj - alpha2) * lambda2;

            if (-beta1min >= beta3min) {
                beta0 = beta1min;
            }
            beta2 = -beta0;
        }

        if ((h_i1star > Tolerances::TOLDRY()) && (h_j3star < -Tolerances::TOLDRY()) && (dt3star < loc_dt)) {
            const double beta1min = -(hi + alpha0) * std::abs(lambda0);
            const double beta3min = -(hj - alpha2) * lambda2;

            if (-beta3min >= beta1min) {
                beta2 = beta3min;
            }
            beta0 = -beta2;
        }
    }

    // --- Allocate temporary structures for rotated updates ---
    amrex::Real F[3];
    amrex::Real S_minus[3] = {0.0, 0.0, 0.0};
    amrex::Real S_plus[3]  = {0.0, 0.0, 0.0};

    // Initialize physical center-flux component
    F[0] = 0.5 * (hi * uhati + hj * uhatj);
    F[1] = 0.5 * (hi * uhati * uhati + 0.5 * PhysConstants::GRAV() * hi * hi +
                  hj * uhatj * uhatj + 0.5 * PhysConstants::GRAV() * hj * hj);
    F[2] = 0.5 * (hi * uhati * vhati + hj * uhatj * vhatj);

    // --- Compute Wave Contributions ---
    for (int k = 0; k < 3; k++) {
        amrex::Real lambda_k = (k == 0) ? lambda0 : ((k == 1) ? lambda1 : lambda2);
        amrex::Real alpha_k  = (k == 0) ? alpha0  : ((k == 1) ? alpha1  : alpha2);
        amrex::Real beta_k   = (k == 0) ? beta0   : ((k == 1) ? beta1   : beta2);

        amrex::Real e0 = (k == 1) ? 0.0    : 1.0;
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

    // --- Assemble D_minus and D_plus ---
    D_minus[0] = F[0] - S_minus[0];
    D_minus[1] = F[1] - S_minus[1];
    D_minus[2] = F[2] - S_minus[2];

    D_plus[0]  = F[0] + S_plus[0];
    D_plus[1]  = F[1] + S_plus[1];
    D_plus[2]  = F[2] + S_plus[2];

    // --- Rotate Back Both Effective Fluxes into Global Coordinates ---
    {
        amrex::Real tmp1 = D_minus[1];
        amrex::Real tmp2 = D_minus[2];
        D_minus[1] = tmp1 * nx - tmp2 * ny;
        D_minus[2] = tmp1 * ny + tmp2 * nx;

        tmp1 = D_plus[1];
        tmp2 = D_plus[2];
        D_plus[1] = tmp1 * nx - tmp2 * ny;
        D_plus[2] = tmp1 * ny + tmp2 * nx;
    }
}

#endif