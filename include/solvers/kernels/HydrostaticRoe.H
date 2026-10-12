#ifndef KERNELS_HYDROSTATIC_ROE_H
#define KERNELS_HYDROSTATIC_ROE_H

#include <AMReX_REAL.H>
#include <utils/Constants.H>

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

    // --- entropy fix Harten Hyman---
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

    amrex::Real alpha0 = (lambda2 * dh - dhu) / denom;
    amrex::Real alpha1 = (-vtilde * dh + dhv) / ctilde;
    amrex::Real alpha2 = (-lambda0 * dh + dhu) / denom;

    // Initialize physical center-flux component
    F[0] = 0.5 * (hi * uhati + hj * uhatj);
    F[1] = 0.5 * (hi * uhati * uhati + 0.5 * PhysConstants::GRAV() * hi * hi +
                  hj * uhatj * uhatj + 0.5 * PhysConstants::GRAV() * hj * hj);
    F[2] = 0.5 * (hi * uhati * vhati + hj * uhatj * vhatj);

    // --- Compute Wave Contributions ---
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