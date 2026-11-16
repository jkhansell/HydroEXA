#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <AMReX_REAL.H>
#include <AMReX_GpuControl.H>

/**
 * \file Constants.H
 * \brief Physical constants and numerical tolerances shared by the solvers.
 *
 * These live in headers rather than in a compiled translation unit because
 * every kernel needs them on the device: constexpr functions are the portable
 * way to expose a value to an AMREX_GPU_DEVICE lambda without an ODR problem.
 */

namespace PhysConstants {

    /**
     * \brief Gravitational acceleration, m/s^2.
     *
     * Used by the flux, wave-speed and well-balanced hydrostatic
     * reconstruction.  Not a tunable: a case that needs a different gravity is
     * a different physical problem, not a different setting.
     */
    AMREX_GPU_DEVICE constexpr amrex::Real GRAV() { return 9.81; }

} // namespace PhysConstants

/**
 * \brief Numerical tolerances, in the units of the quantity they guard.
 *
 * The distinction that matters: DEPTH and TOLDRY decide *when a cell is dry*
 * (a modelling decision), while FLUX and the TOL* family decide *when two
 * floating-point numbers are the same* (a round-off decision).  Conflating the
 * two is how a wetting front starts oscillating.
 */
namespace Tolerances {

    /**
     * \brief Minimum water depth treated as water, metres.
     *
     * Below this the cell is dry: depth is clamped to zero and momentum is
     * discarded.  Larger values stabilise thin films; smaller values preserve
     * run-up extent.
     */
    AMREX_GPU_DEVICE constexpr amrex::Real DEPTH()  { return 1.0e-4; }

    /**
     * \brief Flux magnitude below which a flux difference is noise.
     *
     * Used when deciding whether a cell needs refinement or a diagnostic
     * reports a change; it is a physical-scale threshold, not an epsilon.
     */
    AMREX_GPU_DEVICE constexpr amrex::Real FLUX()   { return 1.0e-7; }

    /**
     * \brief Depth threshold for the "dry" branch of the Riemann solver.
     *
     * Deliberately smaller than DEPTH: a cell must fall further below the
     * wet/dry threshold to be reclassified inside the flux computation than it
     * must to be reported dry, which prevents chattering at the shoreline.
     */
    AMREX_GPU_DEVICE constexpr amrex::Real TOLDRY() { return 1e-6; }

    /**
     * \brief Round-off comparison tolerance (1e-12).
     *
     * For quantities that are O(1), e.g. a normalised residual.
     */
    AMREX_GPU_DEVICE constexpr amrex::Real TOL12()  { return 1e-12; }

    /**
     * \brief Round-off comparison tolerance (1e-10).
     *
     * For accumulated quantities such as a mass balance, where more
     * summations mean more cancellation error.
     */
    AMREX_GPU_DEVICE constexpr amrex::Real TOL10()  { return 1e-10; }

} // namespace Tolerances

#endif
