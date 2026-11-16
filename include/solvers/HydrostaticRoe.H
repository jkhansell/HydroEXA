#ifndef HYDROSTATIC_ROE_H
#define HYDROSTATIC_ROE_H

#include <AMReX_REAL.H>
#include <AMReX_Math.H>
#include <AMReX_TagBox.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_FluxRegister.H>

#include <cmath>

#include <utils/Constants.H>
#include <utils/Params.H>
#include <solvers/BaseSolver.H>

/**
 * \file HydrostaticRoe.H
 * \brief Well-balanced Roe solver with MUSCL reconstruction and hydrostatic correction.
 *
 * The solver to use when a case has a lake at rest, a steep bed, or a
 * shoreline that must not creep; ``Roe`` is the cheaper comparison.
 */

/**
 * \brief Roe that preserves a lake at rest on a sloping bed.
 *
 * The problem this solver exists for: a lake at rest is a steady solution of
 * the shallow-water equations, but a scheme that discretizes the flux and the
 * bed-slope source separately does not preserve it, because the two
 * discretizations do not cancel.  The lake then develops a spurious current
 * that grows without bound.
 *
 * Three ingredients make the cancellation exact:
 *
 * - **MUSCL reconstruction** of the free surface, and of the velocities, with a
 *   van Leer slope.  Slopes are limited so the reconstructed surface cannot
 *   create water where there is none.
 * - **Hydrostatic reconstruction** in the sense of Audusse et al.: the face bed
 *   is ``zf = max(z_left, z_right)``, and the face depths are measured from
 *   ``zf``, not from each cell's own bed.  A cell on a step therefore presents
 *   the wet part of its face rather than the whole column.
 * - **The pressure correction** that puts back the difference between the
 *   uncapped and capped face depths.  It is a normal-momentum flux, and it is
 *   what makes the flux seen by the left cell differ from the one seen by the
 *   right cell.
 *
 * That last point is the visible difference from ``Roe``: this interface
 * returns two effective fluxes per face, ``FL`` and ``FR``, which coincide on a
 * flat bed and differ only by the correction.  There is no separate source
 * decomposition, because the correction *is* the source term.
 */
struct HydrostaticRoe : public BaseSolver<HydrostaticRoe> {

    /**
     * \brief Advance one level: fill ghosts, flux the faces, update, reflux.
     *
     * Same five phases as the classic solver, with two differences that cost
     * time and buy accuracy:
     *
     * - the fill fills eta as well as the momenta, and depth is rebuilt from
     *   the two over the **grown** tile, so the ghost ring is consistent with
     *   this level's own bed rather than inheriting the coarse level's depth;
     * - the reconstruction reads one cell beyond each face on both sides, so
     *   three temporaries are allocated with two ghost cells each.  That is the
     *   extra allocation the profiling phases exist to make visible.
     *
     * The update also applies Manning friction and the dry-cell clamp per cell,
     * so the friction is not a separate pass.
     */
    void compute_fluxes_Impl(
        SolverContext ctx, int lev, amrex::Real dt, amrex::Real time
    );

    /**
     * \brief Evaluate the two effective fluxes over a face box.
     *
     * For each face the kernel reconstructs the free surface and both velocity
     * components from the neighbouring cells, caps the face depths at
     * ``zf = max(z_left, z_right)``, and runs the homogeneous Roe flux on the
     * reconstructed states.  It then adds the hydrostatic pressure correction
     * of each side, rotated back into global coordinates.
     *
     * A face that touches NODATA gets zero flux in both directions: the
     * sentinel edge of the DEM behaves as an impermeable wall.  The classic
     * solver has no such test, because there the sentinel cancels out of
     * ``h = eta - z``; here the reconstruction would instead build a depth from
     * ``eta - zf``, so the case has to be stated.
     *
     * \param face_box Nodal box of the faces to evaluate.
     * \param U        State array; depth is read, not written.
     * \param terrain  Bed elevation (component 0); roughness is not needed here.
     * \param FL       [out] Effective flux seen by the cell on the left.
     * \param FR       [out] Effective flux seen by the cell on the right.
     * \param dt       Timestep, passed to the wave-speed guards.
     * \param dx       Face-normal cell size.
     * \param dir      Direction, 0 or 1.
     */
    void compute_amrex_effective_fluxes(
        const amrex::Box& face_box,
        amrex::Array4<amrex::Real const> const& U,
        amrex::Array4<amrex::Real const> const& terrain,
        amrex::Array4<amrex::Real> const& FL,
        amrex::Array4<amrex::Real> const& FR,
        const amrex::Real dt, const amrex::Real dx, const int dir
    );

    /**
     * \brief Stable timestep for one level, already scaled by \p cfl.
     *
     * Identical in form to ``Roe::compute_dt_Impl``: the same unsplit 2-D
     * bound, the same ``1e30`` for NODATA cells, the same per-level trace and
     * the same abort on a collapsed dt.  Kept as a separate implementation
     * rather than shared because the wave speeds are the solver's business --
     * a variant with a different Riemann problem would not have these.
     */
    amrex::Real compute_dt_Impl(const amrex::Vector<amrex::MultiFab>& U,
                                const amrex::Vector<amrex::MultiFab>& terrain,
                                const amrex::Vector<amrex::Geometry>& geom,
                                int lev, amrex::Real cfl);

    /**
     * \brief Tag cells for refinement from gradients of depth, velocity and bed.
     *
     * Same criteria as the classic solver, with two differences in the
     * details: the NODATA test is a tolerance (``|z + 9999| < TOL10``) rather
     * than an equality, and the bed-slope criterion is only considered when
     * the cell has not already been tagged, so it cannot add work on top of a
     * front that already justifies refinement.
     */
    void tag_cells_Impl(amrex::TagBoxArray& tags,
                        const amrex::MultiFab& U,
                        const amrex::MultiFab& terrain,
                        const amrex::Geometry& geom,
                        int lev,
                        amrex::Real time,
                        const PhysicsParameters& phys_params);
};

#endif 
