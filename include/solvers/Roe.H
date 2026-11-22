#ifndef ROE_H
#define ROE_H

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
 * \file Roe.H
 * \brief Classic Roe solver: first-order face states, sources carried in the waves.
 *
 * The reference scheme of the pair, and the one to read first when comparing
 * accuracy or diagnosing a sloping-bed test failure.
 */

/**
 * \brief First-order Roe with the bed and friction terms split into the waves.
 *
 * This is the reference scheme of the pair.  Face states are the cell values
 * themselves -- no reconstruction -- and the bed-slope and friction terms are
 * not applied as a cell update but decomposed into the same three Roe waves as
 * the flux, then upwinded with them.  That is why the interface produces three
 * arrays per face instead of one: ``F`` is the physical flux, and the two
 * source halves have to travel with the wave that carries them.
 *
 * The practical consequence is that the scheme is exact for a lake at rest only
 * to the extent the source decomposition is: ``HydrostaticRoe`` is the variant
 * that gets that property by construction.  Compare the two when a test fails
 * on a sloping bed.
 */
struct Roe : public BaseSolver<Roe> {

    /**
     * \brief Advance one level: fill ghosts, flux the faces, update, reflux.
     *
     * The body runs in five numbered phases, and the order is load-bearing:
     *
     * 1. swap ``U_new``/``U_old`` so the destination is always ``U_new``, and
     *    pick up the flux registers this level owns;
     * 2. allocate ``F``, ``S_minus`` and ``S_plus`` on the nodal box of each
     *    direction -- one set per direction, because the two sweeps are
     *    unsplit but not staggered;
     * 3. fill the ghost ring from the coarse level, rebuild ``h`` on the grown
     *    tile from the free surface, then evaluate the face kernel;
     * 4. apply the conservative update over the tile, using the side-selected
     *    effective fluxes;
     * 5. hand the same side-selected fluxes to the flux registers, so the
     *    coarse-fine correction matches what entered the cell update.
     */
    void compute_fluxes_Impl(
        SolverContext ctx, int lev, amrex::Real dt, amrex::Real time
    );

    /**
     * \brief Evaluate the Roe flux and its two source halves over a face box.
     *
     * One call per direction, over the nodal box in that direction.  For each
     * face the left and right cell states are passed to the device kernel
     * ``roeSolver``, which returns:
     *
     * - \p F        the physical flux through the face;
     * - \p S_minus  the part of the source carried by right-going waves
     *   (``lambda > 0``), subtracted to form the right cell's effective flux;
     * - \p S_plus   the part carried by left-going waves (``lambda < 0``), added
     *   to form the left cell's effective flux.
     *
     * The two halves are what make the update equal to exact fluctuation
     * differencing: a cell's right face contributes ``F + S_plus`` and its left
     * face ``F - S_minus``, and the flux registers must be given the same pair
     * rather than the bare ``F``.
     *
     * \p U is non-const only because the kernel historically wrote the
     * reconstructed depth back into it.  ``reconstruct_h_from_eta`` now does
     * that over the grown tile, so the two writes in the kernel are
     * self-assignments; that is the one difference from the hydrostatic
     * solver's signature.
     *
     * \param face_box Nodal box of the faces to evaluate.
     * \param U        State array (see above).
     * \param Eta      Free surface, used to rebuild depth.
     * \param terrain  Bed elevation (component 0) and roughness (component 1).
     * \param F        [out] Physical flux, one set per face.
     * \param S_minus  [out] Right-going source half.
     * \param S_plus   [out] Left-going source half.
     * \param dt       Timestep, needed by the kernel's local-dt guards.
     * \param dx       Face-normal cell size.
     * \param dir      Direction, 0 or 1.
     */
    void compute_amrex_effective_fluxes(
        const amrex::Box& face_box,
        amrex::Array4<amrex::Real> const& U,
        amrex::Array4<amrex::Real const> const& Eta,
        amrex::Array4<amrex::Real const> const& terrain,
        amrex::Array4<amrex::Real> const& F,
        amrex::Array4<amrex::Real> const& S_minus,
        amrex::Array4<amrex::Real> const& S_plus,
        const amrex::Real dt, const amrex::Real dx, const int dir
    );

    /**
     * \brief Rebuild depth as ``max(0, eta - z)`` over a box.
     *
     * The conserved state stores depth, but the quantity that is continuous
     * across a coarse-fine boundary is the free surface, so the ghost ring is
     * filled with eta and depth is recomputed here.  Filling depth directly
     * would carry the coarse level's depth onto a differently sloped fine bed,
     * and ``eta_ghost = h_coarse + z_fine`` is not the lake surface -- a
     * mismatch that shows up as a slow, spurious current.
     *
     * The box must include the ghosts the face stencil will read, otherwise the
     * state is rebuilt only inside each tile and the faces at tile borders see
     * a discontinuity that produces fluxes there.
     *
     * \param cell_box Cells and ghosts to overwrite.
     * \param U        State array; component 0 is written.
     * \param Eta      Free surface.
     * \param terrain  Bed elevation.
     */
    void reconstruct_h_from_eta(
        const amrex::Box& cell_box,
        amrex::Array4<amrex::Real> const& U,
        amrex::Array4<amrex::Real const> const& Eta,
        amrex::Array4<amrex::Real const> const& terrain
    );

    /**
     * \brief Stable timestep for one level, already scaled by \p cfl.
     *
     * Reduces the unsplit 2-D bound ``dt/(dx/(|u|+c)) + dt/(dy/(|v|+c)) <= 1``
     * over the valid cells, skipping NODATA.  A sentinel cell returns ``1e30``
     * rather than a small number: invalid terrain must not throttle the
     * timestep of the cells that are real.
     *
     * Prints one line per level per step (``[compute_dt] lev=... dt=...
     * max|vel|=...``).  That is not leftover debug output -- it is the trace
     * you read backwards when the abort below fires, because a collapsing dt is
     * a single-cell velocity explosion and the per-step maximum is how its
     * growth history becomes visible.
     *
     * Aborts instead of returning ~0: a dt of zero would advance nothing and
     * look like a stalled run, and continuing at a tiny dt would mask the
     * offending cell indefinitely.
     *
     * \param U       Per-level state.
     * \param terrain Per-level terrain, for the NODATA test.
     * \param geom    Per-level geometry.
     * \param lev     Level to query.
     * \param cfl     CFL number to apply to the estimate.
     */
    amrex::Real compute_dt_Impl(const amrex::Vector<amrex::MultiFab>& U,
                                const amrex::Vector<amrex::MultiFab>& terrain,
                                const amrex::Vector<amrex::Geometry>& geom,
                                int lev, amrex::Real cfl);

    /**
     * \brief Tag cells for refinement from gradients of depth, velocity and bed.
     *
     * Two details are deliberate:
     *
     * - a cell whose four neighbours are not all valid is never tagged, so
     *   refinement stops at the edge of the DEM instead of following the
     *   sentinel cliff outward;
     * - the bed-slope criterion applies only at ``time <= 1e-8``.  Terrain is
     *   static, so slope alone would tag the same cells at every regrid; doing
     *   it once at ``t = 0`` puts resolution on steep beds before the water
     *   reaches them, and afterwards the flow criteria decide.
     *
     * The velocity criterion uses the gradient of the velocity *magnitude*,
     * with a dry-cell guard, so a reversal with no speed change does not
     * trigger refinement but a front does.
     *
     * \param tags        [in,out] TagBoxArray; cells already set are left alone.
     * \param U           Level state (h, hu, hv).
     * \param terrain     Level terrain.
     * \param geom        Level geometry.
     * \param lev         Level index, also selects the threshold entry.
     * \param time        Current time, for the one-shot terrain criterion.
     * \param phys_params Per-level gradient thresholds.
     */
    void tag_cells_Impl(amrex::TagBoxArray& tags,
                        const amrex::MultiFab& U,
                        const amrex::MultiFab& terrain,
                        const amrex::Geometry& geom,
                        int lev,
                        amrex::Real time,
                        const PhysicsParameters& phys_params);


    /** \brief
     * 
     * 
     * 
     * 
     * 
     * 
    */

    void wet_dry_correction(amrex::Vector<amrex::MultiFab>& U,
                            const amrex::Vector<amrex::MultiFab>& terrain,
                            const amrex::Vector<amrex::Geometry>& geom,
                            int lev);

    amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> flux_fine;
    amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> flux_crse;
};

#endif // ROE_H
