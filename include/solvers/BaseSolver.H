#ifndef BASE_SOLVER_H_
#define BASE_SOLVER_H_

#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_FluxRegister.H>

#include <algorithm>

#include <solvers/SolverContext.H>

/**
 * \file BaseSolver.H
 * \brief The interface every Riemann solver implements.
 *
 * One template, no base state: the contract that lets the AMR state class hold
 * a solver without knowing which model it is.
 */

/**
 * \brief CRTP interface that every Riemann solver implements.
 *
 * A solver answers three questions about one level: what timestep is stable,
 * what the fluxes are across the faces, and which cells should be refined next
 * time.  Everything else -- the AMR hierarchy, the flux registers, the boundary
 * records -- arrives through ``SolverContext``, so a solver is a value type
 * with no members and no pointer back to the mesh.
 *
 * The interface is CRTP rather than virtual.  Concrete solvers are stored in a
 * ``std::variant`` and chosen once at startup, so the four calls per level per
 * step do not need a vtable, and a stateless value type is easier to reason
 * about than a polymorphic one.  The price is that this class cannot *declare*
 * the three ``_Impl`` methods: nothing checks for them until the first call,
 * where a missing override fails to compile as "no member named
 * ``compute_dt_Impl`` in ``Roe``".  Adding a solver therefore means
 * implementing all three and listing the new type in ``SolverVariant.H``.
 */
template <typename Derived>
struct BaseSolver {

    /**
     * \brief Advance one level by \p dt.
     *
     * The implementation owns the whole step for a level: it fills ghosts,
     * computes the dual face fluxes, applies the conservative update, and
     * updates the flux registers.  The context's ``U_new``/``U_old`` are
     * swapped on entry, so the destination after the call is ``U_new``.
     *
     * \param ctx  View of the state, mesh and boundary data.
     * \param lev  Level to advance.
     * \param dt   Timestep for this level, already reduced by CFL.
     * \param time Time at the *start* of the step; ghosts are filled to it.
     */
    void compute_fluxes(
        SolverContext ctx, int lev, amrex::Real dt, amrex::Real time
    ) {
        static_cast<Derived*>(this)->compute_fluxes_Impl(ctx, lev, dt, time);
    }

    /**
     * \brief Stable timestep for one level, already scaled by \p cfl.
     *
     * The base class only forwards to the concrete solver, which knows the
     * wave speeds of its own Riemann problem.  Both solvers here use the
     * unsplit 2-D condition ``dt*(|u|+c)/dx + dt*(|v|+c)/dy <= 1``, so the
     * ``cfl`` passed in the inputs file is the factor it appears to be on a
     * square cell; a split ``min(dx/(|u|+c), dy/(|v|+c))`` would be twice as
     * permissive and make the same keyword mean different things per solver.
     *
     * \param U       Per-level conserved state.
     * \param terrain Per-level terrain.
     * \param geom    Per-level geometry, for the cell size.
     * \param lev     Level to query.
     * \param cfl     CFL number.
     * \return The largest stable timestep for this level.
     */
    amrex::Real compute_dt_level(const amrex::Vector<amrex::MultiFab>& U,
                                 const amrex::Vector<amrex::MultiFab>& terrain,
                                 const amrex::Vector<amrex::Geometry>& geom,
                                 int lev, amrex::Real cfl)
    {
        return static_cast<Derived*>(this)->compute_dt_Impl(U, terrain, geom, lev, cfl);
    }

    /**
     * \brief Timestep for every level, derived from one global bound.
     *
     * This is the only method with a body in the base class, because the AMR
     * time-step protocol is the same whatever the solver is.  It runs in three
     * steps, and the order matters:
     *
     * 1. **Query each level**, then reduce the minimum across ranks *before*
     *    anything is used.  A rank that kept its local minimum would advance
     *    with a different dt from its neighbours, and the next ghost exchange
     *    would deadlock rather than report an error.
     * 2. **Project every level's bound to level 0** by multiplying by the
     *    product of refinement ratios, and take the smallest.  A fine level can
     *    be the restrictive one -- a thin film on the fine grid limits the
     *    coarse step too -- so taking ``dt_tmp[0]`` alone would be wrong.
     * 3. **Downscale by integer ratio**: ``dt[lev] = dt[lev-1] / ratio``.  Fine
     *    levels are never given their own independent bound, so the clocks stay
     *    exactly commensurate and ``ratio`` subcycles land precisely on the
     *    coarse time.  Independent per-level values would drift and the
     *    subcycling loop would no longer close.
     *
     * The ratios are read from component 0 only, so refinement must be
     * isotropic.
     *
     * \param dt           [out] Per-level timesteps, sized ``finest_level+1``.
     * \param U            Per-level conserved state.
     * \param terrain      Per-level terrain.
     * \param geom         Per-level geometry.
     * \param ref_ratio    Refinement ratio of each level against the next coarser.
     * \param finest_level Deepest level in the hierarchy.
     * \param cfl          CFL number.
     */
    void compute_dt(amrex::Vector<amrex::Real>& dt,
                const amrex::Vector<amrex::MultiFab>& U,
                const amrex::Vector<amrex::MultiFab>& terrain,
                const amrex::Vector<amrex::Geometry>& geom,
                const amrex::Vector<amrex::IntVect>& ref_ratio,
                int finest_level, amrex::Real cfl)
    {
        amrex::Vector<amrex::Real> dt_tmp(finest_level + 1);
        for (int lev = 0; lev <= finest_level; ++lev) {
            dt_tmp[lev] = compute_dt_level(U, terrain, geom, lev, cfl);
        }

        // 1. Reduce minimums across ranks FIRST before printing or processing
        amrex::ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

        //if (amrex::ParallelDescriptor::IOProcessor()) {
        //    amrex::Print() << "Computed dt per level:\n";
        //    for (int lev = 0; lev <= finest_level; ++lev) {
        //        amrex::Print() << "  Level " << lev << " : " << dt_tmp[lev] << "\n";
        //    }
        //}

        // 2. Project all level constraints up to Level 0 to find the safest global coarse dt
        amrex::Real dt_0 = dt_tmp[0];
        amrex::Real factor = 1.0;
        for (int lev = 1; lev <= finest_level; ++lev) {
            factor *= static_cast<amrex::Real>(ref_ratio[lev - 1][0]);
            // Scale the fine level constraint up to Level 0 equivalent and cap dt_0
            dt_0 = std::min(dt_0, dt_tmp[lev] * factor);
        }

        // 3. Assign Level 0
        dt[0] = dt_0;

        // 4. Downscale to fine levels using STRICT integer division
        for (int lev = 1; lev <= finest_level; ++lev) {
            dt[lev] = dt[lev - 1] / static_cast<amrex::Real>(ref_ratio[lev - 1][0]);
        }

        // if (amrex::ParallelDescriptor::IOProcessor()) {
        //     amrex::Print() << "Final dt per level:\n";
        //     for (int lev = 0; lev <= finest_level; ++lev) {
        //         amrex::Print() << "  Level " << lev << " : " << dt[lev] << "\n";
        //     }
        // }    
    }

    /**
     * \brief Mark the cells that must be covered by a finer level.
     *
     * Tagging is solver-specific on purpose: the criterion is a property of the
     * solution the solver produces (a wetting front, a hydraulic jump), not of
     * the state container.  Thresholds are per level and live in
     * ``PhysicsParameters``.
     *
     * \param tags        [in,out] TagBoxArray to set; already cleared by caller.
     * \param U           State on the level being tagged.
     * \param terrain     Terrain on that level.
     * \param geom        That level's geometry, for cell sizes.
     * \param lev         Level being tagged.
     * \param time        Current time.
     * \param phys_params Per-level gradient thresholds.
     */
    void tag_cells(amrex::TagBoxArray& tags,
                   const amrex::MultiFab& U,
                   const amrex::MultiFab& terrain,
                   const amrex::Geometry& geom,
                   int lev,
                   amrex::Real time,
                   const PhysicsParameters& phys_params) {
        static_cast<Derived*>(this)->tag_cells_Impl(tags, U, terrain, geom, lev, time, phys_params);
    }

};

#endif // BASE_SOLVER_H_
