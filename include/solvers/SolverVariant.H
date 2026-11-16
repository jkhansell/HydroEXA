#ifndef SOLVER_VARIANT_H_
#define SOLVER_VARIANT_H_

#include <variant>
#include <stdexcept>
#include <type_traits>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_FluxRegister.H>


// Include concrete solvers (or forward declare them if preferred, but since they
// no longer depend on AmrMeshState, they can be directly included safely)
#include <solvers/Roe.H>
#include <solvers/HydrostaticRoe.H>

/**
 * \file SolverVariant.H
 * \brief Runtime selection between the available Riemann solvers.
 *
 * The only place that lists the solvers a build can run, and the only place
 * that has to change when a new one is added.
 */

/**
 * \brief Holds one of the concrete solvers and forwards the four step calls.
 *
 * ``phys.model`` in the inputs file names the solver, but that choice has to
 * be made at run time and then be invisible to the rest of the code.  A
 * ``std::variant`` does both: ``AmrMeshState`` holds one object, calls the same
 * four methods on it, and never asks which model is inside.
 *
 * This class is also the single list of available solvers.  Adding one means
 * adding it to ``SolverType`` here and a branch to
 * ``AmrMeshState::InitializeSolver``; nothing else has to learn about it.
 *
 * A variant rather than a base class with virtual functions because the
 * solvers are stateless value types, so there is nothing to share but the
 * signature, and ``std::visit`` resolves the type once per call site instead of
 * once per object.  The ``std::monostate`` alternative is the "no choice made
 * yet" state, which a pointer would have expressed as ``nullptr``.
 */
class SolverVariant {
public:
    /// Concrete solvers this build can switch between.
    using SolverType = std::variant<std::monostate, Roe, HydrostaticRoe>;

    // Constructors
    /// Construct with no solver; every method throws until one is assigned.
    SolverVariant() : m_solver(std::monostate{}) {}

    /**
     * \brief Take ownership of a concrete solver.
     * \param solver A \c Roe or \c HydrostaticRoe, by value or reference.
     *
     * Implicit, so ``solver = Roe{};`` works in the initialisation code.
     */
    template <typename T>
    SolverVariant(T&& solver) : m_solver(std::forward<T>(solver)) {}

    /**
     * \brief Whether a concrete solver has been selected.
     *
     * ``AmrMeshState::InitializeSolver`` always assigns one or aborts, so this
     * is currently only a guard for callers that want to ask before the first
     * step rather than trust the initialization order.
     */
    bool has_solver() const {
        return !std::holds_alternative<std::monostate>(m_solver);
    }

    /**
     * \brief Advance one level; forwards to the selected solver.
     *
     * Throws rather than returning early when no solver is set: a zero-flux
     * advance would leave the state unchanged, which looks like a stable
     * solution instead of a missing configuration.
     */
    void compute_fluxes(SolverContext ctx, int lev, amrex::Real dt, amrex::Real time) {
        std::visit([&](auto&& solver) {
            using T = std::decay_t<decltype(solver)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error("[SolverVariant Error] Solver not initialized.");
            } else {
                solver.compute_fluxes(ctx, lev, dt, time);
            }
        }, m_solver);
    }

    /// Per-level timesteps for the whole hierarchy; see \c BaseSolver::compute_dt.
    void compute_dt(
        amrex::Vector<amrex::Real>& dt,
        const amrex::Vector<amrex::MultiFab>& U,
        const amrex::Vector<amrex::MultiFab>& terrain,
        const amrex::Vector<amrex::Geometry>& geom,
        const amrex::Vector<amrex::IntVect>& ref_ratio,
        int finest_level, amrex::Real cfl
    ){
        std::visit([&](auto&& solver) {
            using T = std::decay_t<decltype(solver)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error("[SolverVariant Error] Solver not initialized.");
            } else {
                solver.compute_dt(dt, U, terrain, geom, ref_ratio, finest_level, cfl);
            }
        }, m_solver);
    }

    /**
     * \brief Stable timestep for one level; forwards to the selected solver.
     *
     * Mirrors ``BaseSolver::compute_dt_level`` for symmetry.  Nothing in the
     * solver calls it through the variant: ``compute_dt`` already runs the
     * level loop in the base class, so this entry point is unused today.
     */
    amrex::Real compute_dt_level(
        const amrex::Vector<amrex::MultiFab>& U,
        const amrex::Vector<amrex::MultiFab>& terrain,
        const amrex::Vector<amrex::Geometry>& geom,
        int lev, amrex::Real cfl
    ){
        amrex::Real dt_lev = -1.0;
        std::visit([&](auto&& solver) -> amrex::Real {
            using T = std::decay_t<decltype(solver)>;

            if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error("[SolverVariant Error] Solver not initialized.");
            } else {
                dt_lev = solver.compute_dt_level(U, terrain, geom, lev, cfl);
                return dt_lev;
            }
        }, m_solver);
        return dt_lev;
    }

    /// Mark refinement cells using the selected solver's criterion.
    void tag_cells(
        amrex::TagBoxArray& tags,
        const amrex::MultiFab& U,
        const amrex::MultiFab& terrain,
        const amrex::Geometry& geom,
        int lev,
        amrex::Real time,
        const PhysicsParameters& phys_params
    ){
        std::visit([&](auto&& solver) {
            using T = std::decay_t<decltype(solver)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error("[SolverVariant Error] Solver not initialized.");
            } else {
                solver.tag_cells(tags, U, terrain, geom, lev, time, phys_params);
            }
        }, m_solver);
    }

private:
    SolverType m_solver;
};

#endif // SOLVER_VARIANT_H_
