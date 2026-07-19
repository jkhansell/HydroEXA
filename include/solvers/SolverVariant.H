#ifndef SOLVER_VARIANT_H_
#define SOLVER_VARIANT_H_

#include <variant>
#include <stdexcept>
#include <type_traits>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>

// Include concrete solvers (or forward declare them if preferred, but since they
// no longer depend on AmrMeshState, they can be directly included safely)
#include <solvers/Roe.H>

class SolverVariant {
public:
    using SolverType = std::variant<std::monostate, Roe>;

    // Constructors
    SolverVariant() : m_solver(std::monostate{}) {}
    
    template <typename T>
    SolverVariant(T&& solver) : m_solver(std::forward<T>(solver)) {}

    bool has_solver() const {
        return !std::holds_alternative<std::monostate>(m_solver);
    }

    void compute_fluxes(amrex::Vector<amrex::MultiFab>& U,
                        amrex::Vector<amrex::MultiFab>& terrain,
                        const amrex::Vector<amrex::Geometry>& geom,
                        int lev) 
    {
        std::visit([&](auto&& solver) {
            using T = std::decay_t<decltype(solver)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error("[SolverVariant Error] Solver not initialized.");
            } else {
                solver.compute_fluxes(U, terrain, geom, lev);
            }
        }, m_solver);
    }

    void compute_dt(amrex::Vector<amrex::Real>& dt,
                    const amrex::Vector<amrex::MultiFab>& U,
                    const amrex::Vector<amrex::MultiFab>& terrain,
                    const amrex::Vector<amrex::Geometry>& geom,
                    const amrex::Vector<amrex::IntVect>& ref_ratio,
                    int finest_level, amrex::Real cfl) 
    {
        std::visit([&](auto&& solver) {
            using T = std::decay_t<decltype(solver)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error("[SolverVariant Error] Solver not initialized.");
            } else {
                solver.compute_dt(dt, U, terrain, geom, ref_ratio, finest_level, cfl);
            }
        }, m_solver);
    }

private:
    SolverType m_solver;
};

#endif // SOLVER_VARIANT_H_