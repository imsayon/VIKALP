#ifndef VIKALP_SOLVE_RESULT_HPP
#define VIKALP_SOLVE_RESULT_HPP

#include "Model.hpp"

#include <vector>

namespace vikalk {

// Common solver termination status.
enum class SolveStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    NumericalError,
    InvalidModel
};

// Common result returned by solver components.
struct SolveResult {
    SolveStatus status = SolveStatus::NumericalError;

    // Solution in the canonical model coordinates.
    std::vector<Scalar> primal_solution;

    // Objective value of the returned solution.
    Scalar objective_value = 0.0;

    // Bounds used by optimization algorithms.
    Scalar primal_bound = 0.0;
    Scalar dual_bound = 0.0;

    // Verification-related residuals.
    Scalar primal_residual = 0.0;
    Scalar dual_residual = 0.0;
    Scalar complementarity_residual = 0.0;

    // Relative optimality gap.
    Scalar relative_gap = 0.0;
};

}  // namespace vikalk

#endif  // VIKALP_SOLVE_RESULT_HPP