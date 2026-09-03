#ifndef VIKALP_SOLVER_OPTIONS_HPP
#define VIKALP_SOLVER_OPTIONS_HPP

#include "Model.hpp"

namespace vikalk {

// Common solver configuration shared by solver components.
struct SolverOptions {
    // Maximum number of iterations.
    Index max_iterations = 1000;

    // Numerical tolerances.
    Scalar primal_tolerance = 1e-8;
    Scalar dual_tolerance = 1e-8;
    Scalar integrality_tolerance = 1e-8;

    // Maximum allowed running time in seconds.
    Scalar time_limit_seconds = 0.0;
};

}  // namespace vikalk

#endif  // VIKALP_SOLVER_OPTIONS_HPP