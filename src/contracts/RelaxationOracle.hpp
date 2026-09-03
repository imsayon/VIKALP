#ifndef VIKALP_RELAXATION_ORACLE_HPP
#define VIKALP_RELAXATION_ORACLE_HPP

#include "Model.hpp"

namespace vikalk {

struct SolverOptions;
struct SolveResult;

// Common interface for solving a continuous relaxation.
// Used by LP, QP, and nonlinear relaxation components.
class RelaxationOracle {
public:
    virtual ~RelaxationOracle() = default;

    virtual SolveResult solve(
        const Model& model,
        const SolverOptions& options) const = 0;
};

}  // namespace vikalk

#endif  // VIKALP_RELAXATION_ORACLE_HPP