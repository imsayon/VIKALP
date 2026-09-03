// Flow D — Top-level solver with problem-class routing.

#include "vikalp/solver/Solver.hpp"
#include "vikalp/solver/mip/BranchAndBound.hpp"

#include <span>

namespace vikalp {

SolveResult solve(const Model &model,
                  const RelaxationOracle &oracle,
                  const SolverOptions &options) {
    // Validate
    const auto errors = model.validate();
    if (!errors.empty()) {
        SolveResult r;
        r.status = SolveStatus::InvalidModel;
        r.message = errors.front();
        return r;
    }

    const auto pc = model.problem_class();

    switch (pc) {
    case ProblemClass::LP:
    case ProblemClass::QP:
    case ProblemClass::NLP: {
        // Continuous problem → delegate directly to oracle
        SolveResult r = oracle.solve(
            model,
            std::span<const Scalar>{},  // no overrides
            std::span<const Scalar>{},
            std::span<const Scalar>{},  // no warm start
            options);
        r.solver = "vikalp-continuous";
        return r;
    }

    case ProblemClass::MILP:
    case ProblemClass::MIQP: {
        // Mixed-integer linear/quadratic → branch-and-bound
        return solve_mip(model, oracle, options);
    }

    case ProblemClass::MINLP: {
        // Mixed-integer nonlinear → branch-and-bound
        // Full OA would be solve_oa(), but B&B with NLP relaxation is the
        // baseline. When OuterApproximation is ready, this will call solve_oa().
        return solve_mip(model, oracle, options);
    }
    }

    // Unreachable, but satisfy compiler
    SolveResult r;
    r.status = SolveStatus::UnsupportedModel;
    r.message = "Unknown problem class";
    return r;
}

} // namespace vikalp
