// Flow D — Top-level solver with problem-class routing.

#include "vikalp/solver/Solver.hpp"
#include "vikalp/backend/CpuBackend.hpp"
#include "vikalp/backend/CudaBackend.hpp"
#include "vikalp/solver/continuous.hpp"
#include "vikalp/solver/mip/BranchAndBound.hpp"
#include "vikalp/solver/mip/OuterApproximation.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <span>
#include <string>

namespace vikalp {
namespace {

#ifndef VIKALP_CUDA_ENABLED
SolveResult unsupported_cuda() {
    SolveResult result;
    result.status = SolveStatus::UnsupportedModel;
    result.message = "CUDA backend was requested but this build has no CUDA support";
    return result;
}
#endif

class BuiltinRelaxationOracle final : public RelaxationOracle {
public:
    [[nodiscard]] SolveResult solve(
        const Model &model,
        std::span<const Scalar> variable_lower_override,
        std::span<const Scalar> variable_upper_override,
        std::span<const Scalar> warm_start,
        const SolverOptions &options) const override {
        try {
            // ponytail: one backend per relaxation; reuse it only if profiling
            // shows backend construction matters for large branch-and-bound trees.
            std::unique_ptr<ExecutionBackend> backend;
            std::string backend_name;
            if (options.backend == BackendPreference::CUDA) {
#ifdef VIKALP_CUDA_ENABLED
                backend = make_cuda_backend();
                backend_name = "cuda";
#else
                return unsupported_cuda();
#endif
            } else {
                backend = make_cpu_backend();
                backend_name = "cpu";
            }

            Model relaxation = model;
            std::fill(relaxation.variable_types.begin(),
                      relaxation.variable_types.end(), VariableType::Continuous);

            SolveResult result;
            switch (relaxation.problem_class()) {
            case ProblemClass::LP:
                result = solver::solve_lp(relaxation, *backend, options, warm_start,
                                          variable_lower_override,
                                          variable_upper_override);
                break;
            case ProblemClass::QP:
                result = solver::solve_qp(relaxation, *backend, options, warm_start,
                                          variable_lower_override,
                                          variable_upper_override);
                break;
            case ProblemClass::NLP:
                result = solver::solve_nlp(relaxation, *backend, options, warm_start,
                                           variable_lower_override,
                                           variable_upper_override);
                break;
            default:
                result.status = SolveStatus::UnsupportedModel;
                result.message = "No built-in continuous relaxation is available";
                break;
            }
            result.backend = backend_name;
            return result;
        } catch (const std::exception &error) {
            SolveResult result;
            result.status = SolveStatus::NumericalFailure;
            result.message = error.what();
            return result;
        }
    }
};

} // namespace

SolveResult solve(const Model &model, const SolverOptions &options) {
    BuiltinRelaxationOracle oracle;
    SolveResult result = solve(model, oracle, options);
    if (result.backend.empty() && result.status != SolveStatus::UnsupportedModel) {
        result.backend = options.backend == BackendPreference::CUDA ? "cuda" : "cpu";
    }
    return result;
}

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
        return solve_oa(model, oracle, options);
    }
    }

    // Unreachable, but satisfy compiler
    SolveResult r;
    r.status = SolveStatus::UnsupportedModel;
    r.message = "Unknown problem class";
    return r;
}

} // namespace vikalp
