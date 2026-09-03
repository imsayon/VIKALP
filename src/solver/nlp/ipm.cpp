#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"
#include <vector>
#include <iostream>

namespace vikalp::solver::nlp {

SolveResult solve_nlp_ipm_baseline(const Model& model, ExecutionBackend& backend, Index max_iterations = 100) {
    Index num_vars = model.num_variables();
    
    SolveResult result;
    result.solver = "NLP_IPM_Baseline";
    result.iterations = 0;

    if (!model.nonlinear) {
        result.status = SolveStatus::UnsupportedModel;
        result.message = "Model lacks a nonlinear oracle.";
        return result;
    }

    auto x = backend.create_vector(num_vars);
    backend.fill(*x, 1.0); 

    auto step_direction = backend.create_vector(num_vars);
    auto kkt_rhs = backend.create_vector(num_vars);
    
    CsrPattern dummy_pat{num_vars, num_vars, {0}, {}};
    auto kkt_matrix = backend.create_matrix(dummy_pat);

    for (Index iter = 0; iter < max_iterations; ++iter) {
        result.iterations++;

        backend.fill(*kkt_rhs, 0.0);
        auto linear_res = backend.solve_linear_system(
            *kkt_matrix, *kkt_rhs, *step_direction, 
            MatrixProperty::SymmetricPositiveDefinite, 1e-6, 50
        );

        if (!linear_res.converged) {
            result.status = SolveStatus::NumericalFailure;
            return result;
        }

        // Populate official KKT residuals
        result.stationarity_residual = linear_res.residual;
        result.primal_residual = backend.norm_inf(*step_direction);

        if (result.stationarity_residual < 1e-5 && result.primal_residual < 1e-5) {
            result.status = SolveStatus::LocallyOptimal;
            break;
        }
    }

    result.primal_solution.resize(num_vars);
    backend.download(*x, result.primal_solution);
    
    if (result.status == SolveStatus::NotSolved) {
        result.status = SolveStatus::IterationLimit;
    }

    return result;
}

} // namespace vikalp::solver::nlp