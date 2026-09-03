#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"
#include <vector>
#include <cmath>

namespace vikalp::solver::lp {

SolveResult solve_pdhg_baseline(const Model& model, ExecutionBackend& backend, Index max_iterations = 1000) {
    Index num_vars = model.num_variables();
    Index num_cons = model.num_constraints();
    
    SolveResult result;
    result.solver = "PDHG_Baseline";
    result.iterations = 0;

    if (num_vars == 0) {
        result.status = SolveStatus::InvalidModel;
        return result;
    }

    auto x = backend.create_vector(num_vars);
    auto y = backend.create_vector(num_cons);
    auto x_new = backend.create_vector(num_vars);
    auto KTy = backend.create_vector(num_vars);
    auto K_extrap = backend.create_vector(num_cons);
    auto extrapolation = backend.create_vector(num_vars);
    auto primal_res_vec = backend.create_vector(num_cons);
    auto dual_res_vec = backend.create_vector(num_vars);
    
    auto c = backend.create_vector(num_vars);
    auto var_lower = backend.create_vector(num_vars);
    auto var_upper = backend.create_vector(num_vars);
    auto K = backend.create_matrix(model.constraint_matrix.pattern);

    backend.fill(*x, 0.0);
    backend.fill(*y, 0.0);
    backend.upload(model.linear_objective, *c);
    backend.upload(model.variable_lower, *var_lower);
    backend.upload(model.variable_upper, *var_upper);
    backend.set_values(model.constraint_matrix.values, *K);

    double tau = 0.05;  
    double sigma = 0.05;
    const double tolerance = 1e-4;

    for (Index iter = 0; iter < max_iterations; ++iter) {
        backend.multiply(1.0, *K, *y, 0.0, *KTy, Transpose::Yes);
        backend.copy(*x, *x_new);
        backend.axpby(-tau, *c, 1.0, *x_new);
        backend.axpby(tau, *KTy, 1.0, *x_new);
        backend.project_box(*x_new, *var_lower, *var_upper);

        backend.copy(*x, *extrapolation);
        backend.axpby(2.0, *x_new, -1.0, *extrapolation);
        backend.multiply(1.0, *K, *extrapolation, 0.0, *K_extrap, Transpose::No);
        backend.axpby(-sigma, *K_extrap, 1.0, *y);
        
        backend.copy(*x_new, *x);
        result.iterations++;

        // Periodic KKT Residual Check
        if (iter % 20 == 0) {
            backend.multiply(1.0, *K, *x, 0.0, *primal_res_vec, Transpose::No);
            result.primal_residual = backend.norm_inf(*primal_res_vec);

            backend.multiply(1.0, *K, *y, 0.0, *dual_res_vec, Transpose::Yes);
            backend.copy(*c, *dual_res_vec);
            backend.axpby(-1.0, *KTy, 1.0, *dual_res_vec);
            result.stationarity_residual = backend.norm_inf(*dual_res_vec);

            if (result.primal_residual < tolerance && result.stationarity_residual < tolerance) {
                result.status = SolveStatus::Optimal;
                break;
            }
        }
    }

    result.primal_solution.resize(num_vars);
    backend.download(*x, result.primal_solution);
    if (result.status != SolveStatus::Optimal) {
        result.status = SolveStatus::IterationLimit;
    }
    
    return result;
}

} // namespace vikalp::solver::lp