#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/ExecutionBackend.hpp"
#include <vector>
#include <cmath>

namespace vikalp::solver::qp {

SolveResult solve_pdqp_baseline(const Model& model, ExecutionBackend& backend, Index max_iterations = 1000) {
    Index num_vars = model.num_variables();
    Index num_cons = model.num_constraints();
    
    SolveResult result;
    result.solver = "PDQP_Baseline";
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
    auto Qx = backend.create_vector(num_vars);
    auto dual_res_vec = backend.create_vector(num_vars);
    
    auto c = backend.create_vector(num_vars);
    auto var_lower = backend.create_vector(num_vars);
    auto var_upper = backend.create_vector(num_vars);
    auto K = backend.create_matrix(model.constraint_matrix.pattern);
    
    std::unique_ptr<BackendMatrix> Q = nullptr;
    if (model.has_quadratic_objective()) {
        Q = backend.create_matrix(model.quadratic_objective.pattern);
    }

    backend.fill(*x, 0.0);
    backend.fill(*y, 0.0);
    backend.fill(*Qx, 0.0);
    backend.upload(model.linear_objective, *c);
    backend.upload(model.variable_lower, *var_lower);
    backend.upload(model.variable_upper, *var_upper);
    backend.set_values(model.constraint_matrix.values, *K);
    
    if (Q) {
        backend.set_values(model.quadratic_objective.values, *Q);
    }

    double tau = 0.02;  
    double sigma = 0.02;
    const double tolerance = 1e-4;

    for (Index iter = 0; iter < max_iterations; ++iter) {
        backend.multiply(1.0, *K, *y, 0.0, *KTy, Transpose::Yes);
        
        if (Q) {
            backend.multiply(1.0, *Q, *x, 0.0, *Qx, Transpose::No);
        }

        backend.copy(*x, *x_new);
        if (Q) backend.axpby(-tau, *Qx, 1.0, *x_new);
        backend.axpby(-tau, *c, 1.0, *x_new);
        backend.axpby(tau, *KTy, 1.0, *x_new);
        
        backend.project_box(*x_new, *var_lower, *var_upper);

        backend.copy(*x, *extrapolation);
        backend.axpby(2.0, *x_new, -1.0, *extrapolation);
        backend.multiply(1.0, *K, *extrapolation, 0.0, *K_extrap, Transpose::No);
        backend.axpby(-sigma, *K_extrap, 1.0, *y);
        
        backend.copy(*x_new, *x);
        result.iterations++;

        // QP Convergence Check
        if (iter % 20 == 0) {
            if (Q) backend.multiply(1.0, *Q, *x, 0.0, *Qx, Transpose::No);
            
            // Stationarity residual: ||Qx + c - K^Ty||_inf
            backend.copy(*c, *dual_res_vec);
            if (Q) backend.axpby(1.0, *Qx, 1.0, *dual_res_vec);
            backend.axpby(-1.0, *KTy, 1.0, *dual_res_vec);
            result.stationarity_residual = backend.norm_inf(*dual_res_vec);

            if (result.stationarity_residual < tolerance) {
                result.status = SolveStatus::Optimal;
                break;
            }
        }
    }

    result.primal_solution.resize(num_vars);
    backend.download(*x, result.primal_solution);

    // Calculate Final Objective Value: 0.5 * x'Qx + c'x
    Scalar lin_obj = backend.dot(*c, *x);
    Scalar quad_obj = 0.0;
    if (Q) {
        backend.multiply(1.0, *Q, *x, 0.0, *Qx, Transpose::No);
        quad_obj = 0.5 * backend.dot(*x, *Qx);
    }
    result.objective_value = quad_obj + lin_obj;
    result.primal_bound = result.objective_value;

    if (result.status != SolveStatus::Optimal) {
        result.status = SolveStatus::IterationLimit;
    }
    
    return result;
}

} // namespace vikalp::solver::qp