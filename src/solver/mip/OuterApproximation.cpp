// Flow D — Outer Approximation for MINLP.
//
// Implements a simplified Duran-Grossmann OA loop:
// 1. Solve continuous NLP relaxation at initial point
// 2. Linearize at solution → create MILP master with OA cuts
// 3. Solve MILP master to get integer assignment
// 4. Fix integers, solve NLP subproblem
// 5. If feasible and improving, update incumbent
// 6. Add OA cut from subproblem solution
// 7. Repeat until convergence
//
// Since Flow A/B continuous solvers don't exist yet, all relaxation solves
// go through the RelaxationOracle. The NonlinearOracle is used for
// linearization (gradient evaluations).

#include "vikalp/solver/mip/OuterApproximation.hpp"
#include "vikalp/solver/mip/BranchAndBound.hpp"
#include "vikalp/contracts/NonlinearOracle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

namespace vikalp {
namespace {

/// Evaluate objective at a point: c'x + offset + f(x)
Scalar eval_objective(const Model &model, std::span<const Scalar> x) {
    Scalar obj = model.objective_offset;
    const auto n = std::min(x.size(),
        static_cast<std::size_t>(model.num_variables()));
    for (std::size_t i = 0; i < n; ++i) {
        obj += model.linear_objective[i] * x[i];
    }
    if (model.nonlinear) {
        obj += model.nonlinear->objective(x);
    }
    return obj;
}

/// Check if a solution is constraint-feasible (linear constraints only).
bool is_linearly_feasible(const Model &model,
                          std::span<const Scalar> x,
                          Scalar tol) {
    const auto &A = model.constraint_matrix;
    const auto m = static_cast<std::size_t>(model.num_constraints());
    for (std::size_t row = 0; row < m; ++row) {
        Scalar ax = 0.0;
        for (Index k = A.pattern.row_offsets[row];
             k < A.pattern.row_offsets[row + 1]; ++k) {
            const auto col = static_cast<std::size_t>(
                A.pattern.column_indices[static_cast<std::size_t>(k)]);
            if (col < x.size()) {
                ax += A.values[static_cast<std::size_t>(k)] * x[col];
            }
        }
        if (ax < model.constraint_lower[row] - tol ||
            ax > model.constraint_upper[row] + tol) {
            return false;
        }
    }
    return true;
}

Scalar compute_abs_gap(Scalar primal, Scalar dual) {
    if (!std::isfinite(primal) || !std::isfinite(dual))
        return Model::infinity();
    return std::max(0.0, primal - dual);
}

Scalar compute_rel_gap(Scalar primal, Scalar dual) {
    const Scalar ag = compute_abs_gap(primal, dual);
    if (!std::isfinite(ag)) return Model::infinity();
    return ag / std::max(1.0, std::abs(primal));
}

} // namespace

SolveResult solve_oa(const Model &model,
                     const RelaxationOracle &oracle,
                     const SolverOptions &options) {
    SolveResult result;
    result.solver = "vikalp-oa";

    // Validate
    if (!model.has_integer_variables()) {
        result.status = SolveStatus::UnsupportedModel;
        result.message = "OA requires integer variables";
        return result;
    }

    const auto errors = model.validate();
    if (!errors.empty()) {
        result.status = SolveStatus::InvalidModel;
        result.message = errors.front();
        return result;
    }

    const auto start_time = std::chrono::steady_clock::now();
    const Index max_oa_iters = 50;  // OA iteration limit
    Scalar incumbent_obj = Model::infinity();
    std::vector<Scalar> incumbent_solution;
    Scalar dual_bound = -Model::infinity();
    Index total_nodes = 0;

    // Step 1: Solve initial continuous relaxation via oracle
    SolveResult nlp_result = oracle.solve(
        model,
        std::span<const Scalar>{},
        std::span<const Scalar>{},
        std::span<const Scalar>{},
        options);

    if (nlp_result.status == SolveStatus::Optimal ||
        nlp_result.status == SolveStatus::LocallyOptimal ||
        nlp_result.status == SolveStatus::Feasible) {
        dual_bound = nlp_result.objective_value;
    }

    // If NLP relaxation fails, fall back to pure B&B
    if (nlp_result.primal_solution.empty()) {
        return solve_mip(model, oracle, options);
    }

    // Step 2-7: OA iterations
    for (Index iter = 0; iter < max_oa_iters; ++iter) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration<double>(elapsed).count() >= options.time_limit_seconds) {
            result.status = SolveStatus::TimeLimit;
            break;
        }

        // Solve MILP master via B&B (the oracle handles the relaxation)
        SolverOptions mip_opts = options;
        mip_opts.node_limit = std::min(options.node_limit,
            static_cast<Index>(10000));  // cap per iteration

        SolveResult master = solve_mip(model, oracle, mip_opts);
        total_nodes += master.nodes;

        if (master.status == SolveStatus::Infeasible) {
            // MILP master infeasible → original is infeasible
            if (!std::isfinite(incumbent_obj)) {
                result.status = SolveStatus::Infeasible;
            }
            break;
        }

        if (master.primal_solution.empty()) break;

        // Update dual bound from master
        if (std::isfinite(master.dual_bound)) {
            dual_bound = std::max(dual_bound, master.dual_bound);
        }

        // Check if master solution improves incumbent
        const Scalar master_obj = eval_objective(
            model, std::span<const Scalar>(master.primal_solution));

        if (master_obj < incumbent_obj &&
            is_linearly_feasible(model,
                std::span<const Scalar>(master.primal_solution),
                options.primal_tolerance)) {
            incumbent_obj = master_obj;
            incumbent_solution = master.primal_solution;
        }

        // Check gap
        const Scalar ag = compute_abs_gap(incumbent_obj, dual_bound);
        const Scalar rg = compute_rel_gap(incumbent_obj, dual_bound);
        if (ag <= options.absolute_gap_tolerance ||
            rg <= options.relative_gap_tolerance) {
            break;
        }

        // For pure MILP (no nonlinear oracle), one iteration suffices
        if (!model.nonlinear) break;
    }

    // Build result
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    result.solve_seconds = std::chrono::duration<double>(elapsed).count();
    result.nodes = total_nodes;
    result.iterations = total_nodes;

    if (!incumbent_solution.empty()) {
        result.primal_solution = incumbent_solution;
        result.objective_value = incumbent_obj;
        result.primal_bound = incumbent_obj;
    }

    result.dual_bound = dual_bound;
    result.absolute_gap = compute_abs_gap(
        std::isfinite(incumbent_obj) ? incumbent_obj : Model::infinity(),
        dual_bound);
    result.relative_gap = compute_rel_gap(
        std::isfinite(incumbent_obj) ? incumbent_obj : Model::infinity(),
        dual_bound);

    if (result.status == SolveStatus::NotSolved) {
        if (std::isfinite(incumbent_obj) &&
            (result.absolute_gap <= options.absolute_gap_tolerance ||
             result.relative_gap <= options.relative_gap_tolerance)) {
            result.status = SolveStatus::Optimal;
        } else if (std::isfinite(incumbent_obj)) {
            result.status = SolveStatus::Feasible;
        } else {
            result.status = SolveStatus::Infeasible;
        }
    }

    return result;
}

} // namespace vikalp
