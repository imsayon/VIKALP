#include "vikalp/solver/continuous.hpp"
#include "vikalp/contracts/NonlinearOracle.hpp"

#include "../continuous_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vikalp::solver {
namespace {

using detail::Clock;

struct NlpRow {
    std::vector<Scalar> gradient;
    Scalar value = 0.0;
    bool equality = false;
    int nonlinear_sign = 0;
};

struct NlpEvaluation {
    Scalar objective = 0.0;
    std::vector<Scalar> gradient;
    std::vector<Scalar> constraints;
    std::vector<NlpRow> equalities;
    std::vector<NlpRow> inequalities;
};

bool evaluate(const Model &model, std::span<const Scalar> x,
              const detail::Bounds &bounds, NlpEvaluation &evaluation,
              std::string &message) {
    const auto &oracle = *model.nonlinear;
    const Index n = model.num_variables();
    const Index m = model.num_constraints();
    std::vector<Scalar> nonlinear_values(static_cast<std::size_t>(m));
    std::vector<Scalar> nonlinear_gradient(static_cast<std::size_t>(n));
    std::vector<Scalar> jacobian_values(oracle.jacobian_pattern().column_indices.size());
    oracle.constraint_values(x, nonlinear_values);
    oracle.objective_gradient(x, nonlinear_gradient);
    oracle.jacobian_values(x, jacobian_values);
    if (!detail::finite_vector(nonlinear_values) ||
        !detail::finite_vector(nonlinear_gradient) ||
        !detail::finite_vector(jacobian_values)) {
        message = "nonlinear oracle returned non-finite values";
        return false;
    }
    detail::linear_values(model, x, evaluation.constraints);
    for (Index row = 0; row < m; ++row) {
        evaluation.constraints[static_cast<std::size_t>(row)] +=
            nonlinear_values[static_cast<std::size_t>(row)];
    }
    evaluation.objective = detail::objective_value(model, x) + oracle.objective(x);
    evaluation.gradient = model.linear_objective;
    const auto &q = model.quadratic_objective;
    for (Index row = 0; row < q.pattern.rows; ++row) {
        for (Index k = q.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < q.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            evaluation.gradient[static_cast<std::size_t>(row)] +=
                q.values[position] * x[static_cast<std::size_t>(q.pattern.column_indices[position])];
        }
    }
    for (Index i = 0; i < n; ++i) {
        evaluation.gradient[static_cast<std::size_t>(i)] +=
            nonlinear_gradient[static_cast<std::size_t>(i)];
    }
    if (!std::isfinite(evaluation.objective) ||
        !detail::finite_vector(evaluation.gradient)) {
        message = "nonlinear oracle returned a non-finite objective or gradient";
        return false;
    }

    std::vector<std::vector<Scalar>> jacobian(
        static_cast<std::size_t>(m), std::vector<Scalar>(static_cast<std::size_t>(n), 0.0));
    const auto &jacobian_pattern = oracle.jacobian_pattern();
    for (Index row = 0; row < m; ++row) {
        for (Index k = jacobian_pattern.row_offsets[static_cast<std::size_t>(row)];
             k < jacobian_pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            jacobian[static_cast<std::size_t>(row)][static_cast<std::size_t>(
                jacobian_pattern.column_indices[position])] += jacobian_values[position];
        }
        const auto &linear = model.constraint_matrix;
        for (Index k = linear.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < linear.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            jacobian[static_cast<std::size_t>(row)][static_cast<std::size_t>(
                linear.pattern.column_indices[position])] += linear.values[position];
        }
    }

    evaluation.equalities.clear();
    evaluation.inequalities.clear();
    for (Index row = 0; row < m; ++row) {
        const auto lower = model.constraint_lower[static_cast<std::size_t>(row)];
        const auto upper = model.constraint_upper[static_cast<std::size_t>(row)];
        const auto &row_gradient = jacobian[static_cast<std::size_t>(row)];
        if (std::isfinite(lower) && std::isfinite(upper) && lower == upper) {
            evaluation.equalities.push_back({row_gradient,
                                             evaluation.constraints[static_cast<std::size_t>(row)] - lower,
                                             true, 1});
        } else {
            if (std::isfinite(lower)) {
                auto gradient = row_gradient;
                for (auto &value : gradient) value = -value;
                evaluation.inequalities.push_back({
                    std::move(gradient), lower - evaluation.constraints[static_cast<std::size_t>(row)],
                    false, -1});
            }
            if (std::isfinite(upper)) {
                evaluation.inequalities.push_back({
                    row_gradient, evaluation.constraints[static_cast<std::size_t>(row)] - upper,
                    false, 1});
            }
        }
    }
    for (Index i = 0; i < n; ++i) {
        const auto lower = bounds.lower[static_cast<std::size_t>(i)];
        const auto upper = bounds.upper[static_cast<std::size_t>(i)];
        if (std::isfinite(lower)) {
            std::vector<Scalar> gradient(static_cast<std::size_t>(n), 0.0);
            gradient[static_cast<std::size_t>(i)] = -1.0;
            evaluation.inequalities.push_back({
                std::move(gradient), lower - x[static_cast<std::size_t>(i)], false, 0});
        }
        if (std::isfinite(upper)) {
            std::vector<Scalar> gradient(static_cast<std::size_t>(n), 0.0);
            gradient[static_cast<std::size_t>(i)] = 1.0;
            evaluation.inequalities.push_back({
                std::move(gradient), x[static_cast<std::size_t>(i)] - upper, false, 0});
        }
    }
    return true;
}

Scalar infeasibility(const NlpEvaluation &evaluation) {
    Scalar result = 0.0;
    for (const auto &row : evaluation.equalities) result = std::max(result, std::abs(row.value));
    for (const auto &row : evaluation.inequalities) result = std::max(result, std::max<Scalar>(0.0, row.value));
    return result;
}

Scalar dot_row(const std::vector<Scalar> &row, std::span<const Scalar> vector) {
    return detail::dot_host(row, vector);
}

Scalar dual_residual_norm(const NlpEvaluation &evaluation,
                          std::span<const Scalar> lambda,
                          std::span<const Scalar> mu) {
    if (evaluation.gradient.empty()) return 0.0;
    std::vector<Scalar> residual = evaluation.gradient;
    for (std::size_t i = 0; i < evaluation.equalities.size(); ++i) {
        for (std::size_t column = 0; column < residual.size(); ++column) {
            residual[column] += lambda[i] * evaluation.equalities[i].gradient[column];
        }
    }
    for (std::size_t i = 0; i < evaluation.inequalities.size(); ++i) {
        for (std::size_t column = 0; column < residual.size(); ++column) {
            residual[column] += mu[i] * evaluation.inequalities[i].gradient[column];
        }
    }
    Scalar result = 0.0;
    for (const auto value : residual) result = std::max(result, std::abs(value));
    return result;
}

Scalar complementarity_residual(const NlpEvaluation &evaluation,
                                std::span<const Scalar> mu) {
    Scalar result = 0.0;
    for (std::size_t i = 0; i < evaluation.inequalities.size(); ++i) {
        result = std::max(result, std::abs(-evaluation.inequalities[i].value * mu[i]));
    }
    return result;
}

CsrPattern dense_pattern(Index dimension) {
    CsrPattern pattern;
    pattern.rows = dimension;
    pattern.columns = dimension;
    pattern.row_offsets.resize(static_cast<std::size_t>(dimension + 1));
    pattern.column_indices.reserve(static_cast<std::size_t>(dimension * dimension));
    for (Index row = 0; row < dimension; ++row) {
        pattern.row_offsets[static_cast<std::size_t>(row)] = row * dimension;
        for (Index column = 0; column < dimension; ++column) {
            pattern.column_indices.push_back(column);
        }
    }
    pattern.row_offsets[static_cast<std::size_t>(dimension)] = dimension * dimension;
    return pattern;
}

void add_quadratic_hessian(const Model &model, std::vector<Scalar> &hessian) {
    const auto n = static_cast<std::size_t>(model.num_variables());
    for (Index row = 0; row < model.quadratic_objective.pattern.rows; ++row) {
        for (Index k = model.quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < model.quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            hessian[static_cast<std::size_t>(row) * n + static_cast<std::size_t>(
                model.quadratic_objective.pattern.column_indices[position])] +=
                model.quadratic_objective.values[position];
        }
    }
}

} // namespace

SolveResult solve_nlp(const Model &model, ExecutionBackend &backend,
                      const SolverOptions &options,
                      std::span<const Scalar> warm_start,
                      std::span<const Scalar> variable_lower_override,
                      std::span<const Scalar> variable_upper_override) {
    detail::Bounds bounds;
    std::string message;
    if (!detail::copy_bounds(model, variable_lower_override,
                             variable_upper_override, bounds, message)) {
        return detail::invalid_result(message);
    }
    auto validation = detail::validate_continuous(model, ProblemClass::NLP, bounds);
    if (validation.status != SolveStatus::NotSolved) return validation;
    if (model.num_variables() == 0 || !detail::valid_options(options)) {
        return detail::invalid_result("NLP requires variables and non-negative solver limits/tolerances");
    }
    if (options.continuous_method != ContinuousMethod::Auto &&
        options.continuous_method != ContinuousMethod::InteriorPoint) {
        return detail::unsupported_result("requested continuous method is not implemented for NLP");
    }
    const Index n = model.num_variables();
    std::vector<Scalar> x(static_cast<std::size_t>(n));
    if (!warm_start.empty()) {
        if (static_cast<Index>(warm_start.size()) != n || !detail::finite_vector(warm_start)) {
            return detail::invalid_result("warm start must be empty or finite and sized to the model");
        }
        x.assign(warm_start.begin(), warm_start.end());
    } else {
        for (Index i = 0; i < n; ++i) {
            const auto lower = bounds.lower[static_cast<std::size_t>(i)];
            const auto upper = bounds.upper[static_cast<std::size_t>(i)];
            if (std::isfinite(lower) && std::isfinite(upper)) {
                x[static_cast<std::size_t>(i)] = lower + (upper - lower) * 0.5;
            } else if (std::isfinite(lower)) {
                x[static_cast<std::size_t>(i)] = lower + 1.0;
            } else if (std::isfinite(upper)) {
                x[static_cast<std::size_t>(i)] = upper - 1.0;
            } else {
                x[static_cast<std::size_t>(i)] = 0.0;
            }
        }
    }
    for (Index i = 0; i < n; ++i) {
        const auto position = static_cast<std::size_t>(i);
        const auto lower = bounds.lower[position];
        const auto upper = bounds.upper[position];
        if (std::isfinite(lower) && std::isfinite(upper)) {
            const auto margin = std::min(
                (upper - lower) * 0.25,
                std::max<Scalar>(1e-12, (upper - lower) * 1e-6));
            x[position] = std::clamp(x[position], lower + margin, upper - margin);
        } else if (std::isfinite(lower)) {
            x[position] = std::max(x[position], lower + 1e-6);
        } else if (std::isfinite(upper)) {
            x[position] = std::min(x[position], upper - 1e-6);
        }
    }

    NlpEvaluation evaluation;
    if (!evaluate(model, x, bounds, evaluation, message)) {
        return detail::invalid_result(message);
    }
    for (const auto &row : evaluation.inequalities) {
        if (row.value >= 0.0) {
            SolveResult result;
            result.status = SolveStatus::NumericalFailure;
            result.message = "NLP baseline requires a strictly feasible initial point";
            return result;
        }
    }
    std::vector<Scalar> lambda(evaluation.equalities.size(), 0.0);
    std::vector<Scalar> mu(evaluation.inequalities.size(), 1.0);
    const auto started = detail::Clock::now();
    SolveResult result;
    result.solver = "PrimalDualIPM";
    result.backend = "provided";

    for (Index iteration = 0; iteration < options.iteration_limit; ++iteration) {
        if (detail::timed_out(started, options)) {
            result.status = SolveStatus::TimeLimit;
            break;
        }
        if (infeasibility(evaluation) <= options.primal_tolerance &&
            dual_residual_norm(evaluation, lambda, mu) <= options.dual_tolerance &&
            complementarity_residual(evaluation, mu) <= options.primal_tolerance) {
            result.status = SolveStatus::LocallyOptimal;
            break;
        }
        const auto equalities = evaluation.equalities.size();
        const auto inequalities = evaluation.inequalities.size();
        // Rebuild nonlinear multipliers in the same row-bound order used by evaluate().
        std::vector<Scalar> nonlinear_lambda(
            static_cast<std::size_t>(model.num_constraints()), 0.0);
        std::size_t equality_index = 0;
        std::size_t inequality_index = 0;
        for (Index row = 0; row < model.num_constraints(); ++row) {
            const auto lower = model.constraint_lower[static_cast<std::size_t>(row)];
            const auto upper = model.constraint_upper[static_cast<std::size_t>(row)];
            if (std::isfinite(lower) && std::isfinite(upper) && lower == upper) {
                nonlinear_lambda[static_cast<std::size_t>(row)] = lambda[equality_index++];
            } else {
                if (std::isfinite(lower)) nonlinear_lambda[static_cast<std::size_t>(row)] -= mu[inequality_index++];
                if (std::isfinite(upper)) nonlinear_lambda[static_cast<std::size_t>(row)] += mu[inequality_index++];
            }
        }
        std::vector<Scalar> nonlinear_hessian_values(
            model.nonlinear->hessian_pattern().column_indices.size());
        model.nonlinear->hessian_values(x, 1.0, nonlinear_lambda,
                                        nonlinear_hessian_values);
        if (!detail::finite_vector(nonlinear_hessian_values)) {
            result.status = SolveStatus::NumericalFailure;
            result.message = "nonlinear oracle returned a non-finite Hessian";
            break;
        }
        const auto n_size = static_cast<std::size_t>(n);
        std::vector<Scalar> hessian(n_size * n_size, 0.0);
        add_quadratic_hessian(model, hessian);
        const auto &hessian_pattern = model.nonlinear->hessian_pattern();
        for (Index row = 0; row < n; ++row) {
            for (Index k = hessian_pattern.row_offsets[static_cast<std::size_t>(row)];
                 k < hessian_pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
                const auto position = static_cast<std::size_t>(k);
                hessian[static_cast<std::size_t>(row) * n_size + static_cast<std::size_t>(
                    hessian_pattern.column_indices[position])] += nonlinear_hessian_values[position];
            }
        }

        std::vector<Scalar> dual_residual(static_cast<std::size_t>(n), 0.0);
        for (Index i = 0; i < n; ++i) dual_residual[static_cast<std::size_t>(i)] = evaluation.gradient[static_cast<std::size_t>(i)];
        for (std::size_t i = 0; i < equalities; ++i) {
            for (Index column = 0; column < n; ++column) {
                dual_residual[static_cast<std::size_t>(column)] +=
                    lambda[i] * evaluation.equalities[i].gradient[static_cast<std::size_t>(column)];
            }
        }
        for (std::size_t i = 0; i < inequalities; ++i) {
            for (Index column = 0; column < n; ++column) {
                dual_residual[static_cast<std::size_t>(column)] +=
                    mu[i] * evaluation.inequalities[i].gradient[static_cast<std::size_t>(column)];
            }
        }
        std::vector<Scalar> slacks;
        slacks.reserve(inequalities);
        for (const auto &row : evaluation.inequalities) slacks.push_back(-row.value);
        const Scalar complementarity = inequalities == 0
                                           ? 0.0
                                           : detail::dot_host(slacks, mu) /
                                                 static_cast<Scalar>(inequalities);
        // Eliminate inequality slacks/multipliers into the reduced primal KKT system.
        std::vector<Scalar> rhs(static_cast<std::size_t>(n + equalities), 0.0);
        for (Index i = 0; i < n; ++i) rhs[static_cast<std::size_t>(i)] = -dual_residual[static_cast<std::size_t>(i)];
        const Scalar target = 0.1 * complementarity;
        for (std::size_t i = 0; i < inequalities; ++i) {
            const Scalar rc = slacks[i] * mu[i] - target;
            for (Index column = 0; column < n; ++column) {
                rhs[static_cast<std::size_t>(column)] +=
                    evaluation.inequalities[i].gradient[static_cast<std::size_t>(column)] * rc / slacks[i];
            }
        }
        for (std::size_t i = 0; i < equalities; ++i) rhs[static_cast<std::size_t>(n) + i] = -evaluation.equalities[i].value;

        for (std::size_t i = 0; i < inequalities; ++i) {
            const Scalar weight = mu[i] / slacks[i];
            for (Index row = 0; row < n; ++row) {
                for (Index column = 0; column < n; ++column) {
                    hessian[static_cast<std::size_t>(row) * n_size + static_cast<std::size_t>(column)] +=
                        weight * evaluation.inequalities[i].gradient[static_cast<std::size_t>(row)] *
                        evaluation.inequalities[i].gradient[static_cast<std::size_t>(column)];
                }
            }
        }
        const Index dimension = n + static_cast<Index>(equalities);
        std::vector<Scalar> kkt_values(static_cast<std::size_t>(dimension * dimension), 0.0);
        for (Index row = 0; row < n; ++row) {
            for (Index column = 0; column < n; ++column) {
                kkt_values[static_cast<std::size_t>(row * dimension + column)] =
                    hessian[static_cast<std::size_t>(row * n + column)];
            }
            for (std::size_t equality = 0; equality < equalities; ++equality) {
                kkt_values[static_cast<std::size_t>(row * dimension + n + equality)] =
                    evaluation.equalities[equality].gradient[static_cast<std::size_t>(row)];
                kkt_values[static_cast<std::size_t>((n + static_cast<Index>(equality)) * dimension + row)] =
                    evaluation.equalities[equality].gradient[static_cast<std::size_t>(row)];
            }
        }
        auto kkt = backend.create_matrix(dense_pattern(dimension));
        backend.set_values(kkt_values, *kkt);
        auto rhs_vector = backend.create_vector(dimension);
        auto direction_vector = backend.create_vector(dimension);
        backend.upload(rhs, *rhs_vector);
        const auto linear_result = backend.solve_linear_system(
            *kkt, *rhs_vector, *direction_vector,
            MatrixProperty::SymmetricIndefinite, options.dual_tolerance,
            std::max<Index>(50, n * 10));
        if (!linear_result.converged) {
            result.status = SolveStatus::NumericalFailure;
            result.message = "KKT linear solve did not converge";
            break;
        }
        std::vector<Scalar> direction(static_cast<std::size_t>(dimension));
        backend.download(*direction_vector, direction);
        std::vector<Scalar> dx(direction.begin(), direction.begin() + n);
        std::vector<Scalar> dlambda(direction.begin() + n, direction.end());
        std::vector<Scalar> dmu(inequalities, 0.0);
        // Fraction-to-boundary keeps every slack and inequality multiplier positive.
        Scalar alpha = 1.0;
        for (std::size_t i = 0; i < inequalities; ++i) {
            const Scalar ds = -dot_row(evaluation.inequalities[i].gradient, dx);
            const Scalar rc = slacks[i] * mu[i] - target;
            dmu[i] = (mu[i] * dot_row(evaluation.inequalities[i].gradient, dx) - rc) / slacks[i];
            if (ds < 0.0) alpha = std::min(alpha, -0.99 * slacks[i] / ds);
            if (dmu[i] < 0.0) alpha = std::min(alpha, -0.99 * mu[i] / dmu[i]);
        }
        const Scalar current_theta = infeasibility(evaluation);
        const Scalar current_complementarity = complementarity_residual(evaluation, mu);
        bool accepted = false;
        std::vector<Scalar> trial_x(static_cast<std::size_t>(n));
        NlpEvaluation trial;
        // Filter line search accepts progress in feasibility, objective, or barrier complementarity.
        for (int attempt = 0; attempt < 20; ++attempt) {
            for (Index i = 0; i < n; ++i) trial_x[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] + alpha * dx[static_cast<std::size_t>(i)];
            std::string trial_message;
            if (!evaluate(model, trial_x, bounds, trial, trial_message)) {
                alpha *= 0.5;
                continue;
            }
            bool interior = true;
            for (const auto &row : trial.inequalities) interior = interior && row.value < 0.0;
            const Scalar trial_theta = infeasibility(trial);
            Scalar trial_complementarity = 0.0;
            for (std::size_t i = 0; i < inequalities; ++i) {
                trial_complementarity = std::max(
                    trial_complementarity,
                    std::abs(-trial.inequalities[i].value *
                              std::max<Scalar>(1e-12, mu[i] + alpha * dmu[i])));
            }
            const bool filter_accept = trial_theta <= (1.0 - 1e-4) * current_theta ||
                                       trial.objective <= evaluation.objective -
                                           1e-4 * alpha * std::max<Scalar>(1.0, std::abs(evaluation.objective)) ||
                                       trial_complementarity <=
                                           (1.0 - 1e-4 * alpha) * current_complementarity;
            if (interior && filter_accept) {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) {
            result.status = SolveStatus::NumericalFailure;
            result.message = "filter line search could not accept a feasible step";
            break;
        }
        x = std::move(trial_x);
        evaluation = std::move(trial);
        for (std::size_t i = 0; i < lambda.size(); ++i) lambda[i] += alpha * dlambda[i];
        for (std::size_t i = 0; i < mu.size(); ++i) mu[i] = std::max<Scalar>(1e-12, mu[i] + alpha * dmu[i]);
        result.iterations = iteration + 1;

        if (infeasibility(evaluation) <= options.primal_tolerance &&
            dual_residual_norm(evaluation, lambda, mu) <= options.dual_tolerance &&
            complementarity_residual(evaluation, mu) <= options.primal_tolerance) {
            result.status = SolveStatus::LocallyOptimal;
            break;
        }
    }
    result.primal_solution = x;
    result.objective_value = evaluation.objective;
    result.primal_residual = infeasibility(evaluation);
    result.stationarity_residual = dual_residual_norm(evaluation, lambda, mu);
    result.complementarity_residual = complementarity_residual(evaluation, mu);
    if (result.status == SolveStatus::NotSolved) result.status = SolveStatus::IterationLimit;
    if (!detail::finite_solution(result.primal_solution)) {
        result.status = SolveStatus::NumericalFailure;
        result.message = "NLP produced a non-finite iterate";
    }
    result.solve_seconds =
        std::chrono::duration<Scalar>(Clock::now() - started).count();
    return result;
}

} // namespace vikalp::solver
