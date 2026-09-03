#include "vikalp/backend/CpuBackend.hpp"
#include "vikalp/solver/continuous.hpp"

#include "../continuous_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vikalp::solver {
namespace {

using detail::Clock;

SolveResult solve_scaled_qp(const Model &model, ExecutionBackend &backend,
                            const SolverOptions &options,
                            std::span<const Scalar> warm_start,
                            const detail::Bounds &bounds) {
    const auto started = Clock::now();
    const auto scaled = detail::scale_linear_model(model, bounds);
    const Index m = model.num_constraints();
    const Index n = model.num_variables();

    if (!warm_start.empty() &&
        (static_cast<Index>(warm_start.size()) != n ||
         !detail::finite_vector(warm_start))) {
        return detail::invalid_result("warm start must be empty or finite and sized to the model");
    }

    std::vector<Scalar> scaled_q_values = model.quadratic_objective.values;
    for (Index row = 0; row < n; ++row) {
        for (Index k = model.quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < model.quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            const auto column = model.quadratic_objective.pattern.column_indices[position];
            scaled_q_values[position] *=
                scaled.column_scale[static_cast<std::size_t>(row)] *
                scaled.column_scale[static_cast<std::size_t>(column)];
        }
    }

    auto matrix = backend.create_matrix(scaled.pattern);
    auto quadratic = backend.create_matrix(model.quadratic_objective.pattern);
    backend.set_values(scaled.values, *matrix);
    backend.set_values(scaled_q_values, *quadratic);
    auto x = backend.create_vector(n);
    auto x_next = backend.create_vector(n);
    auto x_bar = backend.create_vector(n);
    auto y = backend.create_vector(m);
    auto aty = backend.create_vector(n);
    auto ax_bar = backend.create_vector(m);
    auto y_argument = backend.create_vector(m);
    auto y_projected = backend.create_vector(m);
    auto ax = backend.create_vector(m);
    auto projected_ax = backend.create_vector(m);
    auto qx = backend.create_vector(n);
    auto gradient = backend.create_vector(n);
    auto projected_x = backend.create_vector(n);
    auto residual = backend.create_vector(n);

    auto objective = backend.create_vector(n);
    auto variable_lower = backend.create_vector(n);
    auto variable_upper = backend.create_vector(n);
    auto row_lower = backend.create_vector(m);
    auto row_upper = backend.create_vector(m);
    backend.upload(scaled.objective, *objective);
    backend.upload(scaled.variable_lower, *variable_lower);
    backend.upload(scaled.variable_upper, *variable_upper);
    backend.upload(scaled.row_lower, *row_lower);
    backend.upload(scaled.row_upper, *row_upper);
    if (warm_start.empty()) {
        backend.fill(*x, 0.0);
    } else {
        std::vector<Scalar> scaled_start(warm_start.begin(), warm_start.end());
        for (Index i = 0; i < n; ++i) {
            scaled_start[static_cast<std::size_t>(i)] /=
                scaled.column_scale[static_cast<std::size_t>(i)];
        }
        backend.upload(scaled_start, *x);
    }
    backend.project_box(*x, *variable_lower, *variable_upper);
    backend.fill(*y, 0.0);

    const Scalar matrix_norm = detail::scaled_operator_bound(scaled.pattern, scaled.values);
    Scalar q_norm = 0.0;
    for (Index row = 0; row < n; ++row) {
        Scalar row_sum = 0.0;
        for (Index k = model.quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < model.quadratic_objective.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            row_sum += std::abs(scaled_q_values[static_cast<std::size_t>(k)]);
        }
        q_norm = std::max(q_norm, row_sum);
    }
    const Scalar max_step = 0.99 / std::max<Scalar>(1.0, matrix_norm * matrix_norm + q_norm);
    Scalar tau = max_step;
    Scalar sigma = max_step;
    Scalar previous_metric = std::numeric_limits<Scalar>::infinity();
    bool restart = false;

    SolveResult result;
    result.solver = "PDQP";
    result.backend = "provided";

    // Restarted primal-dual QP iterations share the LP backend contract.
    for (Index iter = 0; iter < options.iteration_limit; ++iter) {
        if (detail::timed_out(started, options)) {
            result.status = SolveStatus::TimeLimit;
            break;
        }
        backend.multiply(1.0, *matrix, *y, 0.0, *aty, Transpose::Yes);
        backend.multiply(1.0, *quadratic, *x, 0.0, *qx, Transpose::No);
        backend.copy(*qx, *gradient);
        backend.axpby(1.0, *objective, 1.0, *gradient);
        backend.axpby(1.0, *aty, 1.0, *gradient);
        backend.copy(*x, *x_next);
        backend.axpby(-tau, *gradient, 1.0, *x_next);
        backend.project_box(*x_next, *variable_lower, *variable_upper);

        if (restart) {
            backend.copy(*x_next, *x_bar);
            restart = false;
        } else {
            backend.copy(*x, *x_bar);
            backend.axpby(2.0, *x_next, -1.0, *x_bar);
        }
        backend.multiply(1.0, *matrix, *x_bar, 0.0, *ax_bar, Transpose::No);
        backend.copy(*y, *y_argument);
        backend.axpby(sigma, *ax_bar, 1.0, *y_argument);
        backend.copy(*y_argument, *y_projected);
        backend.axpby(1.0 / sigma, *y_projected, 0.0, *y_projected);
        backend.project_box(*y_projected, *row_lower, *row_upper);
        backend.copy(*y_argument, *y);
        backend.axpby(-sigma, *y_projected, 1.0, *y);
        backend.copy(*x_next, *x);

        result.iterations = iter + 1;
        backend.multiply(1.0, *matrix, *x, 0.0, *ax, Transpose::No);
        backend.copy(*ax, *projected_ax);
        backend.project_box(*projected_ax, *row_lower, *row_upper);
        backend.axpby(1.0, *projected_ax, -1.0, *ax);
        const Scalar primal_scaled = backend.norm_inf(*ax);

        backend.multiply(1.0, *matrix, *y, 0.0, *aty, Transpose::Yes);
        backend.multiply(1.0, *quadratic, *x, 0.0, *qx, Transpose::No);
        backend.copy(*qx, *gradient);
        backend.axpby(1.0, *objective, 1.0, *gradient);
        backend.axpby(1.0, *aty, 1.0, *gradient);
        backend.copy(*x, *projected_x);
        backend.axpby(-1.0, *gradient, 1.0, *projected_x);
        backend.project_box(*projected_x, *variable_lower, *variable_upper);
        backend.copy(*x, *residual);
        backend.axpby(-1.0, *projected_x, 1.0, *residual);
        const Scalar stationarity_scaled = backend.norm_inf(*residual);
        const Scalar metric = std::max(primal_scaled, stationarity_scaled);

        if (iter > 0 && iter % 50 == 0) {
            if (metric > previous_metric * 1.25) {
                tau *= 0.5;
                sigma *= 0.5;
                restart = true;
            } else if (metric < previous_metric * 0.1) {
                tau = std::min(tau * 1.05, max_step);
                sigma = tau;
            }
            previous_metric = metric;
        } else if (iter == 0) {
            previous_metric = metric;
        }
        if (metric <= std::max(options.primal_tolerance, options.dual_tolerance)) {
            result.status = SolveStatus::Optimal;
            break;
        }
    }

    backend.synchronize();
    std::vector<Scalar> scaled_solution(static_cast<std::size_t>(n));
    backend.download(*x, scaled_solution);
    std::vector<Scalar> solution = scaled_solution;
    for (Index i = 0; i < n; ++i) {
        solution[static_cast<std::size_t>(i)] *=
            scaled.column_scale[static_cast<std::size_t>(i)];
    }
    result.primal_solution = solution;
    result.objective_value = detail::objective_value(model, solution);
    if (!std::isfinite(result.objective_value)) {
        result.status = SolveStatus::NumericalFailure;
        result.message = "PDQP produced a non-finite objective";
    }
    std::vector<Scalar> row_values;
    detail::linear_values(model, solution, row_values);
    result.primal_residual = detail::box_violation(
        row_values, model.constraint_lower, model.constraint_upper);
    result.primal_residual = std::max(
        result.primal_residual,
        detail::box_violation(solution, bounds.lower, bounds.upper));

    std::vector<Scalar> dual(static_cast<std::size_t>(m), 0.0);
    backend.download(*y, dual);
    for (Index row = 0; row < m; ++row) {
        dual[static_cast<std::size_t>(row)] *=
            scaled.row_scale[static_cast<std::size_t>(row)];
    }
    result.constraint_lower_dual.resize(static_cast<std::size_t>(m));
    result.constraint_upper_dual.resize(static_cast<std::size_t>(m));
    for (Index row = 0; row < m; ++row) {
        result.constraint_lower_dual[static_cast<std::size_t>(row)] =
            std::max<Scalar>(0.0, -dual[static_cast<std::size_t>(row)]);
        result.constraint_upper_dual[static_cast<std::size_t>(row)] =
            std::max<Scalar>(0.0, dual[static_cast<std::size_t>(row)]);
    }
    std::vector<Scalar> q_dense;
    detail::csr_to_dense(model.quadratic_objective, q_dense);
    std::vector<Scalar> gradient_host(model.linear_objective);
    for (Index row = 0; row < n; ++row) {
        for (Index column = 0; column < n; ++column) {
            gradient_host[static_cast<std::size_t>(row)] +=
                q_dense[static_cast<std::size_t>(row * n + column)] *
                solution[static_cast<std::size_t>(column)];
        }
    }
    for (Index row = 0; row < m; ++row) {
        const auto multiplier = dual[static_cast<std::size_t>(row)];
        for (Index k = model.constraint_matrix.pattern.row_offsets[static_cast<std::size_t>(row)];
             k < model.constraint_matrix.pattern.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            gradient_host[static_cast<std::size_t>(model.constraint_matrix.pattern.column_indices[position])] +=
                multiplier * model.constraint_matrix.values[position];
        }
    }
    result.stationarity_residual = 0.0;
    result.variable_lower_dual.assign(static_cast<std::size_t>(n), 0.0);
    result.variable_upper_dual.assign(static_cast<std::size_t>(n), 0.0);
    result.complementarity_residual = 0.0;
    const auto active_tolerance = std::max<Scalar>(10.0 * options.primal_tolerance, 1e-8);
    for (Index i = 0; i < n; ++i) {
        const auto position = static_cast<std::size_t>(i);
        const auto projected = std::clamp(
            solution[position] - gradient_host[position], bounds.lower[position],
            bounds.upper[position]);
        result.stationarity_residual = std::max(
            result.stationarity_residual, std::abs(solution[position] - projected));
        if (std::isfinite(bounds.lower[position]) &&
            solution[position] - bounds.lower[position] <= active_tolerance) {
            result.variable_lower_dual[position] = std::max<Scalar>(0.0, gradient_host[position]);
            result.complementarity_residual = std::max(
                result.complementarity_residual,
                std::abs((solution[position] - bounds.lower[position]) *
                         result.variable_lower_dual[position]));
        }
        if (std::isfinite(bounds.upper[position]) &&
            bounds.upper[position] - solution[position] <= active_tolerance) {
            result.variable_upper_dual[position] = std::max<Scalar>(0.0, -gradient_host[position]);
            result.complementarity_residual = std::max(
                result.complementarity_residual,
                std::abs((bounds.upper[position] - solution[position]) *
                         result.variable_upper_dual[position]));
        }
    }
    for (Index row = 0; row < m; ++row) {
        const auto value = row_values[static_cast<std::size_t>(row)];
        const auto lower = model.constraint_lower[static_cast<std::size_t>(row)];
        const auto upper = model.constraint_upper[static_cast<std::size_t>(row)];
        if (std::isfinite(lower)) {
            result.complementarity_residual = std::max(
                result.complementarity_residual,
                std::abs((value - lower) * result.constraint_lower_dual[static_cast<std::size_t>(row)]));
        }
        if (std::isfinite(upper)) {
            result.complementarity_residual = std::max(
                result.complementarity_residual,
                std::abs((upper - value) * result.constraint_upper_dual[static_cast<std::size_t>(row)]));
        }
    }
    if (result.status == SolveStatus::NotSolved) result.status = SolveStatus::IterationLimit;
    if (!detail::finite_solution(result.primal_solution)) {
        result.status = SolveStatus::NumericalFailure;
        result.message = "PDQP produced a non-finite iterate";
    } else if (result.status == SolveStatus::Optimal &&
               (result.primal_residual > options.primal_tolerance ||
                result.stationarity_residual > options.dual_tolerance)) {
        result.status = SolveStatus::NumericalFailure;
        result.message = "PDQP stopped without satisfying original-coordinate residual checks";
    }
    if (result.status == SolveStatus::Optimal) {
        result.primal_bound = result.objective_value;
        result.dual_bound = result.objective_value;
        result.absolute_gap = 0.0;
        result.relative_gap = 0.0;
    }
    result.solve_seconds =
        std::chrono::duration<Scalar>(Clock::now() - started).count();
    return result;
}

} // namespace

SolveResult solve_qp(const Model &model, ExecutionBackend &backend,
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
    auto validation = detail::validate_continuous(model, ProblemClass::QP, bounds);
    if (validation.status != SolveStatus::NotSolved) return validation;
    if (model.num_variables() == 0) {
        return detail::invalid_result("QP requires at least one variable");
    }
    if (!detail::positive_semidefinite(model.quadratic_objective)) {
        return detail::unsupported_result("QP objective matrix is not positive semidefinite");
    }
    if (!detail::valid_options(options)) {
        return detail::invalid_result("solver limits and tolerances must be non-negative");
    }
    if (options.continuous_method != ContinuousMethod::Auto &&
        options.continuous_method != ContinuousMethod::PDHG) {
        return detail::unsupported_result("requested continuous method is not implemented for QP");
    }
    return solve_scaled_qp(model, backend, options, warm_start, bounds);
}

SolveResult QpRelaxationOracle::solve(
    const Model &model, std::span<const Scalar> variable_lower_override,
    std::span<const Scalar> variable_upper_override,
    std::span<const Scalar> warm_start, const SolverOptions &options) const {
    auto backend = make_cpu_backend();
    return solve_qp(model, *backend, options, warm_start,
                    variable_lower_override, variable_upper_override);
}

} // namespace vikalp::solver
