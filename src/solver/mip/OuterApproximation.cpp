// Flow D — convex outer approximation for MINLP.

#include "vikalp/solver/mip/OuterApproximation.hpp"

#include "vikalp/contracts/NonlinearOracle.hpp"
#include "vikalp/solver/mip/BranchAndBound.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace vikalp {
namespace {

Scalar evaluate_objective(const Model &model, std::span<const Scalar> x) {
    Scalar result = model.objective_offset;
    for (std::size_t i = 0; i < x.size(); ++i) {
        result += model.linear_objective[i] * x[i];
    }
    const auto &q = model.quadratic_objective;
    for (Index row = 0; row < q.pattern.rows; ++row) {
        for (Index position = q.pattern.row_offsets[static_cast<std::size_t>(row)];
             position < q.pattern.row_offsets[static_cast<std::size_t>(row + 1)];
             ++position) {
            const auto pos = static_cast<std::size_t>(position);
            result += 0.5 * q.values[pos] * x[static_cast<std::size_t>(row)] *
                      x[static_cast<std::size_t>(q.pattern.column_indices[pos])];
        }
    }
    return result + model.nonlinear->objective(x);
}

bool linearly_feasible(const Model &model, std::span<const Scalar> x,
                       Scalar tolerance) {
    if (x.size() != static_cast<std::size_t>(model.num_variables())) return false;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || x[i] < model.variable_lower[i] - tolerance ||
            x[i] > model.variable_upper[i] + tolerance) {
            return false;
        }
    }
    const auto &matrix = model.constraint_matrix;
    for (std::size_t row = 0;
         row < static_cast<std::size_t>(model.num_constraints()); ++row) {
        Scalar activity = 0.0;
        for (Index position = matrix.pattern.row_offsets[row];
             position < matrix.pattern.row_offsets[row + 1]; ++position) {
            const auto pos = static_cast<std::size_t>(position);
            activity += matrix.values[pos] *
                        x[static_cast<std::size_t>(matrix.pattern.column_indices[pos])];
        }
        if (activity < model.constraint_lower[row] - tolerance ||
            activity > model.constraint_upper[row] + tolerance) {
            return false;
        }
    }
    return true;
}

bool nonlinearly_feasible(const Model &model, std::span<const Scalar> x,
                          Scalar tolerance) {
    if (!model.nonlinear || x.size() !=
        static_cast<std::size_t>(model.num_variables())) return false;
    std::vector<Scalar> values(static_cast<std::size_t>(model.num_constraints()));
    model.nonlinear->constraint_values(x, values);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) ||
            values[i] < model.constraint_lower[i] - tolerance ||
            values[i] > model.constraint_upper[i] + tolerance) {
            return false;
        }
    }
    return true;
}

void append_row(Model &model, const std::vector<Index> &indices,
                const std::vector<Scalar> &coefficients,
                Scalar lower, Scalar upper) {
    model.constraint_lower.push_back(lower);
    model.constraint_upper.push_back(upper);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        model.constraint_matrix.pattern.column_indices.push_back(indices[i]);
        model.constraint_matrix.values.push_back(coefficients[i]);
    }
    model.constraint_matrix.pattern.row_offsets.push_back(
        static_cast<Index>(model.constraint_matrix.pattern.column_indices.size()));
    ++model.constraint_matrix.pattern.rows;
}

bool add_linearizations(Model &master, const Model &original,
                        std::span<const Scalar> x) {
    const auto &oracle = *original.nonlinear;
    const auto n = static_cast<std::size_t>(original.num_variables());
    const auto m = static_cast<std::size_t>(original.num_constraints());
    std::vector<Scalar> values(m);
    std::vector<Scalar> jacobian_values(
        oracle.jacobian_pattern().column_indices.size());
    std::vector<Scalar> gradient(n);
    oracle.constraint_values(x, values);
    oracle.jacobian_values(x, jacobian_values);
    oracle.objective_gradient(x, gradient);
    const Scalar objective_value = oracle.objective(x);
    if (!std::isfinite(objective_value) ||
        !std::all_of(values.begin(), values.end(),
                     [](Scalar value) { return std::isfinite(value); }) ||
        !std::all_of(jacobian_values.begin(), jacobian_values.end(),
                     [](Scalar value) { return std::isfinite(value); }) ||
        !std::all_of(gradient.begin(), gradient.end(),
                     [](Scalar value) { return std::isfinite(value); })) {
        return false;
    }

    const auto &pattern = oracle.jacobian_pattern();
    for (std::size_t row = 0; row < m; ++row) {
        std::vector<Index> indices;
        std::vector<Scalar> coefficients;
        Scalar offset = values[row];
        for (Index position = pattern.row_offsets[row];
             position < pattern.row_offsets[row + 1]; ++position) {
            const auto pos = static_cast<std::size_t>(position);
            const Index column = pattern.column_indices[pos];
            indices.push_back(column);
            coefficients.push_back(jacobian_values[pos]);
            offset -= jacobian_values[pos] * x[static_cast<std::size_t>(column)];
        }
        append_row(master, indices, coefficients,
                   original.constraint_lower[row] - offset,
                   original.constraint_upper[row] - offset);
    }

    // The extra variable is an epigraph for the convex nonlinear objective.
    const Index epigraph = original.num_variables();
    {
        std::vector<Index> indices;
        std::vector<Scalar> coefficients;
        for (Index column = 0; column < original.num_variables(); ++column) {
            if (std::abs(gradient[static_cast<std::size_t>(column)]) > 0.0) {
                indices.push_back(column);
                coefficients.push_back(gradient[static_cast<std::size_t>(column)]);
            }
        }
        indices.push_back(epigraph);
        coefficients.push_back(-1.0);
        Scalar gradient_at_x = 0.0;
        for (std::size_t i = 0; i < n; ++i) gradient_at_x += gradient[i] * x[i];
        append_row(master, indices, coefficients, -Model::infinity(),
                   -objective_value + gradient_at_x);
    }
    return true;
}

Scalar absolute_gap(Scalar primal, Scalar dual) {
    if (!std::isfinite(primal) || !std::isfinite(dual)) return Model::infinity();
    return std::max(0.0, primal - dual);
}

Scalar relative_gap(Scalar primal, Scalar dual) {
    const Scalar absolute = absolute_gap(primal, dual);
    return std::isfinite(absolute) ? absolute / std::max(1.0, std::abs(primal))
                                   : Model::infinity();
}

} // namespace

SolveResult solve_oa(const Model &model, const RelaxationOracle &oracle,
                     const SolverOptions &options) {
    SolveResult result;
    result.solver = "vikalp-oa";
    if (!model.nonlinear || !model.has_integer_variables()) {
        result.status = SolveStatus::UnsupportedModel;
        result.message = "OA requires a mixed-integer nonlinear model";
        return result;
    }
    if (model.has_quadratic_objective()) {
        result.status = SolveStatus::UnsupportedModel;
        result.message = "OA currently supports a linear plus nonlinear objective";
        return result;
    }
    const auto errors = model.validate();
    if (!errors.empty()) {
        result.status = SolveStatus::InvalidModel;
        result.message = errors.front();
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    const Index max_iterations = 50;
    Scalar incumbent_objective = Model::infinity();
    std::vector<Scalar> incumbent_solution;
    Scalar dual_bound = -Model::infinity();
    Index total_nodes = 0;
    Index iterations = 0;
    SolveStatus termination = SolveStatus::IterationLimit;

    SolveResult initial = oracle.solve(model, {}, {}, {}, options);
    if (initial.status == SolveStatus::Infeasible) {
        result.status = SolveStatus::Infeasible;
        result.message = "Nonlinear relaxation is infeasible";
        return result;
    }
    if (initial.status != SolveStatus::Optimal &&
        initial.status != SolveStatus::LocallyOptimal &&
        initial.status != SolveStatus::Feasible) {
        result.status = initial.status;
        result.message = "Nonlinear relaxation did not produce a usable point";
        return result;
    }
    if (initial.primal_solution.size() !=
            static_cast<std::size_t>(model.num_variables()) ||
        !linearly_feasible(model, initial.primal_solution,
                           options.primal_tolerance)) {
        result.status = SolveStatus::NumericalFailure;
        result.message = "Nonlinear relaxation returned an invalid point";
        return result;
    }

    Model master = model;
    master.nonlinear.reset();
    master.linear_objective.push_back(1.0);
    master.variable_lower.push_back(-Model::infinity());
    master.variable_upper.push_back(Model::infinity());
    master.variable_types.push_back(VariableType::Continuous);
    master.constraint_matrix.pattern.columns = master.num_variables();
    if (!add_linearizations(master, model, initial.primal_solution) ||
        !master.validate().empty()) {
        result.status = SolveStatus::NumericalFailure;
        result.message = "Unable to construct the OA master";
        return result;
    }
    if (std::isfinite(initial.dual_bound)) {
        dual_bound = initial.dual_bound;
    } else if (initial.status == SolveStatus::Optimal) {
        dual_bound = initial.objective_value;
    }

    for (Index iteration = 0; iteration < max_iterations; ++iteration) {
        ++iterations;
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= options.time_limit_seconds) {
            termination = SolveStatus::TimeLimit;
            break;
        }
        if (total_nodes >= options.node_limit) {
            termination = SolveStatus::NodeLimit;
            break;
        }

        SolverOptions master_options = options;
        master_options.node_limit = options.node_limit - total_nodes;
        if (std::isfinite(options.time_limit_seconds)) {
            master_options.time_limit_seconds =
                std::max(0.0, options.time_limit_seconds - elapsed);
        }
        SolveResult master_result = solve_mip(master, oracle, master_options);
        total_nodes += master_result.nodes;
        if (master_result.status == SolveStatus::Infeasible) {
            termination = incumbent_solution.empty() ? SolveStatus::Infeasible
                                                      : SolveStatus::IterationLimit;
            break;
        }
        if (master_result.status == SolveStatus::NumericalFailure ||
            master_result.status == SolveStatus::InvalidModel ||
            master_result.status == SolveStatus::UnsupportedModel) {
            termination = master_result.status;
            result.message = master_result.message;
            break;
        }
        if (master_result.status == SolveStatus::NodeLimit ||
            master_result.status == SolveStatus::TimeLimit) {
            termination = master_result.status;
        }
        if (master_result.primal_solution.size() !=
            static_cast<std::size_t>(master.num_variables())) {
            break;
        }
        if (master_result.status == SolveStatus::Optimal &&
            std::isfinite(master_result.dual_bound)) {
            dual_bound = std::max(dual_bound, master_result.dual_bound);
        }

        const auto candidate = std::span<const Scalar>(
            master_result.primal_solution.data(),
            static_cast<std::size_t>(model.num_variables()));
        if (linearly_feasible(model, candidate, options.primal_tolerance) &&
            nonlinearly_feasible(model, candidate, options.primal_tolerance)) {
            const Scalar objective = evaluate_objective(model, candidate);
            if (objective < incumbent_objective) {
                incumbent_objective = objective;
                incumbent_solution.assign(candidate.begin(), candidate.end());
            }
        }

        if (!add_linearizations(master, model, candidate)) {
            termination = SolveStatus::NumericalFailure;
            result.message = "Nonlinear oracle returned invalid derivatives";
            break;
        }
        const Scalar ag = absolute_gap(incumbent_objective, dual_bound);
        const Scalar rg = relative_gap(incumbent_objective, dual_bound);
        if (master_result.status == SolveStatus::Optimal &&
            (ag <= options.absolute_gap_tolerance ||
             rg <= options.relative_gap_tolerance)) {
            termination = SolveStatus::Optimal;
            break;
        }
        if (termination == SolveStatus::NodeLimit ||
            termination == SolveStatus::TimeLimit) {
            break;
        }
    }

    result.solve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    result.iterations = iterations;
    result.nodes = total_nodes;
    if (!incumbent_solution.empty()) {
        result.primal_solution = incumbent_solution;
        result.objective_value = incumbent_objective;
        result.primal_bound = incumbent_objective;
        result.integrality_residual = 0.0;
    }
    result.dual_bound = dual_bound;
    result.absolute_gap = absolute_gap(incumbent_objective, dual_bound);
    result.relative_gap = relative_gap(incumbent_objective, dual_bound);
    result.status = termination;
    if (result.status == SolveStatus::IterationLimit &&
        incumbent_solution.empty()) {
        result.status = SolveStatus::Infeasible;
    }
    return result;
}

} // namespace vikalp
