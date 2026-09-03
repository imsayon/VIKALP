// Flow D — branch-and-bound MIP engine.

#include "vikalp/solver/mip/BranchAndBound.hpp"

#include "vikalp/solver/mip/BbNode.hpp"
#include "vikalp/solver/mip/BbQueue.hpp"
#include "vikalp/solver/mip/CutGenerator.hpp"
#include "vikalp/solver/mip/Pseudocosts.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace vikalp {
namespace {

Scalar integrality_violation(Scalar value) {
    return std::abs(value - std::round(value));
}

bool is_integer_feasible(const Model &model, std::span<const Scalar> x,
                         Scalar tolerance) {
    if (x.size() != static_cast<std::size_t>(model.num_variables())) return false;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (model.variable_types[i] != VariableType::Continuous &&
            integrality_violation(x[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

Scalar max_integrality_residual(const Model &model, std::span<const Scalar> x) {
    Scalar result = 0.0;
    for (std::size_t i = 0; i < x.size() &&
                            i < static_cast<std::size_t>(model.num_variables()); ++i) {
        if (model.variable_types[i] != VariableType::Continuous) {
            result = std::max(result, integrality_violation(x[i]));
        }
    }
    return result;
}

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
    return result;
}

bool is_feasible(const Model &model, std::span<const Scalar> x, Scalar tolerance,
                 const std::vector<Scalar> *lower_override = nullptr,
                 const std::vector<Scalar> *upper_override = nullptr) {
    const auto n = static_cast<std::size_t>(model.num_variables());
    if (x.size() != n) return false;
    const auto &lower = lower_override ? *lower_override : model.variable_lower;
    const auto &upper = upper_override ? *upper_override : model.variable_upper;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || x[i] < lower[i] - tolerance ||
            x[i] > upper[i] + tolerance) {
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

Index select_most_fractional(const Model &model, std::span<const Scalar> x,
                             const std::vector<Scalar> &lower,
                             const std::vector<Scalar> &upper, Scalar tolerance) {
    Index selected = -1;
    Scalar best = -1.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (model.variable_types[i] == VariableType::Continuous ||
            integrality_violation(x[i]) <= tolerance) {
            continue;
        }
        const Scalar floor_value = std::floor(x[i]);
        const Scalar ceil_value = std::ceil(x[i]);
        if (floor_value < lower[i] && ceil_value > upper[i]) continue;
        const Scalar score = 0.5 - std::abs((x[i] - floor_value) - 0.5);
        if (score > best || (score == best && static_cast<Index>(i) < selected)) {
            selected = static_cast<Index>(i);
            best = score;
        }
    }
    return selected;
}

Index select_pseudocost(const Model &model, std::span<const Scalar> x,
                        const std::vector<Scalar> &lower,
                        const std::vector<Scalar> &upper, Scalar tolerance,
                        const PseudocostTracker &pseudocosts) {
    Index selected = -1;
    Scalar best = -1.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (model.variable_types[i] == VariableType::Continuous ||
            integrality_violation(x[i]) <= tolerance ||
            !pseudocosts.is_reliable(static_cast<Index>(i))) {
            continue;
        }
        const Scalar floor_value = std::floor(x[i]);
        const Scalar ceil_value = std::ceil(x[i]);
        if (floor_value < lower[i] && ceil_value > upper[i]) continue;
        const Scalar score = pseudocosts.score(
            static_cast<Index>(i), x[i] - floor_value, ceil_value - x[i]);
        if (score > best || (score == best && static_cast<Index>(i) < selected)) {
            selected = static_cast<Index>(i);
            best = score;
        }
    }
    return selected >= 0 ? selected :
        select_most_fractional(model, x, lower, upper, tolerance);
}

bool try_rounding(const Model &model, std::span<const Scalar> x,
                  const std::vector<Scalar> &lower, const std::vector<Scalar> &upper,
                  std::vector<Scalar> &rounded, Scalar &objective,
                  Scalar int_tolerance, Scalar primal_tolerance) {
    const auto n = static_cast<std::size_t>(model.num_variables());
    if (x.size() != n) return false;
    rounded.assign(x.begin(), x.end());
    for (std::size_t i = 0; i < n; ++i) {
        if (model.variable_types[i] != VariableType::Continuous) {
            rounded[i] = std::round(rounded[i]);
        }
        rounded[i] = std::clamp(rounded[i], lower[i], upper[i]);
    }
    if (!is_integer_feasible(model, rounded, int_tolerance) ||
        !is_feasible(model, rounded, primal_tolerance, &lower, &upper)) {
        return false;
    }
    objective = evaluate_objective(model, rounded);
    return std::isfinite(objective);
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

bool valid_cut(const Model &model, const Cut &cut) {
    if (cut.indices.empty() || cut.indices.size() != cut.coefficients.size() ||
        std::isnan(cut.lower) || std::isnan(cut.upper) || cut.lower > cut.upper) {
        return false;
    }
    Index previous = -1;
    for (std::size_t i = 0; i < cut.indices.size(); ++i) {
        if (cut.indices[i] <= previous ||
            cut.indices[i] < 0 || cut.indices[i] >= model.num_variables() ||
            !std::isfinite(cut.coefficients[i])) {
            return false;
        }
        previous = cut.indices[i];
    }
    return std::isfinite(cut.lower) || std::isfinite(cut.upper);
}

bool append_cuts(Model &model, const std::vector<Cut> &cuts) {
    for (const auto &cut : cuts) {
        if (!valid_cut(model, cut)) return false;
    }
    for (const auto &cut : cuts) {
        model.constraint_lower.push_back(cut.lower);
        model.constraint_upper.push_back(cut.upper);
        for (std::size_t i = 0; i < cut.indices.size(); ++i) {
            model.constraint_matrix.pattern.column_indices.push_back(cut.indices[i]);
            model.constraint_matrix.values.push_back(cut.coefficients[i]);
        }
        model.constraint_matrix.pattern.row_offsets.push_back(
            static_cast<Index>(model.constraint_matrix.pattern.column_indices.size()));
    }
    model.constraint_matrix.pattern.rows += static_cast<Index>(cuts.size());
    return true;
}

bool usable_relaxation(const SolveResult &relaxation, std::size_t variables) {
    if (relaxation.primal_solution.size() != variables ||
        !std::isfinite(relaxation.objective_value)) {
        return false;
    }
    return std::all_of(relaxation.primal_solution.begin(),
                       relaxation.primal_solution.end(),
                       [](Scalar value) { return std::isfinite(value); });
}

SolveResult solve_mip_impl(const Model &model, const RelaxationOracle &oracle,
                           const SolverOptions &options,
                           const CutGenerator *root_cut_generator) {
    SolveResult result;
    result.solver = "vikalp-mip-bnb";

    if (model.nonlinear) {
        result.status = SolveStatus::UnsupportedModel;
        result.message = "MIP branch-and-bound requires a linear or quadratic model";
        return result;
    }
    if (!model.has_integer_variables()) {
        result.status = SolveStatus::UnsupportedModel;
        result.message = "Model has no integer variables; use a continuous solver";
        return result;
    }
    const auto errors = model.validate();
    if (!errors.empty()) {
        result.status = SolveStatus::InvalidModel;
        result.message = errors.front();
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    const Scalar int_tolerance = options.integrality_tolerance;
    Model working_model = model;
    BbQueue queue;
    PseudocostTracker pseudocosts(model.num_variables());
    Scalar incumbent_objective = Model::infinity();
    std::vector<Scalar> incumbent_solution;
    bool root_cuts_applied = root_cut_generator == nullptr;
    bool all_bounds_certified = true;
    SolveStatus termination = SolveStatus::NotSolved;

    BbNode root;
    root.id = queue.next_id();
    queue.push(std::move(root));

    while (!queue.empty()) {
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= options.time_limit_seconds) {
            termination = SolveStatus::TimeLimit;
            break;
        }
        if (queue.nodes_processed >= options.node_limit) {
            termination = SolveStatus::NodeLimit;
            break;
        }

        BbNode node = queue.pop();
        ++queue.nodes_processed;
        const auto lower = effective_lower_bounds(working_model, node);
        const auto upper = effective_upper_bounds(working_model, node);
        bool bounds_infeasible = false;
        for (std::size_t i = 0; i < lower.size(); ++i) {
            if (lower[i] > upper[i] + options.primal_tolerance) {
                bounds_infeasible = true;
                break;
            }
        }
        if (bounds_infeasible) {
            ++queue.nodes_infeasible;
            continue;
        }

        const Scalar parent_bound = node.relaxation_bound;
        const bool has_parent_solution = node.parent_id >= 0 &&
                                          node.relaxation_solution.size() == lower.size();
        const Scalar parent_branch_value =
            has_parent_solution && node.branching_variable >= 0
                ? node.relaxation_solution[static_cast<std::size_t>(node.branching_variable)]
                : 0.0;

        SolveResult relaxation = oracle.solve(
            working_model, lower, upper, node.relaxation_solution, options);
        if (relaxation.status == SolveStatus::Infeasible) {
            ++queue.nodes_infeasible;
            continue;
        }
        if (relaxation.status == SolveStatus::Unbounded ||
            relaxation.status == SolveStatus::UnsupportedModel ||
            relaxation.status == SolveStatus::InvalidModel ||
            relaxation.status == SolveStatus::NumericalFailure ||
            relaxation.status == SolveStatus::NotSolved) {
            termination = relaxation.status;
            result.message = "Relaxation oracle failed at node " + std::to_string(node.id);
            break;
        }
        if (relaxation.status != SolveStatus::Optimal &&
            relaxation.status != SolveStatus::LocallyOptimal &&
            relaxation.status != SolveStatus::Feasible) {
            termination = relaxation.status;
            result.message = "Relaxation oracle did not produce a solved relaxation";
            break;
        }
        if (!usable_relaxation(relaxation, lower.size()) ||
            !is_feasible(working_model, relaxation.primal_solution,
                         options.primal_tolerance, &lower, &upper)) {
            termination = SolveStatus::NumericalFailure;
            result.message = "Relaxation oracle returned an invalid primal solution";
            break;
        }

        if (!root_cuts_applied && node.parent_id < 0) {
            root_cuts_applied = true;
            const auto cuts = root_cut_generator->generate(
                working_model, relaxation.primal_solution);
            if (!append_cuts(working_model, cuts)) {
                termination = SolveStatus::InvalidModel;
                result.message = "Root cut generator returned an invalid cut";
                break;
            }
            if (!cuts.empty()) {
                relaxation = oracle.solve(working_model, lower, upper,
                                          node.relaxation_solution, options);
                if (relaxation.status != SolveStatus::Optimal &&
                    relaxation.status != SolveStatus::LocallyOptimal &&
                    relaxation.status != SolveStatus::Feasible) {
                    termination = relaxation.status;
                    result.message = "Relaxation oracle failed after root cuts";
                    break;
                }
                if (!usable_relaxation(relaxation, lower.size()) ||
                    !is_feasible(working_model, relaxation.primal_solution,
                                 options.primal_tolerance, &lower, &upper)) {
                    termination = SolveStatus::NumericalFailure;
                    result.message = "Relaxation oracle returned an invalid cut relaxation";
                    break;
                }
            }
        }

        const bool bound_certified = relaxation.status == SolveStatus::Optimal ||
                                     std::isfinite(relaxation.dual_bound);
        all_bounds_certified = all_bounds_certified && bound_certified;
        const Scalar node_bound = std::isfinite(relaxation.dual_bound)
            ? relaxation.dual_bound
            : (relaxation.status == SolveStatus::Optimal
                   ? relaxation.objective_value : -Model::infinity());
        if (has_parent_solution && std::isfinite(parent_bound) &&
            node.branching_variable >= 0) {
            const Scalar delta = std::max(0.0, node_bound - parent_bound);
            const Scalar distance = node.branch_direction == BranchDirection::Down
                ? parent_branch_value - std::floor(parent_branch_value)
                : std::ceil(parent_branch_value) - parent_branch_value;
            if (node.branch_direction == BranchDirection::Down) {
                pseudocosts.record_down(node.branching_variable, delta, distance);
            } else {
                pseudocosts.record_up(node.branching_variable, delta, distance);
            }
        }

        if (std::isfinite(incumbent_objective) && std::isfinite(node_bound) &&
            node_bound >= incumbent_objective - options.absolute_gap_tolerance) {
            ++queue.nodes_pruned;
            continue;
        }

        const auto update_incumbent = [&](Scalar objective, std::vector<Scalar> solution) {
            if (std::isfinite(objective) && objective < incumbent_objective) {
                incumbent_objective = objective;
                incumbent_solution = std::move(solution);
                ++queue.incumbent_updates;
            }
        };

        if (is_integer_feasible(working_model, relaxation.primal_solution,
                                int_tolerance)) {
            ++queue.nodes_integral;
            update_incumbent(evaluate_objective(working_model, relaxation.primal_solution),
                             relaxation.primal_solution);
        } else {
            std::vector<Scalar> rounded;
            Scalar rounded_objective = Model::infinity();
            if (try_rounding(working_model, relaxation.primal_solution, lower, upper,
                             rounded, rounded_objective, int_tolerance,
                             options.primal_tolerance)) {
                update_incumbent(rounded_objective, std::move(rounded));
            }

            const Index branch_variable = queue.nodes_processed <= 8
                ? select_most_fractional(working_model, relaxation.primal_solution,
                                         lower, upper, int_tolerance)
                : select_pseudocost(working_model, relaxation.primal_solution,
                                    lower, upper, int_tolerance, pseudocosts);
            if (branch_variable < 0) {
                termination = SolveStatus::NumericalFailure;
                result.message = "Relaxation oracle returned a non-integral solution without a branch";
                break;
            }

            const auto variable = static_cast<std::size_t>(branch_variable);
            const Scalar floor_value = std::floor(relaxation.primal_solution[variable]);
            const Scalar ceil_value = std::ceil(relaxation.primal_solution[variable]);
            if (floor_value >= lower[variable]) {
                BbNode down;
                down.id = queue.next_id();
                down.parent_id = node.id;
                down.depth = node.depth + 1;
                down.lower_overrides = node.lower_overrides;
                down.upper_overrides = node.upper_overrides;
                down.upper_overrides.push_back({branch_variable, floor_value});
                down.relaxation_bound = node_bound;
                down.relaxation_solution = relaxation.primal_solution;
                down.branching_variable = branch_variable;
                down.branch_direction = BranchDirection::Down;
                queue.push(std::move(down));
            }
            if (ceil_value <= upper[variable]) {
                BbNode up;
                up.id = queue.next_id();
                up.parent_id = node.id;
                up.depth = node.depth + 1;
                up.lower_overrides = node.lower_overrides;
                up.upper_overrides = node.upper_overrides;
                up.lower_overrides.push_back({branch_variable, ceil_value});
                up.relaxation_bound = node_bound;
                up.relaxation_solution = relaxation.primal_solution;
                up.branching_variable = branch_variable;
                up.branch_direction = BranchDirection::Up;
                queue.push(std::move(up));
            }
        }

        const Scalar best_open_bound = queue.empty() ? incumbent_objective
                                                       : queue.best_bound();
        if (all_bounds_certified && std::isfinite(incumbent_objective) &&
            std::isfinite(node_bound) &&
            (absolute_gap(incumbent_objective, best_open_bound) <=
                 options.absolute_gap_tolerance ||
             relative_gap(incumbent_objective, best_open_bound) <=
                 options.relative_gap_tolerance)) {
            termination = SolveStatus::Optimal;
            break;
        }
    }

    result.solve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    result.nodes = queue.nodes_processed;
    result.iterations = queue.nodes_processed;
    if (!incumbent_solution.empty()) {
        result.primal_solution = incumbent_solution;
        result.objective_value = incumbent_objective;
        result.primal_bound = incumbent_objective;
        result.integrality_residual = max_integrality_residual(
            working_model, incumbent_solution);
    }

    result.dual_bound = queue.empty()
        ? (std::isfinite(incumbent_objective) ? incumbent_objective : Model::infinity())
        : queue.best_bound();
    result.absolute_gap = absolute_gap(
        std::isfinite(incumbent_objective) ? incumbent_objective : Model::infinity(),
        result.dual_bound);
    result.relative_gap = relative_gap(
        std::isfinite(incumbent_objective) ? incumbent_objective : Model::infinity(),
        result.dual_bound);

    if (termination == SolveStatus::NotSolved) {
        if (queue.empty() && incumbent_solution.empty()) {
            termination = SolveStatus::Infeasible;
        } else if (queue.empty() && all_bounds_certified) {
            termination = SolveStatus::Optimal;
        } else {
            termination = SolveStatus::Feasible;
        }
    }
    result.status = termination;
    if (termination == SolveStatus::Optimal &&
        (!all_bounds_certified ||
         (result.absolute_gap > options.absolute_gap_tolerance &&
          result.relative_gap > options.relative_gap_tolerance))) {
        result.status = incumbent_solution.empty() ? SolveStatus::Infeasible
                                                    : SolveStatus::Feasible;
    }
    return result;
}

} // namespace

SolveResult solve_mip(const Model &model, const RelaxationOracle &oracle,
                      const SolverOptions &options) {
    return solve_mip_impl(model, oracle, options, nullptr);
}

SolveResult solve_mip(const Model &model, const RelaxationOracle &oracle,
                      const SolverOptions &options,
                      const CutGenerator &root_cut_generator) {
    return solve_mip_impl(model, oracle, options, &root_cut_generator);
}

} // namespace vikalp
