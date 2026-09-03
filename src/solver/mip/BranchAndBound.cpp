// Flow D — Branch-and-bound MIP engine implementation.
//
// Processing loop:
//   1. Create root node
//   2. While queue is non-empty and limits not reached:
//      a. Pop best-bound node
//      b. Solve relaxation via oracle
//      c. Check infeasibility → prune
//      d. Check bound pruning → prune
//      e. Check integrality → candidate incumbent
//      f. Try rounding heuristic
//      g. Select branching variable (most-fractional)
//      h. Create two children with tightened bounds
//   3. Compute final gap and return SolveResult

#include "vikalp/solver/mip/BranchAndBound.hpp"
#include "vikalp/solver/mip/BbNode.hpp"
#include "vikalp/solver/mip/BbQueue.hpp"
#include "vikalp/solver/mip/Pseudocosts.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

namespace vikalp {
namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Integrality helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Distance of x from the nearest integer.
inline Scalar integrality_violation(Scalar x) {
    return std::abs(x - std::round(x));
}

/// True if all integer/binary variables in the solution are integral.
bool is_integer_feasible(const Model &model,
                         std::span<const Scalar> x,
                         Scalar tol) {
    const auto n = static_cast<std::size_t>(model.num_variables());
    if (x.size() != n) return false;
    for (std::size_t i = 0; i < n; ++i) {
        if (model.variable_types[i] == VariableType::Continuous) continue;
        if (integrality_violation(x[i]) > tol) return false;
    }
    return true;
}

/// Maximum integrality violation across integer/binary variables.
Scalar max_integrality_residual(const Model &model,
                                std::span<const Scalar> x) {
    Scalar worst = 0.0;
    const auto n = static_cast<std::size_t>(model.num_variables());
    for (std::size_t i = 0; i < n && i < x.size(); ++i) {
        if (model.variable_types[i] == VariableType::Continuous) continue;
        worst = std::max(worst, integrality_violation(x[i]));
    }
    return worst;
}

// ─────────────────────────────────────────────────────────────────────────────
// Branching
// ─────────────────────────────────────────────────────────────────────────────

/// Select the most-fractional integer variable.
/// Returns -1 if no fractional integer variable exists.
Index select_most_fractional(const Model &model,
                             std::span<const Scalar> x,
                             const std::vector<Scalar> &eff_lo,
                             const std::vector<Scalar> &eff_hi,
                             Scalar int_tol) {
    Index best = -1;
    Scalar best_frac = -1.0;

    const auto n = static_cast<std::size_t>(model.num_variables());
    for (std::size_t i = 0; i < n && i < x.size(); ++i) {
        if (model.variable_types[i] == VariableType::Continuous) continue;

        const Scalar val = x[i];
        const Scalar viol = integrality_violation(val);
        if (viol <= int_tol) continue;

        const Scalar floor_val = std::floor(val);
        const Scalar ceil_val = std::ceil(val);
        if (floor_val < eff_lo[i] && ceil_val > eff_hi[i]) continue;

        const Scalar frac_part = val - floor_val;
        const Scalar closeness_to_half = 0.5 - std::abs(frac_part - 0.5);

        if (closeness_to_half > best_frac ||
            (closeness_to_half == best_frac && static_cast<Index>(i) < best)) {
            best_frac = closeness_to_half;
            best = static_cast<Index>(i);
        }
    }
    return best;
}

/// Select branching variable using pseudocost scores.
/// Falls back to most-fractional if no variable has reliable pseudocosts.
Index select_pseudocost(const Model &model,
                        std::span<const Scalar> x,
                        const std::vector<Scalar> &eff_lo,
                        const std::vector<Scalar> &eff_hi,
                        Scalar int_tol,
                        const PseudocostTracker &psc) {
    Index best = -1;
    Scalar best_score = -1.0;

    const auto n = static_cast<std::size_t>(model.num_variables());
    for (std::size_t i = 0; i < n && i < x.size(); ++i) {
        if (model.variable_types[i] == VariableType::Continuous) continue;

        const Scalar val = x[i];
        const Scalar viol = integrality_violation(val);
        if (viol <= int_tol) continue;

        const Scalar floor_val = std::floor(val);
        const Scalar ceil_val = std::ceil(val);
        if (floor_val < eff_lo[i] && ceil_val > eff_hi[i]) continue;

        const Scalar frac_down = val - floor_val;
        const Scalar frac_up = ceil_val - val;
        const Scalar sc = psc.score(static_cast<Index>(i), frac_down, frac_up);

        if (sc > best_score ||
            (sc == best_score && static_cast<Index>(i) < best)) {
            best_score = sc;
            best = static_cast<Index>(i);
        }
    }
    return best;
}

constexpr int PSEUDOCOST_WARMUP_NODES = 8;  // use most-fractional for first N nodes

// ─────────────────────────────────────────────────────────────────────────────
// Rounding heuristic
// ─────────────────────────────────────────────────────────────────────────────

/// Try to round a relaxation solution to integer feasibility.
/// Returns true if rounding produced a valid candidate; fills rounded_obj.
bool try_rounding(const Model &model,
                  std::span<const Scalar> x,
                  const std::vector<Scalar> &eff_lo,
                  const std::vector<Scalar> &eff_hi,
                  std::vector<Scalar> &rounded,
                  Scalar &rounded_obj,
                  Scalar int_tol,
                  Scalar primal_tol) {
    const auto n = static_cast<std::size_t>(model.num_variables());
    rounded.resize(n);

    // 1. Copy and round integer variables
    for (std::size_t i = 0; i < n; ++i) {
        Scalar val = x[i];
        if (model.variable_types[i] != VariableType::Continuous) {
            val = std::round(val);
        }
        // 2. Clamp to effective bounds
        val = std::max(val, eff_lo[i]);
        val = std::min(val, eff_hi[i]);
        rounded[i] = val;
    }

    // 3. Verify integrality
    if (!is_integer_feasible(model, std::span<const Scalar>(rounded), int_tol))
        return false;

    // 4. Verify constraint feasibility: Ax must be within [constraint_lower, constraint_upper]
    const auto &A = model.constraint_matrix;
    const auto m = static_cast<std::size_t>(model.num_constraints());
    for (std::size_t row = 0; row < m; ++row) {
        Scalar ax = 0.0;
        for (Index k = A.pattern.row_offsets[row];
             k < A.pattern.row_offsets[row + 1]; ++k) {
            const auto col = static_cast<std::size_t>(
                A.pattern.column_indices[static_cast<std::size_t>(k)]);
            ax += A.values[static_cast<std::size_t>(k)] * rounded[col];
        }
        if (ax < model.constraint_lower[row] - primal_tol ||
            ax > model.constraint_upper[row] + primal_tol) {
            return false;  // violated
        }
    }

    // 5. Compute objective: c'x + offset (ignore quadratic for simplicity)
    rounded_obj = model.objective_offset;
    for (std::size_t i = 0; i < n; ++i) {
        rounded_obj += model.linear_objective[i] * rounded[i];
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Gap computation
// ─────────────────────────────────────────────────────────────────────────────

Scalar compute_absolute_gap(Scalar primal, Scalar dual) {
    if (!std::isfinite(primal) || !std::isfinite(dual))
        return Model::infinity();
    return std::max(0.0, primal - dual);
}

Scalar compute_relative_gap(Scalar primal, Scalar dual) {
    const Scalar ag = compute_absolute_gap(primal, dual);
    if (!std::isfinite(ag)) return Model::infinity();
    return ag / std::max(1.0, std::abs(primal));
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Main solver
// ─────────────────────────────────────────────────────────────────────────────

SolveResult solve_mip(const Model &model,
                      const RelaxationOracle &oracle,
                      const SolverOptions &options) {
    SolveResult result;
    result.solver = "vikalp-mip-bnb";

    // ── Validate ─────────────────────────────────────────────────────────────
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

    // ── Setup ────────────────────────────────────────────────────────────────
    const auto start_time = std::chrono::steady_clock::now();
    const Scalar int_tol = options.integrality_tolerance;
    const Scalar abs_gap_tol = options.absolute_gap_tolerance;
    const Scalar rel_gap_tol = options.relative_gap_tolerance;

    Scalar incumbent_obj = Model::infinity();
    std::vector<Scalar> incumbent_solution;
    Scalar global_dual_bound = -Model::infinity();

    BbQueue queue;
    PseudocostTracker pseudocosts(model.num_variables());

    // ── Root node ────────────────────────────────────────────────────────────
    BbNode root;
    root.id = queue.next_id();
    root.parent_id = -1;
    root.depth = 0;
    root.status = NodeStatus::Open;
    queue.push(std::move(root));

    // ── Main loop ────────────────────────────────────────────────────────────
    SolveStatus termination = SolveStatus::Optimal;

    while (!queue.empty()) {
        // Time check
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        double elapsed_sec = std::chrono::duration<double>(elapsed).count();
        if (elapsed_sec >= options.time_limit_seconds) {
            termination = SolveStatus::TimeLimit;
            break;
        }

        // Node limit check
        if (queue.nodes_processed >= options.node_limit) {
            termination = SolveStatus::NodeLimit;
            break;
        }

        // Pop best-bound node
        BbNode node = queue.pop();
        queue.nodes_processed++;

        // Build effective bounds
        auto eff_lo = effective_lower_bounds(model, node);
        auto eff_hi = effective_upper_bounds(model, node);

        // Check for infeasible bounds (lo > hi)
        bool bounds_infeasible = false;
        for (std::size_t i = 0; i < eff_lo.size(); ++i) {
            if (eff_lo[i] > eff_hi[i] + options.primal_tolerance) {
                bounds_infeasible = true;
                break;
            }
        }
        if (bounds_infeasible) {
            node.status = NodeStatus::Infeasible;
            queue.nodes_infeasible++;
            continue;
        }

        // ── Solve relaxation ─────────────────────────────────────────────────
        SolveResult rel = oracle.solve(
            model,
            std::span<const Scalar>(eff_lo),
            std::span<const Scalar>(eff_hi),
            std::span<const Scalar>(node.relaxation_solution),  // warm start
            options);

        // ── Infeasibility pruning ────────────────────────────────────────────
        if (rel.status == SolveStatus::Infeasible ||
            rel.status == SolveStatus::InvalidModel) {
            node.status = NodeStatus::Infeasible;
            queue.nodes_infeasible++;
            continue;
        }

        // ── Numerical failure → skip ─────────────────────────────────────────
        if (rel.status == SolveStatus::NumericalFailure ||
            rel.status == SolveStatus::NotSolved) {
            node.status = NodeStatus::Pruned;
            queue.nodes_pruned++;
            continue;
        }

        // Use the relaxation objective as the node bound
        node.relaxation_bound = rel.objective_value;
        node.relaxation_solution = rel.primal_solution;

        // ── Bound pruning ────────────────────────────────────────────────────
        if (node.relaxation_bound >= incumbent_obj - abs_gap_tol) {
            node.status = NodeStatus::Pruned;
            queue.nodes_pruned++;
            continue;
        }

        // ── Integrality check ────────────────────────────────────────────────
        if (is_integer_feasible(model,
                                std::span<const Scalar>(rel.primal_solution),
                                int_tol)) {
            node.status = NodeStatus::Integral;
            queue.nodes_integral++;
            if (rel.objective_value < incumbent_obj) {
                incumbent_obj = rel.objective_value;
                incumbent_solution = rel.primal_solution;
                queue.incumbent_updates++;
            }

            // Check gap termination after incumbent update
            global_dual_bound = queue.empty() ?
                incumbent_obj : std::max(global_dual_bound, queue.best_bound());
            const Scalar ag = compute_absolute_gap(incumbent_obj, global_dual_bound);
            const Scalar rg = compute_relative_gap(incumbent_obj, global_dual_bound);
            if (ag <= abs_gap_tol || rg <= rel_gap_tol)
                break;

            continue;
        }

        // ── Rounding heuristic ───────────────────────────────────────────────
        {
            std::vector<Scalar> rounded;
            Scalar rounded_obj = 0.0;
            if (try_rounding(model, std::span<const Scalar>(rel.primal_solution),
                             eff_lo, eff_hi, rounded, rounded_obj,
                             int_tol, options.primal_tolerance)) {
                if (rounded_obj < incumbent_obj) {
                    incumbent_obj = rounded_obj;
                    incumbent_solution = std::move(rounded);
                    queue.incumbent_updates++;
                }
            }
        }

        // ── Record pseudocost observation from parent branching ───────────
        if (node.branching_variable >= 0 && node.parent_id >= 0) {
            const Scalar parent_bound = node.relaxation_bound;
            const Scalar delta = std::max(0.0, rel.objective_value - parent_bound);
            const auto bv = static_cast<std::size_t>(node.branching_variable);
            if (bv < node.relaxation_solution.size()) {
                const Scalar parent_val = node.relaxation_solution[bv];
                if (node.branch_direction == BranchDirection::Down) {
                    const Scalar frac = parent_val - std::floor(parent_val);
                    pseudocosts.record_down(node.branching_variable, delta, frac);
                } else {
                    const Scalar frac = std::ceil(parent_val) - parent_val;
                    pseudocosts.record_up(node.branching_variable, delta, frac);
                }
            }
        }

        // ── Branching ────────────────────────────────────────────────────────
        Index branch_var;
        if (queue.nodes_processed <= PSEUDOCOST_WARMUP_NODES) {
            branch_var = select_most_fractional(
                model, std::span<const Scalar>(rel.primal_solution),
                eff_lo, eff_hi, int_tol);
        } else {
            branch_var = select_pseudocost(
                model, std::span<const Scalar>(rel.primal_solution),
                eff_lo, eff_hi, int_tol, pseudocosts);
        }

        if (branch_var < 0) {
            // No fractional variable found but not integer-feasible?
            // This shouldn't happen, but treat as pruned.
            node.status = NodeStatus::Pruned;
            queue.nodes_pruned++;
            continue;
        }

        node.status = NodeStatus::Branched;
        const Scalar val = rel.primal_solution[static_cast<std::size_t>(branch_var)];
        const Scalar floor_val = std::floor(val);
        const Scalar ceil_val = std::ceil(val);

        // Down child: x_j <= floor(val)
        if (floor_val >= eff_lo[static_cast<std::size_t>(branch_var)]) {
            BbNode down;
            down.id = queue.next_id();
            down.parent_id = node.id;
            down.depth = node.depth + 1;
            down.lower_overrides = node.lower_overrides;
            down.upper_overrides = node.upper_overrides;
            down.upper_overrides.push_back({branch_var, floor_val});
            down.relaxation_bound = node.relaxation_bound;
            down.relaxation_solution = rel.primal_solution;  // warm start
            down.branching_variable = branch_var;
            down.branch_direction = BranchDirection::Down;
            down.status = NodeStatus::Open;
            queue.push(std::move(down));
        }

        // Up child: x_j >= ceil(val)
        if (ceil_val <= eff_hi[static_cast<std::size_t>(branch_var)]) {
            BbNode up;
            up.id = queue.next_id();
            up.parent_id = node.id;
            up.depth = node.depth + 1;
            up.lower_overrides = node.lower_overrides;
            up.upper_overrides = node.upper_overrides;
            up.lower_overrides.push_back({branch_var, ceil_val});
            up.relaxation_bound = node.relaxation_bound;
            up.relaxation_solution = rel.primal_solution;  // warm start
            up.branching_variable = branch_var;
            up.branch_direction = BranchDirection::Up;
            up.status = NodeStatus::Open;
            queue.push(std::move(up));
        }

        // Update global dual bound
        if (!queue.empty()) {
            global_dual_bound = queue.best_bound();
        }

        // Check gap termination
        if (std::isfinite(incumbent_obj)) {
            const Scalar ag = compute_absolute_gap(incumbent_obj, global_dual_bound);
            const Scalar rg = compute_relative_gap(incumbent_obj, global_dual_bound);
            if (ag <= abs_gap_tol || rg <= rel_gap_tol)
                break;
        }
    }

    // ── Build result ─────────────────────────────────────────────────────────
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    result.solve_seconds = std::chrono::duration<double>(elapsed).count();
    result.nodes = queue.nodes_processed;
    result.iterations = queue.nodes_processed;

    if (!incumbent_solution.empty()) {
        result.primal_solution = incumbent_solution;
        result.objective_value = incumbent_obj;
        result.primal_bound = incumbent_obj;
        result.integrality_residual = max_integrality_residual(
            model, std::span<const Scalar>(incumbent_solution));
    }

    // Global dual bound: best open node, or incumbent if queue exhausted
    if (queue.empty() && std::isfinite(incumbent_obj)) {
        global_dual_bound = incumbent_obj;
    } else if (!queue.empty()) {
        global_dual_bound = queue.best_bound();
    }
    result.dual_bound = global_dual_bound;

    result.absolute_gap = compute_absolute_gap(
        std::isfinite(incumbent_obj) ? incumbent_obj : Model::infinity(),
        global_dual_bound);
    result.relative_gap = compute_relative_gap(
        std::isfinite(incumbent_obj) ? incumbent_obj : Model::infinity(),
        global_dual_bound);

    // Set status
    if (termination == SolveStatus::TimeLimit) {
        result.status = std::isfinite(incumbent_obj) ?
            SolveStatus::TimeLimit : SolveStatus::TimeLimit;
    } else if (termination == SolveStatus::NodeLimit) {
        result.status = SolveStatus::NodeLimit;
    } else if (std::isfinite(incumbent_obj) &&
               (result.absolute_gap <= abs_gap_tol ||
                result.relative_gap <= rel_gap_tol)) {
        result.status = SolveStatus::Optimal;
    } else if (std::isfinite(incumbent_obj)) {
        result.status = SolveStatus::Feasible;
    } else {
        // Queue exhausted, no incumbent found
        result.status = SolveStatus::Infeasible;
    }

    return result;
}

} // namespace vikalp
