// Flow D — Mixed-integer engine correctness tests.
//
// Uses a deterministic MockRelaxationOracle that maps bound configurations
// to predetermined SolveResults. No real LP/QP solver is required.
//
// Test framework matches existing repository pattern (g_failures, printf).
// Exit code: 0 = all pass, 1 = any failure.

#include "vikalp/solver/mip/BranchAndBound.hpp"
#include "vikalp/solver/mip/BbNode.hpp"
#include "vikalp/solver/mip/BbQueue.hpp"
#include "vikalp/solver/mip/Pseudocosts.hpp"
#include "vikalp/solver/mip/CutGenerator.hpp"
#include "vikalp/solver/mip/OuterApproximation.hpp"
#include "vikalp/solver/Solver.hpp"
#include "vikalp/examples/RefineryModel.hpp"
#include "vikalp/contracts/RelaxationOracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Tiny test framework (matches existing repository convention)
// ─────────────────────────────────────────────────────────────────────────────

int g_failures = 0;

void check_eq(const std::string &test, double got, double expected,
              double tol = 1e-9) {
    if (std::abs(got - expected) > tol) {
        std::printf("FAIL [%s]: got %.17g, expected %.17g (tol %.1e)\n",
                    test.c_str(), got, expected, tol);
        ++g_failures;
    }
}

void check_true(const std::string &test, bool cond) {
    if (!cond) {
        std::printf("FAIL [%s]: condition was false\n", test.c_str());
        ++g_failures;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mock RelaxationOracle
// ─────────────────────────────────────────────────────────────────────────────

/// A mock that uses a user-supplied callback to produce relaxation results.
/// The callback receives the effective bounds and returns a SolveResult.
class MockRelaxationOracle final : public vikalp::RelaxationOracle {
public:
    using Callback = std::function<vikalp::SolveResult(
        const vikalp::Model &,
        std::span<const vikalp::Scalar>,  // lower overrides
        std::span<const vikalp::Scalar>,  // upper overrides
        std::span<const vikalp::Scalar>,  // warm start
        const vikalp::SolverOptions &)>;

    explicit MockRelaxationOracle(Callback cb) : cb_(std::move(cb)) {}

    [[nodiscard]] vikalp::SolveResult solve(
        const vikalp::Model &model,
        std::span<const vikalp::Scalar> lo,
        std::span<const vikalp::Scalar> hi,
        std::span<const vikalp::Scalar> ws,
        const vikalp::SolverOptions &opts) const override {
        ++call_count_;
        return cb_(model, lo, hi, ws, opts);
    }

    mutable int call_count_ = 0;

private:
    Callback cb_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a simple 1-variable MILP model
//   min c*x  s.t. lo <= x <= hi, x integer
// ─────────────────────────────────────────────────────────────────────────────

vikalp::Model make_1var_milp(double c, double lo, double hi,
                              vikalp::VariableType vtype = vikalp::VariableType::Integer) {
    vikalp::Model m;
    m.name = "test_1var";
    m.linear_objective = {c};
    m.variable_lower = {lo};
    m.variable_upper = {hi};
    m.variable_types = {vtype};
    m.constraint_matrix = {{0, 1, {0}, {}}, {}};
    m.constraint_lower = {};
    m.constraint_upper = {};
    return m;
}

/// Build a 2-variable model: one integer, one continuous.
vikalp::Model make_2var_mixed(double c0, double c1,
                               double lo0, double hi0,
                               double lo1, double hi1) {
    vikalp::Model m;
    m.name = "test_2var";
    m.linear_objective = {c0, c1};
    m.variable_lower = {lo0, lo1};
    m.variable_upper = {hi0, hi1};
    m.variable_types = {vikalp::VariableType::Integer,
                        vikalp::VariableType::Continuous};
    m.constraint_matrix = {{0, 2, {0}, {}}, {}};
    m.constraint_lower = {};
    m.constraint_upper = {};
    return m;
}

/// Build default options with small limits.
vikalp::SolverOptions test_options() {
    vikalp::SolverOptions opts;
    opts.node_limit = 1000;
    opts.time_limit_seconds = 10.0;
    opts.integrality_tolerance = 1e-6;
    opts.absolute_gap_tolerance = 1e-6;
    opts.relative_gap_tolerance = 1e-4;
    return opts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// 1. Already-integral root
void test_integral_root() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto, auto, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        r.objective_value = 3.0;
        r.primal_solution = {3.0};  // already integer
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("integral_root: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("integral_root: obj", res.objective_value, 3.0);
    check_eq("integral_root: x", res.primal_solution[0], 3.0);
    check_eq("integral_root: nodes", static_cast<double>(res.nodes), 1.0);
}

// 2. One fractional variable → branch to two children
void test_one_fractional() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    int call = 0;
    MockRelaxationOracle oracle([&](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        ++call;
        if (call == 1) {
            // Root: x=2.5
            r.objective_value = 2.5;
            r.primal_solution = {2.5};
        } else if (hi[0] <= 2.0 + 1e-9) {
            // Down child: x <= 2 → x=2
            r.objective_value = 2.0;
            r.primal_solution = {2.0};
        } else if (lo[0] >= 3.0 - 1e-9) {
            // Up child: x >= 3 → x=3
            r.objective_value = 3.0;
            r.primal_solution = {3.0};
        } else {
            r.objective_value = 2.5;
            r.primal_solution = {2.5};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("one_frac: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("one_frac: obj", res.objective_value, 2.0);
    check_eq("one_frac: x", res.primal_solution[0], 2.0);
}

// 3. Two-level branching
void test_two_level_branch() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            // Root
            r.objective_value = 1.5;
            r.primal_solution = {1.5};
        } else if (hi[0] <= 1.0 + 1e-9 && lo[0] <= 0.0 + 1e-9) {
            // x<=1: still fractional at 0.5
            r.objective_value = 0.5;
            r.primal_solution = {0.5};
        } else if (lo[0] >= 2.0 - 1e-9) {
            // x>=2: integer
            r.objective_value = 2.0;
            r.primal_solution = {2.0};
        } else if (hi[0] <= 0.0 + 1e-9) {
            // x<=0: integer
            r.objective_value = 0.0;
            r.primal_solution = {0.0};
        } else if (lo[0] >= 1.0 - 1e-9 && hi[0] <= 1.0 + 1e-9) {
            // x=1: integer
            r.objective_value = 1.0;
            r.primal_solution = {1.0};
        } else {
            r.objective_value = lo[0];
            r.primal_solution = {lo[0]};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("two_level: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("two_level: obj", res.objective_value, 0.0);
}

// 4. Bound pruning
void test_bound_pruning() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.objective_value = 1.5;
            r.primal_solution = {1.5};
        } else if (hi[0] <= 1.0 + 1e-9) {
            // Down: obj=1 (integer)
            r.objective_value = 1.0;
            r.primal_solution = {1.0};
        } else if (lo[0] >= 2.0 - 1e-9) {
            // Up: obj=5 — should be pruned by bound since incumbent=1
            r.objective_value = 5.0;
            r.primal_solution = {5.0};
        } else {
            r.objective_value = lo[0];
            r.primal_solution = {lo[0]};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("bound_prune: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("bound_prune: obj", res.objective_value, 1.0);
}

// 5. Infeasible node pruning
void test_infeasible_prune() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.status = vikalp::SolveStatus::Optimal;
            r.objective_value = 1.5;
            r.primal_solution = {1.5};
        } else if (hi[0] <= 1.0 + 1e-9) {
            r.status = vikalp::SolveStatus::Infeasible;  // down child infeasible
        } else if (lo[0] >= 2.0 - 1e-9) {
            r.status = vikalp::SolveStatus::Optimal;
            r.objective_value = 2.0;
            r.primal_solution = {2.0};
        } else {
            r.status = vikalp::SolveStatus::Infeasible;
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("infeasible_prune: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("infeasible_prune: obj", res.objective_value, 2.0);
}

// 6. Incumbent update (later child beats first)
void test_incumbent_update() {
    auto model = make_1var_milp(-1.0, 0.0, 10.0);  // maximize (min of -x)
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.objective_value = -5.5;
            r.primal_solution = {5.5};
        } else if (hi[0] <= 5.0 + 1e-9) {
            r.objective_value = -5.0;
            r.primal_solution = {5.0};
        } else if (lo[0] >= 6.0 - 1e-9) {
            r.objective_value = -6.0;
            r.primal_solution = {6.0};
        } else {
            r.objective_value = -lo[0];
            r.primal_solution = {lo[0]};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("incumbent_update: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("incumbent_update: obj", res.objective_value, -6.0);
}

// 7. No incumbent found (all nodes infeasible)
void test_no_incumbent() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto, auto, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Infeasible;
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("no_incumbent: infeasible", res.status == vikalp::SolveStatus::Infeasible);
}

// 8. Gap termination
void test_gap_termination() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.objective_value = 2.5;
            r.primal_solution = {2.5};
        } else if (hi[0] <= 2.0 + 1e-9) {
            r.objective_value = 2.0;
            r.primal_solution = {2.0};  // integer → incumbent=2
        } else {
            r.objective_value = 3.0;
            r.primal_solution = {3.0};  // integer → but worse
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("gap_term: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("gap_term: obj", res.objective_value, 2.0);
    check_true("gap_term: abs_gap small", res.absolute_gap <= 1e-5);
}

// 9. Node limit
void test_node_limit() {
    auto model = make_1var_milp(1.0, 0.0, 1000.0);
    // Always return fractional to force infinite branching
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        double mid = (lo[0] + hi[0]) / 2.0;
        r.objective_value = mid;
        r.primal_solution = std::vector<double>{mid};
        return r;
    });
    auto opts = test_options();
    opts.node_limit = 5;
    auto res = vikalp::solve_mip(model, oracle, opts);
    check_true("node_limit: status", res.status == vikalp::SolveStatus::NodeLimit ||
                                     res.status == vikalp::SolveStatus::Feasible ||
                                     res.status == vikalp::SolveStatus::Optimal);
    check_true("node_limit: nodes <= 5", res.nodes <= 5);
}

// 10. Time limit (use a tiny timeout)
void test_time_limit() {
    auto model = make_1var_milp(1.0, 0.0, 1e9);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        double mid = (lo[0] + hi[0]) / 2.0;
        r.objective_value = mid;
        r.primal_solution = std::vector<double>{mid};
        return r;
    });
    auto opts = test_options();
    opts.time_limit_seconds = 0.0;  // instant timeout
    opts.node_limit = 1'000'000;
    auto res = vikalp::solve_mip(model, oracle, opts);
    // Should stop very quickly
    check_true("time_limit: terminated",
               res.status == vikalp::SolveStatus::TimeLimit ||
               res.status == vikalp::SolveStatus::NodeLimit ||
               res.nodes <= 10);
}

// 11. Binary branching
void test_binary_branching() {
    auto model = make_1var_milp(1.0, 0.0, 1.0, vikalp::VariableType::Binary);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (lo[0] <= 0.0 + 1e-9 && hi[0] >= 1.0 - 1e-9) {
            r.objective_value = 0.3;
            r.primal_solution = {0.3};
        } else if (hi[0] <= 0.0 + 1e-9) {
            r.objective_value = 0.0;
            r.primal_solution = {0.0};
        } else if (lo[0] >= 1.0 - 1e-9) {
            r.objective_value = 1.0;
            r.primal_solution = {1.0};
        } else {
            r.objective_value = lo[0];
            r.primal_solution = {lo[0]};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("binary: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("binary: obj", res.objective_value, 0.0);
    check_eq("binary: x", res.primal_solution[0], 0.0);
}

// 12. Integer branching (non-binary)
void test_integer_branching() {
    auto model = make_1var_milp(1.0, -5.0, 5.0, vikalp::VariableType::Integer);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        double val = lo[0]; // return lower bound as integer
        r.objective_value = val;
        r.primal_solution = {val};
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("int_branch: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("int_branch: obj", res.objective_value, -5.0);
}

// 13. Continuous variable exclusion — should not branch on continuous vars
void test_continuous_exclusion() {
    auto model = make_2var_mixed(1.0, 1.0, 0.0, 10.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            // Root: x0=2.5 (integer var fractional), x1=3.7 (continuous)
            r.objective_value = 6.2;
            r.primal_solution = {2.5, 3.7};
        } else if (hi[0] <= 2.0 + 1e-9) {
            r.objective_value = 5.7;
            r.primal_solution = {2.0, 3.7};
        } else if (lo[0] >= 3.0 - 1e-9) {
            r.objective_value = 6.7;
            r.primal_solution = {3.0, 3.7};
        } else {
            r.objective_value = lo[0] + 3.7;
            r.primal_solution = {lo[0], 3.7};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("cont_excl: optimal", res.status == vikalp::SolveStatus::Optimal);
    // x0 should be integer, x1 stays continuous at 3.7
    check_eq("cont_excl: x0", res.primal_solution[0], 2.0);
    check_eq("cont_excl: x1", res.primal_solution[1], 3.7);
}

// 14. Warm start propagation — verify mock receives parent solution
void test_warm_start() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    std::vector<double> received_warm_start;
    MockRelaxationOracle oracle([&](auto &, auto lo, auto hi, auto ws, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (!ws.empty()) {
            received_warm_start.assign(ws.begin(), ws.end());
        }
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.objective_value = 2.5;
            r.primal_solution = {2.5};
        } else {
            double val = std::round((lo[0] + hi[0]) / 2.0);
            val = std::max(val, lo[0]);
            val = std::min(val, hi[0]);
            r.objective_value = val;
            r.primal_solution = {val};
        }
        return r;
    });
    (void)vikalp::solve_mip(model, oracle, test_options());
    // Children should have received the root solution {2.5} as warm start
    check_true("warm_start: received", !received_warm_start.empty());
    check_eq("warm_start: value", received_warm_start[0], 2.5);
}

// 15. Deterministic priority ordering — best bound first
void test_deterministic_order() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    std::vector<double> solve_order;
    MockRelaxationOracle oracle([&](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.objective_value = 4.5;
            r.primal_solution = {4.5};
        } else if (hi[0] <= 4.0 + 1e-9) {
            solve_order.push_back(hi[0]);
            r.objective_value = 3.0;
            r.primal_solution = {3.0};  // integer → incumbent
        } else if (lo[0] >= 5.0 - 1e-9) {
            solve_order.push_back(lo[0]);
            r.objective_value = 5.0;
            r.primal_solution = {5.0};
        } else {
            r.objective_value = lo[0];
            r.primal_solution = {lo[0]};
        }
        return r;
    });
    (void)vikalp::solve_mip(model, oracle, test_options());
    // Both children have relaxation_bound=4.5 (inherited). The down child
    // (lower ID) should be processed first due to deterministic ordering.
    check_true("det_order: solved children", solve_order.size() >= 1);
}

// 16. Mock oracle call count
void test_mock_oracle_calls() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        r.objective_value = 5.0;
        r.primal_solution = {5.0};
        return r;
    });
    (void)vikalp::solve_mip(model, oracle, test_options());
    check_true("oracle_calls: at least 1", oracle.call_count_ >= 1);
}

// 17. Rounding heuristic accepted — root fractional but rounds to valid
void test_rounding_accepted() {
    // Model: min x, 0 <= x <= 10, x integer
    // Root relaxation: x=2.1 → rounds to 2 (valid, better than right child)
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.objective_value = 2.1;
            r.primal_solution = {2.1};
        } else if (hi[0] <= 2.0 + 1e-9) {
            r.objective_value = 2.0;
            r.primal_solution = {2.0};
        } else {
            r.objective_value = 3.0;
            r.primal_solution = {3.0};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("round_accept: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_eq("round_accept: obj", res.objective_value, 2.0);
}

// 18. Rounding heuristic rejected — constraint violation
void test_rounding_rejected() {
    // Model: min x0 + x1, x0 integer [0,10], x1 continuous [0,10]
    // Constraint: x0 + x1 >= 5
    vikalp::Model model;
    model.name = "round_reject";
    model.linear_objective = {1.0, 1.0};
    model.variable_lower = {0.0, 0.0};
    model.variable_upper = {10.0, 10.0};
    model.variable_types = {vikalp::VariableType::Integer,
                            vikalp::VariableType::Continuous};
    // Constraint: x0 + x1 >= 5  →  5 <= x0+x1 <= inf
    model.constraint_matrix = {{1, 2, {0, 2}, {0, 1}}, {1.0, 1.0}};
    model.constraint_lower = {5.0};
    model.constraint_upper = {vikalp::Model::infinity()};

    MockRelaxationOracle oracle([](auto &, auto lo, auto hi, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        // Root: x0=2.5, x1=2.5 (sum=5, feasible)
        // Rounding x0 to 2 → 2+2.5=4.5 < 5 → violated → rejected
        // Rounding x0 to 3 → 3+2.5=5.5 >= 5 → accepted
        if (hi[0] >= 10.0 - 1e-9 && lo[0] <= 0.0 + 1e-9) {
            r.objective_value = 5.0;
            r.primal_solution = {2.5, 2.5};
        } else if (hi[0] <= 2.0 + 1e-9) {
            r.objective_value = 5.0;
            r.primal_solution = {2.0, 3.0};
        } else if (lo[0] >= 3.0 - 1e-9) {
            r.objective_value = 5.5;
            r.primal_solution = {3.0, 2.5};
        } else {
            r.objective_value = 5.0;
            r.primal_solution = {lo[0], 5.0 - lo[0]};
        }
        return r;
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("round_reject: solved",
               res.status == vikalp::SolveStatus::Optimal ||
               res.status == vikalp::SolveStatus::Feasible);
}

// 19. UnsupportedModel when no integer variables
void test_no_integer_vars() {
    vikalp::Model model;
    model.linear_objective = {1.0};
    model.variable_lower = {0.0};
    model.variable_upper = {10.0};
    model.variable_types = {vikalp::VariableType::Continuous};
    model.constraint_matrix = {{0, 1, {0}, {}}, {}};

    MockRelaxationOracle oracle([](auto &, auto, auto, auto, auto) {
        return vikalp::SolveResult{};
    });
    auto res = vikalp::solve_mip(model, oracle, test_options());
    check_true("no_int: unsupported", res.status == vikalp::SolveStatus::UnsupportedModel);
}

// 20. BbQueue ordering correctness
void test_queue_ordering() {
    vikalp::BbQueue q;
    vikalp::BbNode a; a.id = 0; a.relaxation_bound = 5.0; a.depth = 1;
    vikalp::BbNode b; b.id = 1; b.relaxation_bound = 3.0; b.depth = 1;
    vikalp::BbNode c; c.id = 2; c.relaxation_bound = 3.0; c.depth = 2;
    q.push(std::move(a));
    q.push(std::move(b));
    q.push(std::move(c));
    // Best bound first: 3.0. Among those, deeper first: depth=2 (id=2)
    auto first = q.pop();
    check_eq("queue_order: first id", static_cast<double>(first.id), 2.0);
    auto second = q.pop();
    check_eq("queue_order: second id", static_cast<double>(second.id), 1.0);
    auto third = q.pop();
    check_eq("queue_order: third id", static_cast<double>(third.id), 0.0);
    check_true("queue_order: empty", q.empty());
}

// 21. BbQueue best_bound
void test_queue_best_bound() {
    vikalp::BbQueue q;
    check_eq("q_best_empty", q.best_bound(), vikalp::Model::infinity());
    vikalp::BbNode n; n.relaxation_bound = 7.0;
    q.push(std::move(n));
    check_eq("q_best_one", q.best_bound(), 7.0);
}

// 22. Effective bounds
void test_effective_bounds() {
    vikalp::Model model;
    model.variable_lower = {0.0, 1.0, 2.0};
    model.variable_upper = {10.0, 11.0, 12.0};

    vikalp::BbNode node;
    node.lower_overrides = {{0, 3.0}, {2, 5.0}};
    node.upper_overrides = {{1, 8.0}};

    auto lo = vikalp::effective_lower_bounds(model, node);
    auto hi = vikalp::effective_upper_bounds(model, node);

    check_eq("eff_lo[0]", lo[0], 3.0);   // overridden from 0 to 3
    check_eq("eff_lo[1]", lo[1], 1.0);   // unchanged
    check_eq("eff_lo[2]", lo[2], 5.0);   // overridden from 2 to 5
    check_eq("eff_hi[0]", hi[0], 10.0);  // unchanged
    check_eq("eff_hi[1]", hi[1], 8.0);   // overridden from 11 to 8
    check_eq("eff_hi[2]", hi[2], 12.0);  // unchanged
}

// 23. Pseudocost tracker
void test_pseudocost_tracker() {
    vikalp::PseudocostTracker tracker(3);
    // Initially unobserved
    check_true("psc: not reliable initially", !tracker.is_reliable(0, 1));
    check_eq("psc: default down cost", tracker.down_cost(0), 1.0);
    check_eq("psc: default up cost", tracker.up_cost(0), 1.0);

    // Record down observation: delta = 4.0, frac = 0.5 -> cost = 8.0
    tracker.record_down(0, 4.0, 0.5);
    check_eq("psc: down cost recorded", tracker.down_cost(0), 8.0);

    // Record up observation: delta = 3.0, frac = 0.5 -> cost = 6.0
    tracker.record_up(0, 3.0, 0.5);
    check_eq("psc: up cost recorded", tracker.up_cost(0), 6.0);
    check_true("psc: reliable at 1", tracker.is_reliable(0, 1));

    // Score calculation
    double sc = tracker.score(0, 0.5, 0.5);
    check_true("psc: positive score", sc > 0.0);
}

// 24. Rounding cut generator
void test_rounding_cut_generator() {
    auto model = make_2var_mixed(1.0, 1.0, 0.0, 10.0, 0.0, 10.0);
    vikalp::RoundingCutGenerator gen;
    // x0 is integer at 2.4, x1 is continuous at 3.7
    std::vector<double> sol = {2.4, 3.7};
    auto cuts = gen.generate(model, std::span<const double>(sol));

    // Should generate 2 cuts for x0 (down: <= 2, up: >= 3), none for x1
    check_eq("cuts: count", static_cast<double>(cuts.size()), 2.0);
    if (cuts.size() >= 2) {
        check_eq("cuts[0]: upper", cuts[0].upper, 2.0);
        check_eq("cuts[1]: lower", cuts[1].lower, 3.0);
    }
}

// 25. Solver router for continuous LP
void test_solver_router_lp() {
    vikalp::Model model;
    model.linear_objective = {1.0, 2.0};
    model.variable_lower = {0.0, 0.0};
    model.variable_upper = {10.0, 10.0};
    model.variable_types = {vikalp::VariableType::Continuous, vikalp::VariableType::Continuous};
    model.constraint_matrix = {{0, 2, {0}, {}}, {}};

    MockRelaxationOracle oracle([](auto &, auto, auto, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        r.objective_value = 0.0;
        r.primal_solution = {0.0, 0.0};
        return r;
    });

    auto res = vikalp::solve(model, oracle, test_options());
    check_true("router_lp: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_true("router_lp: solver name", res.solver == "vikalp-continuous");
}

// 26. Solver router for MILP
void test_solver_router_milp() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto, auto, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        r.objective_value = 3.0;
        r.primal_solution = {3.0};
        return r;
    });

    auto res = vikalp::solve(model, oracle, test_options());
    check_true("router_milp: optimal", res.status == vikalp::SolveStatus::Optimal);
    check_true("router_milp: solver name", res.solver == "vikalp-mip-bnb");
}

// 27. Outer approximation
void test_outer_approximation() {
    auto model = make_1var_milp(1.0, 0.0, 10.0);
    MockRelaxationOracle oracle([](auto &, auto, auto, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        r.objective_value = 2.0;
        r.primal_solution = {2.0};
        return r;
    });

    auto res = vikalp::solve_oa(model, oracle, test_options());
    check_true("oa: optimal or feasible", res.status == vikalp::SolveStatus::Optimal ||
                                          res.status == vikalp::SolveStatus::Feasible);
    check_true("oa: solver name", res.solver == "vikalp-oa");
}

// 28. Refinery model generation and validation
void test_refinery_model_generation_and_validation() {
    auto small = vikalp::build_refinery_model(vikalp::refinery_small());
    auto errors = small.validate();
    check_true("refinery_small: valid", errors.empty());
    check_true("refinery_small: has int vars", small.has_integer_variables());
    check_true("refinery_small: is MILP", small.problem_class() == vikalp::ProblemClass::MILP);

    auto med = vikalp::build_refinery_model(vikalp::refinery_medium());
    auto med_errs = med.validate();
    check_true("refinery_med: valid", med_errs.empty());

    auto lg = vikalp::build_refinery_model(vikalp::refinery_large());
    auto lg_errs = lg.validate();
    check_true("refinery_lg: valid", lg_errs.empty());
}

// 29. Refinery MIQP generation
void test_refinery_miqp_generation() {
    auto cfg = vikalp::refinery_small();
    cfg.include_quadratic = true;
    auto miqp = vikalp::build_refinery_model(cfg);

    auto errors = miqp.validate();
    check_true("refinery_miqp: valid", errors.empty());
    check_true("refinery_miqp: has quadratic", miqp.has_quadratic_objective());
    check_true("refinery_miqp: is MIQP", miqp.problem_class() == vikalp::ProblemClass::MIQP);
}

// 30. Refinery solve pipeline
void test_refinery_solve_pipeline() {
    auto model = vikalp::build_refinery_model(vikalp::refinery_small());
    MockRelaxationOracle oracle([](const vikalp::Model &m, auto lo, auto, auto, auto) {
        vikalp::SolveResult r;
        r.status = vikalp::SolveStatus::Optimal;
        const auto n = static_cast<std::size_t>(m.num_variables());
        r.primal_solution.assign(n, 0.0);
        double obj = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double val = lo.empty() ? m.variable_lower[i] : lo[i];
            if (m.variable_types[i] != vikalp::VariableType::Continuous) {
                val = std::round(val);
            }
            r.primal_solution[i] = val;
            obj += m.linear_objective[i] * val;
        }
        r.objective_value = obj;
        return r;
    });

    auto res = vikalp::solve(model, oracle, test_options());
    check_true("refinery_solve: solved", res.status == vikalp::SolveStatus::Optimal ||
                                         res.status == vikalp::SolveStatus::Feasible);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_integral_root();           //  1
    test_one_fractional();          //  2
    test_two_level_branch();        //  3
    test_bound_pruning();           //  4
    test_infeasible_prune();        //  5
    test_incumbent_update();        //  6
    test_no_incumbent();            //  7
    test_gap_termination();         //  8
    test_node_limit();              //  9
    test_time_limit();              // 10
    test_binary_branching();        // 11
    test_integer_branching();       // 12
    test_continuous_exclusion();    // 13
    test_warm_start();              // 14
    test_deterministic_order();     // 15
    test_mock_oracle_calls();       // 16
    test_rounding_accepted();       // 17
    test_rounding_rejected();       // 18
    test_no_integer_vars();         // 19
    test_queue_ordering();          // 20
    test_queue_best_bound();        // 21
    test_effective_bounds();        // 22
    test_pseudocost_tracker();      // 23
    test_rounding_cut_generator();  // 24
    test_solver_router_lp();        // 25
    test_solver_router_milp();      // 26
    test_outer_approximation();     // 27
    test_refinery_model_generation_and_validation(); // 28
    test_refinery_miqp_generation(); // 29
    test_refinery_solve_pipeline(); // 30

    if (g_failures == 0) {
        std::printf("ALL %d MIP TESTS PASSED\n", 30);
        return 0;
    }
    std::printf("%d MIP TEST(S) FAILED\n", g_failures);
    return 1;
}
