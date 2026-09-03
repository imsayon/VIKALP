#include "vikalp/backend/CpuBackend.hpp"
#include "vikalp/contracts/NonlinearOracle.hpp"
#include "vikalp/solver/continuous.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_near(double actual, double expected, double tolerance,
                const char *message) {
    expect(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
           message);
}

vikalp::Model bounded_linear(double objective, double lower, double upper) {
    vikalp::Model model;
    model.name = "bounded-linear";
    model.linear_objective = {objective};
    model.variable_lower = {lower};
    model.variable_upper = {upper};
    model.variable_types = {vikalp::VariableType::Continuous};
    model.constraint_matrix.pattern = {0, 1, {0}, {}};
    return model;
}

class QuadraticOracle final : public vikalp::NonlinearOracle {
public:
    explicit QuadraticOracle(bool equality) : equality_(equality) {}

    [[nodiscard]] vikalp::Index variables() const noexcept override { return 1; }
    [[nodiscard]] vikalp::Index constraints() const noexcept override {
        return equality_ ? 1 : 0;
    }
    [[nodiscard]] vikalp::Scalar objective(std::span<const vikalp::Scalar> x) const override {
        const auto error = x[0] - 2.0;
        return error * error;
    }
    void objective_gradient(std::span<const vikalp::Scalar> x,
                            std::span<vikalp::Scalar> gradient) const override {
        gradient[0] = 2.0 * (x[0] - 2.0);
    }
    void constraint_values(std::span<const vikalp::Scalar> x,
                           std::span<vikalp::Scalar> values) const override {
        if (equality_) values[0] = x[0];
    }
    [[nodiscard]] const vikalp::CsrPattern &jacobian_pattern() const noexcept override {
        return jacobian_;
    }
    void jacobian_values(std::span<const vikalp::Scalar>,
                         std::span<vikalp::Scalar> values) const override {
        if (equality_) values[0] = 1.0;
    }
    [[nodiscard]] const vikalp::CsrPattern &hessian_pattern() const noexcept override {
        return hessian_;
    }
    void hessian_values(std::span<const vikalp::Scalar>, vikalp::Scalar,
                        std::span<const vikalp::Scalar>,
                        std::span<vikalp::Scalar> values) const override {
        values[0] = 2.0;
    }

private:
    bool equality_;
    vikalp::CsrPattern jacobian_{
        equality_ ? 1 : 0, 1, equality_ ? std::vector<vikalp::Index>{0, 1}
                                        : std::vector<vikalp::Index>{0},
        equality_ ? std::vector<vikalp::Index>{0} : std::vector<vikalp::Index>{}};
    vikalp::CsrPattern hessian_{1, 1, {0, 1}, {0}};
};

vikalp::Model nonlinear_model(bool equality) {
    vikalp::Model model;
    model.name = equality ? "constrained-nlp" : "unconstrained-nlp";
    model.linear_objective = {0.0};
    model.variable_lower = {-10.0};
    model.variable_upper = {10.0};
    model.variable_types = {vikalp::VariableType::Continuous};
    model.constraint_matrix.pattern = {equality ? 1 : 0, 1,
                                       equality ? std::vector<vikalp::Index>{0, 1}
                                                : std::vector<vikalp::Index>{0},
                                       equality ? std::vector<vikalp::Index>{0}
                                                : std::vector<vikalp::Index>{}};
    model.constraint_matrix.values = equality ? std::vector<vikalp::Scalar>{0.0}
                                              : std::vector<vikalp::Scalar>{};
    model.constraint_lower = equality ? std::vector<vikalp::Scalar>{1.0}
                                      : std::vector<vikalp::Scalar>{};
    model.constraint_upper = model.constraint_lower;
    model.nonlinear = std::make_shared<QuadraticOracle>(equality);
    return model;
}

void test_lp(vikalp::ExecutionBackend &backend) {
    vikalp::SolverOptions options;
    options.iteration_limit = 5000;
    options.primal_tolerance = 1e-5;
    options.dual_tolerance = 1e-5;

    auto model = bounded_linear(-1.0, 0.0, 10.0);
    const auto result = vikalp::solver::solve_lp(model, backend, options);
    expect(result.status == vikalp::SolveStatus::Optimal, "LP reaches optimal status");
    expect_near(result.primal_solution[0], 10.0, 1e-5, "LP respects active upper bound");
    expect_near(result.objective_value, -10.0, 1e-5, "LP objective is correct");
    expect(result.primal_residual <= options.primal_tolerance, "LP residual is reported");

    model.variable_upper = {20.0};
    model.constraint_matrix.pattern = {1, 1, {0, 1}, {0}};
    model.constraint_matrix.values = {1.0};
    model.constraint_lower = {-vikalp::Model::infinity()};
    model.constraint_upper = {3.0};
    const auto constrained = vikalp::solver::solve_lp(model, backend, options);
    expect(constrained.status == vikalp::SolveStatus::Optimal,
           "LP handles two-sided row bounds");
    expect_near(constrained.primal_solution[0], 3.0, 1e-4,
                "LP enforces the row upper bound");

    const std::vector<vikalp::Scalar> warm_start = {2.0};
    const std::vector<vikalp::Scalar> upper_override = {4.0};
    vikalp::solver::LpRelaxationOracle oracle;
    const auto relaxed = oracle.solve(model, {}, upper_override, warm_start, options);
    expect(relaxed.status == vikalp::SolveStatus::Optimal,
           "LP relaxation oracle returns a solved result");
    expect_near(relaxed.primal_solution[0], 3.0, 1e-4,
                "LP relaxation accepts warm start and bound override");
}

void test_qp(vikalp::ExecutionBackend &backend) {
    auto model = bounded_linear(-10.0, 0.0, 10.0);
    model.name = "convex-qp";
    model.quadratic_objective.pattern = {1, 1, {0, 1}, {0}};
    model.quadratic_objective.values = {2.0};
    vikalp::SolverOptions options;
    options.iteration_limit = 1000;
    options.primal_tolerance = 1e-6;
    options.dual_tolerance = 1e-6;
    const auto result = vikalp::solver::solve_qp(model, backend, options);
    expect(result.status == vikalp::SolveStatus::Optimal, "convex QP reaches optimal status");
    expect_near(result.primal_solution[0], 5.0, 1e-4, "QP reaches analytic minimizer");
    expect_near(result.objective_value, -25.0, 1e-4, "QP objective is correct");
    const std::vector<vikalp::Scalar> upper_override = {4.0};
    const std::vector<vikalp::Scalar> warm_start = {1.0};
    vikalp::solver::QpRelaxationOracle oracle;
    const auto relaxed = oracle.solve(model, {}, upper_override, warm_start, options);
    expect(relaxed.status == vikalp::SolveStatus::Optimal,
           "QP relaxation oracle returns a solved result");
    expect_near(relaxed.primal_solution[0], 4.0, 1e-4,
                "QP relaxation applies a warm start and bound override");

    model.quadratic_objective.values = {-2.0};
    const auto nonconvex = vikalp::solver::solve_qp(model, backend, options);
    expect(nonconvex.status == vikalp::SolveStatus::UnsupportedModel,
           "nonconvex QP is rejected explicitly");
}

void test_nlp(vikalp::ExecutionBackend &backend) {
    vikalp::SolverOptions options;
    options.iteration_limit = 50;
    options.primal_tolerance = 1e-6;
    options.dual_tolerance = 1e-6;

    const auto unconstrained = vikalp::solver::solve_nlp(
        nonlinear_model(false), backend, options);
    expect(unconstrained.status == vikalp::SolveStatus::LocallyOptimal,
           "NLP IPM solves an analytic smooth objective");
    expect_near(unconstrained.primal_solution[0], 2.0, 1e-6,
                "NLP reaches analytic minimizer");

    const auto constrained = vikalp::solver::solve_nlp(
        nonlinear_model(true), backend, options);
    expect(constrained.status == vikalp::SolveStatus::LocallyOptimal,
           "NLP IPM handles an analytic equality constraint");
    expect_near(constrained.primal_solution[0], 1.0, 1e-5,
                "NLP satisfies equality constraint");
    expect(constrained.primal_residual <= options.primal_tolerance,
           "NLP reports equality residual");
}

void test_failure_paths(vikalp::ExecutionBackend &backend) {
    vikalp::SolverOptions options;
    options.iteration_limit = 20;
    auto invalid = bounded_linear(1.0, 3.0, 2.0);
    const auto invalid_result = vikalp::solver::solve_lp(invalid, backend, options);
    expect(invalid_result.status == vikalp::SolveStatus::InvalidModel,
           "invalid variable bounds are rejected");

    auto valid = bounded_linear(1.0, 0.0, 2.0);
    const std::vector<vikalp::Scalar> non_finite_start = {
        vikalp::Model::infinity()};
    const auto bad_start = vikalp::solver::solve_lp(
        valid, backend, options, non_finite_start);
    expect(bad_start.status == vikalp::SolveStatus::InvalidModel,
           "non-finite warm starts are rejected");

    valid.quadratic_objective.pattern = {1, 1, {0, 1}, {0}};
    valid.quadratic_objective.values = {1.0};
    const auto wrong_solver = vikalp::solver::solve_lp(valid, backend, options);
    expect(wrong_solver.status == vikalp::SolveStatus::UnsupportedModel,
           "LP solver rejects a quadratic model");

    options.iteration_limit = -1;
    valid.quadratic_objective = {};
    const auto bad_options = vikalp::solver::solve_lp(valid, backend, options);
    expect(bad_options.status == vikalp::SolveStatus::InvalidModel,
           "negative iteration limits are rejected");
}

} // namespace

int main() {
    auto backend = vikalp::make_cpu_backend();
    test_lp(*backend);
    test_qp(*backend);
    test_nlp(*backend);
    test_failure_paths(*backend);
    if (failures != 0) {
        std::cerr << failures << " Flow C checks failed\n";
        return 1;
    }
    std::cout << "ALL FLOW C CHECKS PASSED\n";
    return 0;
}
