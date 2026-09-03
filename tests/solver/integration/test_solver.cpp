#include "vikalp/contracts/NonlinearOracle.hpp"
#include "vikalp/io/MpsReader.hpp"
#include "vikalp/solver/Solver.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <span>

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

vikalp::SolverOptions options() {
    vikalp::SolverOptions result;
    result.iteration_limit = 10'000;
    result.node_limit = 100;
    result.time_limit_seconds = 10.0;
    result.primal_tolerance = 1e-5;
    result.dual_tolerance = 1e-5;
    return result;
}

vikalp::Model one_variable(vikalp::VariableType type) {
    vikalp::Model model;
    model.linear_objective = {-1.0};
    model.variable_lower = {0.0};
    model.variable_upper = {10.0};
    model.variable_types = {type};
    model.constraint_matrix.pattern = {0, 1, {0}, {}};
    return model;
}

class QuadraticObjective final : public vikalp::NonlinearOracle {
public:
    [[nodiscard]] vikalp::Index variables() const noexcept override { return 1; }
    [[nodiscard]] vikalp::Index constraints() const noexcept override { return 0; }
    [[nodiscard]] vikalp::Scalar objective(
        std::span<const vikalp::Scalar> x) const override {
        const auto offset = x[0] - 0.6;
        return offset * offset;
    }
    void objective_gradient(std::span<const vikalp::Scalar> x,
                            std::span<vikalp::Scalar> gradient) const override {
        gradient[0] = 2.0 * (x[0] - 0.6);
    }
    void constraint_values(std::span<const vikalp::Scalar>,
                           std::span<vikalp::Scalar>) const override {}
    [[nodiscard]] const vikalp::CsrPattern &jacobian_pattern() const noexcept override {
        return jacobian_;
    }
    void jacobian_values(std::span<const vikalp::Scalar>,
                         std::span<vikalp::Scalar>) const override {}
    [[nodiscard]] const vikalp::CsrPattern &hessian_pattern() const noexcept override {
        return hessian_;
    }
    void hessian_values(std::span<const vikalp::Scalar>, vikalp::Scalar objective_weight,
                        std::span<const vikalp::Scalar>,
                        std::span<vikalp::Scalar> values) const override {
        values[0] = 2.0 * objective_weight;
    }

private:
    vikalp::CsrPattern jacobian_{0, 1, {0}, {}};
    vikalp::CsrPattern hessian_{1, 1, {0, 1}, {0}};
};

void test_lp() {
    const auto result = vikalp::solve(one_variable(vikalp::VariableType::Continuous),
                                      options());
    expect(result.status == vikalp::SolveStatus::Optimal, "LP integrated route");
    expect_near(result.primal_solution[0], 10.0, 1e-4, "LP solution");
    expect(result.backend == "cpu", "LP backend label");
}

void test_mps_to_solution() {
    vikalp::MpsReader reader;
    vikalp::Model model;
    expect(reader.read("tests/io/simple_lp.mps", model), "MPS input route");
    const auto result = vikalp::solve(model, options());
    expect(result.status == vikalp::SolveStatus::Optimal, "MPS solve route");
    expect_near(result.primal_solution[0], 0.0, 1e-5, "MPS solution");
}

void test_qp(vikalp::VariableType type, const char *route) {
    auto model = one_variable(type);
    model.linear_objective = {-1.2};
    model.variable_upper = {2.0};
    model.quadratic_objective.pattern = {1, 1, {0, 1}, {0}};
    model.quadratic_objective.values = {2.0};
    const auto result = vikalp::solve(model, options());
    expect(result.status == vikalp::SolveStatus::Optimal, route);
    expect_near(result.primal_solution[0],
                type == vikalp::VariableType::Continuous ? 0.6 : 1.0,
                1e-4, route);
}

void test_milp() {
    auto model = one_variable(vikalp::VariableType::Integer);
    model.linear_objective = {1.0};
    model.variable_upper = {2.0};
    const auto result = vikalp::solve(model, options());
    expect(result.status == vikalp::SolveStatus::Optimal, "MILP integrated route");
    expect_near(result.primal_solution[0], 0.0, 1e-5, "MILP solution");
}

void test_nonlinear(vikalp::VariableType type, const char *route) {
    auto model = one_variable(type);
    model.linear_objective = {0.0};
    model.variable_upper = {2.0};
    model.nonlinear = std::make_shared<QuadraticObjective>();
    const auto result = vikalp::solve(model, options());
    const auto expected_status = type == vikalp::VariableType::Continuous
                                     ? vikalp::SolveStatus::LocallyOptimal
                                     : vikalp::SolveStatus::Optimal;
    expect(result.status == expected_status, route);
    expect_near(result.primal_solution[0],
                type == vikalp::VariableType::Continuous ? 0.6 : 1.0,
                1e-4, route);
}

void test_unavailable_cuda() {
#ifndef VIKALP_CUDA_ENABLED
    auto requested = options();
    requested.backend = vikalp::BackendPreference::CUDA;
    const auto result = vikalp::solve(
        one_variable(vikalp::VariableType::Continuous), requested);
    expect(result.status == vikalp::SolveStatus::UnsupportedModel,
           "unavailable CUDA is reported");
#endif
}

} // namespace

int main() {
    test_mps_to_solution();
    test_lp();
    test_qp(vikalp::VariableType::Continuous, "QP integrated route");
    test_nonlinear(vikalp::VariableType::Continuous, "NLP integrated route");
    test_milp();
    test_qp(vikalp::VariableType::Integer, "MIQP integrated route");
    test_nonlinear(vikalp::VariableType::Integer, "MINLP integrated route");
    test_unavailable_cuda();
    if (failures != 0) return 1;
    std::cout << "ALL INTEGRATION CHECKS PASSED\n";
    return 0;
}
