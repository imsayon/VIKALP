#include "vikalp/contracts/ExecutionBackend.hpp"
#include "vikalp/contracts/NonlinearOracle.hpp"
#include "vikalp/contracts/RelaxationOracle.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"
#include "vikalp/contracts/Verifier.hpp"

#include <cassert>
#include <memory>
#include <type_traits>

namespace {

class DummyNonlinearOracle final : public vikalp::NonlinearOracle {
public:
    vikalp::Index variables() const noexcept override { return 1; }
    vikalp::Index constraints() const noexcept override { return 1; }
    vikalp::Scalar objective(std::span<const vikalp::Scalar> x) const override {
        return x[0] * x[0];
    }
    void objective_gradient(std::span<const vikalp::Scalar> x,
                            std::span<vikalp::Scalar> gradient) const override {
        gradient[0] = 2.0 * x[0];
    }
    void constraint_values(std::span<const vikalp::Scalar> x,
                           std::span<vikalp::Scalar> values) const override {
        values[0] = x[0] * x[0];
    }
    const vikalp::CsrPattern &jacobian_pattern() const noexcept override {
        return pattern_;
    }
    void jacobian_values(std::span<const vikalp::Scalar> x,
                         std::span<vikalp::Scalar> values) const override {
        values[0] = 2.0 * x[0];
    }
    const vikalp::CsrPattern &hessian_pattern() const noexcept override {
        return pattern_;
    }
    void hessian_values(std::span<const vikalp::Scalar>,
                        vikalp::Scalar sigma,
                        std::span<const vikalp::Scalar> lambda,
                        std::span<vikalp::Scalar> values) const override {
        values[0] = 2.0 * (sigma + lambda[0]);
    }

private:
    vikalp::CsrPattern pattern_{1, 1, {0, 1}, {0}};
};

vikalp::Model valid_model() {
    vikalp::Model model;
    model.linear_objective = {1.0};
    model.constraint_matrix = {{1, 1, {0, 1}, {0}}, {1.0}};
    model.constraint_lower = {0.0};
    model.constraint_upper = {1.0};
    model.variable_lower = {0.0};
    model.variable_upper = {1.0};
    model.variable_types = {vikalp::VariableType::Continuous};
    return model;
}

} // namespace

int main() {
    static_assert(std::is_abstract_v<vikalp::ExecutionBackend>);
    static_assert(std::is_abstract_v<vikalp::RelaxationOracle>);
    static_assert(std::is_abstract_v<vikalp::Verifier>);

    auto model = valid_model();
    assert(model.validate().empty());
    assert(model.problem_class() == vikalp::ProblemClass::LP);

    model.variable_types[0] = vikalp::VariableType::Integer;
    assert(model.problem_class() == vikalp::ProblemClass::MILP);

    model.quadratic_objective = {{1, 1, {0, 1}, {0}}, {2.0}};
    assert(model.problem_class() == vikalp::ProblemClass::MIQP);
    model.variable_types[0] = vikalp::VariableType::Continuous;
    assert(model.problem_class() == vikalp::ProblemClass::QP);
    assert(model.validate().empty());

    model.nonlinear = std::make_shared<DummyNonlinearOracle>();
    assert(model.problem_class() == vikalp::ProblemClass::NLP);
    model.variable_types[0] = vikalp::VariableType::Binary;
    assert(model.problem_class() == vikalp::ProblemClass::MINLP);
    assert(model.validate().empty());

    model.constraint_matrix.pattern.column_indices[0] = 1;
    assert(!model.validate().empty());

    model = valid_model();
    model.quadratic_objective = {{1, 1, {0, 2}, {0}}, {1.0}};
    assert(!model.validate().empty());

    const vikalp::SolverOptions options;
    const vikalp::SolveResult result;
    assert(options.time_limit_seconds == vikalp::Model::infinity());
    assert(result.status == vikalp::SolveStatus::NotSolved);
}
