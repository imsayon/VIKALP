#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/NonlinearOracle.hpp"

#include <cassert>
#include <memory>
#include <span>
#include <vector>

class TestMinlpOracle : public vikalp::NonlinearOracle {
public:
    [[nodiscard]] vikalp::Index variables() const noexcept override {
        return 2;
    }

    [[nodiscard]] vikalp::Index constraints() const noexcept override {
        return 1;
    }

    [[nodiscard]] vikalp::Scalar objective(
        std::span<const vikalp::Scalar> x) const override {
        return x[0] * x[0] + x[1];
    }

    void objective_gradient(
        std::span<const vikalp::Scalar> x,
        std::span<vikalp::Scalar> gradient) const override {
        gradient[0] = 2.0 * x[0];
        gradient[1] = 1.0;
    }

    void constraint_values(
        std::span<const vikalp::Scalar> x,
        std::span<vikalp::Scalar> values) const override {
        values[0] = x[0] + x[1];
    }

    [[nodiscard]] const vikalp::CsrPattern &
    jacobian_pattern() const noexcept override {
        return jacobian_pattern_;
    }

    void jacobian_values(
        std::span<const vikalp::Scalar>,
        std::span<vikalp::Scalar> values) const override {
        values[0] = 1.0;
        values[1] = 1.0;
    }

    [[nodiscard]] const vikalp::CsrPattern &
    hessian_pattern() const noexcept override {
        return hessian_pattern_;
    }

    void hessian_values(
        std::span<const vikalp::Scalar>,
        vikalp::Scalar,
        std::span<const vikalp::Scalar>,
        std::span<vikalp::Scalar> values) const override {
        values[0] = 2.0;
    }

private:
    vikalp::CsrPattern jacobian_pattern_{
        1,
        2,
        {0, 2},
        {0, 1}
    };

    vikalp::CsrPattern hessian_pattern_{
        2,
        2,
        {0, 1, 1},
        {0}
    };
};

int main() {
    vikalp::Model model;

    model.name = "SIMPLE_MINLP";

    model.linear_objective = {1.0, 1.0};

    model.constraint_matrix.pattern.rows = 1;
    model.constraint_matrix.pattern.columns = 2;
    model.constraint_matrix.pattern.row_offsets = {0, 2};
    model.constraint_matrix.pattern.column_indices = {0, 1};
    model.constraint_matrix.values = {1.0, 1.0};

    model.constraint_lower = {0.0};
    model.constraint_upper = {5.0};

    model.variable_lower = {0.0, 0.0};
    model.variable_upper = {10.0, 10.0};

    model.variable_types = {
        vikalp::VariableType::Integer,
        vikalp::VariableType::Continuous
    };

    model.nonlinear = std::make_shared<TestMinlpOracle>();

    assert(model.validate().empty());

    assert(model.num_variables() == 2);
    assert(model.num_constraints() == 1);

    assert(model.has_integer_variables());
    assert(model.nonlinear != nullptr);

    assert(model.problem_class() == vikalp::ProblemClass::MINLP);

    return 0;
}

