#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/NonlinearOracle.hpp"

#include <cassert>
#include <span>
#include <vector>

class TestNlpOracle : public vikalp::NonlinearOracle {
public:
    [[nodiscard]] vikalp::Index variables() const noexcept override {
        return 1;
    }

    [[nodiscard]] vikalp::Index constraints() const noexcept override {
        return 1;
    }

    [[nodiscard]] vikalp::Scalar objective(
        std::span<const vikalp::Scalar> x) const override {
        return x[0] * x[0];
    }

    void objective_gradient(
        std::span<const vikalp::Scalar> x,
        std::span<vikalp::Scalar> gradient) const override {
        gradient[0] = 2.0 * x[0];
    }

    void constraint_values(
        std::span<const vikalp::Scalar> x,
        std::span<vikalp::Scalar> values) const override {
        values[0] = x[0] - 1.0;
    }

    [[nodiscard]] const vikalp::CsrPattern &
    jacobian_pattern() const noexcept override {
        return jacobian_pattern_;
    }

    void jacobian_values(
        std::span<const vikalp::Scalar>,
        std::span<vikalp::Scalar> values) const override {
        values[0] = 1.0;
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
        1,
        {0, 1},
        {0}
    };

    vikalp::CsrPattern hessian_pattern_{
        1,
        1,
        {0, 1},
        {0}
    };
};

int main() {
    TestNlpOracle oracle;

    assert(oracle.variables() == 1);
    assert(oracle.constraints() == 1);

    std::vector<vikalp::Scalar> x = {2.0};

    assert(
        oracle.objective(
            std::span<const vikalp::Scalar>(x)) == 4.0);

    std::vector<vikalp::Scalar> gradient(1);

    oracle.objective_gradient(
        std::span<const vikalp::Scalar>(x),
        std::span<vikalp::Scalar>(gradient));

    assert(gradient[0] == 4.0);

    std::vector<vikalp::Scalar> constraint_values(1);

    oracle.constraint_values(
        std::span<const vikalp::Scalar>(x),
        std::span<vikalp::Scalar>(constraint_values));

    assert(constraint_values[0] == 1.0);

    std::vector<vikalp::Scalar> jacobian_values(1);

    oracle.jacobian_values(
        std::span<const vikalp::Scalar>(x),
        std::span<vikalp::Scalar>(jacobian_values));

    assert(jacobian_values[0] == 1.0);

    std::vector<vikalp::Scalar> multipliers = {0.0};
    std::vector<vikalp::Scalar> hessian_values(1);

    oracle.hessian_values(
        std::span<const vikalp::Scalar>(x),
        1.0,
        std::span<const vikalp::Scalar>(multipliers),
        std::span<vikalp::Scalar>(hessian_values));

    assert(hessian_values[0] == 2.0);

    return 0;
}
