#pragma once

#include "vikalp/contracts/Model.hpp"

#include <span>

namespace vikalp {

// Derivative sparsity is fixed for the oracle lifetime; calls only fill values.
class NonlinearOracle {
public:
    virtual ~NonlinearOracle() = default;

    [[nodiscard]] virtual Index variables() const noexcept = 0;
    [[nodiscard]] virtual Index constraints() const noexcept = 0;
    [[nodiscard]] virtual Scalar objective(std::span<const Scalar> x) const = 0;
    virtual void objective_gradient(std::span<const Scalar> x,
                                    std::span<Scalar> gradient) const = 0;
    virtual void constraint_values(std::span<const Scalar> x,
                                   std::span<Scalar> values) const = 0;
    [[nodiscard]] virtual const CsrPattern &jacobian_pattern() const noexcept = 0;
    virtual void jacobian_values(std::span<const Scalar> x,
                                 std::span<Scalar> values) const = 0;
    // Full symmetric Hessian of sigma*f(x) + lambda'g(x), in CSR order.
    [[nodiscard]] virtual const CsrPattern &hessian_pattern() const noexcept = 0;
    virtual void hessian_values(std::span<const Scalar> x,
                                Scalar sigma,
                                std::span<const Scalar> lambda,
                                std::span<Scalar> values) const = 0;
};

} // namespace vikalp
