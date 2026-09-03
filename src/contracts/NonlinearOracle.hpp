#ifndef VIKALP_NONLINEAR_ORACLE_HPP
#define VIKALP_NONLINEAR_ORACLE_HPP

#include "Model.hpp"

#include <vector>

namespace vikalk {

// Interface for user-supplied nonlinear objective and constraint functions.
//
// Nonlinear models enter VIKALP through this interface because general
// nonlinear functions cannot be represented by the MPS input format.
class NonlinearOracle {
public:
    virtual ~NonlinearOracle() = default;

    // Nonlinear objective value f_nl(x).
    virtual Scalar value_f(const std::vector<Scalar>& x) const = 0;

    // Gradient of the nonlinear objective.
    virtual std::vector<Scalar> gradient_f(
        const std::vector<Scalar>& x) const = 0;

    // Nonlinear constraint values g_nl(x).
    virtual std::vector<Scalar> constraints_g(
        const std::vector<Scalar>& x) const = 0;

    // Jacobian of the nonlinear constraints.
    virtual std::vector<SparseEntry> jacobian_g(
        const std::vector<Scalar>& x) const = 0;

    // Hessian of the Lagrangian.
    virtual std::vector<SparseEntry> hessian_lagrangian(
        const std::vector<Scalar>& x,
        const std::vector<Scalar>& lambda,
        Scalar sigma) const = 0;
};

}  // namespace vikalk

#endif  // VIKALP_NONLINEAR_ORACLE_HPP