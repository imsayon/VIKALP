#ifndef VIKALP_EXECUTION_BACKEND_HPP
#define VIKALP_EXECUTION_BACKEND_HPP

#include "Model.hpp"

#include <vector>

namespace vikalk {

// Common execution interface used by solver components.
// The backend performs numerical operations but does not contain
// solver policy.
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;

    // Matrix-vector product: y = A*x.
    virtual std::vector<Scalar> matvec(
        const std::vector<SparseEntry>& matrix,
        const std::vector<Scalar>& x) const = 0;

    // Transpose matrix-vector product: y = A^T*x.
    virtual std::vector<Scalar> transpose_matvec(
        const std::vector<SparseEntry>& matrix,
        const std::vector<Scalar>& x) const = 0;

    // Vector operation: y = alpha*x + beta*y.
    virtual void axpby(
        Scalar alpha,
        const std::vector<Scalar>& x,
        Scalar beta,
        std::vector<Scalar>& y) const = 0;

    // Dot product.
    virtual Scalar dot(
        const std::vector<Scalar>& x,
        const std::vector<Scalar>& y) const = 0;

    // Infinity norm.
    virtual Scalar norm_inf(
        const std::vector<Scalar>& x) const = 0;
};

}  // namespace vikalk

#endif  // VIKALP_EXECUTION_BACKEND_HPP