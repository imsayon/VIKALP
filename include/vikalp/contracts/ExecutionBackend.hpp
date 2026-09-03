#pragma once

#include "vikalp/contracts/Model.hpp"

#include <memory>
#include <span>

namespace vikalp {

class BackendVector {
public:
    virtual ~BackendVector() = default;
    [[nodiscard]] virtual Index size() const noexcept = 0;
};

class BackendMatrix {
public:
    virtual ~BackendMatrix() = default;
    [[nodiscard]] virtual Index rows() const noexcept = 0;
    [[nodiscard]] virtual Index columns() const noexcept = 0;
};

enum class Transpose { No,
                       Yes };
enum class MatrixProperty { General,
                            SymmetricPositiveDefinite,
                            SymmetricIndefinite };

struct LinearSolveResult {
    bool converged = false;
    Index iterations = 0;
    Scalar residual = Model::infinity();
};

// Owns CPU or GPU-resident buffers. Solver policy stays outside this interface.
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;

    [[nodiscard]] virtual std::unique_ptr<BackendVector> create_vector(Index size) = 0;
    [[nodiscard]] virtual std::unique_ptr<BackendMatrix> create_matrix(
        const CsrPattern &pattern) = 0;
    virtual void upload(std::span<const Scalar> source, BackendVector &target) = 0;
    virtual void download(const BackendVector &source, std::span<Scalar> target) const = 0;
    virtual void set_values(std::span<const Scalar> values, BackendMatrix &target) = 0;
    virtual void fill(BackendVector &target, Scalar value) = 0;
    virtual void copy(const BackendVector &source, BackendVector &target) = 0;
    virtual void axpby(Scalar alpha, const BackendVector &x,
                       Scalar beta, BackendVector &y) = 0;
    [[nodiscard]] virtual Scalar dot(const BackendVector &x,
                                     const BackendVector &y) = 0;
    [[nodiscard]] virtual Scalar norm_inf(const BackendVector &x) = 0;
    virtual void project_box(BackendVector &x,
                             const BackendVector &lower,
                             const BackendVector &upper) = 0;
    virtual void multiply(Scalar alpha, const BackendMatrix &matrix,
                          const BackendVector &x, Scalar beta,
                          BackendVector &y, Transpose transpose) = 0;
    [[nodiscard]] virtual LinearSolveResult solve_linear_system(
        const BackendMatrix &matrix, const BackendVector &rhs,
        BackendVector &solution, MatrixProperty property,
        Scalar tolerance, Index iteration_limit) = 0;
    virtual void synchronize() = 0;
};

} // namespace vikalp
