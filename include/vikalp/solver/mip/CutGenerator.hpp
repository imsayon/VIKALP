#pragma once

// Flow D — Cut generator interface.
//
// Cuts are linear constraints added to tighten the relaxation.
// A CutGenerator inspects a relaxation solution and produces zero
// or more cuts. Each cut is a row: lower <= a'x <= upper.

#include "vikalp/contracts/Model.hpp"

#include <span>
#include <vector>

namespace vikalp {

/// A single linear cut: lower <= coefficients'x <= upper.
struct Cut {
    std::vector<Index> indices;       // variable indices (sorted, unique)
    std::vector<Scalar> coefficients; // same length as indices
    Scalar lower = -Model::infinity();
    Scalar upper = Model::infinity();
};

/// Abstract interface for cut generators (Gomory, MIR, etc.)
class CutGenerator {
public:
    virtual ~CutGenerator() = default;

    /// Generate cuts given the current relaxation solution.
    /// model: the original MILP
    /// x: relaxation primal solution
    /// Returns a (possibly empty) vector of cuts.
    [[nodiscard]] virtual std::vector<Cut> generate(
        const Model &model,
        std::span<const Scalar> x) const = 0;
};

/// Trivial rounding cut generator.
/// For each fractional integer variable x_j, it generates the pair of
/// cuts x_j <= floor(x_j*) and x_j >= ceil(x_j*) as candidate separators.
/// This is a teaching-quality generator, not production-grade.
class RoundingCutGenerator final : public CutGenerator {
public:
    explicit RoundingCutGenerator(Scalar int_tol = 1e-6) : int_tol_(int_tol) {}

    [[nodiscard]] std::vector<Cut> generate(
        const Model &model,
        std::span<const Scalar> x) const override;

private:
    Scalar int_tol_;
};

} // namespace vikalp
