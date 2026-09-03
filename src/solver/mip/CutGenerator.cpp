// Flow D — Cut generator implementations.

#include "vikalp/solver/mip/CutGenerator.hpp"

#include <algorithm>
#include <cmath>

namespace vikalp {

std::vector<Cut> RoundingCutGenerator::generate(
    const Model &model,
    std::span<const Scalar> x) const {
    std::vector<Cut> cuts;
    const auto n = static_cast<std::size_t>(model.num_variables());
    const auto m = static_cast<std::size_t>(model.num_constraints());

    if (x.size() != n) return cuts;

    const auto &pattern = model.constraint_matrix.pattern;
    const auto &values = model.constraint_matrix.values;

    for (std::size_t row = 0; row < m; ++row) {
        const Index begin = pattern.row_offsets[row];
        const Index end = pattern.row_offsets[row + 1];
        bool integral_row = begin != end;
        for (Index position = begin; position < end && integral_row; ++position) {
            const auto pos = static_cast<std::size_t>(position);
            const auto col = static_cast<std::size_t>(pattern.column_indices[pos]);
            integral_row = model.variable_types[col] != VariableType::Continuous &&
                           std::abs(values[pos] - std::round(values[pos])) <= int_tol_;
        }
        if (!integral_row) continue;

        std::vector<Index> indices;
        std::vector<Scalar> coefficients;
        Scalar activity = 0.0;
        for (Index position = begin; position < end; ++position) {
            const auto pos = static_cast<std::size_t>(position);
            indices.push_back(pattern.column_indices[pos]);
            coefficients.push_back(values[pos]);
            activity += values[pos] * x[static_cast<std::size_t>(pattern.column_indices[pos])];
        }

        auto make_cut = [&](Scalar lower, Scalar upper) {
            Cut cut;
            cut.indices = indices;
            cut.coefficients = coefficients;
            cut.lower = lower;
            cut.upper = upper;
            cuts.push_back(std::move(cut));
        };

        if (std::isfinite(model.constraint_upper[row])) {
            const Scalar upper = std::floor(model.constraint_upper[row]);
            if (upper < model.constraint_upper[row] - int_tol_ &&
                activity > upper + int_tol_) {
                make_cut(-Model::infinity(), upper);
            }
        }
        if (std::isfinite(model.constraint_lower[row])) {
            const Scalar lower = std::ceil(model.constraint_lower[row]);
            if (lower > model.constraint_lower[row] + int_tol_ &&
                activity < lower - int_tol_) {
                make_cut(lower, Model::infinity());
            }
        }
    }

    return cuts;
}

} // namespace vikalp
