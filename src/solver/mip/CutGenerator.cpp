// Flow D — Cut generator implementations.

#include "vikalp/solver/mip/CutGenerator.hpp"

#include <cmath>

namespace vikalp {

std::vector<Cut> RoundingCutGenerator::generate(
    const Model &model,
    std::span<const Scalar> x) const {
    std::vector<Cut> cuts;
    const auto n = static_cast<std::size_t>(model.num_variables());

    for (std::size_t i = 0; i < n && i < x.size(); ++i) {
        if (model.variable_types[i] == VariableType::Continuous) continue;

        const Scalar val = x[i];
        const Scalar frac = std::abs(val - std::round(val));
        if (frac <= int_tol_) continue;

        const Index idx = static_cast<Index>(i);
        const Scalar floor_val = std::floor(val);
        const Scalar ceil_val = std::ceil(val);

        // Down cut: x_j <= floor(val)
        if (floor_val >= model.variable_lower[i]) {
            Cut down;
            down.indices = {idx};
            down.coefficients = {1.0};
            down.lower = -Model::infinity();
            down.upper = floor_val;
            cuts.push_back(std::move(down));
        }

        // Up cut: x_j >= ceil(val)
        if (ceil_val <= model.variable_upper[i]) {
            Cut up;
            up.indices = {idx};
            up.coefficients = {1.0};
            up.lower = ceil_val;
            up.upper = Model::infinity();
            cuts.push_back(std::move(up));
        }
    }
    return cuts;
}

} // namespace vikalp
