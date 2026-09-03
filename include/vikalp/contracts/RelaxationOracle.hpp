#pragma once

#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"

#include <span>

namespace vikalp {

class RelaxationOracle {
public:
    virtual ~RelaxationOracle() = default;

    // Override and warm-start spans are either empty or num_variables() long.
    // Empty overrides use model bounds; an empty warm start means cold start.
    [[nodiscard]] virtual SolveResult solve(
        const Model &model,
        std::span<const Scalar> variable_lower_override,
        std::span<const Scalar> variable_upper_override,
        std::span<const Scalar> warm_start,
        const SolverOptions &options) const = 0;
};

} // namespace vikalp
