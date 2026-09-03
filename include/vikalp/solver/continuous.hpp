#pragma once

#include "vikalp/contracts/ExecutionBackend.hpp"
#include "vikalp/contracts/RelaxationOracle.hpp"

#include <span>

namespace vikalp::solver {

SolveResult solve_lp(
    const Model &model, ExecutionBackend &backend,
    const SolverOptions &options = {},
    std::span<const Scalar> warm_start = {},
    std::span<const Scalar> variable_lower_override = {},
    std::span<const Scalar> variable_upper_override = {});

SolveResult solve_qp(
    const Model &model, ExecutionBackend &backend,
    const SolverOptions &options = {},
    std::span<const Scalar> warm_start = {},
    std::span<const Scalar> variable_lower_override = {},
    std::span<const Scalar> variable_upper_override = {});

SolveResult solve_nlp(
    const Model &model, ExecutionBackend &backend,
    const SolverOptions &options = {},
    std::span<const Scalar> warm_start = {},
    std::span<const Scalar> variable_lower_override = {},
    std::span<const Scalar> variable_upper_override = {});

class LpRelaxationOracle final : public RelaxationOracle {
public:
    [[nodiscard]] SolveResult solve(
        const Model &model,
        std::span<const Scalar> variable_lower_override,
        std::span<const Scalar> variable_upper_override,
        std::span<const Scalar> warm_start,
        const SolverOptions &options) const override;
};

class QpRelaxationOracle final : public RelaxationOracle {
public:
    [[nodiscard]] SolveResult solve(
        const Model &model,
        std::span<const Scalar> variable_lower_override,
        std::span<const Scalar> variable_upper_override,
        std::span<const Scalar> warm_start,
        const SolverOptions &options) const override;
};

} // namespace vikalp::solver
