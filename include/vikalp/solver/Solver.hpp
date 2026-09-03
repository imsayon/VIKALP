#pragma once

// Flow D — Top-level solver with problem-class routing.
//
// Routes LP/QP/NLP to the RelaxationOracle directly (continuous problems).
// Routes MILP/MIQP to solve_mip (branch-and-bound).
// Routes MINLP to outer approximation (when available).
//
// This is the single public entry point for solving any Model.

#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/RelaxationOracle.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"

namespace vikalp {

/// Solve a model with VIKALP's built-in continuous relaxation solvers.
[[nodiscard]] SolveResult solve(
    const Model &model,
    const SolverOptions &options = {});

/// Solve a model by routing to the appropriate solver based on problem class.
/// Continuous problems (LP/QP/NLP) → RelaxationOracle directly.
/// Mixed-integer (MILP/MIQP) → Branch-and-bound.
/// MINLP → Outer approximation.
[[nodiscard]] SolveResult solve(
    const Model &model,
    const RelaxationOracle &oracle,
    const SolverOptions &options);

} // namespace vikalp
