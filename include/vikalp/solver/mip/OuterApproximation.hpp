#pragma once

// Flow D — Outer Approximation for MINLP problems.
//
// Algorithm:
//   1. Solve NLP relaxation (fix integers to initial guess or LP solution)
//   2. Linearize nonlinear functions at NLP solution → add OA cuts to MILP master
//   3. Solve MILP master → get new integer assignment
//   4. Fix integers, solve NLP subproblem
//   5. Update incumbent if feasible; add new OA cuts
//   6. Repeat until gap closes or iteration limit
//
// This is the Duran-Grossmann OA method adapted for the VIKALP contract system.

#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/RelaxationOracle.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"

namespace vikalp {

/// Solve a MINLP using outer approximation.
/// Requires: model.nonlinear is set, model has integer variables.
/// The oracle solves the MILP master problems.
[[nodiscard]] SolveResult solve_oa(
    const Model &model,
    const RelaxationOracle &oracle,
    const SolverOptions &options);

} // namespace vikalp
