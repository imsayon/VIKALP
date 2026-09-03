#pragma once

// Flow D — Branch-and-bound MIP engine.
//
// Orchestrates:
//   1. Node selection (best-bound-first)
//   2. Relaxation solve via RelaxationOracle
//   3. Integrality check
//   4. Most-fractional branching (Phase 7)
//   5. Bound pruning, infeasibility pruning, incumbent acceptance (Phase 8)
//   6. Gap termination (Phase 9)
//   7. Warm-start propagation (Phase 10)
//   8. Bounded rounding heuristic (Phase 11)
//
// Does NOT implement LP/QP/NLP numerical mathematics.
// Consumes relaxation results through the RelaxationOracle interface.

#include "vikalp/contracts/Model.hpp"
#include "vikalp/contracts/RelaxationOracle.hpp"
#include "vikalp/contracts/SolveResult.hpp"
#include "vikalp/contracts/SolverOptions.hpp"

namespace vikalp {

/// Solve a mixed-integer program using branch-and-bound.
/// Requires a RelaxationOracle that solves the continuous relaxation.
/// The engine is stateless: all state lives in the returned SolveResult.
[[nodiscard]] SolveResult solve_mip(
    const Model &model,
    const RelaxationOracle &oracle,
    const SolverOptions &options);

} // namespace vikalp
