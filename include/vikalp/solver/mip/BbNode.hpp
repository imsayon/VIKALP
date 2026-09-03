#pragma once

// Flow D — Branch-and-bound node representation.
//
// Each node represents a subproblem derived from the original Model by
// tightening variable bounds. Nodes store only the bound modifications
// (overrides), not a copy of the entire model. This is memory-efficient
// and makes the branching history transparent.
//
// Node state transitions:
//   Open → Processing → Pruned | Integral | Branched | Infeasible

#include "vikalp/contracts/Model.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace vikalp {

enum class NodeStatus {
    Open,        // Awaiting processing
    Branched,    // Relaxation solved, children created
    Pruned,      // Dominated by incumbent or infeasible
    Integral,    // Relaxation solution is integer-feasible (candidate incumbent)
    Infeasible   // Relaxation is infeasible at this node
};

enum class BranchDirection {
    Down,  // upper bound tightened: x_j <= floor(x_j*)
    Up     // lower bound tightened: x_j >= ceil(x_j*)
};

struct BbNode {
    // ── Identity ─────────────────────────────────────────────────────────────
    std::int64_t id = 0;
    std::int64_t parent_id = -1;   // -1 for root
    Index depth = 0;

    // ── Bound overrides ──────────────────────────────────────────────────────
    // Pairs of (variable_index, bound_value). Applied on top of the model's
    // original bounds. Both vectors are sorted by variable_index for
    // deterministic behavior.
    std::vector<std::pair<Index, Scalar>> lower_overrides;
    std::vector<std::pair<Index, Scalar>> upper_overrides;

    // ── Relaxation result ────────────────────────────────────────────────────
    Scalar relaxation_bound = -Model::infinity();  // Best dual bound from relaxation
    std::vector<Scalar> relaxation_solution;        // Primal from relaxation (warm start source)

    // ── Branching info ───────────────────────────────────────────────────────
    Index branching_variable = -1;           // Which variable was branched on to create this node
    BranchDirection branch_direction = BranchDirection::Down;

    // ── Status ───────────────────────────────────────────────────────────────
    NodeStatus status = NodeStatus::Open;
};

// ── Utilities ────────────────────────────────────────────────────────────────

/// Build effective lower bounds for a node by applying overrides to model bounds.
/// Result is num_variables() long.
inline std::vector<Scalar> effective_lower_bounds(const Model &model,
                                                   const BbNode &node) {
    auto bounds = model.variable_lower;
    for (const auto &[idx, val] : node.lower_overrides) {
        auto i = static_cast<std::size_t>(idx);
        if (i < bounds.size() && val > bounds[i])
            bounds[i] = val;
    }
    return bounds;
}

/// Build effective upper bounds for a node by applying overrides to model bounds.
inline std::vector<Scalar> effective_upper_bounds(const Model &model,
                                                   const BbNode &node) {
    auto bounds = model.variable_upper;
    for (const auto &[idx, val] : node.upper_overrides) {
        auto i = static_cast<std::size_t>(idx);
        if (i < bounds.size() && val < bounds[i])
            bounds[i] = val;
    }
    return bounds;
}

} // namespace vikalp
