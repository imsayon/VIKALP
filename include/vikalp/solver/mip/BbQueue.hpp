#pragma once

// Flow D — Deterministic priority queue for branch-and-bound nodes.
//
// Ordering for minimization (best-bound-first):
//   1. Lower relaxation_bound is better (explore most promising first)
//   2. Tie: deeper node first (dive toward integer feasibility)
//   3. Tie: lower node ID first (deterministic)
//
// The queue also tracks statistics: total processed, pruned, integral, etc.

#include "vikalp/solver/mip/BbNode.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

namespace vikalp {

/// Comparator: returns true when `a` has LOWER priority than `b`.
/// std::priority_queue is a max-heap, so "lower priority" means popped later.
struct BbNodeCompare {
    bool operator()(const BbNode &a, const BbNode &b) const noexcept {
        // 1. Higher relaxation_bound = lower priority (we want the smallest first)
        if (a.relaxation_bound != b.relaxation_bound)
            return a.relaxation_bound > b.relaxation_bound;
        // 2. Shallower node = lower priority (we prefer deeper for diving)
        if (a.depth != b.depth)
            return a.depth < b.depth;
        // 3. Higher ID = lower priority (deterministic)
        return a.id > b.id;
    }
};

class BbQueue {
public:
    void push(BbNode node) {
        queue_.push(std::move(node));
    }

    [[nodiscard]] BbNode pop() {
        BbNode top = std::move(const_cast<BbNode &>(queue_.top()));
        queue_.pop();
        return top;
    }

    [[nodiscard]] bool empty() const noexcept {
        return queue_.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return queue_.size();
    }

    /// Best (lowest) relaxation bound among all open nodes.
    /// Returns +infinity if queue is empty.
    [[nodiscard]] Scalar best_bound() const noexcept {
        if (queue_.empty()) return Model::infinity();
        return queue_.top().relaxation_bound;
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    std::int64_t nodes_created = 0;
    std::int64_t nodes_processed = 0;
    std::int64_t nodes_pruned = 0;
    std::int64_t nodes_integral = 0;
    std::int64_t nodes_infeasible = 0;
    std::int64_t incumbent_updates = 0;

    [[nodiscard]] std::int64_t next_id() noexcept {
        return nodes_created++;
    }

private:
    std::priority_queue<BbNode, std::vector<BbNode>, BbNodeCompare> queue_;
};

} // namespace vikalp
