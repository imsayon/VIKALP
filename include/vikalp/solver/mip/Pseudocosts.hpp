#pragma once

// Flow D — Pseudocost branching score tracker.
//
// Pseudocosts estimate the per-unit change in relaxation bound when
// branching on an integer variable. After sufficient observations,
// pseudocost-based branching outperforms most-fractional branching.
//
// Score for variable j:
//   score(j) = (1 - mu) * min(down_cost, up_cost) + mu * max(down_cost, up_cost)
// where mu = 1/6 (product scoring weight, standard in SCIP/Gurobi).

#include "vikalp/contracts/Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace vikalp {

class PseudocostTracker {
public:
    explicit PseudocostTracker(Index num_variables)
        : down_sum_(static_cast<std::size_t>(num_variables), 0.0),
          up_sum_(static_cast<std::size_t>(num_variables), 0.0),
          down_count_(static_cast<std::size_t>(num_variables), 0),
          up_count_(static_cast<std::size_t>(num_variables), 0) {}

    /// Record an observation: branching on variable j in direction down/up
    /// changed the relaxation bound by delta over a fractional distance of frac.
    /// Pseudocost unit = delta / frac.
    void record_down(Index j, Scalar delta, Scalar frac) {
        if (frac > 1e-12) {
            auto idx = static_cast<std::size_t>(j);
            down_sum_[idx] += delta / frac;
            down_count_[idx]++;
        }
    }

    void record_up(Index j, Scalar delta, Scalar frac) {
        if (frac > 1e-12) {
            auto idx = static_cast<std::size_t>(j);
            up_sum_[idx] += delta / frac;
            up_count_[idx]++;
        }
    }

    /// Average pseudocost for branching down on variable j.
    [[nodiscard]] Scalar down_cost(Index j) const noexcept {
        auto idx = static_cast<std::size_t>(j);
        return down_count_[idx] > 0 ? down_sum_[idx] / down_count_[idx] : default_cost_;
    }

    /// Average pseudocost for branching up on variable j.
    [[nodiscard]] Scalar up_cost(Index j) const noexcept {
        auto idx = static_cast<std::size_t>(j);
        return up_count_[idx] > 0 ? up_sum_[idx] / up_count_[idx] : default_cost_;
    }

    /// True if variable j has at least min_reliable observations in both directions.
    [[nodiscard]] bool is_reliable(Index j, int min_reliable = 4) const noexcept {
        auto idx = static_cast<std::size_t>(j);
        return down_count_[idx] >= min_reliable && up_count_[idx] >= min_reliable;
    }

    /// Combined branching score using product scoring.
    /// score = (1-mu)*min(d,u) + mu*max(d,u), where d = down_cost * frac_down,
    /// u = up_cost * frac_up.
    [[nodiscard]] Scalar score(Index j, Scalar frac_down, Scalar frac_up) const noexcept {
        const Scalar d = down_cost(j) * frac_down;
        const Scalar u = up_cost(j) * frac_up;
        constexpr Scalar mu = 1.0 / 6.0;
        return (1.0 - mu) * std::min(d, u) + mu * std::max(d, u);
    }

    /// Number of observations for variable j.
    [[nodiscard]] int total_observations(Index j) const noexcept {
        auto idx = static_cast<std::size_t>(j);
        return down_count_[idx] + up_count_[idx];
    }

    void set_default_cost(Scalar cost) noexcept { default_cost_ = cost; }

private:
    std::vector<Scalar> down_sum_;
    std::vector<Scalar> up_sum_;
    std::vector<int> down_count_;
    std::vector<int> up_count_;
    Scalar default_cost_ = 1.0;  // used when no observations exist
};

} // namespace vikalp
