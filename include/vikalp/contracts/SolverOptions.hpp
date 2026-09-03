#pragma once

#include "vikalp/contracts/Model.hpp"

namespace vikalp {

enum class BackendPreference { Auto,
                               CPU,
                               CUDA };
enum class ContinuousMethod { Auto,
                              PDHG,
                              RevisedSimplex,
                              InteriorPoint };

struct SolverOptions {
    BackendPreference backend = BackendPreference::Auto;
    ContinuousMethod continuous_method = ContinuousMethod::Auto;
    Index iteration_limit = 100'000;
    Index node_limit = 100'000;
    Scalar time_limit_seconds = Model::infinity();
    Scalar primal_tolerance = 1e-7;
    Scalar dual_tolerance = 1e-7;
    Scalar integrality_tolerance = 1e-6;
    Scalar absolute_gap_tolerance = 1e-6;
    Scalar relative_gap_tolerance = 1e-4;
    bool deterministic = true;
};

} // namespace vikalp
