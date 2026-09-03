#pragma once

#include "vikalp/contracts/Model.hpp"

#include <limits>
#include <string>
#include <vector>

namespace vikalp {

enum class SolveStatus {
    NotSolved,
    Optimal,
    LocallyOptimal,
    Feasible,
    Infeasible,
    Unbounded,
    IterationLimit,
    NodeLimit,
    TimeLimit,
    NumericalFailure,
    UnsupportedModel,
    InvalidModel
};

struct SolveResult {
    SolveStatus status = SolveStatus::NotSolved;
    std::vector<Scalar> primal_solution;
    std::vector<Scalar> constraint_lower_dual;
    std::vector<Scalar> constraint_upper_dual;
    std::vector<Scalar> variable_lower_dual;
    std::vector<Scalar> variable_upper_dual;
    std::vector<Scalar> primal_ray;
    std::vector<Scalar> dual_ray;
    Scalar objective_value = std::numeric_limits<Scalar>::quiet_NaN();
    Scalar primal_bound = std::numeric_limits<Scalar>::quiet_NaN();
    Scalar dual_bound = std::numeric_limits<Scalar>::quiet_NaN();
    Scalar primal_residual = Model::infinity();
    Scalar stationarity_residual = Model::infinity();
    Scalar complementarity_residual = Model::infinity();
    Scalar integrality_residual = Model::infinity();
    Scalar absolute_gap = Model::infinity();
    Scalar relative_gap = Model::infinity();
    Index iterations = 0;
    Index nodes = 0;
    Scalar solve_seconds = 0.0;
    std::string solver;
    std::string backend;
    bool verified = false;
    std::string message;
};

} // namespace vikalp
