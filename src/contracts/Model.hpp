#ifndef VIKALP_MODEL_HPP
#define VIKALP_MODEL_HPP

#include <cstdint>
#include <limits>
#include <vector>

namespace vikalk {

// Gate 0 common scalar/index types.
using Index = std::int64_t;
using Scalar = double;

// Variable types supported by the common model contract.
enum class VariableType {
    Continuous,
    Integer
};

// One sparse matrix entry.
// The same representation is used for linear model data.
struct SparseEntry {
    Index row;
    Index column;
    Scalar value;
};

// Canonical optimization model.
//
// All models use minimization internally:
//
//   minimize  c^T x + 1/2 x^T Q x + f_nl(x)
//
// subject to
//
//   lower_constraint <= A x + g_nl(x) <= upper_constraint
//
//   lower_bound <= x <= upper_bound
//
struct Model {
    // Number of variables and constraints.
    Index num_variables = 0;
    Index num_constraints = 0;

    // Linear objective: c^T x
    std::vector<Scalar> objective_linear;

    // Quadratic objective: 1/2 x^T Q x
    std::vector<SparseEntry> objective_quadratic;

    // Linear constraint matrix A.
    std::vector<SparseEntry> constraint_matrix;

    // Constraint bounds:
    // lower_constraint <= A x <= upper_constraint
    std::vector<Scalar> lower_constraint;
    std::vector<Scalar> upper_constraint;

    // Variable bounds:
    // lower_bound <= x <= upper_bound
    std::vector<Scalar> lower_bound;
    std::vector<Scalar> upper_bound;

    // Variable types.
    std::vector<VariableType> variable_type;

    // Returns the default value used for an unbounded side.
    static constexpr Scalar infinity() {
        return std::numeric_limits<Scalar>::infinity();
    }
};

}  // namespace vikalk

#endif  // VIKALP_MODEL_HPP